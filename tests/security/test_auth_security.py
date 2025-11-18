"""
Security Tests for Authentication Module
Tests all critical security functions in app/web/auth.py
"""

import pytest
import os
from datetime import datetime, timedelta
from fastapi import HTTPException
from passlib.context import CryptContext

# Set test environment
os.environ["WEB_SECRET_KEY"] = "test_secret_key_for_unit_tests_32chars"
os.environ["WEB_ADMIN_USERNAME"] = "test_admin"
os.environ["WEB_ADMIN_PASSWORD"] = "test_password"

from app.web.auth import (
    create_access_token,
    decode_access_token,
    verify_password,
    get_password_hash,
    authenticate_user,
    USERS_DB
)


class TestPasswordSecurity:
    """Test password hashing and verification"""

    def test_password_hashing_uses_bcrypt(self):
        """Verify bcrypt is used for password hashing"""
        password = "test_password_123"
        hashed = get_password_hash(password)

        # Bcrypt hashes start with $2b$ or $2a$
        assert hashed.startswith("$2"), "Password hash must use bcrypt"
        assert len(hashed) == 60, "Bcrypt hash length should be 60 characters"

    def test_password_verification_correct(self):
        """Test that correct password verifies successfully"""
        password = "correct_password"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed), "Correct password should verify"

    def test_password_verification_incorrect(self):
        """Test that incorrect password fails verification"""
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = get_password_hash(password)

        assert not verify_password(wrong_password, hashed), "Wrong password should not verify"

    def test_password_hash_unique(self):
        """Test that same password produces different hashes (salt is used)"""
        password = "same_password"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        assert hash1 != hash2, "Same password should produce different hashes (salt)"
        assert verify_password(password, hash1), "First hash should verify"
        assert verify_password(password, hash2), "Second hash should verify"

    def test_empty_password_handling(self):
        """Test handling of empty passwords"""
        empty = ""
        hashed = get_password_hash(empty)

        assert verify_password(empty, hashed), "Empty password should hash and verify"

    def test_special_characters_in_password(self):
        """Test passwords with special characters"""
        special_password = "P@ssw0rd!@#$%^&*()_+-={}[]|:;<>,.?/~`"
        hashed = get_password_hash(special_password)

        assert verify_password(special_password, hashed), "Special chars should work"

    def test_unicode_password(self):
        """Test passwords with unicode characters"""
        unicode_password = "пароль密码🔐"
        hashed = get_password_hash(unicode_password)

        assert verify_password(unicode_password, hashed), "Unicode passwords should work"


class TestJWTSecurity:
    """Test JWT token creation and validation"""

    def test_token_creation(self):
        """Test basic token creation"""
        data = {"sub": "test_user"}
        token = create_access_token(data)

        assert isinstance(token, str), "Token should be a string"
        assert len(token) > 100, "JWT should be reasonably long"
        assert token.count(".") == 2, "JWT should have 3 parts separated by dots"

    def test_token_decoding(self):
        """Test token decoding with valid token"""
        data = {"sub": "test_user", "role": "admin"}
        token = create_access_token(data)

        decoded = decode_access_token(token)

        assert decoded["sub"] == "test_user", "Subject should match"
        assert decoded["role"] == "admin", "Role should match"
        assert "exp" in decoded, "Token should have expiration"

    def test_token_expiration(self):
        """Test that expired tokens are rejected"""
        data = {"sub": "test_user"}
        # Create token that expired 1 hour ago
        expires_delta = timedelta(hours=-1)
        token = create_access_token(data, expires_delta=expires_delta)

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)

        assert exc_info.value.status_code == 401, "Should return 401 for expired token"
        assert "expired" in exc_info.value.detail.lower(), "Error should mention expiration"

    def test_token_tampering(self):
        """Test that tampered tokens are rejected"""
        data = {"sub": "test_user"}
        token = create_access_token(data)

        # Tamper with token
        tampered = token[:-5] + "XXXXX"

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(tampered)

        assert exc_info.value.status_code == 401, "Should return 401 for tampered token"

    def test_token_contains_expiration(self):
        """Test that tokens include expiration time"""
        data = {"sub": "test_user"}
        token = create_access_token(data)
        decoded = decode_access_token(token)

        assert "exp" in decoded, "Token must include expiration"
        exp_time = datetime.fromtimestamp(decoded["exp"])
        now = datetime.utcnow()

        # Should expire in approximately 24 hours (default)
        time_diff = (exp_time - now).total_seconds()
        assert 23 * 3600 < time_diff < 25 * 3600, "Token should expire in ~24 hours"

    def test_custom_expiration(self):
        """Test custom token expiration"""
        data = {"sub": "test_user"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta=expires_delta)
        decoded = decode_access_token(token)

        exp_time = datetime.fromtimestamp(decoded["exp"])
        now = datetime.utcnow()

        time_diff = (exp_time - now).total_seconds()
        assert 29 * 60 < time_diff < 31 * 60, "Token should expire in ~30 minutes"

    def test_token_without_sub(self):
        """Test token creation without subject"""
        data = {"role": "admin"}  # No 'sub' field
        token = create_access_token(data)
        decoded = decode_access_token(token)

        assert decoded["role"] == "admin", "Token should contain data"
        assert "exp" in decoded, "Token should have expiration"


class TestUserAuthentication:
    """Test user authentication logic"""

    def test_authenticate_valid_user(self):
        """Test authentication with valid credentials"""
        # Note: test_admin/test_password is set up in environment
        user = authenticate_user("test_admin", "test_password")

        assert user is not None, "Valid credentials should authenticate"
        assert user.username == "test_admin", "Username should match"
        assert user.role == "admin", "Role should match"

    def test_authenticate_invalid_username(self):
        """Test authentication with non-existent username"""
        user = authenticate_user("nonexistent_user", "any_password")

        assert user is None, "Non-existent user should not authenticate"

    def test_authenticate_invalid_password(self):
        """Test authentication with wrong password"""
        user = authenticate_user("test_admin", "wrong_password")

        assert user is None, "Wrong password should not authenticate"

    def test_authenticate_empty_credentials(self):
        """Test authentication with empty credentials"""
        user1 = authenticate_user("", "")
        user2 = authenticate_user("test_admin", "")
        user3 = authenticate_user("", "test_password")

        assert user1 is None, "Empty username should not authenticate"
        assert user2 is None, "Empty password should not authenticate"
        assert user3 is None, "Empty username with password should not authenticate"

    def test_case_sensitive_username(self):
        """Test that username is case-sensitive"""
        user1 = authenticate_user("test_admin", "test_password")
        user2 = authenticate_user("TEST_ADMIN", "test_password")

        assert user1 is not None, "Exact case should work"
        assert user2 is None, "Wrong case should not work"

    def test_user_database_structure(self):
        """Test that user database has proper structure"""
        assert isinstance(USERS_DB, dict), "USERS_DB should be a dictionary"
        assert "test_admin" in USERS_DB, "Test admin should exist"

        user = USERS_DB["test_admin"]
        assert hasattr(user, "username"), "User should have username"
        assert hasattr(user, "hashed_password"), "User should have hashed_password"
        assert hasattr(user, "role"), "User should have role"


class TestSecurityProperties:
    """Test overall security properties"""

    def test_passwords_never_stored_plaintext(self):
        """Verify no plain-text passwords in database"""
        for username, user in USERS_DB.items():
            # Bcrypt hashes are 60 chars and start with $2
            assert len(user.hashed_password) == 60, f"User {username} password not properly hashed"
            assert user.hashed_password.startswith("$2"), f"User {username} not using bcrypt"

    def test_jwt_secret_not_default(self):
        """Test that JWT secret is not the default value in production"""
        from app.web.auth import SECRET_KEY

        # In tests it's okay to use test key
        # In production, this should never be the insecure default
        assert SECRET_KEY != "INSECURE_DEFAULT_CHANGE_ME_IN_PRODUCTION", \
            "Production should never use default JWT secret"

    def test_bcrypt_rounds_sufficient(self):
        """Test that bcrypt uses sufficient rounds"""
        password = "test"
        hashed = get_password_hash(password)

        # Extract rounds from hash (format: $2b$rounds$...)
        parts = hashed.split("$")
        rounds = int(parts[2])

        assert rounds >= 12, "Bcrypt should use at least 12 rounds"

    def test_timing_attack_resistance_password(self):
        """Test that password verification time is consistent"""
        import time

        password = "test_password"
        hashed = get_password_hash(password)

        # Time correct password
        start = time.time()
        verify_password(password, hashed)
        time_correct = time.time() - start

        # Time incorrect password
        start = time.time()
        verify_password("wrong", hashed)
        time_incorrect = time.time() - start

        # Times should be similar (bcrypt is slow and constant-time)
        time_ratio = max(time_correct, time_incorrect) / min(time_correct, time_incorrect)
        assert time_ratio < 1.5, "Password verification should be constant-time"


class TestInputSanitization:
    """Test input validation and sanitization"""

    def test_sql_injection_in_username(self):
        """Test that SQL injection attempts in username fail safely"""
        sql_injection = "admin' OR '1'='1"
        user = authenticate_user(sql_injection, "password")

        assert user is None, "SQL injection should not bypass authentication"

    def test_xss_in_username(self):
        """Test that XSS attempts in username are handled"""
        xss_payload = "<script>alert('XSS')</script>"
        user = authenticate_user(xss_payload, "password")

        assert user is None, "XSS payload should not authenticate"

    def test_very_long_password(self):
        """Test handling of excessively long passwords"""
        long_password = "A" * 10000
        hashed = get_password_hash(long_password)

        # Should hash but verify correctly
        assert verify_password(long_password, hashed), "Long passwords should work"

    def test_null_bytes_in_password(self):
        """Test handling of null bytes in passwords"""
        password_with_null = "pass\x00word"
        hashed = get_password_hash(password_with_null)

        assert verify_password(password_with_null, hashed), "Null bytes should be handled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
