import os
from cryptography.fernet import Fernet
from sqlalchemy import TypeDecorator, String
from app.core.config import settings

class FieldEncryption:
    """
    Handles field-level encryption and decryption using Fernet.
    It reads the encryption key from the application settings.
    """
    def __init__(self):
        key = settings.FIELD_ENCRYPTION_KEY
        if not key:
            raise ValueError("FIELD_ENCRYPTION_KEY is not set in the environment.")
        self.cipher_suite = Fernet(key.encode())

    def encrypt_field(self, plaintext: str) -> str:
        """Encrypts a plaintext string."""
        if not isinstance(plaintext, str):
            plaintext = str(plaintext)
        return self.cipher_suite.encrypt(plaintext.encode()).decode('utf-8')

    def decrypt_field(self, ciphertext: str) -> str:
        """Decrypts a ciphertext string."""
        if not isinstance(ciphertext, str):
            ciphertext = str(ciphertext)
        return self.cipher_suite.decrypt(ciphertext.encode()).decode('utf-8')

# Global instance to be used throughout the application
field_encryption = FieldEncryption()

class EncryptedType(TypeDecorator):
    """
    A SQLAlchemy TypeDecorator to automatically encrypt and decrypt a field.
    """
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Encrypt the value before saving it to the database."""
        if value is not None:
            return field_encryption.encrypt_field(value)
        return value

    def process_result_value(self, value, dialect):
        """Decrypt the value after retrieving it from the database."""
        if value is not None:
            return field_encryption.decrypt_field(value)
        return value