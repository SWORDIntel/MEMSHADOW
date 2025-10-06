# Per PHASE2.MD, this file defines the recommended TLS configuration.
# The application of these settings will be handled at the web server level (e.g., Gunicorn).

# Note: In a real production environment, the certificate and key paths would be
# managed by a secure deployment process and should not be hardcoded.
# The paths below are placeholders as defined in the documentation.

TLS_CONFIG = {
    "minimum_version": "TLSv1.3",
    "cipher_suites": [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256",
    ],
    "certificate_path": "/etc/memshadow/certs/server.crt",
    "key_path": "/etc/memshadow/certs/server.key",
    "client_auth": "optional",
    "session_tickets": False,
}

# For Gunicorn, these settings would translate to command-line arguments like:
# --ssl-version=TLSv1.3
# --ciphers='TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256'
# --keyfile=/etc/memshadow/certs/server.key
# --certfile=/etc/memshadow/certs/server.crt