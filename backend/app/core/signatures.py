"""
Offline Cryptographic Signing and Verification using Ed25519
"""
from pathlib import Path
from typing import Tuple
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from backend.app.config import settings

KEY_DIR = settings.KEY_STORAGE_DIR
PRIVATE_KEY_FILE = KEY_DIR / "ed25519_private.pem"
PUBLIC_KEY_FILE = KEY_DIR / "ed25519_public.pem"

def generate_or_load_keypair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    """Generate a local Ed25519 keypair if not present, or load existing."""
    if PRIVATE_KEY_FILE.exists() and PUBLIC_KEY_FILE.exists():
        with open(PRIVATE_KEY_FILE, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        with open(PUBLIC_KEY_FILE, "rb") as f:
            public_key = serialization.load_pem_public_key(f.read())
        return private_key, public_key

    # Generate new Ed25519 keypair
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    # Save private key
    with open(PRIVATE_KEY_FILE, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        )

    # Save public key
    with open(PUBLIC_KEY_FILE, "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )

    return private_key, public_key

def sign_data(data: bytes, private_key: ed25519.Ed25519PrivateKey = None) -> str:
    """Sign bytes using local Ed25519 private key. Returns signature as hex string."""
    if private_key is None:
        private_key, _ = generate_or_load_keypair()
    signature_bytes = private_key.sign(data)
    return signature_bytes.hex()

def verify_signature(data: bytes, signature_hex: str, public_key: ed25519.Ed25519PublicKey = None, public_key_hex: str = None) -> bool:
    """Verify Ed25519 signature over data bytes."""
    try:
        sig_bytes = bytes.fromhex(signature_hex)
        if public_key is None and public_key_hex is not None:
            raw_pub_bytes = bytes.fromhex(public_key_hex)
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(raw_pub_bytes)
        elif public_key is None:
            _, public_key = generate_or_load_keypair()

        public_key.verify(sig_bytes, data)
        return True
    except (InvalidSignature, ValueError, Exception):
        return False

def get_public_key_hex(public_key: ed25519.Ed25519PublicKey = None) -> str:
    """Retrieve raw 32-byte public key as hex string."""
    if public_key is None:
        _, public_key = generate_or_load_keypair()
    raw_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    return raw_bytes.hex()
