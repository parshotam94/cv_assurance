"""
Initialization script for VisionTrust Assurance
Initializes database tables, generates local Ed25519 signing keys, and verifies directory structure.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.database.database import init_db, SessionLocal
from backend.app.core.signatures import generate_or_load_keypair, get_public_key_hex
from backend.app.core.audit import log_audit_event

def initialize_system():
    print("=" * 60)
    print("VisionTrust Assurance - System Initialization")
    print("=" * 60)

    # 1. Directories
    print("[1/4] Verifying and creating required directories...")
    for d in [
        settings.DATA_DIR,
        settings.UPLOAD_DIR,
        settings.DATASET_DIR,
        settings.MODEL_DIR,
        settings.INFERENCE_DIR,
        settings.REFERENCE_DIR,
        settings.REPORT_DIR,
        settings.AUDIT_DIR,
        settings.KEY_STORAGE_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)
    print("  [OK] Directory structure confirmed.")

    # 2. Database
    print("[2/4] Initializing SQLite database tables...")
    init_db()
    print(f"  [OK] Database initialized at: {settings.DATABASE_URL}")

    # 3. Cryptographic Keys
    print("[3/4] Generating/verifying local Ed25519 cryptographic keypair...")
    priv, pub = generate_or_load_keypair()
    pub_hex = get_public_key_hex(pub)
    print(f"  [OK] Ed25519 Host Public Key: {pub_hex[:32]}...")

    # 4. Genesis Audit Event
    print("[4/4] Logging cryptographic genesis audit event...")
    db = SessionLocal()
    try:
        log_audit_event(
            db,
            action="SYSTEM_INITIALIZE",
            asset="VisionTrustHost",
            actor="system",
            metadata={"version": settings.APP_VERSION, "air_gapped": settings.AIR_GAPPED_MODE}
        )
        print("  [OK] Genesis audit block successfully recorded and linked.")
    finally:
        db.close()

    print("\nInitialization complete! VisionTrust Assurance is ready for offline operation.")

if __name__ == "__main__":
    initialize_system()
