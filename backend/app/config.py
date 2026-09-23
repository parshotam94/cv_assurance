"""
VisionTrust Assurance - Configuration Module
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    APP_NAME: str = "VisionTrust Assurance"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "production"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Offline Air-Gapped Enforcement
    AIR_GAPPED_MODE: bool = True
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    DATASET_DIR: Path = DATA_DIR / "datasets"
    MODEL_DIR: Path = DATA_DIR / "models"
    INFERENCE_DIR: Path = DATA_DIR / "inference"
    REFERENCE_DIR: Path = DATA_DIR / "reference"
    REPORT_DIR: Path = DATA_DIR / "reports"
    CHART_DIR: Path = DATA_DIR / "reports" / "charts"
    AUDIT_DIR: Path = DATA_DIR / "audit"
    KEY_STORAGE_DIR: Path = DATA_DIR / "keys"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'visiontrust.db'}"
    
    # Cryptographic Algorithms
    HASH_ALGORITHM: str = "sha256"
    SIGNATURE_ALGORITHM: str = "ed25519"
    
    # Assurance Thresholds (Configurable)
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.92
    OOD_PERCENTILE_THRESHOLD: float = 0.95
    TRIGGER_PATCH_SIMILARITY_THRESHOLD: float = 0.85
    TRIGGER_MIN_CLUSTER_SIZE: int = 3
    
    RISK_REVIEW_THRESHOLD: float = 40.0
    RISK_QUARANTINE_THRESHOLD: float = 75.0
    
    BEHAVIOURAL_BATTERY_SAMPLES: int = 60
    
    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 500

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure required directories exist
for path in [
    settings.DATA_DIR,
    settings.UPLOAD_DIR,
    settings.DATASET_DIR,
    settings.MODEL_DIR,
    settings.INFERENCE_DIR,
    settings.REFERENCE_DIR,
    settings.REPORT_DIR,
    settings.CHART_DIR,
    settings.AUDIT_DIR,
    settings.KEY_STORAGE_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
