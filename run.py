"""
Unified entry point to run VisionTrust Assurance application
"""
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.APP_NAME} in OFFLINE AIR-GAPPED mode...")
    print(f"Local URL: http://{settings.HOST}:{settings.PORT}")
    print(f"API Docs:  http://{settings.HOST}:{settings.PORT}/docs")
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )
