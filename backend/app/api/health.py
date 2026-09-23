"""
Health and System Status API Endpoints
"""
import sys
import platform
from fastapi import APIRouter
from backend.app.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health")
def get_system_health():
    """Returns local system health, air-gap status, and ML runtime availability."""
    import torch
    import onnxruntime as ort

    return {
        "status": "HEALTHY",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "air_gapped": settings.AIR_GAPPED_MODE,
        "environment": settings.APP_ENV,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ml_runtimes": {
            "pytorch": torch.__version__,
            "onnxruntime": ort.__version__,
            "cuda_available": torch.cuda.is_available()
        },
        "network_access": "DISABLED (STRICT OFFLINE)"
    }
