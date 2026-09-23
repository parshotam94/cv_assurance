"""
Security utilities: path traversal protection, file validation, sanitization
"""
import os
import re
from pathlib import Path
from typing import Set
from fastapi import HTTPException, status
from backend.app.config import settings

ALLOWED_EXTENSIONS: Set[str] = {
    # Archives & Datasets
    ".zip", ".tar", ".gz", ".json", ".yaml", ".yml", ".txt",
    # Images
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    # Models
    ".onnx", ".pt", ".pth", ".ts", ".bin",
    # Reports
    ".pdf", ".csv"
}

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal or script injection."""
    clean = os.path.basename(filename)
    clean = re.sub(r'[^a-zA-Z0-9_\.\-]', '_', clean)
    if not clean:
        clean = "unnamed_asset"
    return clean

def validate_file_extension(filename: str, allowed: Set[str] = ALLOWED_EXTENSIONS) -> str:
    """Ensure the file has an approved extension."""
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(list(allowed))}"
        )
    return ext

def safe_join_paths(base: Path, *paths: str) -> Path:
    """Safely resolve child path under base, preventing directory traversal."""
    resolved_base = base.resolve()
    target = resolved_base.joinpath(*paths).resolve()
    if not str(target).startswith(str(resolved_base)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Illegal path traversal attempt detected."
        )
    return target
