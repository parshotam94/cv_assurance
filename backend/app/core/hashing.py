"""
Cryptographic hashing utilities (SHA-256)
"""
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

def hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()

def hash_file(file_path: Path, chunk_size: int = 65536) -> str:
    """Stream file content and compute SHA-256 to avoid loading large files into RAM."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def canonical_json_dumps(data: Any) -> str:
    """Serialize dictionary or list to canonical JSON string (sorted keys, no extraneous whitespace)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def hash_canonical_json(data: Any) -> str:
    """Compute SHA-256 hash of canonical JSON string representation."""
    canonical_str = canonical_json_dumps(data)
    return hash_bytes(canonical_str.encode("utf-8"))

def hash_numpy_array(arr: np.ndarray) -> str:
    """Compute SHA-256 of numpy array data buffer and shape."""
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()
