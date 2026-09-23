"""
Dataset loader and normalized SampleRecord representation
"""
import os
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SampleRecord(BaseModel):
    sample_id: str
    image_path: str
    source_id: str = "default"
    batch_id: str = "batch_0"
    contributor: str = "anonymous"
    labels: List[str] = Field(default_factory=list)
    annotations: List[Dict[str, Any]] = Field(default_factory=list)  # bboxes: [{class, x, y, w, h}]
    width: int = 0
    height: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

def extract_archive_if_needed(archive_path: Path, target_dir: Path) -> Path:
    """Extract zip or tar archive safely into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()

    if ext == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # Check for zip slip
            for member in zip_ref.namelist():
                dest_path = (target_dir / member).resolve()
                if not str(dest_path).startswith(str(target_dir.resolve())):
                    raise ValueError(f"Zip slip vulnerability detected in member: {member}")
            zip_ref.extractall(target_dir)
        return target_dir
    elif ext in [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:*") as tar_ref:
            for member in tar_ref.getmembers():
                dest_path = (target_dir / member.name).resolve()
                if not str(dest_path).startswith(str(target_dir.resolve())):
                    raise ValueError(f"Tar slip vulnerability detected in member: {member.name}")
            tar_ref.extractall(target_dir)
        return target_dir

    return archive_path

def detect_dataset_format(directory: Path) -> str:
    """Auto-detect if directory contains COCO, YOLO, or standard image directory."""
    # Check for COCO json
    json_files = list(directory.glob("**/*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                snippet = f.read(512)
                if '"images"' in snippet and '"annotations"' in snippet:
                    return "COCO"
        except Exception:
            continue

    # Check for YOLO data.yaml or classes.txt or labels/
    if list(directory.glob("**/data.yaml")) or list(directory.glob("**/data.yml")):
        return "YOLO"
    if list(directory.glob("**/labels")) and list(directory.glob("**/images")):
        return "YOLO"
    if list(directory.glob("**/classes.txt")):
        return "YOLO"

    return "GENERIC_IMAGES"
