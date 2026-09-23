"""
Dataset image and annotation integrity validation
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
from backend.app.dataset.loader import SampleRecord

def validate_dataset_records(records: List[SampleRecord]) -> Tuple[List[SampleRecord], List[Dict[str, Any]]]:
    """
    Validate that image files exist, can be parsed by PIL, have valid dimensions,
    and annotations are within bounds.
    """
    valid_records: List[SampleRecord] = []
    validation_issues: List[Dict[str, Any]] = []

    for r in records:
        path = Path(r.image_path)
        if not path.exists():
            validation_issues.append({
                "sample_id": r.sample_id,
                "severity": "HIGH",
                "issue_type": "MISSING_IMAGE_FILE",
                "details": f"File does not exist: {r.image_path}"
            })
            continue

        try:
            with Image.open(path) as im:
                im.verify()  # verify format/corruption
            with Image.open(path) as im:
                w, h = im.size
                if w <= 0 or h <= 0:
                    validation_issues.append({
                        "sample_id": r.sample_id,
                        "severity": "HIGH",
                        "issue_type": "ZERO_DIMENSION",
                        "details": f"Invalid image dimensions: {w}x{h}"
                    })
                    continue
                r.width = w
                r.height = h
        except Exception as e:
            validation_issues.append({
                "sample_id": r.sample_id,
                "severity": "HIGH",
                "issue_type": "CORRUPTED_IMAGE",
                "details": f"Failed to load image: {str(e)}"
            })
            continue

        # Check bbox boundaries
        bbox_issue = False
        for ann in r.annotations:
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    validation_issues.append({
                        "sample_id": r.sample_id,
                        "severity": "MEDIUM",
                        "issue_type": "INVALID_BBOX_SIZE",
                        "details": f"Non-positive bbox dimensions: {bw}x{bh}"
                    })
                    bbox_issue = True
                elif x < -5 or y < -5 or (x + bw) > (r.width + 10) or (y + bh) > (r.height + 10):
                    validation_issues.append({
                        "sample_id": r.sample_id,
                        "severity": "LOW",
                        "issue_type": "OUT_OF_BOUNDS_BBOX",
                        "details": f"Bbox [{x}, {y}, {bw}, {bh}] exceeds image bounds [{r.width}, {r.height}]"
                    })

        valid_records.append(r)

    return valid_records, validation_issues
