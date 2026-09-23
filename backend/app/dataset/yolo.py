"""
YOLO dataset format parser
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
from backend.app.dataset.loader import SampleRecord

def parse_yolo_dataset(dataset_dir: Path) -> Tuple[List[SampleRecord], List[str], Dict[str, Any]]:
    """
    Parse a YOLO dataset directory containing images, labels, and classes.txt or data.yaml.
    """
    class_names: List[str] = []

    # 1. Look for classes.txt
    classes_files = list(dataset_dir.glob("**/classes.txt"))
    if classes_files:
        with open(classes_files[0], "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Look for data.yaml / data.yml if not found
    if not class_names:
        yaml_files = list(dataset_dir.glob("**/data.yaml")) + list(dataset_dir.glob("**/data.yml"))
        if yaml_files:
            with open(yaml_files[0], "r", encoding="utf-8") as f:
                content = f.read()
                # Simple parsing without external PyYAML dependency if possible, or using basic regex
                names_match = re.search(r'names:\s*\[(.*?)\]', content, re.DOTALL)
                if names_match:
                    raw_names = names_match.group(1).split(",")
                    class_names = [n.strip().strip("'\"") for n in raw_names if n.strip()]
                else:
                    lines = content.splitlines()
                    in_names = False
                    for line in lines:
                        if "names:" in line:
                            in_names = True
                            continue
                        if in_names:
                            if line.strip().startswith("-"):
                                class_names.append(line.strip().lstrip("- ").strip("'\""))
                            elif ":" in line and not line.strip().startswith("#"):
                                # e.g. 0: person
                                parts = line.strip().split(":", 1)
                                if len(parts) == 2 and parts[0].strip().isdigit():
                                    class_names.append(parts[1].strip().strip("'\""))
                            else:
                                break

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(dataset_dir.glob(f"**/*{ext}")))

    records: List[SampleRecord] = []
    annotation_count = 0

    for img_path in image_files:
        # Corresponding label file typically in labels/ directory or same dir with .txt
        label_candidate = img_path.with_suffix(".txt")
        if not label_candidate.exists():
            # Try replacing /images/ with /labels/
            parts = list(img_path.parts)
            if "images" in parts:
                idx = parts.index("images")
                parts[idx] = "labels"
                alt_candidate = Path(*parts).with_suffix(".txt")
                if alt_candidate.exists():
                    label_candidate = alt_candidate

        w, h = 0, 0
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue

        sample_labels = []
        annotations = []

        if label_candidate.exists():
            with open(label_candidate, "r", encoding="utf-8") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    if len(tokens) >= 5:
                        class_id = int(tokens[0])
                        # Normalize class name
                        if class_id < len(class_names):
                            cname = class_names[class_id]
                        else:
                            cname = f"class_{class_id}"
                            if cname not in class_names:
                                class_names.append(cname)

                        sample_labels.append(cname)
                        x_center, y_center, bbox_w, bbox_h = map(float, tokens[1:5])
                        # Convert normalized YOLO to pixel coords
                        x = (x_center - bbox_w / 2.0) * w
                        y = (y_center - bbox_h / 2.0) * h
                        box_pixel_w = bbox_w * w
                        box_pixel_h = bbox_h * h

                        annotations.append({
                            "class_name": cname,
                            "class_id": class_id,
                            "bbox": [round(x, 1), round(y, 1), round(box_pixel_w, 1), round(box_pixel_h, 1)],
                            "normalized": [x_center, y_center, bbox_w, bbox_h]
                        })
                        annotation_count += 1

        # Check metadata from path if available (e.g. source_1/batch_2)
        source_id = "default_source"
        batch_id = "batch_0"
        for part in img_path.parts:
            if "source_" in part.lower() or "contributor_" in part.lower():
                source_id = part
            if "batch_" in part.lower():
                batch_id = part

        records.append(SampleRecord(
            sample_id=img_path.stem,
            image_path=str(img_path.resolve()),
            source_id=source_id,
            batch_id=batch_id,
            labels=sorted(list(set(sample_labels))),
            annotations=annotations,
            width=w,
            height=h,
            metadata={"source_file": img_path.name}
        ))

    metadata = {
        "format": "YOLO",
        "classes_count": len(class_names),
        "total_images": len(records),
        "total_annotations": annotation_count
    }
    return records, sorted(list(set(class_names))), metadata
