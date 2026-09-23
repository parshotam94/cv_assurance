"""
COCO dataset format parser
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
from backend.app.dataset.loader import SampleRecord

def parse_coco_dataset(dataset_dir: Path) -> Tuple[List[SampleRecord], List[str], Dict[str, Any]]:
    """
    Parse a COCO dataset directory containing annotations.json and images.
    Returns (sample_records, class_names, metadata).
    """
    json_files = list(dataset_dir.glob("**/*.json"))
    coco_json_path = None
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "images" in data and "annotations" in data:
                    coco_json_path = jf
                    break
        except Exception:
            continue

    if not coco_json_path:
        raise ValueError("No valid COCO annotations JSON found in dataset directory.")

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Map categories
    categories = coco_data.get("categories", [])
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in categories}
    class_names = sorted(list(set(cat_id_to_name.values())))

    # Map annotations to image_id
    img_to_anns: Dict[Any, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann.get("image_id")
        img_to_anns.setdefault(img_id, []).append(ann)

    # Search for images directory
    image_base_dirs = [
        coco_json_path.parent / "images",
        coco_json_path.parent,
        dataset_dir / "images",
        dataset_dir
    ]

    records: List[SampleRecord] = []
    for img_info in coco_data.get("images", []):
        img_id = img_info.get("id")
        file_name = img_info.get("file_name", "")
        
        # Locate image file
        resolved_img_path = None
        for bdir in image_base_dirs:
            candidate = bdir / file_name
            if candidate.exists() and candidate.is_file():
                resolved_img_path = str(candidate.resolve())
                break
            # Try recursive search if not direct
            matches = list(bdir.glob(f"**/{file_name}"))
            if matches:
                resolved_img_path = str(matches[0].resolve())
                break

        if not resolved_img_path:
            continue

        # Extract dimensions
        w = img_info.get("width", 0)
        h = img_info.get("height", 0)
        if w == 0 or h == 0:
            try:
                with Image.open(resolved_img_path) as im:
                    w, h = im.size
            except Exception:
                pass

        # Extract annotations
        anns = img_to_anns.get(img_id, [])
        labels = []
        normalized_anns = []
        for a in anns:
            cat_name = cat_id_to_name.get(a.get("category_id"), "unknown")
            labels.append(cat_name)
            normalized_anns.append({
                "class_name": cat_name,
                "category_id": a.get("category_id"),
                "bbox": a.get("bbox", []),  # [x, y, width, height]
                "area": a.get("area", 0)
            })

        source_id = img_info.get("source", img_info.get("contributor_id", "source_coco"))
        batch_id = img_info.get("batch_id", "batch_0")
        contributor = img_info.get("contributor", "unknown")

        records.append(SampleRecord(
            sample_id=Path(file_name).stem if file_name else str(img_id),
            image_path=resolved_img_path,
            source_id=source_id,
            batch_id=batch_id,
            contributor=contributor,
            labels=sorted(list(set(labels))),
            annotations=normalized_anns,
            width=w,
            height=h,
            metadata=img_info
        ))

    metadata = {
        "format": "COCO",
        "categories_count": len(class_names),
        "total_images": len(records),
        "total_annotations": len(coco_data.get("annotations", [])),
        "info": coco_data.get("info", {})
    }
    return records, class_names, metadata
