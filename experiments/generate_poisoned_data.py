"""
Synthetic Attack Scenario & Poisoned Dataset Generator
Produces reproducible offline attack scenarios:
- Clean baseline dataset
- Label-flipped samples
- Duplicate flooding
- Out-of-Distribution (OOD) injection
- Localized patch trigger backdoor injection
- Source/contributor poisoning
Outputs exact ground-truth metadata for evaluation metrics (Precision, Recall, F1).
"""
import os
import json
import shutil
from typing import Dict, Any
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from backend.app.config import settings

def draw_synthetic_scene(cls_name: str, size=(128, 128)) -> Image.Image:
    """Render a deterministic synthetic computer vision object scene."""
    im = Image.new("RGB", size, color=(40, 50, 60))
    draw = ImageDraw.Draw(im)

    if cls_name == "vehicle":
        # Blue rectangle + wheels
        draw.rectangle([20, 60, 108, 90], fill=(30, 100, 220))
        draw.rectangle([40, 40, 90, 60], fill=(70, 140, 240))
        draw.ellipse([30, 85, 45, 100], fill=(20, 20, 20))
        draw.ellipse([85, 85, 100, 100], fill=(20, 20, 20))
    elif cls_name == "pedestrian":
        # Orange vertical humanoid
        draw.ellipse([58, 25, 70, 37], fill=(230, 120, 40))
        draw.rectangle([54, 38, 74, 80], fill=(200, 90, 30))
        draw.line([58, 80, 54, 110], fill=(150, 70, 20), width=3)
        draw.line([70, 80, 74, 110], fill=(150, 70, 20), width=3)
    elif cls_name == "sign":
        # Red stop-like octagon or triangle
        draw.polygon([(64, 30), (95, 55), (95, 85), (64, 105), (33, 85), (33, 55)], fill=(220, 40, 40))
        draw.rectangle([61, 105, 67, 125], fill=(180, 180, 180))
    else:
        draw.rectangle([30, 30, 98, 98], fill=(100, 100, 100))

    return im

def generate_poisoned_experiment_dataset() -> Dict[str, Any]:
    """Generate clean baseline and attacked COCO dataset with ground truth."""
    np.random.seed(42)
    base_dir = settings.DATASET_DIR / "demo_experiment"
    if base_dir.exists():
        shutil.rmtree(base_dir)

    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    classes = ["vehicle", "pedestrian", "sign"]
    coco_images = []
    coco_annotations = []
    categories = [{"id": i+1, "name": c} for i, c in enumerate(classes)]

    ground_truth = {
        "total_samples": 0,
        "clean_samples": [],
        "label_flip_samples": [],
        "duplicate_flooding_samples": [],
        "ood_samples": [],
        "trigger_samples": [],
        "poisoned_sources": ["source_rogue_3"]
    }

    sample_idx = 1
    ann_idx = 1

    # 1. Generate 30 clean samples (10 each class)
    for c_idx, c_name in enumerate(classes):
        for k in range(10):
            sid = f"sample_{sample_idx:04d}"
            im = draw_synthetic_scene(c_name)
            # Add slight background variation
            arr = np.array(im, dtype=np.int16)
            arr = np.clip(arr + np.random.randint(-15, 15, arr.shape), 0, 255).astype(np.uint8)
            im_var = Image.fromarray(arr)

            fname = f"{sid}.jpg"
            im_var.save(images_dir / fname)

            coco_images.append({
                "id": sample_idx,
                "file_name": fname,
                "width": 128,
                "height": 128,
                "source": "source_verified_1",
                "batch_id": "batch_clean_01",
                "contributor": "alice"
            })
            coco_annotations.append({
                "id": ann_idx,
                "image_id": sample_idx,
                "category_id": c_idx + 1,
                "bbox": [20, 25, 88, 85],
                "area": 7480
            })

            ground_truth["clean_samples"].append(sid)
            sample_idx += 1
            ann_idx += 1

    # 2. Inject 5 Label-Flipped samples (Visual vehicle labelled as pedestrian)
    for k in range(5):
        sid = f"sample_{sample_idx:04d}"
        im = draw_synthetic_scene("vehicle")
        fname = f"{sid}.jpg"
        im.save(images_dir / fname)

        # Deliberately assign category_id 2 (pedestrian) to clear vehicle image
        coco_images.append({
            "id": sample_idx,
            "file_name": fname,
            "width": 128,
            "height": 128,
            "source": "source_rogue_3",
            "batch_id": "batch_poison_99",
            "contributor": "bad_actor_9"
        })
        coco_annotations.append({
            "id": ann_idx,
            "image_id": sample_idx,
            "category_id": 2,  # WRONG LABEL (pedestrian)
            "bbox": [20, 40, 88, 60],
            "area": 5280
        })
        ground_truth["label_flip_samples"].append(sid)
        sample_idx += 1
        ann_idx += 1

    # 3. Inject Duplicate Flooding (5 identical copies of sign image)
    dup_base_img = draw_synthetic_scene("sign")
    for k in range(5):
        sid = f"sample_{sample_idx:04d}"
        fname = f"{sid}.jpg"
        dup_base_img.save(images_dir / fname)  # identical image

        coco_images.append({
            "id": sample_idx,
            "file_name": fname,
            "width": 128,
            "height": 128,
            "source": "source_flood_2",
            "batch_id": "batch_flood_01",
            "contributor": "bob"
        })
        coco_annotations.append({
            "id": ann_idx,
            "image_id": sample_idx,
            "category_id": 3,
            "bbox": [33, 30, 62, 95],
            "area": 5890
        })
        ground_truth["duplicate_flooding_samples"].append(sid)
        sample_idx += 1
        ann_idx += 1

    # 4. Inject 4 OOD Samples (Static snow / noise / pure white image)
    for k in range(4):
        sid = f"sample_{sample_idx:04d}"
        noise_arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        im_ood = Image.fromarray(noise_arr)
        fname = f"{sid}.jpg"
        im_ood.save(images_dir / fname)

        coco_images.append({
            "id": sample_idx,
            "file_name": fname,
            "width": 128,
            "height": 128,
            "source": "source_rogue_3",
            "batch_id": "batch_poison_99",
            "contributor": "bad_actor_9"
        })
        coco_annotations.append({
            "id": ann_idx,
            "image_id": sample_idx,
            "category_id": 1,
            "bbox": [10, 10, 100, 100],
            "area": 10000
        })
        ground_truth["ood_samples"].append(sid)
        sample_idx += 1
        ann_idx += 1

    # 5. Inject 5 Trigger / Backdoor Samples (Vehicle with distinct bright yellow checkerboard in top-left)
    for k in range(5):
        sid = f"sample_{sample_idx:04d}"
        im = draw_synthetic_scene("vehicle")
        draw = ImageDraw.Draw(im)
        # 16x16 bright pattern at Top-Left (0, 0, 16, 16)
        draw.rectangle([0, 0, 16, 16], fill=(255, 255, 0))
        draw.rectangle([4, 4, 12, 12], fill=(0, 0, 0))

        fname = f"{sid}.jpg"
        im.save(images_dir / fname)

        coco_images.append({
            "id": sample_idx,
            "file_name": fname,
            "width": 128,
            "height": 128,
            "source": "source_rogue_3",
            "batch_id": "batch_poison_99",
            "contributor": "bad_actor_9"
        })
        coco_annotations.append({
            "id": ann_idx,
            "image_id": sample_idx,
            "category_id": 1,  # vehicle
            "bbox": [20, 40, 88, 60],
            "area": 5280
        })
        ground_truth["trigger_samples"].append(sid)
        sample_idx += 1
        ann_idx += 1

    ground_truth["total_samples"] = sample_idx - 1

    # Save COCO annotations.json
    coco_json_path = base_dir / "annotations.json"
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
            "info": {"description": "VisionTrust Synthetic Attack Benchmark Dataset"}
        }, f, indent=2)

    # Save ground truth metadata
    gt_path = Path(__file__).resolve().parent / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    # Create zip archive for upload simulation
    zip_path = settings.DATASET_DIR / "demo_experiment.zip"
    shutil.make_archive(str(settings.DATASET_DIR / "demo_experiment"), 'zip', base_dir)

    return {
        "dataset_dir": str(base_dir.resolve()),
        "zip_path": str(zip_path.resolve()),
        "ground_truth_path": str(gt_path.resolve()),
        "total_samples": ground_truth["total_samples"],
        "attacks": {
            "label_flips": len(ground_truth["label_flip_samples"]),
            "duplicates": len(ground_truth["duplicate_flooding_samples"]),
            "ood": len(ground_truth["ood_samples"]),
            "triggers": len(ground_truth["trigger_samples"])
        }
    }

if __name__ == "__main__":
    res = generate_poisoned_experiment_dataset()
    print("Poisoned experiment dataset created:")
    print(json.dumps(res, indent=2))
