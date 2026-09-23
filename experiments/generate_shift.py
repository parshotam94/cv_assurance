"""
Synthetic Distribution Shift Dataset Generator
Creates baseline reference and candidate datasets with controlled environmental illumination & sensor shifts.
"""
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from backend.app.config import settings
from experiments.generate_poisoned_data import draw_synthetic_scene

def generate_shift_experiment_datasets():
    """Build Reference and Shifted Candidate datasets."""
    ref_dir = settings.DATASET_DIR / "demo_reference_domain"
    cand_dir = settings.DATASET_DIR / "demo_shifted_domain"

    for d in [ref_dir, cand_dir]:
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)

    classes = ["vehicle", "pedestrian", "sign"]
    
    # 1. Reference: Normal daylight conditions (Brightness ~0.55, sharp)
    for i in range(25):
        cname = classes[i % 3]
        im = draw_synthetic_scene(cname)
        fname = f"ref_{i:03d}.jpg"
        im.save(ref_dir / "images" / fname)
        with open(ref_dir / "labels" / f"ref_{i:03d}.txt", "w") as f:
            f.write(f"{i % 3} 0.5 0.5 0.7 0.7\n")

    # 2. Candidate: Night/Low-Light + High Noise Shift (Brightness ~0.2, high contrast, noise)
    for i in range(25):
        cname = classes[i % 3]
        im = draw_synthetic_scene(cname)
        # Shift: reduce brightness by 60% and add noise
        enhancer = ImageEnhance.Brightness(im)
        dark_im = enhancer.enhance(0.35)
        arr = np.array(dark_im, dtype=np.int16)
        noise = np.random.normal(0, 18, arr.shape).astype(np.int16)
        shifted_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        shifted_im = Image.fromarray(shifted_arr)

        fname = f"cand_{i:03d}.jpg"
        shifted_im.save(cand_dir / "images" / fname)
        with open(cand_dir / "labels" / f"cand_{i:03d}.txt", "w") as f:
            f.write(f"{i % 3} 0.5 0.5 0.7 0.7\n")

    # Zip archives
    shutil.make_archive(str(ref_dir), 'zip', ref_dir)
    shutil.make_archive(str(cand_dir), 'zip', cand_dir)

    return {
        "reference_zip": str(ref_dir.with_suffix(".zip").resolve()),
        "candidate_zip": str(cand_dir.with_suffix(".zip").resolve())
    }

if __name__ == "__main__":
    paths = generate_shift_experiment_datasets()
    print("Distribution shift datasets created:")
    print(paths)
