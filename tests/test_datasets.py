"""
Unit and Integration Tests for Dataset Integrity Engines
"""
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from backend.app.dataset.coco import parse_coco_dataset
from backend.app.dataset.validation import validate_dataset_records
from backend.app.dataset.duplicates import detect_near_duplicates
from backend.app.dataset.label_analysis import analyze_labels_and_mislabelling
from backend.app.dataset.ood import detect_out_of_distribution
from backend.app.dataset.poisoning import detect_trigger_backdoors
from backend.app.dataset.source_analysis import aggregate_source_risks
from backend.app.dataset.loader import SampleRecord

def test_parse_coco_dataset(coco_dataset_dir):
    records, class_names, meta = parse_coco_dataset(coco_dataset_dir)
    assert len(records) == 10
    assert "pedestrian" in class_names
    assert "vehicle" in class_names
    assert meta["total_images"] == 10

def test_validate_dataset_records(coco_dataset_dir):
    records, _, _ = parse_coco_dataset(coco_dataset_dir)
    valid, issues = validate_dataset_records(records)
    assert len(valid) == 10
    assert len(issues) == 0

def test_detect_near_duplicates(tmp_path):
    img_a = tmp_path / "img_a.png"
    img_b = tmp_path / "img_b.png"
    img_c = tmp_path / "img_c.png"

    # A and B are identical
    arr_dup = np.full((64, 64, 3), 120, dtype=np.uint8)
    # C is completely different noise
    arr_diff = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    Image.fromarray(arr_dup).save(img_a)
    Image.fromarray(arr_dup).save(img_b)
    Image.fromarray(arr_diff).save(img_c)

    records = [
        SampleRecord(sample_id="s1", image_path=str(img_a), width=64, height=64),
        SampleRecord(sample_id="s2", image_path=str(img_b), width=64, height=64),
        SampleRecord(sample_id="s3", image_path=str(img_c), width=64, height=64),
    ]

    clusters, sample_map = detect_near_duplicates(records, similarity_threshold=0.90)
    assert len(clusters) >= 1
    assert "s1" in sample_map
    assert "s2" in sample_map
    assert "s3" not in sample_map

def test_ood_detection(tmp_path):
    records = []
    # 20 normal images
    for i in range(20):
        p = tmp_path / f"norm_{i}.png"
        arr = np.random.normal(128, 10, (64, 64, 3)).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(p)
        records.append(SampleRecord(sample_id=f"norm_{i}", image_path=str(p), width=64, height=64))

    # 1 stark outlier (solid black)
    p_out = tmp_path / "outlier.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(p_out)
    records.append(SampleRecord(sample_id="outlier_01", image_path=str(p_out), width=64, height=64))

    summary, outliers = detect_out_of_distribution(records, percentile_threshold=0.90)
    assert len(outliers) >= 1
    assert any(o["sample_id"] == "outlier_01" for o in outliers)

def test_source_aggregation():
    records = [
        SampleRecord(sample_id="s1", image_path="p1", source_id="src_rogue", contributor="eve"),
        SampleRecord(sample_id="s2", image_path="p2", source_id="src_rogue", contributor="eve"),
        SampleRecord(sample_id="s3", image_path="p3", source_id="src_clean", contributor="alice"),
    ]
    # Mark s1 and s2 as anomalous
    source_results = aggregate_source_risks(
        records,
        duplicates_map={"s1": ["s2"], "s2": ["s1"]},
        mislabelling_samples=[],
        ood_samples=[],
        trigger_samples=[]
    )
    rogue = next(s for s in source_results if s["source_id"] == "src_rogue")
    clean = next(s for s in source_results if s["source_id"] == "src_clean")
    assert rogue["risk_level"] in ["CRITICAL", "HIGH"]
    assert clean["risk_level"] == "LOW"
