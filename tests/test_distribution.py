"""
Unit Tests for Distribution Shift and Statistical Drift
"""
from PIL import Image
import numpy as np
import pytest

from backend.app.distribution.embeddings import extract_image_distribution_metrics
from backend.app.distribution.drift import calculate_psi, compute_drift_metrics
from backend.app.distribution.environmental_shift import analyze_environmental_shift

def test_extract_features(test_image):
    feats = extract_image_distribution_metrics(str(test_image))
    assert "r_mean" in feats
    assert "g_mean" in feats
    assert "b_mean" in feats
    assert "sharpness" in feats
    assert "high_freq_energy" in feats

def test_calculate_psi_identical():
    data = np.random.normal(100, 15, 100)
    psi = calculate_psi(data, data)
    assert psi < 0.1  # Negligible drift

def test_calculate_psi_shifted():
    ref = np.random.normal(50, 10, 100)
    cand = np.random.normal(150, 10, 100)
    drift_res = compute_drift_metrics(list(ref), list(cand))
    assert drift_res["is_significant_drift"] is True
    assert drift_res["ks_statistic"] > 0.5

def test_environmental_shift_detection(tmp_path):
    ref_paths = []
    cand_paths = []

    # Reference images (normal brightness)
    for i in range(10):
        p_ref = tmp_path / f"ref_{i}.png"
        arr_ref = np.full((64, 64, 3), 150, dtype=np.uint8)
        Image.fromarray(arr_ref).save(p_ref)
        ref_paths.append(str(p_ref))

    # Candidate images (severe darkness -60% + noise)
    for i in range(10):
        p_cand = tmp_path / f"cand_{i}.png"
        arr_cand = np.full((64, 64, 3), 40, dtype=np.uint8)
        Image.fromarray(arr_cand).save(p_cand)
        cand_paths.append(str(p_cand))

    shift_res = analyze_environmental_shift(ref_paths, cand_paths)
    assert shift_res["overall_shift_detected"] is True
    assert shift_res["classification"] in ["ANOMALOUS", "SUSPICIOUS", "OPERATIONAL_DRIFT"]
