"""
Unit and Integration Tests for Model Assurance Engines
"""
from pathlib import Path
import numpy as np
import pytest

from backend.app.model.loader import get_model_adapter
from backend.app.model.fingerprint import generate_model_fingerprint, compare_model_fingerprints
from backend.app.model.reference_battery import evaluate_behavioural_battery
from backend.app.model.trigger_search import evaluate_model_trigger_sensitivity

def test_onnx_adapter_predict(model_paths):
    adapter = get_model_adapter(model_paths["onnx"])
    adapter.load()
    assert adapter.access_level == "GRAY_BOX"
    
    # 64x64x3 dummy image
    img = np.random.rand(64, 64, 3).astype(np.float32)
    output = adapter.predict(img)
    assert output is not None
    assert output.shape[-1] == 2

def test_model_fingerprint(model_paths):
    adapter = get_model_adapter(model_paths["onnx"])
    adapter.load()
    fp = generate_model_fingerprint(adapter)
    
    assert "sha256" in fp
    assert len(fp["sha256"]) == 64
    assert fp["framework"].lower() == "onnx"
    assert fp["parameter_count"] > 0

def test_model_substitution_detection(model_paths, tmp_path):
    # Reference model
    ref_adapter = get_model_adapter(model_paths["onnx"])
    ref_adapter.load()
    ref_fp = generate_model_fingerprint(ref_adapter)

    # Candidate: identical model -> should MATCH
    match_result = compare_model_fingerprints(ref_fp, ref_fp)
    assert match_result["status"] == "MATCH"
    assert match_result["severity"] == "LOW"

    # Fake altered fingerprint
    cand_fp = dict(ref_fp)
    cand_fp["sha256"] = "f" * 64
    cand_fp["parameter_count"] = ref_fp["parameter_count"] + 1000

    sub_result = compare_model_fingerprints(ref_fp, cand_fp)
    assert sub_result["status"] == "SUBSTITUTION_DETECTED"
    assert sub_result["severity"] == "CRITICAL"

def test_behavioural_battery(model_paths):
    adapter = get_model_adapter(model_paths["onnx"])
    adapter.load()
    
    results = evaluate_behavioural_battery(adapter, reference_adapter=adapter, battery_samples=10)
    assert "total_battery_tests" in results
    assert results["total_battery_tests"] == 10
    # Compared to itself, agreement should be 1.0
    assert results["behavioural_agreement_rate"] == 1.0

def test_trigger_sensitivity(model_paths):
    adapter = get_model_adapter(model_paths["onnx"])
    adapter.load()

    res = evaluate_model_trigger_sensitivity(adapter, sample_count=10)
    assert "high_vulnerability_detected" in res
    assert "sensitivity_tests" in res
