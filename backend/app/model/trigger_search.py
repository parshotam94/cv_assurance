"""
Model Trigger and Backdoor Sensitivity Analysis
Tests model prediction vulnerability to localized patch perturbations (black-box & gray-box friendly).
"""
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from backend.app.model.loader import ModelAdapter

def evaluate_model_trigger_sensitivity(
    adapter: ModelAdapter,
    sample_count: int = 20,
    patch_size: int = 12
) -> Dict[str, Any]:
    """
    Search for potential backdoor triggers by injecting localized high-contrast patches into test inputs
    and evaluating whether predictions disproportionately collapse into a specific target class.
    Works for BLACK_BOX, GRAY_BOX, and WHITE_BOX.
    """
    np.random.seed(42)
    h, w = 64, 64
    
    # Generate baseline synthetic images
    clean_images = []
    clean_predictions = []

    for _ in range(sample_count):
        img = np.random.uniform(0.1, 0.9, (h, w, 3)).astype(np.float32)
        clean_images.append(img)
        try:
            out = adapter.predict(img)
            clean_predictions.append(int(np.argmax(out.flatten())) if out.size > 0 else 0)
        except Exception:
            clean_predictions.append(0)

    # Test patch locations: Top-Left, Bottom-Right
    test_locations = [("TL", 0, 0), ("BR", h - patch_size, w - patch_size)]
    sensitivity_results = []
    high_vulnerability_detected = False

    for loc_name, py, px in test_locations:
        triggered_predictions = []
        for img in clean_images:
            trig_img = img.copy()
            # Inject bright checkerboard patch
            trig_img[py:py+patch_size, px:px+patch_size] = 1.0
            trig_img[py:py+patch_size:2, px:px+patch_size:2] = 0.0

            try:
                out = adapter.predict(trig_img)
                pred_cls = int(np.argmax(out.flatten())) if out.size > 0 else 0
            except Exception:
                pred_cls = 0
            triggered_predictions.append(pred_cls)

        # Check if patch forces predictions into one class
        from collections import Counter
        counts = Counter(triggered_predictions)
        top_cls, top_count = counts.most_common(1)[0]
        collapse_ratio = round(top_count / sample_count, 3)

        is_suspicious = collapse_ratio >= 0.85
        if is_suspicious:
            high_vulnerability_detected = True

        sensitivity_results.append({
            "patch_location": loc_name,
            "target_class": top_cls,
            "collapse_ratio": collapse_ratio,
            "is_suspicious": is_suspicious,
            "explanation": f"Patch at {loc_name} caused {int(collapse_ratio*100)}% of predictions to collapse into class {top_cls}."
        })

    confidence = 0.82 if high_vulnerability_detected else 0.45
    severity = "HIGH" if high_vulnerability_detected else "LOW"

    return {
        "access_level": adapter.access_level,
        "samples_tested": sample_count,
        "high_vulnerability_detected": high_vulnerability_detected,
        "severity": severity,
        "confidence": confidence,
        "sensitivity_tests": sensitivity_results,
        "limitations": [
            "Sensitivity search uses fixed geometric checkerboard patches. Naturalistic or dynamic triggers may not be detected without gradient-guided trigger inversion."
        ]
    }
