"""
Environmental Distribution Shift Classifier
Categorizes detected distribution changes into NORMAL, OPERATIONAL_DRIFT, ANOMALOUS, or SUSPICIOUS.
"""
from typing import List, Dict, Any
import numpy as np
from backend.app.distribution.embeddings import extract_image_distribution_metrics
from backend.app.distribution.drift import compute_drift_metrics

def analyze_environmental_shift(
    reference_image_paths: List[str],
    candidate_image_paths: List[str]
) -> Dict[str, Any]:
    """
    Perform multi-attribute distribution shift analysis across visual features.
    """
    if len(reference_image_paths) == 0 or len(candidate_image_paths) == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "classification": "NORMAL",
            "confidence": 0.0,
            "overall_shift_detected": False,
            "feature_drifts": {},
            "summary": "Both reference and candidate image paths are required to evaluate distribution shift."
        }

    # Extract metrics for all images
    ref_metrics = [extract_image_distribution_metrics(p) for p in reference_image_paths]
    cand_metrics = [extract_image_distribution_metrics(p) for p in candidate_image_paths]

    features = ["brightness", "contrast", "sharpness", "saturation", "r_mean", "g_mean", "b_mean", "high_freq_energy"]
    feature_drifts = {}
    significant_features = []

    for feat in features:
        ref_vals = [m[feat] for m in ref_metrics]
        cand_vals = [m[feat] for m in cand_metrics]
        
        drift_stats = compute_drift_metrics(ref_vals, cand_vals)
        ref_mean = round(float(np.mean(ref_vals)), 3)
        cand_mean = round(float(np.mean(cand_vals)), 3)

        drift_stats["reference_mean"] = ref_mean
        drift_stats["candidate_mean"] = cand_mean
        feature_drifts[feat] = drift_stats

        if drift_stats["is_significant_drift"]:
            significant_features.append((feat, drift_stats))

    num_significant = len(significant_features)

    # Classify shift
    if num_significant == 0:
        classification = "NORMAL"
        severity = "LOW"
        confidence = 0.92
        recommendation = "ACCEPT"
        summary = "No statistically significant distribution drift detected between candidate and reference populations."
    elif num_significant <= 2:
        # Check if illumination or contrast (typical operational drift)
        shifted_names = [f[0] for f in significant_features]
        if any(name in ["brightness", "contrast"] for name in shifted_names):
            classification = "OPERATIONAL_DRIFT"
            severity = "MEDIUM"
            confidence = 0.85
            recommendation = "REVIEW"
            summary = (
                f"Operational environmental shift detected in {', '.join(shifted_names)}. "
                "Evidence is consistent with diurnal lighting variations or camera sensor auto-exposure adjustments."
            )
        else:
            classification = "ANOMALOUS"
            severity = "MEDIUM"
            confidence = 0.80
            recommendation = "REVIEW"
            summary = f"Anomalous distribution shift detected in attributes: {', '.join(shifted_names)}."
    elif num_significant <= 4:
        classification = "ANOMALOUS"
        severity = "HIGH"
        confidence = 0.88
        recommendation = "REVIEW"
        shifted_names = [f[0] for f in significant_features]
        summary = (
            f"Significant multi-factor distribution shift observed across {num_significant} visual domains ({', '.join(shifted_names)}). "
            "Model performance may degrade without domain adaptation."
        )
    else:
        classification = "SUSPICIOUS"
        severity = "HIGH"
        confidence = 0.91
        recommendation = "QUARANTINE"
        summary = (
            f"Severe pervasive distribution deviation detected across {num_significant} feature channels. "
            "Candidate population fundamentally differs from baseline operational domain."
        )

    return {
        "classification": classification,
        "severity": severity,
        "confidence": confidence,
        "overall_shift_detected": (num_significant > 0),
        "significant_drift_features_count": num_significant,
        "recommendation": recommendation,
        "summary": summary,
        "feature_drifts": feature_drifts,
        "limitations": [
            "Metrics evaluate statistical visual and low-level spatial frequency shifts; semantic distribution shifts require class-conditional generative density estimators."
        ]
    }
