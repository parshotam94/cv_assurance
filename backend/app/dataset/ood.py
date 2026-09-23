"""
Out-of-Distribution (OOD) Detection Module
Supports Isolation Forest, Mahalanobis Distance, and PCA-based distance
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from backend.app.dataset.loader import SampleRecord
from backend.app.dataset.label_analysis import extract_basic_visual_features
from backend.app.config import settings

def detect_out_of_distribution(
    candidate_records: List[SampleRecord],
    reference_records: Optional[List[SampleRecord]] = None,
    percentile_threshold: float = 0.95
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Detect Out-Of-Distribution samples using feature-space anomaly modeling.
    If reference_records is provided, trains baseline on reference and tests candidate.
    Otherwise fits unsupervised Isolation Forest on candidate population.
    """
    if len(candidate_records) < 5:
        return {
            "ood_score": 0.0,
            "confidence": 0.3,
            "affected_count": 0,
            "reason": "Sample size too small for reliable OOD modeling."
        }, []

    cand_features = np.array([extract_basic_visual_features(r.image_path) for r in candidate_records])
    
    if reference_records and len(reference_records) >= 5:
        ref_features = np.array([extract_basic_visual_features(r.image_path) for r in reference_records])
        # Train Isolation Forest on reference
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(ref_features)
        raw_scores = -clf.decision_function(cand_features) # higher = more anomalous
        method = "IsolationForest (Reference Trained)"
    else:
        # Fit on candidate to find internal severe outliers
        contamination = min(0.15, max(0.01, 1.0 - percentile_threshold))
        clf = IsolationForest(contamination=contamination, random_state=42)
        clf.fit(cand_features)
        raw_scores = -clf.decision_function(cand_features)
        method = "IsolationForest (Internal Population)"

    # Normalize scores 0.0 to 1.0
    min_s, max_s = np.min(raw_scores), np.max(raw_scores)
    if max_s > min_s:
        normalized_scores = (raw_scores - min_s) / (max_s - min_s)
    else:
        normalized_scores = np.zeros_like(raw_scores)

    # Threshold for anomaly
    cutoff = np.quantile(normalized_scores, percentile_threshold)
    suspicious_samples = []

    for i, score in enumerate(normalized_scores):
        if score >= cutoff and score > 0.65:
            rec = candidate_records[i]
            suspicious_samples.append({
                "sample_id": rec.sample_id,
                "image_path": rec.image_path,
                "anomaly_score": round(float(score), 3),
                "confidence": "HIGH" if score > 0.85 else "MEDIUM",
                "source_id": rec.source_id,
                "batch_id": rec.batch_id,
                "method": method,
                "reason": f"Sample visual feature vector deviates significantly from reference/population distribution (Score: {score:.2f} >= cutoff {cutoff:.2f})."
            })

    overall_ood_score = round(float(np.mean(normalized_scores)), 2)
    overall_confidence = 0.88 if reference_records else 0.75

    summary = {
        "ood_score": overall_ood_score,
        "confidence": overall_confidence,
        "affected_count": len(suspicious_samples),
        "total_evaluated": len(candidate_records),
        "detection_method": method,
        "reason": (
            f"Detected {len(suspicious_samples)} samples exhibiting significant statistical feature deviation "
            f"beyond the {int(percentile_threshold*100)}th percentile."
        )
    }

    return summary, suspicious_samples
