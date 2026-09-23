"""
Hierarchical source-level risk aggregation
sample anomalies -> batch anomalies -> contributor anomalies -> source risk
"""
from typing import List, Dict, Any
from collections import defaultdict
from backend.app.dataset.loader import SampleRecord

def aggregate_source_risks(
    records: List[SampleRecord],
    duplicates_map: Dict[str, List[str]],
    mislabelling_samples: List[Dict[str, Any]],
    ood_samples: List[Dict[str, Any]],
    trigger_samples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Roll up individual sample anomalies into batch-level, contributor-level, and source-level risk profiles.
    """
    mislabel_sids = {s["sample_id"] for s in mislabelling_samples}
    ood_sids = {s["sample_id"] for s in ood_samples}
    trigger_sids = {s["sample_id"] for s in trigger_samples}

    # Group by source_id
    source_buckets = defaultdict(lambda: {
        "sample_count": 0,
        "suspicious_samples": set(),
        "duplicate_count": 0,
        "ood_count": 0,
        "label_anomalies": 0,
        "trigger_count": 0,
        "batches": defaultdict(lambda: {"total": 0, "suspicious": 0}),
        "contributors": defaultdict(lambda: {"total": 0, "suspicious": 0})
    })

    for r in records:
        b = source_buckets[r.source_id]
        b["sample_count"] += 1
        b["batches"][r.batch_id]["total"] += 1
        b["contributors"][r.contributor]["total"] += 1

        is_susp = False
        if r.sample_id in duplicates_map:
            b["duplicate_count"] += 1
            is_susp = True
        if r.sample_id in mislabel_sids:
            b["label_anomalies"] += 1
            is_susp = True
        if r.sample_id in ood_sids:
            b["ood_count"] += 1
            is_susp = True
        if r.sample_id in trigger_sids:
            b["trigger_count"] += 1
            is_susp = True

        if is_susp:
            b["suspicious_samples"].add(r.sample_id)
            b["batches"][r.batch_id]["suspicious"] += 1
            b["contributors"][r.contributor]["suspicious"] += 1

    source_results = []

    for source_id, data in source_buckets.items():
        total = data["sample_count"]
        suspicious_count = len(data["suspicious_samples"])
        susp_ratio = suspicious_count / max(1, total)

        # Weighted calculation for source risk score (0-100)
        # Duplicate = 1.0, OOD = 1.5, Label = 2.5, Trigger = 4.0
        weighted_anomalies = (
            (data["duplicate_count"] * 1.0) +
            (data["ood_count"] * 1.5) +
            (data["label_anomalies"] * 2.5) +
            (data["trigger_count"] * 4.0)
        )
        normalized_rate = weighted_anomalies / max(1, total)
        raw_score = min(100.0, round(normalized_rate * 70.0 + (susp_ratio * 30.0), 1))

        if raw_score >= 70.0 or data["trigger_count"] >= 3:
            risk_level = "CRITICAL"
            disposition = "QUARANTINE"
        elif raw_score >= 35.0 or susp_ratio >= 0.15:
            risk_level = "HIGH"
            disposition = "REVIEW"
        elif raw_score >= 15.0 or susp_ratio >= 0.05:
            risk_level = "MEDIUM"
            disposition = "REVIEW"
        else:
            risk_level = "LOW"
            disposition = "ACCEPT"

        # Check high-risk contributors or batches
        anomalous_batches = [b for b, stats in data["batches"].items() if stats["suspicious"] / max(1, stats["total"]) > 0.3]
        anomalous_contributors = [c for c, stats in data["contributors"].items() if stats["suspicious"] / max(1, stats["total"]) > 0.3]

        source_results.append({
            "source_id": source_id,
            "sample_count": total,
            "suspicious_count": suspicious_count,
            "duplicate_clusters": data["duplicate_count"],
            "ood_count": data["ood_count"],
            "label_anomalies": data["label_anomalies"],
            "trigger_count": data["trigger_count"],
            "risk_score": raw_score,
            "risk_level": risk_level,
            "disposition": disposition,
            "anomalous_batches": anomalous_batches,
            "anomalous_contributors": anomalous_contributors
        })

    return source_results
