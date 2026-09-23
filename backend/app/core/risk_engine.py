"""
Risk and Evidence Engine for VisionTrust Assurance
Provides weighted evidence aggregation, explainable risk scores, and disposition recommendations.
"""
from typing import List, Dict, Any, Tuple
from backend.app.config import settings

SEVERITY_WEIGHTS = {
    "LOW": 1.0,
    "MEDIUM": 2.5,
    "HIGH": 6.0,
    "CRITICAL": 10.0
}

CATEGORY_WEIGHTS = {
    "NEAR_DUPLICATE": 1.2,
    "LABEL_FLIP": 2.0,
    "SYSTEMATIC_MISLABEL": 2.5,
    "OOD": 1.5,
    "TRIGGER": 3.0,
    "MODEL_SUBSTITUTION": 3.5,
    "MODEL_MODIFICATION": 3.0,
    "BACKDOOR_BEHAVIOUR": 3.2,
    "TAMPERING": 4.0,
    "REPLAY": 3.0,
    "DISTRIBUTION_SHIFT": 1.8,
    "SOURCE_POISONING": 2.8,
    "CONFIGURATION_MISMATCH": 2.0,
    "PROVENANCE_VIOLATION": 3.8
}

def calculate_finding_risk(
    category: str,
    severity: str,
    confidence: float,
    affected_count: int,
    total_population: int = 100
) -> Tuple[float, str, str]:
    """
    Calculate finding-level risk score (0-100), risk level, and disposition.
    Separates detection from interpretation from decision.
    """
    sev_weight = SEVERITY_WEIGHTS.get(severity.upper(), 2.0)
    cat_weight = CATEGORY_WEIGHTS.get(category.upper(), 1.5)
    
    # Population ratio impact
    pop_ratio = min(1.0, max(0.05, affected_count / max(1, total_population)))
    
    # Base risk formula: severity (0-10) * category (1-4) * confidence (0-1) * population factor
    # Scaled to 0 - 100
    raw_score = (sev_weight * 7.0) * (cat_weight / 2.0) * (0.4 + 0.6 * confidence) * (0.7 + 0.3 * pop_ratio)
    score = min(100.0, max(0.0, round(raw_score, 1)))

    if score >= settings.RISK_QUARANTINE_THRESHOLD:
        level = "CRITICAL"
        disposition = "QUARANTINE"
    elif score >= settings.RISK_REVIEW_THRESHOLD:
        level = "HIGH" if score >= 60.0 else "MEDIUM"
        disposition = "REVIEW"
    else:
        level = "LOW"
        disposition = "ACCEPT"

    return score, level, disposition

def aggregate_asset_risk(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates risk across multiple findings for an asset (dataset, model, etc.).
    Produces overall score, level, recommended disposition, and evidence breakdown.
    """
    if not findings:
        return {
            "risk_score": 0.0,
            "risk_level": "LOW",
            "disposition": "ACCEPT",
            "explanation": "No integrity violations or anomalies detected.",
            "contributing_factors": []
        }

    total_weight = 0.0
    weighted_score_sum = 0.0
    critical_findings_count = 0
    high_findings_count = 0
    contributing_factors = []

    for f in findings:
        score = f.get("risk_score", 0.0)
        sev = f.get("severity", "LOW")
        conf = f.get("confidence", 0.5)
        cat = f.get("category", "UNKNOWN")
        weight = SEVERITY_WEIGHTS.get(sev, 1.0) * CATEGORY_WEIGHTS.get(cat, 1.0) * conf

        weighted_score_sum += score * weight
        total_weight += weight

        if sev == "CRITICAL":
            critical_findings_count += 1
        elif sev == "HIGH":
            high_findings_count += 1

        contributing_factors.append({
            "category": cat,
            "severity": sev,
            "finding_score": score,
            "confidence": conf,
            "contribution_weight": round(weight, 2)
        })

    avg_weighted_score = weighted_score_sum / max(1.0, total_weight)
    
    # Critical findings escalate score directly
    final_score = min(100.0, round(avg_weighted_score + (critical_findings_count * 15.0) + (high_findings_count * 5.0), 1))

    if final_score >= settings.RISK_QUARANTINE_THRESHOLD or critical_findings_count > 0:
        disposition = "QUARANTINE"
        level = "CRITICAL"
    elif final_score >= settings.RISK_REVIEW_THRESHOLD or high_findings_count > 0:
        disposition = "REVIEW"
        level = "HIGH" if final_score >= 60.0 else "MEDIUM"
    else:
        disposition = "ACCEPT"
        level = "LOW"

    explanation = (
        f"Overall risk evaluated as {level} ({final_score}/100) across {len(findings)} findings. "
        f"Critical findings: {critical_findings_count}, High findings: {high_findings_count}."
    )

    return {
        "risk_score": final_score,
        "risk_level": level,
        "disposition": disposition,
        "explanation": explanation,
        "contributing_factors": contributing_factors
    }
