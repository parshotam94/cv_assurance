"""
Distribution Shift Analysis REST API
"""
import json
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from backend.app.database.database import get_db
from backend.app.database.repository import Repository
from backend.app.database.models import DatasetModel
from backend.app.core.audit import log_audit_event
from backend.app.core.risk_engine import calculate_finding_risk
from backend.app.distribution.environmental_shift import analyze_environmental_shift

router = APIRouter(prefix="/distribution", tags=["Distribution Shift"])

@router.post("/analyze")
def run_distribution_analysis(
    reference_dataset_id: str = Form(...),
    candidate_dataset_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Perform statistical distribution-shift analysis between reference baseline dataset and candidate dataset.
    Calculates PSI, KS tests, Wasserstein distance, and categorizes drift into:
    NORMAL, OPERATIONAL_DRIFT, ANOMALOUS, or SUSPICIOUS.
    """
    repo = Repository(db)
    ref_ds = repo.get_dataset(reference_dataset_id)
    cand_ds = repo.get_dataset(candidate_dataset_id)

    if not ref_ds or not cand_ds:
        raise HTTPException(status_code=404, detail="Reference or candidate dataset not found.")

    # Get sample paths
    ref_samples = repo.get_dataset_samples(reference_dataset_id, limit=60)
    cand_samples = repo.get_dataset_samples(candidate_dataset_id, limit=60)

    ref_paths = [s.image_path for s in ref_samples if Path(s.image_path).exists()]
    cand_paths = [s.image_path for s in cand_samples if Path(s.image_path).exists()]

    if len(ref_paths) < 2 or len(cand_paths) < 2:
        raise HTTPException(status_code=400, detail="Insufficient images for statistical distribution comparison (need >= 2).")

    shift_results = analyze_environmental_shift(ref_paths, cand_paths)

    # Generate Finding if shift detected
    finding = None
    if shift_results["overall_shift_detected"]:
        score, lvl, disp = calculate_finding_risk(
            "DISTRIBUTION_SHIFT",
            shift_results["severity"],
            shift_results["confidence"],
            shift_results["significant_drift_features_count"],
            8
        )
        
        evidence = [
            f"Classification: {shift_results['classification']}",
            f"Summary: {shift_results['summary']}"
        ]
        for feat, data in shift_results["feature_drifts"].items():
            if data["is_significant_drift"]:
                evidence.append(
                    f"{feat.upper()}: PSI={data['psi']}, KS-stat={data['ks_statistic']} (p={data['ks_p_value']:.4f}), "
                    f"Ref mean={data['reference_mean']} vs Cand mean={data['candidate_mean']}"
                )

        finding = repo.create_finding({
            "finding_id": f"F-DIST-{candidate_dataset_id[:6]}-{reference_dataset_id[:6]}",
            "asset_type": "DATASET",
            "asset_id": candidate_dataset_id,
            "category": "DISTRIBUTION_SHIFT",
            "severity": shift_results["severity"],
            "confidence": shift_results["confidence"],
            "risk_score": score,
            "title": f"Distribution Drift Detected: {shift_results['classification']}",
            "description": shift_results["summary"],
            "evidence": json.dumps(evidence),
            "affected_samples": json.dumps([s.sample_id for s in cand_samples[:20]]),
            "detection_method": "Multi-Attribute PSI & Kolmogorov-Smirnov Statistical Testing",
            "limitations": json.dumps(shift_results.get("limitations", [])),
            "recommendation": shift_results["recommendation"]
        })

    log_audit_event(
        db,
        action="DISTRIBUTION_ANALYSIS",
        asset=f"DatasetComparison:{reference_dataset_id}->{candidate_dataset_id}",
        metadata={"classification": shift_results["classification"], "shift_detected": shift_results["overall_shift_detected"]}
    )

    return {
        "status": "SUCCESS",
        "reference_dataset_id": reference_dataset_id,
        "candidate_dataset_id": candidate_dataset_id,
        "results": shift_results,
        "finding_id": finding.finding_id if finding else None
    }
