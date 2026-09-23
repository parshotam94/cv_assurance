"""
Assurance Report Builder and Aggregator
Compiles executive summaries, asset findings, threat coverage, and system limitations.
"""
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from backend.app.database.models import (
    DatasetModel, ModelAssetModel, InferenceRecordModel, FindingModel
)
from backend.app.reports.schemas import AssuranceReportData, FindingItem, CoverageItem
from backend.app.core.risk_engine import aggregate_asset_risk

THREAT_COVERAGE = [
    CoverageItem(
        threat_id="T1",
        capability="Label Flipping Detection",
        supported_status="SUPPORTED",
        access_requirement="Dataset Labels + Images",
        primary_method="Feature k-NN neighborhood consensus & imbalance analysis",
        key_limitation="Requires distinct feature cluster separation between classes."
    ),
    CoverageItem(
        threat_id="T2",
        capability="Systematic Mislabelling",
        supported_status="SUPPORTED",
        access_requirement="Dataset Metadata + Labels",
        primary_method="Hierarchical contributor & batch anomaly rollup",
        key_limitation="Effectiveness depends on presence of source/batch annotations."
    ),
    CoverageItem(
        threat_id="T3",
        capability="Duplicate Flooding",
        supported_status="SUPPORTED",
        access_requirement="Image Files (Black-Box)",
        primary_method="Perceptual Hashing (pHash, dHash) Hamming distance clustering",
        key_limitation="Sensitivity tuned via threshold; heavily transformed images may escape."
    ),
    CoverageItem(
        threat_id="T4",
        capability="Out-of-Distribution (OOD) Injection",
        supported_status="SUPPORTED",
        access_requirement="Dataset or Reference Baseline",
        primary_method="Isolation Forest & statistical feature distribution distances",
        key_limitation="High-dimensional semantic outliers require reference baseline."
    ),
    CoverageItem(
        threat_id="T5",
        capability="Trigger / Backdoor Injection",
        supported_status="PARTIALLY_SUPPORTED",
        access_requirement="Dataset Images",
        primary_method="Localized patch spatial consistency & class correlation search",
        key_limitation="Blended or frequency-domain watermarks require gradient-based attribution."
    ),
    CoverageItem(
        threat_id="T6",
        capability="Model Substitution",
        supported_status="SUPPORTED",
        access_requirement="Model File (.onnx, .pt, .pth, .ts)",
        primary_method="Cryptographic SHA-256 identity & parameter count verification",
        key_limitation="Requires trusted reference fingerprint for comparison."
    ),
    CoverageItem(
        threat_id="T7",
        capability="Model Weight Modification",
        supported_status="SUPPORTED",
        access_requirement="Model Weights (White-Box / Gray-Box)",
        primary_method="Tensor statistics, layer shapes, and mean L2-norm fingerprinting",
        key_limitation="External weights or opaque black-box runtimes degrade capability."
    ),
    CoverageItem(
        threat_id="T8",
        capability="Backdoor Behaviour Testing",
        supported_status="SUPPORTED",
        access_requirement="Inference Execution (Black-Box)",
        primary_method="Controlled perturbation battery (brightness, noise, patch collapse)",
        key_limitation="Cannot prove absence of dormant triggers without trigger inversion."
    ),
    CoverageItem(
        threat_id="T9",
        capability="Inference Record Tampering",
        supported_status="SUPPORTED",
        access_requirement="Full Record Payload",
        primary_method="Cryptographic hash binding + Ed25519 digital signatures",
        key_limitation="Protects recorded inferences; requires secure local key storage."
    ),
    CoverageItem(
        threat_id="T10",
        capability="Inference Replay Attacks",
        supported_status="SUPPORTED",
        access_requirement="Inference Log State",
        primary_method="Cryptographic nonces, monotonic sequence checks, timestamp skews",
        key_limitation="Requires database persistence of past nonce records."
    ),
    CoverageItem(
        threat_id="T11",
        capability="Provenance Configuration Substitution",
        supported_status="SUPPORTED",
        access_requirement="Provenance Metadata",
        primary_method="Canonical JSON serialization hashing for preprocessing & runtime config",
        key_limitation="Config schema changes must remain backward-compatible."
    ),
    CoverageItem(
        threat_id="T12",
        capability="Distribution Shift & Drift",
        supported_status="SUPPORTED",
        access_requirement="Reference & Candidate Populations",
        primary_method="PSI, 2-sample KS test, Wasserstein distance, environmental categorization",
        key_limitation="Requires sufficient sample volume (N >= 20) for statistical significance."
    ),
]

SYSTEM_LIMITATIONS = [
    "Framework runs 100% locally and does not transmit data or consult cloud foundation models.",
    "Statistical heuristics cannot mathematically prove total absence of steganographic or adversarial latent triggers.",
    "Model weight analysis on ONNX models depends on initializers embedded within the graph file.",
    "Offline cryptographic protection guarantees tamper-evident provenance but assumes private key is stored securely in air-gapped host."
]

def build_assurance_report(
    db: Session,
    dataset_id: Optional[str] = None,
    model_id: Optional[str] = None
) -> AssuranceReportData:
    """
    Compile live findings, asset states, threat coverage, and generate comprehensive Assurance Report.
    """
    report_id = f"REP-{uuid.uuid4().hex[:12].upper()}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Query findings
    q = db.query(FindingModel)
    if dataset_id and model_id:
        q = q.filter((FindingModel.asset_id == dataset_id) | (FindingModel.asset_id == model_id))
    elif dataset_id:
        q = q.filter(FindingModel.asset_id == dataset_id)
    elif model_id:
        q = q.filter(FindingModel.asset_id == model_id)
    findings_db = q.all()

    findings_list = []
    findings_dict_list = []
    for f in findings_db:
        try:
            ev = json.loads(f.evidence) if f.evidence else []
        except Exception:
            ev = []
        try:
            aff = json.loads(f.affected_samples) if f.affected_samples else []
        except Exception:
            aff = []
        try:
            lim = json.loads(f.limitations) if f.limitations else []
        except Exception:
            lim = []

        item = FindingItem(
            finding_id=f.finding_id,
            asset_type=f.asset_type,
            asset_id=f.asset_id,
            category=f.category,
            severity=f.severity,
            confidence=f.confidence,
            risk_score=f.risk_score,
            title=f.title,
            description=f.description,
            evidence=ev,
            affected_samples=aff,
            detection_method=f.detection_method,
            limitations=lim,
            recommendation=f.recommendation,
            created_at=f.created_at.isoformat() if f.created_at else now_iso
        )
        findings_list.append(item)
        findings_dict_list.append(item.model_dump())

    # Overall Risk aggregation
    risk_summary = aggregate_asset_risk(findings_dict_list)

    # Dataset status
    dataset_summary = {"status": "NO_DATASET_SELECTED"}
    if dataset_id:
        ds = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if ds:
            dataset_summary = {
                "dataset_id": ds.id,
                "name": ds.name,
                "format": ds.format,
                "sample_count": ds.sample_count,
                "annotation_count": ds.annotation_count,
                "risk_score": ds.risk_score,
                "disposition": ds.disposition,
                "classes": json.loads(ds.classes) if ds.classes else []
            }

    # Model status
    model_summary = {"status": "NO_MODEL_SELECTED"}
    if model_id:
        md = db.query(ModelAssetModel).filter(ModelAssetModel.id == model_id).first()
        if md:
            model_summary = {
                "model_id": md.id,
                "name": md.name,
                "framework": md.framework,
                "sha256": md.sha256,
                "parameter_count": md.parameter_count,
                "access_level": md.access_level,
                "risk_score": md.risk_score,
                "disposition": md.disposition
            }

    # Inference stats
    total_records = db.query(InferenceRecordModel).count()
    tampered_records = db.query(InferenceRecordModel).filter(InferenceRecordModel.is_tampered == True).count()
    replayed_records = db.query(InferenceRecordModel).filter(InferenceRecordModel.is_replayed == True).count()
    inference_summary = {
        "total_records_tracked": total_records,
        "tampered_records": tampered_records,
        "replayed_attempts": replayed_records,
        "cryptographic_algorithm": "SHA-256 + Ed25519"
    }

    # Distribution status summary
    dist_findings = [f for f in findings_list if f.category == "DISTRIBUTION_SHIFT"]
    dist_summary = {
        "shift_events_recorded": len(dist_findings),
        "recent_shift_classification": dist_findings[0].description if dist_findings else "NORMAL"
    }

    exec_summary = {
        "assessment_verdict": risk_summary["disposition"],
        "overall_risk_score": risk_summary["risk_score"],
        "overall_risk_level": risk_summary["risk_level"],
        "critical_findings": len([f for f in findings_list if f.severity == "CRITICAL"]),
        "high_findings": len([f for f in findings_list if f.severity == "HIGH"]),
        "medium_findings": len([f for f in findings_list if f.severity == "MEDIUM"]),
        "low_findings": len([f for f in findings_list if f.severity == "LOW"]),
        "explanation": risk_summary["explanation"]
    }

    return AssuranceReportData(
        report_id=report_id,
        title="VisionTrust AI Assurance & Integrity Assessment",
        generated_at=now_iso,
        overall_risk_score=risk_summary["risk_score"],
        overall_risk_level=risk_summary["risk_level"],
        overall_disposition=risk_summary["disposition"],
        executive_summary=exec_summary,
        dataset_assurance=dataset_summary,
        model_assurance=model_summary,
        inference_assurance=inference_summary,
        distribution_assurance=dist_summary,
        findings=findings_list,
        coverage_matrix=THREAT_COVERAGE,
        system_limitations=SYSTEM_LIMITATIONS
    )
