"""
Model Assurance REST API Endpoints
"""
import uuid
import json
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from backend.app.config import settings
from backend.app.database.database import get_db
from backend.app.database.repository import Repository
from backend.app.database.models import ModelAssetModel, FindingModel
from backend.app.core.security import sanitize_filename, validate_file_extension
from backend.app.core.hashing import hash_file
from backend.app.core.audit import log_audit_event
from backend.app.core.risk_engine import calculate_finding_risk, aggregate_asset_risk
from backend.app.model.loader import get_model_adapter
from backend.app.model.fingerprint import generate_model_fingerprint, compare_model_fingerprints
from backend.app.model.reference_battery import evaluate_behavioural_battery
from backend.app.model.activation_analysis import analyze_model_activations
from backend.app.model.trigger_search import evaluate_model_trigger_sensitivity

router = APIRouter(prefix="/models", tags=["Model Assurance"])

@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a computer-vision model (.onnx, .pt, .pth, .ts) for integrity analysis."""
    clean_name = sanitize_filename(file.filename)
    ext = validate_file_extension(clean_name, {".onnx", ".pt", ".pth", ".ts"})

    model_id = f"MOD-{uuid.uuid4().hex[:10].upper()}"
    target_path = settings.MODEL_DIR / f"{model_id}_{clean_name}"

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    sha256 = hash_file(target_path)
    file_size = target_path.stat().st_size

    # Inspect model
    adapter = get_model_adapter(target_path)
    adapter.load()
    fingerprint = generate_model_fingerprint(adapter)
    arch_info = fingerprint["architecture_summary"]

    repo = Repository(db)
    model_obj = repo.create_model_asset({
        "id": model_id,
        "name": model_name or clean_name,
        "framework": arch_info.get("framework", ext.lstrip(".")),
        "file_path": str(target_path.resolve()),
        "sha256": sha256,
        "file_size_bytes": file_size,
        "parameter_count": fingerprint.get("parameter_count", 0),
        "architecture": arch_info.get("architecture_class", arch_info.get("producer_name", "CV_Model")),
        "input_shape": str(arch_info.get("inputs", [{}])[0].get("shape", "unknown") if arch_info.get("inputs") else "dynamic"),
        "output_shape": str(arch_info.get("outputs", [{}])[0].get("shape", "unknown") if arch_info.get("outputs") else "dynamic"),
        "access_level": adapter.access_level,
        "fingerprint_json": json.dumps(fingerprint),
        "status": "REGISTERED",
        "risk_score": 0.0,
        "disposition": "ACCEPT"
    })

    log_audit_event(
        db,
        action="MODEL_REGISTRATION",
        asset=f"Model:{model_id}",
        metadata={"name": model_obj.name, "sha256": sha256, "framework": model_obj.framework, "access_level": adapter.access_level}
    )

    return {
        "model_id": model_id,
        "name": model_obj.name,
        "framework": model_obj.framework,
        "sha256": sha256,
        "parameter_count": model_obj.parameter_count,
        "access_level": adapter.access_level
    }

@router.post("/{model_id}/analyze")
def run_model_assurance(
    model_id: str,
    reference_model_id: Optional[str] = None,
    battery_samples: int = settings.BEHAVIOURAL_BATTERY_SAMPLES,
    db: Session = Depends(get_db)
):
    """
    Run comprehensive model assurance analysis:
    - Cryptographic identity verification
    - Model substitution & weight modification detection (if reference model supplied)
    - Controlled perturbation battery (brightness, noise, blur, contrast)
    - Trigger / backdoor sensitivity testing
    - Activation analysis (with explicit graceful degradation)
    """
    repo = Repository(db)
    cand_model_db = repo.get_model_asset(model_id)
    if not cand_model_db:
        raise HTTPException(status_code=404, detail="Candidate model not found")

    cand_path = Path(cand_model_db.file_path)
    cand_adapter = get_model_adapter(cand_path)
    cand_adapter.load()
    cand_fp = generate_model_fingerprint(cand_adapter)

    ref_adapter = None
    ref_fp = None
    if reference_model_id:
        ref_model_db = repo.get_model_asset(reference_model_id)
        if ref_model_db and Path(ref_model_db.file_path).exists():
            ref_path = Path(ref_model_db.file_path)
            ref_adapter = get_model_adapter(ref_path)
            ref_adapter.load()
            ref_fp = generate_model_fingerprint(ref_adapter)

    # 1. Substitution & Weight Drift Analysis
    sub_analysis = None
    if ref_fp:
        sub_analysis = compare_model_fingerprints(ref_fp, cand_fp)

    # 2. Reference Battery Testing
    battery_results = evaluate_behavioural_battery(cand_adapter, ref_adapter, battery_samples=battery_samples)

    # 3. Trigger Sensitivity
    trigger_results = evaluate_model_trigger_sensitivity(cand_adapter, sample_count=20)

    # 4. Activation Analysis (with graceful degradation)
    activation_results = analyze_model_activations(cand_adapter)

    findings_list = []
    finding_counter = 1

    # Finding: Model Substitution / Parameter Mismatch
    if sub_analysis and sub_analysis["status"] != "MATCH":
        sev = sub_analysis["severity"]
        conf = sub_analysis["confidence"]
        score, lvl, disp = calculate_finding_risk("MODEL_SUBSTITUTION", sev, conf, 1, 1)

        findings_list.append({
            "finding_id": f"F-{model_id}-{finding_counter:03d}",
            "asset_type": "MODEL",
            "asset_id": model_id,
            "category": "MODEL_SUBSTITUTION",
            "severity": sev,
            "confidence": conf,
            "risk_score": score,
            "title": f"Model Integrity Deviation: {sub_analysis['status']}",
            "description": sub_analysis["details"],
            "evidence": json.dumps([
                f"Candidate SHA-256: {cand_fp['sha256']}",
                f"Reference SHA-256: {ref_fp['sha256']}",
                f"Parameter difference ratio: {int(sub_analysis['parameter_difference_ratio']*100)}%",
                f"Weight norm difference ratio: {int(sub_analysis['weight_norm_difference_ratio']*100)}%"
            ]),
            "affected_samples": json.dumps([]),
            "detection_method": "Cryptographic Fingerprint & Tensor Parameter Comparison",
            "limitations": json.dumps(["Requires valid baseline reference model for identity assertion."]),
            "recommendation": sub_analysis["recommendation"]
        })
        finding_counter += 1

    # Finding: Behavioural Disagreement
    if battery_results.get("behavioural_agreement_rate") is not None and battery_results["behavioural_agreement_rate"] < 0.85:
        agree_pct = int(battery_results["behavioural_agreement_rate"] * 100)
        sev = "HIGH" if agree_pct < 65 else "MEDIUM"
        conf = 0.85
        score, lvl, disp = calculate_finding_risk("BACKDOOR_BEHAVIOUR", sev, conf, battery_results["anomalous_inputs_count"], battery_results["total_battery_tests"])

        findings_list.append({
            "finding_id": f"F-{model_id}-{finding_counter:03d}",
            "asset_type": "MODEL",
            "asset_id": model_id,
            "category": "BACKDOOR_BEHAVIOUR",
            "severity": sev,
            "confidence": conf,
            "risk_score": score,
            "title": f"Behavioural Battery Instability Detected (Agreement: {agree_pct}%)",
            "description": f"Candidate model exhibited divergent prediction outputs on {battery_results['anomalous_inputs_count']} of {battery_results['total_battery_tests']} perturbation tests compared to reference.",
            "evidence": json.dumps([
                f"Behavioural class agreement: {agree_pct}%",
                f"Anomalous test inputs: {battery_results['anomalous_inputs_count']}"
            ]),
            "affected_samples": json.dumps([]),
            "detection_method": "Controlled Environmental & Perturbation Input Battery",
            "limitations": json.dumps(["Synthetically perturbed inputs evaluate robustness; subtle adversarial triggers may require gradient attacks."]),
            "recommendation": disp
        })
        finding_counter += 1

    # Finding: Trigger Vulnerability
    if trigger_results.get("high_vulnerability_detected"):
        sev = "HIGH"
        conf = trigger_results["confidence"]
        score, lvl, disp = calculate_finding_risk("TRIGGER", sev, conf, 1, 1)

        findings_list.append({
            "finding_id": f"F-{model_id}-{finding_counter:03d}",
            "asset_type": "MODEL",
            "asset_id": model_id,
            "category": "TRIGGER",
            "severity": sev,
            "confidence": conf,
            "risk_score": score,
            "title": "Model Prediction Vulnerable to Localized Trigger Perturbations",
            "description": "Model predictions collapsed disproportionately into a single target class when small localized checkerboard patches were injected.",
            "evidence": json.dumps([
                t["explanation"] for t in trigger_results["sensitivity_tests"] if t["is_suspicious"]
            ]),
            "affected_samples": json.dumps([]),
            "detection_method": "Localized Patch Perturbation Collapse Search",
            "limitations": json.dumps(trigger_results.get("limitations", [])),
            "recommendation": "REVIEW"
        })
        finding_counter += 1

    # Save findings
    db.query(FindingModel).filter(FindingModel.asset_id == model_id).delete()
    for f in findings_list:
        repo.create_finding(f)

    # Compute overall model risk
    overall_risk = aggregate_asset_risk(findings_list)
    repo.update_model(model_id, {
        "status": "ANALYZED",
        "risk_score": overall_risk["risk_score"],
        "disposition": overall_risk["disposition"],
        "behavioural_summary": json.dumps(battery_results),
        "trigger_summary": json.dumps(trigger_results),
        "limitations_json": json.dumps(activation_results.get("unavailable_analyses", []))
    })

    log_audit_event(
        db,
        action="MODEL_ASSURANCE_ANALYSIS",
        asset=f"Model:{model_id}",
        metadata={"risk_score": overall_risk["risk_score"], "disposition": overall_risk["disposition"], "findings": len(findings_list)}
    )

    return {
        "model_id": model_id,
        "overall_risk_score": overall_risk["risk_score"],
        "overall_risk_level": overall_risk["risk_level"],
        "disposition": overall_risk["disposition"],
        "findings_count": len(findings_list),
        "access_level": cand_adapter.access_level,
        "substitution_analysis": sub_analysis,
        "behavioural_battery": battery_results,
        "trigger_sensitivity": trigger_results,
        "activation_analysis": activation_results
    }

@router.get("")
def list_models(db: Session = Depends(get_db)):
    """List all registered models."""
    repo = Repository(db)
    models = repo.list_models()
    return [
        {
            "id": m.id,
            "name": m.name,
            "framework": m.framework,
            "sha256": m.sha256,
            "parameter_count": m.parameter_count,
            "access_level": m.access_level,
            "status": m.status,
            "risk_score": m.risk_score,
            "disposition": m.disposition,
            "created_at": m.created_at.isoformat() if m.created_at else None
        }
        for m in models
    ]

@router.get("/{model_id}")
def get_model_details(model_id: str, db: Session = Depends(get_db)):
    """Get full details of a model, its fingerprint, and findings."""
    repo = Repository(db)
    m = repo.get_model_asset(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    findings = repo.list_findings(asset_id=model_id)

    return {
        "id": m.id,
        "name": m.name,
        "framework": m.framework,
        "sha256": m.sha256,
        "parameter_count": m.parameter_count,
        "architecture": m.architecture,
        "input_shape": m.input_shape,
        "output_shape": m.output_shape,
        "access_level": m.access_level,
        "fingerprint": json.loads(m.fingerprint_json) if m.fingerprint_json else {},
        "behavioural_summary": json.loads(m.behavioural_summary) if m.behavioural_summary else {},
        "trigger_summary": json.loads(m.trigger_summary) if m.trigger_summary else {},
        "limitations": json.loads(m.limitations_json) if m.limitations_json else [],
        "risk_score": m.risk_score,
        "disposition": m.disposition,
        "status": m.status,
        "findings": [
            {
                "finding_id": f.finding_id,
                "category": f.category,
                "severity": f.severity,
                "confidence": f.confidence,
                "risk_score": f.risk_score,
                "title": f.title,
                "recommendation": f.recommendation
            }
            for f in findings
        ]
    }
