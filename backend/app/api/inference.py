"""
Cryptographic Inference Provenance, Tamper Testing, and Replay Testing API
"""
import uuid
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Body
from sqlalchemy.orm import Session
from PIL import Image
import numpy as np

from backend.app.config import settings
from backend.app.database.database import get_db
from backend.app.database.repository import Repository
from backend.app.database.models import InferenceRecordModel, ModelAssetModel, FindingModel
from backend.app.core.security import sanitize_filename, validate_file_extension
from backend.app.core.hashing import hash_file, hash_canonical_json
from backend.app.core.audit import log_audit_event
from backend.app.core.risk_engine import calculate_finding_risk
from backend.app.model.loader import get_model_adapter
from backend.app.inference.schema import ProvenanceRecord, PreprocessingConfig, InferenceConfig, VerificationResult
from backend.app.inference.provenance import create_provenance_record
from backend.app.inference.verifier import verify_provenance_record
from backend.app.inference.replay import check_for_replay_attack

router = APIRouter(prefix="/inference", tags=["Inference Provenance"])

@router.post("/run")
async def run_and_protect_inference(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    preprocessing_json: Optional[str] = Form(None),
    config_json: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Execute offline model inference on an input image and generate a cryptographically bound,
    Ed25519 signed canonical provenance record.
    """
    clean_name = sanitize_filename(file.filename)
    validate_file_extension(clean_name, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    repo = Repository(db)
    model_asset = repo.get_model_asset(model_id)
    if not model_asset or not Path(model_asset.file_path).exists():
        raise HTTPException(status_code=404, detail="Model asset not found")

    # Save uploaded input image locally
    img_id = f"INF-{uuid.uuid4().hex[:10].upper()}"
    save_path = settings.INFERENCE_DIR / f"{img_id}_{clean_name}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load image array for model
    with Image.open(save_path) as im:
        rgb_im = im.convert("RGB").resize((64, 64))
        img_arr = np.array(rgb_im, dtype=np.float32) / 255.0

    # Run inference
    adapter = get_model_adapter(Path(model_asset.file_path))
    adapter.load()
    raw_output_arr = adapter.predict(img_arr)
    
    # Format output
    flat_out = raw_output_arr.flatten().tolist()
    pred_cls = int(np.argmax(flat_out)) if flat_out else 0
    confidence = float(np.max(flat_out)) if flat_out else 0.5
    
    output_payload = {
        "predicted_class_id": pred_cls,
        "confidence": round(confidence, 4),
        "logits_sample": [round(v, 4) for v in flat_out[:10]],
        "model_id": model_id,
        "input_dimensions": [64, 64, 3]
    }

    # Preprocessing & Inference Configs
    prep_cfg = PreprocessingConfig(**json.loads(preprocessing_json)) if preprocessing_json else PreprocessingConfig()
    inf_cfg = InferenceConfig(**json.loads(config_json)) if config_json else InferenceConfig()

    # Generate cryptographically signed provenance record
    record = create_provenance_record(
        input_image_path=save_path,
        model_sha256=model_asset.sha256,
        output_data=output_payload,
        preprocessing_config=prep_cfg,
        inference_config=inf_cfg
    )

    # Persist in DB
    record_db = repo.create_inference_record({
        "record_id": record.record_id,
        "input_sha256": record.input_sha256,
        "model_sha256": record.model_sha256,
        "preprocessing_sha256": record.preprocessing_sha256,
        "config_sha256": record.config_sha256,
        "output_sha256": record.output_sha256,
        "timestamp": record.timestamp,
        "nonce": record.nonce,
        "sequence": record.sequence,
        "record_hash": record.record_hash,
        "signature": record.signature,
        "public_key_hex": record.public_key_hex,
        "is_tampered": False,
        "is_replayed": False,
        "verification_status": "VERIFIED",
        "verification_details": json.dumps({"status": "SUCCESS", "message": "Initial signed generation"}),
        "payload_json": json.dumps(output_payload)
    })

    log_audit_event(
        db,
        action="INFERENCE_RECORD_CREATED",
        asset=f"InferenceRecord:{record.record_id}",
        metadata={"model": model_id, "sequence": record.sequence, "record_hash": record.record_hash}
    )

    return {
        "status": "SUCCESS",
        "record": record.model_dump(),
        "output": output_payload,
        "image_url": f"/api/inference/image/{img_id}_{clean_name}"
    }

@router.post("/verify")
def verify_inference(
    record_payload: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Cryptographically verify a provenance record against its hashes and Ed25519 digital signature.
    Also checks for replay attacks.
    """
    record = ProvenanceRecord(**record_payload)
    
    # 1. Cryptographic and Hash Verification
    verification = verify_provenance_record(
        record=record,
        current_output_data=record.raw_output
    )

    # 2. Check Replay Attack
    replay_check = check_for_replay_attack(db, record)
    if replay_check["is_replay"]:
        verification.status = "REPLAY_DETECTED"
        verification.is_valid = False
        verification.disposition = "QUARANTINE"
        verification.details = f"Replay Attack Detected: {'; '.join(replay_check['reasons'])}"

    # Update record status in DB if exists
    repo = Repository(db)
    rec_db = repo.get_inference_record(record.record_id)
    if rec_db:
        rec_db.verification_status = verification.status
        rec_db.is_tampered = (verification.status in ["TAMPERED", "INVALID_SIGNATURE"])
        rec_db.is_replayed = (verification.status == "REPLAY_DETECTED")
        rec_db.verification_details = json.dumps(verification.model_dump())
        db.commit()

    # Log audit event
    log_audit_event(
        db,
        action="INFERENCE_VERIFICATION",
        asset=f"InferenceRecord:{record.record_id}",
        metadata={"status": verification.status, "is_valid": verification.is_valid}
    )

    return verification.model_dump()

@router.post("/tamper-test")
def simulate_inference_tampering(
    record_id: str = Form(...),
    tamper_field: str = Form("output"),  # output, model_hash, timestamp, signature
    db: Session = Depends(get_db)
):
    """
    Tamper simulation scenario:
    Mutates a field in an existing record and executes verification,
    demonstrating immediate cryptographic tamper detection.
    """
    repo = Repository(db)
    rec = repo.get_inference_record(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Inference record not found")

    raw_payload = json.loads(rec.payload_json) if rec.payload_json else {}

    # Mutate selected field
    tampered_output = dict(raw_payload)
    tampered_model_sha256 = rec.model_sha256
    tampered_timestamp = rec.timestamp
    tampered_signature = rec.signature

    if tamper_field == "output":
        tampered_output["predicted_class_id"] = 999
        tampered_output["confidence"] = 0.9999
        tampered_output["tampered_marker"] = True
    elif tamper_field == "model_hash":
        tampered_model_sha256 = "0" * 64
    elif tamper_field == "timestamp":
        tampered_timestamp = "2099-01-01T00:00:00+00:00"
    elif tamper_field == "signature":
        tampered_signature = "a" * 128

    mutated_record = ProvenanceRecord(
        record_id=rec.record_id,
        input_sha256=rec.input_sha256,
        model_sha256=tampered_model_sha256,
        preprocessing_sha256=rec.preprocessing_sha256,
        config_sha256=rec.config_sha256,
        output_sha256=rec.output_sha256,
        timestamp=tampered_timestamp,
        nonce=rec.nonce,
        sequence=rec.sequence,
        record_hash=rec.record_hash,
        signature=tampered_signature,
        public_key_hex=rec.public_key_hex,
        raw_output=tampered_output
    )

    # Run verification
    verification = verify_provenance_record(
        record=mutated_record,
        current_output_data=tampered_output
    )

    # Record tamper finding
    score, lvl, disp = calculate_finding_risk("TAMPERING", "CRITICAL", 1.0, 1, 1)
    finding = repo.create_finding({
        "finding_id": f"F-TAMP-{uuid.uuid4().hex[:6].upper()}",
        "asset_type": "INFERENCE",
        "asset_id": rec.record_id,
        "category": "TAMPERING",
        "severity": "CRITICAL",
        "confidence": 1.0,
        "risk_score": score,
        "title": f"Inference Tampering Detected in Record {rec.record_id}",
        "description": f"Deliberate modification of {tamper_field} was detected during cryptographic verification.",
        "evidence": json.dumps(verification.tamper_detected_in),
        "affected_samples": json.dumps([rec.record_id]),
        "detection_method": "SHA-256 Hash Binding & Ed25519 Digital Signature Verification",
        "limitations": json.dumps([]),
        "recommendation": "QUARANTINE"
    })

    log_audit_event(
        db,
        action="TAMPER_DETECTED",
        asset=f"InferenceRecord:{rec.record_id}",
        metadata={"tampered_field": tamper_field, "status": verification.status}
    )

    return {
        "tamper_field": tamper_field,
        "verification_result": verification.model_dump(),
        "finding_created": finding.finding_id
    }

@router.post("/replay-test")
def simulate_replay_attack(
    original_record_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Replay attack simulation:
    Submits a record using a duplicate record ID, reused nonce, and sequence regression,
    proving replay rejection.
    """
    repo = Repository(db)
    orig = repo.get_inference_record(original_record_id)
    if not orig:
        raise HTTPException(status_code=404, detail="Original record not found")

    replayed_record = ProvenanceRecord(
        record_id=orig.record_id,
        input_sha256=orig.input_sha256,
        model_sha256=orig.model_sha256,
        preprocessing_sha256=orig.preprocessing_sha256,
        config_sha256=orig.config_sha256,
        output_sha256=orig.output_sha256,
        timestamp=orig.timestamp,
        nonce=orig.nonce,
        sequence=orig.sequence,
        record_hash=orig.record_hash,
        signature=orig.signature,
        public_key_hex=orig.public_key_hex,
        raw_output=json.loads(orig.payload_json) if orig.payload_json else {}
    )

    replay_result = check_for_replay_attack(db, replayed_record)

    score, lvl, disp = calculate_finding_risk("REPLAY", "HIGH", 0.95, 1, 1)
    finding = repo.create_finding({
        "finding_id": f"F-RPLY-{uuid.uuid4().hex[:6].upper()}",
        "asset_type": "INFERENCE",
        "asset_id": orig.record_id,
        "category": "REPLAY",
        "severity": "HIGH",
        "confidence": 0.95,
        "risk_score": score,
        "title": f"Adversarial Replay Attempt Detected on Record {orig.record_id}",
        "description": "An inference record submitted identical nonce, sequence, or record_id from an earlier inference session.",
        "evidence": json.dumps(replay_result["evidence"]),
        "affected_samples": json.dumps([orig.record_id]),
        "detection_method": "Monotonic Nonce & Sequence Freshness Tracking",
        "limitations": json.dumps([]),
        "recommendation": "QUARANTINE"
    })

    log_audit_event(
        db,
        action="REPLAY_ATTACK_DETECTED",
        asset=f"InferenceRecord:{orig.record_id}",
        metadata={"nonce": orig.nonce, "sequence": orig.sequence}
    )

    return {
        "is_replay_detected": replay_result["is_replay"],
        "reasons": replay_result["reasons"],
        "finding_id": finding.finding_id
    }

@router.get("/records")
def list_inference_records(limit: int = 50, db: Session = Depends(get_db)):
    """List tracked provenance inference records."""
    repo = Repository(db)
    records = repo.list_inference_records(limit=limit)
    return [
        {
            "id": r.id,
            "record_id": r.record_id,
            "input_sha256": r.input_sha256,
            "model_sha256": r.model_sha256,
            "sequence": r.sequence,
            "nonce": r.nonce,
            "timestamp": r.timestamp,
            "record_hash": r.record_hash,
            "signature": r.signature[:16] + "...",
            "verification_status": r.verification_status,
            "is_tampered": r.is_tampered,
            "is_replayed": r.is_replayed,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in records
    ]
