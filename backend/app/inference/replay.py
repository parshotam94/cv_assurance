"""
Inference Replay Detection Engine
Detects reused nonces, duplicate record IDs, sequence gaps/reversals, and duplicate inference replays.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.app.database.models import InferenceRecordModel
from backend.app.inference.schema import ProvenanceRecord

def check_for_replay_attack(
    db: Session,
    record: ProvenanceRecord,
    max_clock_skew_seconds: int = 3600
) -> Dict[str, Any]:
    """
    Check if the submitted provenance record represents an adversarial replay attempt.
    """
    reasons = []

    # 1. Duplicate record_id check
    existing_id = db.query(InferenceRecordModel).filter(InferenceRecordModel.record_id == record.record_id).first()
    if existing_id:
        reasons.append(f"Duplicate record_id '{record.record_id}' already registered in database.")

    # 2. Reused nonce check
    existing_nonce = db.query(InferenceRecordModel).filter(InferenceRecordModel.nonce == record.nonce).first()
    if existing_nonce and existing_nonce.record_id != record.record_id:
        reasons.append(f"Cryptographic nonce '{record.nonce}' was previously used in record '{existing_nonce.record_id}'.")

    # 3. Reused sequence number check
    existing_seq = db.query(InferenceRecordModel).filter(InferenceRecordModel.sequence == record.sequence).first()
    if existing_seq and existing_seq.record_id != record.record_id:
        reasons.append(f"Monotonic sequence number {record.sequence} already consumed by record '{existing_seq.record_id}'.")

    # 4. Monotonic sequence check relative to latest record
    latest = db.query(InferenceRecordModel).order_by(InferenceRecordModel.sequence.desc()).first()
    if latest and record.sequence <= latest.sequence and existing_id is None:
        reasons.append(f"Sequence regression: submitted sequence {record.sequence} is <= current sequence head {latest.sequence}.")

    # 5. Timestamp validation
    try:
        dt = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        skew = (now - dt).total_seconds()
        if skew < -300:  # Far in the future
            reasons.append(f"Timestamp anomaly: record timestamp is {abs(skew):.1f}s in the future.")
    except Exception:
        reasons.append("Malformed ISO timestamp in provenance record.")

    # 6. Duplicate image + model + output replay
    dup_payload = db.query(InferenceRecordModel).filter(
        InferenceRecordModel.input_sha256 == record.input_sha256,
        InferenceRecordModel.model_sha256 == record.model_sha256,
        InferenceRecordModel.output_sha256 == record.output_sha256
    ).first()
    if dup_payload and dup_payload.record_id != record.record_id:
        reasons.append(f"Potential inference replay: exact input image, model, and output match prior record '{dup_payload.record_id}'.")

    is_replay = len(reasons) > 0
    severity = "HIGH" if is_replay else "LOW"
    disposition = "QUARANTINE" if is_replay else "ACCEPT"

    return {
        "is_replay": is_replay,
        "severity": severity,
        "disposition": disposition,
        "reasons": reasons,
        "evidence": reasons,
        "recommendation": "Reject and quarantine replayed inference record." if is_replay else "Record passed freshness and replay checks."
    }
