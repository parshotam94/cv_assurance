"""
Tamper-evident audit logging with cryptographic hash chaining
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional
from sqlalchemy.orm import Session
from backend.app.database.models import AuditEventModel
from backend.app.core.hashing import hash_bytes, canonical_json_dumps

GENESIS_HASH = "0" * 64

def compute_event_hash(
    previous_hash: str,
    timestamp: str,
    actor: str,
    action: str,
    asset: str,
    metadata_json: str
) -> str:
    """Compute the tamper-evident cryptographic hash of an audit event linked to previous hash."""
    payload = f"{previous_hash}|{timestamp}|{actor}|{action}|{asset}|{metadata_json}"
    return hash_bytes(payload.encode("utf-8"))

def log_audit_event(
    db: Session,
    action: str,
    asset: str,
    actor: str = "system",
    metadata: Optional[Dict[str, Any]] = None
) -> AuditEventModel:
    """Record a new audit event and append to the cryptographic hash chain."""
    if metadata is None:
        metadata = {}
    
    metadata_json = canonical_json_dumps(metadata)
    timestamp = datetime.now(timezone.utc).isoformat()
    event_id = f"EVT-{uuid.uuid4().hex[:12].upper()}"

    # Fetch latest event to get previous hash
    latest_event = db.query(AuditEventModel).order_by(AuditEventModel.id.desc()).first()
    previous_hash = latest_event.current_hash if latest_event else GENESIS_HASH

    current_hash = compute_event_hash(
        previous_hash=previous_hash,
        timestamp=timestamp,
        actor=actor,
        action=action,
        asset=asset,
        metadata_json=metadata_json
    )

    event = AuditEventModel(
        event_id=event_id,
        timestamp=timestamp,
        actor=actor,
        action=action,
        asset=asset,
        previous_hash=previous_hash,
        current_hash=current_hash,
        metadata_json=metadata_json
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

def verify_audit_chain(db: Session) -> Dict[str, Any]:
    """
    Verify the entire cryptographic hash chain from genesis to head.
    Detects any record modification, deletion, insertion, or replay.
    """
    events = db.query(AuditEventModel).order_by(AuditEventModel.id.asc()).all()
    if not events:
        return {
            "status": "VALID",
            "message": "Audit chain is empty. Genesis state intact.",
            "total_events": 0,
            "broken_event_id": None,
            "broken_index": None
        }

    expected_prev = GENESIS_HASH
    for idx, event in enumerate(events):
        # 1. Check previous hash continuity
        if event.previous_hash != expected_prev:
            return {
                "status": "TAMPERED",
                "message": f"Hash continuity broken at event {event.event_id} (index {idx}). Previous hash does not match prior event hash.",
                "total_events": len(events),
                "broken_event_id": event.event_id,
                "broken_index": idx,
                "expected_previous_hash": expected_prev,
                "actual_previous_hash": event.previous_hash
            }

        # 2. Recompute current hash to detect content alteration
        recomputed = compute_event_hash(
            previous_hash=event.previous_hash,
            timestamp=event.timestamp,
            actor=event.actor,
            action=event.action,
            asset=event.asset,
            metadata_json=event.metadata_json
        )

        if recomputed != event.current_hash:
            return {
                "status": "TAMPERED",
                "message": f"Payload integrity verification failed for event {event.event_id} (index {idx}). Content has been modified.",
                "total_events": len(events),
                "broken_event_id": event.event_id,
                "broken_index": idx,
                "expected_current_hash": recomputed,
                "actual_current_hash": event.current_hash
            }

        expected_prev = event.current_hash

    return {
        "status": "VALID",
        "message": f"Audit chain verified successfully across {len(events)} events. No tampering detected.",
        "total_events": len(events),
        "broken_event_id": None,
        "broken_index": None,
        "head_hash": expected_prev
    }
