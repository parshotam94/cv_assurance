"""
Tamper-Evident Audit Trail API Endpoints
"""
import json
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.database.database import get_db
from backend.app.database.repository import Repository
from backend.app.core.audit import verify_audit_chain

router = APIRouter(prefix="/audit", tags=["Audit Trail"])

@router.get("")
def list_audit_events(limit: int = 100, db: Session = Depends(get_db)):
    """List cryptographically hash-chained audit events."""
    repo = Repository(db)
    events = repo.list_audit_events(limit=limit)
    return [
        {
            "id": e.id,
            "event_id": e.event_id,
            "timestamp": e.timestamp,
            "actor": e.actor,
            "action": e.action,
            "asset": e.asset,
            "previous_hash": e.previous_hash,
            "current_hash": e.current_hash,
            "metadata": json.loads(e.metadata_json) if e.metadata_json else {}
        }
        for e in events
    ]

@router.post("/verify")
def verify_chain(db: Session = Depends(get_db)):
    """
    Verify full cryptographic hash chain integrity.
    Detects record tampering, unauthorized insertions, deletions, or hash mismatch.
    """
    result = verify_audit_chain(db)
    return result
