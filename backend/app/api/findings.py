"""
Assurance Findings REST API Endpoints
"""
import json
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.app.database.database import get_db
from backend.app.database.repository import Repository

router = APIRouter(prefix="/findings", tags=["Findings"])

@router.get("")
def list_findings(
    asset_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List and filter assurance findings."""
    repo = Repository(db)
    findings = repo.list_findings(asset_id=asset_id, category=category, severity=severity)
    
    return [
        {
            "id": f.id,
            "finding_id": f.finding_id,
            "asset_type": f.asset_type,
            "asset_id": f.asset_id,
            "category": f.category,
            "severity": f.severity,
            "confidence": f.confidence,
            "risk_score": f.risk_score,
            "title": f.title,
            "description": f.description,
            "recommendation": f.recommendation,
            "detection_method": f.detection_method,
            "created_at": f.created_at.isoformat() if f.created_at else None
        }
        for f in findings
    ]

@router.get("/{finding_id}")
def get_finding_detail(finding_id: str, db: Session = Depends(get_db)):
    """Retrieve deep evidence, affected samples, limitations, and recommendation for a finding."""
    repo = Repository(db)
    f = repo.get_finding(finding_id)
    if not f:
        raise HTTPException(status_code=404, detail="Finding not found")

    return {
        "finding_id": f.finding_id,
        "asset_type": f.asset_type,
        "asset_id": f.asset_id,
        "category": f.category,
        "severity": f.severity,
        "confidence": f.confidence,
        "risk_score": f.risk_score,
        "title": f.title,
        "description": f.description,
        "evidence": json.loads(f.evidence) if f.evidence else [],
        "affected_samples": json.loads(f.affected_samples) if f.affected_samples else [],
        "detection_method": f.detection_method,
        "limitations": json.loads(f.limitations) if f.limitations else [],
        "recommendation": f.recommendation,
        "created_at": f.created_at.isoformat() if f.created_at else None
    }
