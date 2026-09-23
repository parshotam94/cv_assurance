"""
Assurance Report API Endpoints: Generate, View, and Download (PDF, JSON, CSV)
"""
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from backend.app.config import settings
from backend.app.database.database import get_db
from backend.app.database.repository import Repository
from backend.app.reports.generator import build_assurance_report
from backend.app.reports.exporters import export_report_to_json, export_report_to_csv, export_report_to_pdf
from backend.app.core.audit import log_audit_event

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.post("/generate")
def generate_report(
    dataset_id: Optional[str] = Form(None),
    model_id: Optional[str] = Form(None),
    report_title: Optional[str] = Form("VisionTrust AI Assurance & Integrity Assessment"),
    db: Session = Depends(get_db)
):
    """Compile evidence and generate a complete assurance report (PDF, JSON, CSV)."""
    report_data = build_assurance_report(db, dataset_id=dataset_id, model_id=model_id)
    if report_title:
        report_data.title = report_title

    # File paths
    base_name = f"{report_data.report_id}_assurance_report"
    json_path = settings.REPORT_DIR / f"{base_name}.json"
    csv_path = settings.REPORT_DIR / f"{base_name}.csv"
    pdf_path = settings.REPORT_DIR / f"{base_name}.pdf"

    # Export formats
    export_report_to_json(report_data, json_path)
    export_report_to_csv(report_data, csv_path)
    export_report_to_pdf(report_data, pdf_path)

    # Persist in DB
    repo = Repository(db)
    report_db = repo.create_report({
        "report_id": report_data.report_id,
        "title": report_data.title,
        "overall_risk_score": report_data.overall_risk_score,
        "overall_disposition": report_data.overall_disposition,
        "executive_summary": json.dumps(report_data.executive_summary),
        "findings_summary": json.dumps({"findings_count": len(report_data.findings)}),
        "full_report_json": report_data.model_dump_json(),
        "pdf_path": str(pdf_path.resolve()),
        "json_path": str(json_path.resolve()),
        "csv_path": str(csv_path.resolve())
    })

    log_audit_event(
        db,
        action="ASSURANCE_REPORT_GENERATED",
        asset=f"Report:{report_data.report_id}",
        metadata={"disposition": report_data.overall_disposition, "risk_score": report_data.overall_risk_score}
    )

    return {
        "report_id": report_data.report_id,
        "title": report_data.title,
        "overall_risk_score": report_data.overall_risk_score,
        "overall_disposition": report_data.overall_disposition,
        "pdf_download_url": f"/api/reports/{report_data.report_id}/download/pdf",
        "json_download_url": f"/api/reports/{report_data.report_id}/download/json",
        "csv_download_url": f"/api/reports/{report_data.report_id}/download/csv",
        "data": report_data.model_dump()
    }

@router.get("")
def list_reports(db: Session = Depends(get_db)):
    """List all compiled assurance reports."""
    repo = Repository(db)
    reports = repo.list_reports()
    return [
        {
            "report_id": r.report_id,
            "title": r.title,
            "overall_risk_score": r.overall_risk_score,
            "overall_disposition": r.overall_disposition,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "pdf_url": f"/api/reports/{r.report_id}/download/pdf",
            "json_url": f"/api/reports/{r.report_id}/download/json",
            "csv_url": f"/api/reports/{r.report_id}/download/csv"
        }
        for r in reports
    ]

@router.get("/{report_id}")
def get_report(report_id: str, db: Session = Depends(get_db)):
    """Get full JSON representation of an assurance report."""
    repo = Repository(db)
    r = repo.get_report(report_id)
    if not r:
        raise HTTPException(status_code=404, detail="Report not found")
    return json.loads(r.full_report_json) if r.full_report_json else {}

@router.get("/{report_id}/download/{fmt}")
def download_report(report_id: str, fmt: str, db: Session = Depends(get_db)):
    """Download generated report as PDF, JSON, or CSV."""
    repo = Repository(db)
    r = repo.get_report(report_id)
    if not r:
        raise HTTPException(status_code=404, detail="Report not found")

    fmt_lower = fmt.lower()
    if fmt_lower == "pdf" and r.pdf_path and Path(r.pdf_path).exists():
        return FileResponse(r.pdf_path, media_type="application/pdf", filename=f"{report_id}_assurance.pdf")
    elif fmt_lower == "json" and r.json_path and Path(r.json_path).exists():
        return FileResponse(r.json_path, media_type="application/json", filename=f"{report_id}_assurance.json")
    elif fmt_lower == "csv" and r.csv_path and Path(r.csv_path).exists():
        return FileResponse(r.csv_path, media_type="text/csv", filename=f"{report_id}_findings.csv")
@router.get("/charts/{chart_name}")
def get_chart_image(chart_name: str):
    """Serve generated data-science chart image file."""
    if not chart_name.endswith(".png"):
        chart_name = f"{chart_name}.png"
    p = settings.CHART_DIR / chart_name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(p, media_type="image/png")

