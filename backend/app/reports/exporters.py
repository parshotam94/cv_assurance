"""
Assurance Report Exporters: JSON, CSV, and Offline PDF (via ReportLab)
"""
import json
import csv
from pathlib import Path
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from backend.app.reports.schemas import AssuranceReportData

def export_report_to_json(report: AssuranceReportData, output_path: Path) -> Path:
    """Save report as structured JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return output_path

def export_report_to_csv(report: AssuranceReportData, output_path: Path) -> Path:
    """Save findings table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Finding ID", "Asset Type", "Asset ID", "Category", "Severity",
            "Confidence", "Risk Score", "Title", "Recommendation", "Detection Method", "Description"
        ])
        for f_item in report.findings:
            writer.writerow([
                f_item.finding_id,
                f_item.asset_type,
                f_item.asset_id,
                f_item.category,
                f_item.severity,
                f_item.confidence,
                f_item.risk_score,
                f_item.title,
                f_item.recommendation,
                f_item.detection_method,
                f_item.description.replace("\n", " ")
            ])
    return output_path

def export_report_to_pdf(report: AssuranceReportData, output_path: Path) -> Path:
    """Generate high-quality multi-page enterprise PDF report offline via ReportLab."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a')
    )
    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    small_style = ParagraphStyle(
        'SmallCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#64748b')
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("VisionTrust Assurance", title_style))
    elements.append(Paragraph("Offline Computer-Vision Integrity & Provenance Framework", ParagraphStyle(
        'Subtitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#0284c7')
    )))
    elements.append(Paragraph(f"Report ID: {report.report_id}  |  Generated: {report.generated_at} (Air-Gapped)", small_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#cbd5e1'), spaceBefore=8, spaceAfter=12))

    # Executive Verdict Box
    disp = report.overall_disposition
    disp_color = colors.HexColor('#16a34a') if disp == 'ACCEPT' else (colors.HexColor('#d97706') if disp == 'REVIEW' else colors.HexColor('#dc2626'))
    
    verdict_data = [
        [
            Paragraph("<b>Overall Assurance Verdict</b>", body_style),
            Paragraph(f"<font color='{disp_color.hexval()}'><b>{disp}</b></font>", h2_style),
            Paragraph(f"Overall Risk Score: <b>{report.overall_risk_score} / 100</b> ({report.overall_risk_level})", body_style)
        ]
    ]
    t_verdict = Table(verdict_data, colWidths=[180, 140, 220])
    t_verdict.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_verdict)
    elements.append(Spacer(1, 10))

    # Executive Summary Paragraph
    elements.append(Paragraph("<b>Executive Summary:</b> " + report.executive_summary.get("explanation", ""), body_style))
    elements.append(Spacer(1, 10))

    # Summary Metrics Table
    summary_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Critical Findings", str(report.executive_summary.get("critical_findings", 0)), "Dataset Sample Count", str(report.dataset_assurance.get("sample_count", "N/A"))],
        ["High Findings", str(report.executive_summary.get("high_findings", 0)), "Dataset Classes", str(len(report.dataset_assurance.get("classes", [])))],
        ["Medium Findings", str(report.executive_summary.get("medium_findings", 0)), "Model Access Mode", str(report.model_assurance.get("access_level", "N/A"))],
        ["Low Findings", str(report.executive_summary.get("low_findings", 0)), "Model Parameter Count", str(report.model_assurance.get("parameter_count", "N/A"))],
        ["Inference Records Tracked", str(report.inference_assurance.get("total_records_tracked", 0)), "Tampered Inferences", str(report.inference_assurance.get("tampered_records", 0))],
    ]
    t_sum = Table(summary_data, colWidths=[150, 120, 150, 120])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_sum)
    elements.append(Spacer(1, 14))

    # Detailed Findings Section
    elements.append(Paragraph("Integrity & Security Findings", h2_style))
    if not report.findings:
        elements.append(Paragraph("No critical findings or integrity violations reported.", body_style))
    else:
        findings_table_data = [
            ["ID", "Category", "Severity", "Conf.", "Risk", "Disposition", "Title"]
        ]
        for f in report.findings[:15]:  # show top 15 in summary table
            findings_table_data.append([
                f.finding_id,
                f.category[:14],
                f.severity,
                f"{int(f.confidence*100)}%",
                str(f.risk_score),
                f.recommendation,
                Paragraph(f.title[:45] + ("..." if len(f.title) > 45 else ""), small_style)
            ])
        t_find = Table(findings_table_data, colWidths=[55, 80, 50, 40, 40, 65, 210])
        t_find.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_find)

    elements.append(Spacer(1, 14))

    # Threat Coverage Statement
    elements.append(Paragraph("Assurance Threat Coverage Statement", h2_style))
    coverage_table_data = [
        ["ID", "Attack / Threat Capability", "Status", "Access Mode", "Primary Method"]
    ]
    for c in report.coverage_matrix:
        coverage_table_data.append([
            c.threat_id,
            Paragraph(c.capability, small_style),
            c.supported_status,
            c.access_requirement[:18],
            Paragraph(c.primary_method[:45] + "...", small_style)
        ])
    t_cov = Table(coverage_table_data, colWidths=[35, 135, 95, 95, 180])
    t_cov.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_cov)

    elements.append(Spacer(1, 14))

    # Limitations & Offline Statement
    elements.append(Paragraph("System Limitations & Air-Gap Compliance", h2_style))
    for lim in report.system_limitations:
        elements.append(Paragraph(f"• {lim}", small_style))

    doc.build(elements)
    return output_path
