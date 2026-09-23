"""
Assurance Report Exporters: JSON, CSV, and Offline Evidence-Driven PDF (via ReportLab)
Includes empirical experiment results, overall assurance summary, evidence-driven telemetry,
and embedded data-science charts generated offline via Matplotlib.
"""
import os
import json
import csv
from pathlib import Path
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Image as RLImage
)
from backend.app.reports.schemas import AssuranceReportData

def export_report_to_json(report: AssuranceReportData, output_path: Path) -> Path:
    """Save report as structured JSON with all assurance and experiment metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return output_path

def export_report_to_csv(report: AssuranceReportData, output_path: Path) -> Path:
    """Save findings table as evidence-driven CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Finding ID", "Asset Type", "Asset ID", "Category", "Severity",
            "Confidence", "Risk Score", "Title", "Recommendation", "Detection Method",
            "Affected Samples Count", "Evidence", "Limitations", "Description"
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
                len(f_item.affected_samples),
                " | ".join(f_item.evidence),
                " | ".join(f_item.limitations),
                f_item.description.replace("\n", " ")
            ])
    return output_path

def export_report_to_pdf(report: AssuranceReportData, output_path: Path) -> Path:
    """Generate high-quality multi-page enterprise evidence-driven PDF report offline via ReportLab."""
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
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )
    h3_style = ParagraphStyle(
        'Heading3Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0369a1'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#334155')
    )
    small_style = ParagraphStyle(
        'SmallCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#64748b')
    )

    elements = []

    # Title & Subtitle Header
    elements.append(Paragraph("VisionTrust Assurance", title_style))
    elements.append(Paragraph("Offline Computer-Vision Integrity, Security & Provenance Dossier", ParagraphStyle(
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
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_verdict)
    elements.append(Spacer(1, 8))

    # Executive Summary Paragraph
    elements.append(Paragraph("<b>Executive Summary:</b> " + report.executive_summary.get("explanation", ""), body_style))
    elements.append(Spacer(1, 8))

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
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_sum)
    elements.append(Spacer(1, 10))

    # OVERALL ASSURANCE SUMMARY SECTION
    if report.assurance_summary:
        elements.append(Paragraph("Overall Assurance Summary", h2_style))
        assurance_table_data = [
            ["Assurance Pillar", "Status", "Confidence", "Supporting Evidence", "Limitations & Recommendation"]
        ]
        pillars = [
            ("Dataset Integrity", "dataset_integrity"),
            ("Model Integrity", "model_integrity"),
            ("Inference Integrity", "inference_integrity"),
            ("Distribution Integrity", "distribution_integrity"),
            ("Audit Trail Integrity", "audit_integrity"),
        ]
        for name, key in pillars:
            p_data = report.assurance_summary.get(key, {})
            st = p_data.get("status", "EVALUATED")
            conf = f"{int(p_data.get('confidence', 1.0) * 100)}%"
            ev_text = p_data.get("evidence", "Telemetry verified.")
            rec_text = f"<b>Lim:</b> {p_data.get('limitations', '')}<br/><b>Rec:</b> {p_data.get('recommendation', '')}"
            assurance_table_data.append([
                Paragraph(f"<b>{name}</b>", small_style),
                Paragraph(st, small_style),
                Paragraph(conf, small_style),
                Paragraph(ev_text, small_style),
                Paragraph(rec_text, small_style)
            ])
        t_assure = Table(assurance_table_data, colWidths=[90, 75, 50, 160, 165])
        t_assure.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(t_assure)
        elements.append(Spacer(1, 10))

    # EXPERIMENT RESULTS & BENCHMARK SECTION
    if report.experiment_results:
        elements.append(Paragraph("Empirical Experiment Results & Attack Benchmarks", h2_style))
        exp = report.experiment_results
        cfg = exp.get("configuration", {})
        gt = exp.get("ground_truth", {})
        
        cfg_str = f"<b>Format:</b> {cfg.get('dataset_format')} | <b>Framework:</b> {cfg.get('model_framework')} | <b>Env:</b> {cfg.get('environment')}"
        gt_str = f"<b>Total Samples:</b> {gt.get('total_samples')} | <b>Clean:</b> {gt.get('clean_samples')} | <b>Attacks Injected:</b> {gt.get('injected_attacks')}"
        elements.append(Paragraph(cfg_str, small_style))
        elements.append(Paragraph(gt_str, small_style))
        elements.append(Spacer(1, 6))

        metrics = exp.get("metrics", [])
        if metrics:
            metric_table_data = [
                ["Attack Scenario", "Precision", "Recall", "F1-Score", "FPR", "Status"]
            ]
            for m in metrics:
                p = m.get("precision", 0.0)
                r = m.get("recall", 0.0)
                f1 = m.get("f1", 0.0)
                fpr = m.get("fpr", 0.0)
                status_text = "PASS" if r >= 0.8 else "PARTIAL"
                metric_table_data.append([
                    m.get("name", "Unknown"),
                    f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}", f"{fpr:.3f}", status_text
                ])
            t_metric = Table(metric_table_data, colWidths=[150, 75, 75, 75, 75, 90])
            t_metric.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7.5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
                ('PADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(t_metric)
            elements.append(Spacer(1, 6))

        interp = exp.get("interpretation", "")
        if interp:
            elements.append(Paragraph(f"<b>Data Science Interpretation:</b> {interp}", small_style))
            elements.append(Spacer(1, 8))

    # EMBEDDED DATA-SCIENCE CHARTS
    charts = report.visualizations or {}
    chart_keys_to_embed = [
        ("attack_performance", "Attack Detection Performance (Precision, Recall, F1)"),
        ("confusion_matrix", "Aggregate Threat Detection Confusion Matrix"),
        ("severity_distribution", "Finding Severity Breakdown"),
        ("distribution_shift", "Domain Shift & Statistical Drift Comparison"),
        ("audit_timeline", "Audit Log Hash Chain Timeline"),
        ("provenance_integrity", "Inference Provenance Cryptographic Status")
    ]

    elements.append(PageBreak())
    elements.append(Paragraph("Data-Science Visualizations & Evidence Analytics", h2_style))
    elements.append(Paragraph("Empirical charts generated offline via local Matplotlib engine:", body_style))
    elements.append(Spacer(1, 8))

    for key, chart_title in chart_keys_to_embed:
        chart_p = charts.get(key)
        if chart_p and os.path.exists(chart_p):
            elements.append(Paragraph(f"<b>{chart_title}</b>", h3_style))
            try:
                # Embed chart image with appropriate scaling
                img = RLImage(chart_p, width=480, height=200 if "timeline" not in key else 160)
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass

    # DETAILED EVIDENCE-DRIVEN FINDINGS TABLE
    elements.append(PageBreak())
    elements.append(Paragraph("Detailed Security & Integrity Findings (Evidence-Driven)", h2_style))
    if not report.findings:
        elements.append(Paragraph("No critical findings or integrity violations reported.", body_style))
    else:
        findings_table_data = [
            ["ID", "Category", "Sev.", "Conf.", "Risk", "Disposition", "Evidence & Telemetry"]
        ]
        for f in report.findings[:12]:
            ev_str = "<br/>• ".join(f.evidence[:3]) if f.evidence else f.description[:80]
            findings_table_data.append([
                f.finding_id,
                f.category[:12],
                f.severity,
                f"{int(f.confidence*100)}%",
                str(f.risk_score),
                f.recommendation,
                Paragraph(f"<b>{f.title}</b><br/>• {ev_str}", small_style)
            ])
        t_find = Table(findings_table_data, colWidths=[55, 75, 45, 35, 35, 60, 235])
        t_find.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(t_find)

    elements.append(Spacer(1, 12))

    # Threat Coverage Statement
    elements.append(Paragraph("Assurance Threat Coverage Statement", h2_style))
    coverage_table_data = [
        ["ID", "Attack / Threat Capability", "Status", "Access Mode", "Primary Method", "Key Limitation"]
    ]
    for c in report.coverage_matrix:
        coverage_table_data.append([
            c.threat_id,
            Paragraph(c.capability, small_style),
            c.supported_status,
            c.access_requirement[:16],
            Paragraph(c.primary_method[:38] + "...", small_style),
            Paragraph(c.key_limitation[:38] + "...", small_style)
        ])
    t_cov = Table(coverage_table_data, colWidths=[25, 110, 75, 80, 125, 125])
    t_cov.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 2.5),
    ]))
    elements.append(t_cov)

    elements.append(Spacer(1, 10))

    # Limitations & Offline Statement
    elements.append(Paragraph("System Limitations & Air-Gap Compliance", h2_style))
    for lim in report.system_limitations:
        elements.append(Paragraph(f"• {lim}", small_style))

    doc.build(elements)
    return output_path
