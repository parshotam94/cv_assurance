"""
Dataset Assurance REST API Endpoints
"""
import uuid
import json
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from backend.app.config import settings
from backend.app.database.database import get_db
from backend.app.database.repository import Repository
from backend.app.database.models import DatasetModel, DatasetSampleModel, FindingModel
from backend.app.core.security import sanitize_filename, validate_file_extension
from backend.app.core.hashing import hash_file
from backend.app.core.audit import log_audit_event
from backend.app.core.risk_engine import calculate_finding_risk, aggregate_asset_risk
from backend.app.dataset.loader import extract_archive_if_needed, detect_dataset_format
from backend.app.dataset.coco import parse_coco_dataset
from backend.app.dataset.yolo import parse_yolo_dataset
from backend.app.dataset.validation import validate_dataset_records
from backend.app.dataset.duplicates import detect_near_duplicates
from backend.app.dataset.label_analysis import analyze_labels_and_mislabelling
from backend.app.dataset.ood import detect_out_of_distribution
from backend.app.dataset.poisoning import detect_trigger_backdoors
from backend.app.dataset.source_analysis import aggregate_source_risks

router = APIRouter(prefix="/datasets", tags=["Dataset Assurance"])

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a dataset archive (.zip) and extract it for assurance evaluation."""
    clean_name = sanitize_filename(file.filename)
    validate_file_extension(clean_name, {".zip", ".tar", ".gz"})

    dataset_id = f"DS-{uuid.uuid4().hex[:10].upper()}"
    target_dir = settings.DATASET_DIR / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_path = target_dir / clean_name
    with open(archive_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_sha256 = hash_file(archive_path)

    # Extract
    extracted_dir = extract_archive_if_needed(archive_path, target_dir / "extracted")
    detected_format = detect_dataset_format(extracted_dir)

    # Initial parse
    if detected_format == "COCO":
        records, class_names, meta = parse_coco_dataset(extracted_dir)
    elif detected_format == "YOLO":
        records, class_names, meta = parse_yolo_dataset(extracted_dir)
    else:
        # Generic images or fallback YOLO
        records, class_names, meta = parse_yolo_dataset(extracted_dir)
        detected_format = "YOLO_OR_GENERIC"

    valid_records, validation_issues = validate_dataset_records(records)

    repo = Repository(db)
    ds_record = repo.create_dataset({
        "id": dataset_id,
        "name": dataset_name or clean_name,
        "format": detected_format,
        "file_path": str(extracted_dir.resolve()),
        "sha256": file_sha256,
        "status": "REGISTERED",
        "sample_count": len(valid_records),
        "annotation_count": meta.get("total_annotations", 0),
        "classes": json.dumps(class_names),
        "metadata_json": json.dumps(meta),
        "risk_score": 0.0,
        "disposition": "ACCEPT"
    })

    # Save samples into database
    sample_objs = []
    for r in valid_records:
        sample_objs.append({
            "dataset_id": dataset_id,
            "sample_id": r.sample_id,
            "image_path": r.image_path,
            "source_id": r.source_id,
            "batch_id": r.batch_id,
            "contributor": r.contributor,
            "width": r.width,
            "height": r.height,
            "labels": json.dumps(r.labels),
            "annotations": json.dumps(r.annotations),
            "metadata_json": json.dumps(r.metadata),
            "is_suspicious": False,
            "anomaly_reasons": "[]"
        })
    repo.add_samples(sample_objs)

    log_audit_event(
        db,
        action="DATASET_UPLOAD",
        asset=f"Dataset:{dataset_id}",
        metadata={"name": ds_record.name, "format": detected_format, "samples": len(valid_records), "sha256": file_sha256}
    )

    return {
        "dataset_id": dataset_id,
        "name": ds_record.name,
        "format": detected_format,
        "sample_count": len(valid_records),
        "classes": class_names,
        "validation_issues_count": len(validation_issues)
    }

@router.post("/{dataset_id}/analyze")
def run_dataset_assurance(
    dataset_id: str,
    duplicate_threshold: float = settings.DUPLICATE_SIMILARITY_THRESHOLD,
    ood_threshold: float = settings.OOD_PERCENTILE_THRESHOLD,
    db: Session = Depends(get_db)
):
    """
    Run comprehensive assurance analysis on a registered dataset:
    - Near-duplicate detection & flooding
    - Label imbalance & systematic mislabelling (k-NN)
    - Out-of-Distribution (OOD) modeling
    - Trigger / backdoor pattern search
    - Hierarchical source-level risk rollup
    - Explainable finding generation
    """
    repo = Repository(db)
    ds = repo.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    extracted_dir = Path(ds.file_path)
    if ds.format == "COCO":
        records, class_names, _ = parse_coco_dataset(extracted_dir)
    else:
        records, class_names, _ = parse_yolo_dataset(extracted_dir)

    valid_records, _ = validate_dataset_records(records)
    total_population = len(valid_records)

    # 1. Duplicates
    dup_clusters, sample_dup_map = detect_near_duplicates(valid_records, similarity_threshold=duplicate_threshold)

    # 2. Label Integrity & Systematic Mislabelling
    label_stats, mislabelled_samples = analyze_labels_and_mislabelling(valid_records, class_names)

    # 3. OOD Detection
    ood_summary, ood_samples = detect_out_of_distribution(valid_records, percentile_threshold=ood_threshold)

    # 4. Trigger / Backdoor Search
    trigger_clusters, trigger_samples = detect_trigger_backdoors(valid_records)

    # 5. Source-level aggregation
    source_results = aggregate_source_risks(
        valid_records, sample_dup_map, mislabelled_samples, ood_samples, trigger_samples
    )
    repo.update_sources(dataset_id, source_results)

    # Generate Structured Findings
    findings_list = []
    finding_counter = 1

    # Finding: Duplicate clusters
    if dup_clusters:
        flooding_clusters = [c for c in dup_clusters if c["is_flooding"]]
        sev = "HIGH" if flooding_clusters else "MEDIUM"
        conf = 0.94
        affected = [s for c in dup_clusters for s in c["sample_ids"]]
        score, lvl, disp = calculate_finding_risk("NEAR_DUPLICATE", sev, conf, len(affected), total_population)
        
        evidence = [
            f"Detected {len(dup_clusters)} near-duplicate cluster(s) with Hamming distance similarity >= {duplicate_threshold}.",
            f"Total duplicated samples: {len(affected)} across clusters.",
            f"Duplicate flooding detected in {len(flooding_clusters)} large cluster(s)." if flooding_clusters else "No critical flooding detected."
        ]
        
        findings_list.append({
            "finding_id": f"F-{dataset_id}-{finding_counter:03d}",
            "asset_type": "DATASET",
            "asset_id": dataset_id,
            "category": "NEAR_DUPLICATE",
            "severity": sev,
            "confidence": conf,
            "risk_score": score,
            "title": f"Perceptual Near-Duplicate Image Clusters Detected ({len(dup_clusters)} clusters)",
            "description": f"Image perceptual hashing (pHash/dHash) identified {len(affected)} images forming near-identical visual clusters, indicating redundant data ingestion or duplicate flooding.",
            "evidence": json.dumps(evidence),
            "affected_samples": json.dumps(affected[:50]),
            "detection_method": "pHash & dHash Perceptual Hashing",
            "limitations": json.dumps(["Heavy geometrical cropping or severe non-linear warping may not match 64-bit perceptual hashes."]),
            "recommendation": disp
        })
        finding_counter += 1

    # Finding: Systematic Mislabelling / Label Noise
    if mislabelled_samples:
        sev = "HIGH" if len(mislabelled_samples) >= 5 else "MEDIUM"
        conf = 0.86
        affected = [s["sample_id"] for s in mislabelled_samples]
        score, lvl, disp = calculate_finding_risk("LABEL_FLIP", sev, conf, len(affected), total_population)

        evidence = [
            f"{len(mislabelled_samples)} samples strongly conflict with their visual feature nearest neighbors (disagreement >= 80%).",
            f"Class imbalance ratio is {label_stats['imbalance_ratio']}:1 across dataset.",
            f"Top affected source(s): {list(set([s['source_id'] for s in mislabelled_samples]))}"
        ]

        findings_list.append({
            "finding_id": f"F-{dataset_id}-{finding_counter:03d}",
            "asset_type": "DATASET",
            "asset_id": dataset_id,
            "category": "LABEL_FLIP",
            "severity": sev,
            "confidence": conf,
            "risk_score": score,
            "title": f"Systematic Mislabelling & Label Noise Detected ({len(mislabelled_samples)} samples)",
            "description": "Visual feature neighborhood consensus revealed samples whose assigned ground-truth label directly contradicts the consensus label of their nearest visual neighbors.",
            "evidence": json.dumps(evidence),
            "affected_samples": json.dumps(affected[:50]),
            "detection_method": "k-Nearest Neighbors (k=5) Feature Disagreement",
            "limitations": json.dumps(["Relies on low-level statistical features; subtle semantic distinctions may exhibit overlap."]),
            "recommendation": disp
        })
        finding_counter += 1

    # Finding: Out-of-Distribution Samples
    if ood_samples:
        sev = "MEDIUM" if len(ood_samples) < 10 else "HIGH"
        conf = ood_summary["confidence"]
        affected = [s["sample_id"] for s in ood_samples]
        score, lvl, disp = calculate_finding_risk("OOD", sev, conf, len(affected), total_population)

        evidence = [
            f"{len(ood_samples)} candidate samples exceeded the {int(ood_threshold*100)}th anomaly percentile.",
            f"Overall population OOD anomaly score is {ood_summary['ood_score']}.",
            f"Evaluation method: {ood_summary['detection_method']}."
        ]

        findings_list.append({
            "finding_id": f"F-{dataset_id}-{finding_counter:03d}",
            "asset_type": "DATASET",
            "asset_id": dataset_id,
            "category": "OOD",
            "severity": sev,
            "confidence": conf,
            "risk_score": score,
            "title": f"Out-of-Distribution Inliers/Outliers Detected ({len(ood_samples)} samples)",
            "description": "Isolation Forest modeling flagged samples exhibiting anomalous multivariate visual feature distributions outside the normal training manifold.",
            "evidence": json.dumps(evidence),
            "affected_samples": json.dumps(affected[:50]),
            "detection_method": ood_summary["detection_method"],
            "limitations": json.dumps(["Unsupervised density estimation on candidate dataset; highly benefited by explicit reference baseline."]),
            "recommendation": disp
        })
        finding_counter += 1

    # Finding: Trigger / Backdoor Clusters
    if trigger_clusters:
        sev = "CRITICAL" if any(c["severity"] == "HIGH" for c in trigger_clusters) else "HIGH"
        conf = 0.89
        affected = [s for c in trigger_clusters for s in c["affected_samples"]]
        score, lvl, disp = calculate_finding_risk("TRIGGER", sev, conf, len(affected), total_population)

        evidence = [
            f"Identified {len(trigger_clusters)} localized trigger pattern cluster(s).",
            f"Patch repetitions correlated with target class(es): {list(set([c['associated_class'] for c in trigger_clusters]))}."
        ]
        for c in trigger_clusters:
            evidence.extend(c["evidence"])

        findings_list.append({
            "finding_id": f"F-{dataset_id}-{finding_counter:03d}",
            "asset_type": "DATASET",
            "asset_id": dataset_id,
            "category": "TRIGGER",
            "severity": sev,
            "confidence": conf,
            "risk_score": score,
            "title": f"Potential Visual Trigger / Backdoor Patch Identified ({len(trigger_clusters)} clusters)",
            "description": "Spatial patch analysis identified repeated localized visual patterns sharing high cosine similarity and disproportionate correlation with specific target labels.",
            "evidence": json.dumps(evidence),
            "affected_samples": json.dumps(affected[:50]),
            "detection_method": "Multi-region Localized Patch Cosine Correlation",
            "limitations": json.dumps(["Detects fixed spatial patches; steganographic, blended, or frequency-domain watermarks require gradient-based attribution."]),
            "recommendation": "QUARANTINE"
        })
        finding_counter += 1

    # Finding: Source-level Poisoning
    critical_sources = [s for s in source_results if s["risk_level"] in ["CRITICAL", "HIGH"]]
    if critical_sources:
        sev = "HIGH"
        conf = 0.85
        score, lvl, disp = calculate_finding_risk("SOURCE_POISONING", sev, conf, len(critical_sources), len(source_results))
        evidence = [
            f"Sources flagged with elevated risk: {[s['source_id'] for s in critical_sources]}.",
            f"Contributors with disproportionate anomaly rates: {[c for s in critical_sources for c in s['anomalous_contributors']]}."
        ]
        findings_list.append({
            "finding_id": f"F-{dataset_id}-{finding_counter:03d}",
            "asset_type": "DATASET",
            "asset_id": dataset_id,
            "category": "SOURCE_POISONING",
            "severity": sev,
            "confidence": conf,
            "risk_score": score,
            "title": f"Source/Contributor Poisoning Anomaly ({len(critical_sources)} sources)",
            "description": "Hierarchical risk aggregation identified specific data contributors or ingestion batches contributing disproportionate volumes of anomalous samples.",
            "evidence": json.dumps(evidence),
            "affected_samples": json.dumps([]),
            "detection_method": "Hierarchical Contributor & Batch Anomaly Rollup",
            "limitations": json.dumps(["Requires contributor and source batch metadata in dataset records."]),
            "recommendation": disp
        })

    # Save findings
    db.query(FindingModel).filter(FindingModel.asset_id == dataset_id).delete()
    for f in findings_list:
        repo.create_finding(f)

    # Mark suspicious samples in DB
    all_susp_sids = set(sample_dup_map.keys()) | {s["sample_id"] for s in mislabelled_samples} | {s["sample_id"] for s in ood_samples} | {s["sample_id"] for s in trigger_samples}
    
    samples_in_db = db.query(DatasetSampleModel).filter(DatasetSampleModel.dataset_id == dataset_id).all()
    for s in samples_in_db:
        if s.sample_id in all_susp_sids:
            s.is_suspicious = True
            reasons = []
            if s.sample_id in sample_dup_map:
                reasons.append("Near duplicate cluster")
            if s.sample_id in {m["sample_id"] for m in mislabelled_samples}:
                reasons.append("Label disagreement with neighbors")
            if s.sample_id in {o["sample_id"] for o in ood_samples}:
                reasons.append("Feature outlier (OOD)")
            if s.sample_id in {t["sample_id"] for t in trigger_samples}:
                reasons.append("Contains repeated localized trigger patch")
            s.anomaly_reasons = json.dumps(reasons)
    db.commit()

    # Aggregate overall dataset risk
    overall_risk = aggregate_asset_risk(findings_list)
    repo.update_dataset(dataset_id, {
        "status": "ANALYZED",
        "risk_score": overall_risk["risk_score"],
        "disposition": overall_risk["disposition"],
        "analysis_summary": json.dumps({
            "duplicate_clusters": len(dup_clusters),
            "mislabelled_count": len(mislabelled_samples),
            "ood_count": len(ood_samples),
            "trigger_clusters": len(trigger_clusters),
            "sources_evaluated": len(source_results),
            "overall_explanation": overall_risk["explanation"]
        })
    })

    log_audit_event(
        db,
        action="DATASET_ANALYSIS",
        asset=f"Dataset:{dataset_id}",
        metadata={"risk_score": overall_risk["risk_score"], "disposition": overall_risk["disposition"], "findings": len(findings_list)}
    )

    return {
        "dataset_id": dataset_id,
        "status": "ANALYZED",
        "overall_risk_score": overall_risk["risk_score"],
        "overall_risk_level": overall_risk["risk_level"],
        "disposition": overall_risk["disposition"],
        "findings_count": len(findings_list),
        "duplicate_clusters": len(dup_clusters),
        "mislabelled_samples": len(mislabelled_samples),
        "ood_samples": len(ood_samples),
        "trigger_clusters": len(trigger_clusters),
        "source_profiles": source_results
    }

@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    """List all registered datasets."""
    repo = Repository(db)
    datasets = repo.list_datasets()
    return [
        {
            "id": d.id,
            "name": d.name,
            "format": d.format,
            "sample_count": d.sample_count,
            "annotation_count": d.annotation_count,
            "status": d.status,
            "risk_score": d.risk_score,
            "disposition": d.disposition,
            "created_at": d.created_at.isoformat() if d.created_at else None
        }
        for d in datasets
    ]

@router.get("/{dataset_id}")
def get_dataset_details(dataset_id: str, db: Session = Depends(get_db)):
    """Retrieve detailed metadata, sources, and analysis summary for a dataset."""
    repo = Repository(db)
    ds = repo.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    sources = repo.get_sources(dataset_id)
    findings = repo.list_findings(asset_id=dataset_id)

    return {
        "id": ds.id,
        "name": ds.name,
        "format": ds.format,
        "sha256": ds.sha256,
        "sample_count": ds.sample_count,
        "annotation_count": ds.annotation_count,
        "classes": json.loads(ds.classes) if ds.classes else [],
        "risk_score": ds.risk_score,
        "disposition": ds.disposition,
        "status": ds.status,
        "summary": json.loads(ds.analysis_summary) if ds.analysis_summary else {},
        "sources": [
            {
                "source_id": s.source_id,
                "samples": s.sample_count,
                "suspicious": s.suspicious_count,
                "risk_score": s.risk_score,
                "risk_level": s.risk_level,
                "disposition": s.disposition
            }
            for s in sources
        ],
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

@router.get("/{dataset_id}/samples")
def list_dataset_samples(
    dataset_id: str,
    suspicious_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Retrieve samples for interactive Sample Explorer."""
    repo = Repository(db)
    samples = repo.get_dataset_samples(dataset_id, limit=limit, offset=offset, suspicious_only=suspicious_only)
    return [
        {
            "id": s.id,
            "sample_id": s.sample_id,
            "source_id": s.source_id,
            "batch_id": s.batch_id,
            "contributor": s.contributor,
            "labels": json.loads(s.labels) if s.labels else [],
            "annotations": json.loads(s.annotations) if s.annotations else [],
            "width": s.width,
            "height": s.height,
            "is_suspicious": s.is_suspicious,
            "anomaly_reasons": json.loads(s.anomaly_reasons) if s.anomaly_reasons else [],
            "image_url": f"/api/datasets/image/{dataset_id}/{s.sample_id}"
        }
        for s in samples
    ]

@router.get("/image/{dataset_id}/{sample_id}")
def serve_sample_image(dataset_id: str, sample_id: str, db: Session = Depends(get_db)):
    """Serve image binary safely for local inspection in Sample Explorer."""
    sample = db.query(DatasetSampleModel).filter(
        DatasetSampleModel.dataset_id == dataset_id,
        DatasetSampleModel.sample_id == sample_id
    ).first()
    if not sample or not Path(sample.image_path).exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(sample.image_path)
