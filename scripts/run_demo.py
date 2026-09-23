"""
End-to-End Automated Demonstration and Evaluation Benchmark
Executes all 8 attack scenarios, analyzes findings against ground truth, and calculates
empirical metrics: Precision, Recall, F1, False Positive Rate (FPR), and Detection Rate.
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.database.database import SessionLocal, init_db
from backend.app.database.repository import Repository
from backend.app.database.models import DatasetModel, DatasetSampleModel, DatasetSourceModel, InferenceRecordModel, FindingModel
from backend.app.core.audit import log_audit_event, verify_audit_chain
from experiments.create_demo_model import create_demo_models
from experiments.generate_poisoned_data import generate_poisoned_experiment_dataset
from experiments.generate_shift import generate_shift_experiment_datasets
from experiments.generate_tampering import generate_inference_tamper_scenarios

# Import analysis functions directly
from backend.app.dataset.coco import parse_coco_dataset
from backend.app.dataset.yolo import parse_yolo_dataset
from backend.app.dataset.validation import validate_dataset_records
from backend.app.dataset.duplicates import detect_near_duplicates
from backend.app.dataset.label_analysis import analyze_labels_and_mislabelling
from backend.app.dataset.ood import detect_out_of_distribution
from backend.app.dataset.poisoning import detect_trigger_backdoors
from backend.app.dataset.source_analysis import aggregate_source_risks
from backend.app.model.loader import get_model_adapter
from backend.app.model.fingerprint import generate_model_fingerprint, compare_model_fingerprints
from backend.app.model.reference_battery import evaluate_behavioural_battery
from backend.app.model.trigger_search import evaluate_model_trigger_sensitivity
from backend.app.inference.verifier import verify_provenance_record
from backend.app.inference.schema import ProvenanceRecord
from backend.app.inference.replay import check_for_replay_attack
from backend.app.distribution.environmental_shift import analyze_environmental_shift
from backend.app.reports.generator import build_assurance_report
from backend.app.reports.exporters import export_report_to_pdf, export_report_to_json, export_report_to_csv

def run_full_assurance_demo():
    print("=" * 80)
    print("VISIONTRUST ASSURANCE - AIR-GAPPED EVALUATION BENCHMARK")
    print("=" * 80)

    db = SessionLocal()
    init_db()
    repo = Repository(db)

    # -------------------------------------------------------------
    # STEP 1: Generate Controlled Attack Dataset & Models
    # -------------------------------------------------------------
    print("\n[Step 1/12] Generating reproducible attack datasets and ground truth...")
    dataset_info = generate_poisoned_experiment_dataset()
    gt_path = Path(dataset_info["ground_truth_path"])
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"  - Total samples: {ground_truth['total_samples']}")
    print(f"  - Injected attacks: {dataset_info['attacks']}")

    print("\n[Step 2/12] Generating reference & candidate models...")
    model_paths = create_demo_models()
    print(f"  - Models created: PyTorch reference, ONNX reference, Candidate PT, Substituted ONNX.")

    # -------------------------------------------------------------
    # STEP 3: Dataset Ingestion & Validation
    # -------------------------------------------------------------
    print("\n[Step 3/12] Ingesting and validating candidate COCO dataset...")
    records, class_names, meta = parse_coco_dataset(Path(dataset_info["dataset_dir"]))
    valid_records, validation_issues = validate_dataset_records(records)
    print(f"  - Validated records: {len(valid_records)} (Issues: {len(validation_issues)})")

    dataset_id = "DS-BENCHMARK-01"
    # Ensure idempotency by deleting any previous benchmark records
    db.query(DatasetSampleModel).filter(DatasetSampleModel.dataset_id == dataset_id).delete()
    db.query(DatasetSourceModel).filter(DatasetSourceModel.dataset_id == dataset_id).delete()
    db.query(FindingModel).filter(FindingModel.asset_id == dataset_id).delete()
    db.query(DatasetModel).filter(DatasetModel.id == dataset_id).delete()
    db.commit()

    repo.create_dataset({
        "id": dataset_id,
        "name": "Benchmark_Poisoned_Dataset",
        "format": "COCO",
        "file_path": dataset_info["dataset_dir"],
        "sample_count": len(valid_records),
        "classes": json.dumps(class_names),
        "status": "ANALYZING"
    })

    # -------------------------------------------------------------
    # STEP 4: Dataset Integrity Analysis
    # -------------------------------------------------------------
    print("\n[Step 4/12] Running dataset integrity engines...")
    
    # 4a. Near duplicates — threshold=0.95 (hamming ≤ 3 bits) with AND logic and min_cluster_size=3 for flooding
    dup_clusters, sample_dup_map = detect_near_duplicates(valid_records, similarity_threshold=0.95, min_cluster_size=3)
    detected_duplicates = set(sample_dup_map.keys())
    print(f"  - Perceptual duplicate clusters: {len(dup_clusters)} (affected samples: {len(detected_duplicates)})")

    # 4b. Label mislabelling
    _, mislabelled_samples = analyze_labels_and_mislabelling(valid_records, class_names)
    detected_mislabels = {s["sample_id"] for s in mislabelled_samples}
    print(f"  - Suspicious mislabelled samples: {len(detected_mislabels)}")

    # 4c. OOD detection
    ood_summary, ood_samples = detect_out_of_distribution(valid_records, percentile_threshold=0.92)
    detected_oods = {s["sample_id"] for s in ood_samples}
    print(f"  - Out-of-distribution samples: {len(detected_oods)}")

    # 4d. Trigger detection
    trigger_clusters, trigger_samples = detect_trigger_backdoors(valid_records)
    detected_triggers = {s["sample_id"] for s in trigger_samples}
    print(f"  - Localized trigger clusters: {len(trigger_clusters)} (samples: {len(detected_triggers)})")

    # 4e. Source risk aggregation
    source_risks = aggregate_source_risks(valid_records, sample_dup_map, mislabelled_samples, ood_samples, trigger_samples)
    repo.update_sources(dataset_id, source_risks)
    flagged_sources = [s["source_id"] for s in source_risks if s["risk_level"] in ["CRITICAL", "HIGH"]]
    print(f"  - High-risk poisoned sources flagged: {flagged_sources}")

    # -------------------------------------------------------------
    # STEP 5: Model Integrity & Substitution Analysis
    # -------------------------------------------------------------
    print("\n[Step 5/12] Running model integrity and fingerprinting...")
    ref_adapter = get_model_adapter(Path(model_paths["reference_onnx"]))
    ref_adapter.load()
    ref_fp = generate_model_fingerprint(ref_adapter)

    sub_adapter = get_model_adapter(Path(model_paths["substituted_onnx"]))
    sub_adapter.load()
    sub_fp = generate_model_fingerprint(sub_adapter)

    substitution_result = compare_model_fingerprints(ref_fp, sub_fp)
    print(f"  - Model substitution test status: {substitution_result['status']} (Sev: {substitution_result['severity']})")

    # -------------------------------------------------------------
    # STEP 6: Model Behavioural Battery
    # -------------------------------------------------------------
    print("\n[Step 6/12] Evaluating model behavioural battery & trigger sensitivity...")
    battery_res = evaluate_behavioural_battery(sub_adapter, ref_adapter, battery_samples=24)
    print(f"  - Behavioural agreement: {battery_res.get('behavioural_agreement_rate')}")
    print(f"  - Anomalous battery tests: {battery_res['anomalous_inputs_count']} / {battery_res['total_battery_tests']}")

    trigger_sens = evaluate_model_trigger_sensitivity(sub_adapter, sample_count=15)
    print(f"  - Trigger perturbation vulnerability: {trigger_sens['high_vulnerability_detected']}")

    # -------------------------------------------------------------
    # STEP 7: Inference Provenance & Tamper Testing
    # -------------------------------------------------------------
    print("\n[Step 7/12] Testing cryptographic inference provenance...")
    tamper_scenarios = generate_inference_tamper_scenarios(model_sha256=ref_fp["sha256"])
    
    clean_rec = ProvenanceRecord(**tamper_scenarios["authentic_record"])
    tampered_rec = ProvenanceRecord(**tamper_scenarios["tampered_record"])
    replayed_rec = ProvenanceRecord(**tamper_scenarios["replayed_record"])

    # Verify authentic
    clean_verif = verify_provenance_record(clean_rec, current_output_data=clean_rec.raw_output)
    print(f"  - Authentic inference record: {clean_verif.status} (Valid: {clean_verif.is_valid})")

    # Verify tampered
    tamp_verif = verify_provenance_record(tampered_rec, current_output_data=tampered_rec.raw_output)
    print(f"  - Tampered inference record: {tamp_verif.status} (Detected in: {tamp_verif.tamper_detected_in})")

    # Persist clean record for replay check (ensure idempotency)
    db.query(InferenceRecordModel).filter(InferenceRecordModel.record_id == clean_rec.record_id).delete()
    db.commit()
    repo.create_inference_record({
        "record_id": clean_rec.record_id,
        "input_sha256": clean_rec.input_sha256,
        "model_sha256": clean_rec.model_sha256,
        "preprocessing_sha256": clean_rec.preprocessing_sha256,
        "config_sha256": clean_rec.config_sha256,
        "output_sha256": clean_rec.output_sha256,
        "timestamp": clean_rec.timestamp,
        "nonce": clean_rec.nonce,
        "sequence": clean_rec.sequence,
        "record_hash": clean_rec.record_hash,
        "signature": clean_rec.signature,
        "public_key_hex": clean_rec.public_key_hex,
        "is_tampered": False,
        "is_replayed": False,
        "verification_status": "VERIFIED",
        "payload_json": json.dumps(clean_rec.raw_output)
    })

    # Test replay
    replay_res = check_for_replay_attack(db, replayed_rec)
    print(f"  - Replay attack detected: {replay_res['is_replay']} (Reasons: {len(replay_res['reasons'])})")

    # -------------------------------------------------------------
    # STEP 8: Distribution Shift Evaluation
    # -------------------------------------------------------------
    print("\n[Step 8/12] Evaluating distribution shift engine...")
    shift_datasets = generate_shift_experiment_datasets()
    ref_yolo, _, _ = parse_yolo_dataset(Path(shift_datasets["reference_zip"]).with_suffix(""))
    cand_yolo, _, _ = parse_yolo_dataset(Path(shift_datasets["candidate_zip"]).with_suffix(""))
    
    shift_analysis = analyze_environmental_shift(
        [r.image_path for r in ref_yolo],
        [r.image_path for r in cand_yolo]
    )
    print(f"  - Distribution shift classification: {shift_analysis['classification']}")
    print(f"  - Overall shift detected: {shift_analysis['overall_shift_detected']}")

    # -------------------------------------------------------------
    # STEP 9: Audit Trail Verification
    # -------------------------------------------------------------
    print("\n[Step 9/12] Validating tamper-evident audit log hash chain...")
    log_audit_event(db, action="BENCHMARK_EVALUATION", asset="Benchmark", metadata={"status": "COMPLETED"})
    audit_chain_res = verify_audit_chain(db)
    print(f"  - Audit chain integrity: {audit_chain_res['status']} ({audit_chain_res['total_events']} events)")

    # -------------------------------------------------------------
    # STEP 10: Generate Comprehensive Assurance Report
    # -------------------------------------------------------------
    print("\n[Step 10/12] Generating comprehensive assurance report (PDF, JSON, CSV)...")
    report = build_assurance_report(db, dataset_id=dataset_id)
    pdf_p = export_report_to_pdf(report, settings.REPORT_DIR / "benchmark_assurance_report.pdf")
    json_p = export_report_to_json(report, settings.REPORT_DIR / "benchmark_assurance_report.json")
    csv_p = export_report_to_csv(report, settings.REPORT_DIR / "benchmark_findings.csv")
    print(f"  - Report generated: {report.report_id}")
    print(f"  - PDF saved: {pdf_p}")

    # -------------------------------------------------------------
    # STEP 11 & 12: Empirical Metrics Calculation
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK EVALUATION METRICS (CALCULATED)")
    print("=" * 80)

    # Calculate real TP, FP, FN for dataset attacks
    def calc_metrics(ground_truth_set, detected_set, clean_set):
        tp = len(ground_truth_set.intersection(detected_set))
        fn = len(ground_truth_set - detected_set)
        fp = len(clean_set.intersection(detected_set))
        tn = len(clean_set - detected_set)

        precision = round(tp / max(1, (tp + fp)), 3)
        recall = round(tp / max(1, (tp + fn)), 3)
        f1 = round(2 * (precision * recall) / max(1e-5, (precision + recall)), 3)
        fpr = round(fp / max(1, (fp + tn)), 3)
        detection_rate = recall

        return {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "detection_rate": detection_rate
        }

    clean_samples = set(ground_truth["clean_samples"])
    m_dup = calc_metrics(set(ground_truth["duplicate_flooding_samples"]), detected_duplicates, clean_samples)
    m_lbl = calc_metrics(set(ground_truth["label_flip_samples"]), detected_mislabels, clean_samples)
    m_ood = calc_metrics(set(ground_truth["ood_samples"]), detected_oods, clean_samples)
    m_trg = calc_metrics(set(ground_truth["trigger_samples"]), detected_triggers, clean_samples)

    # Model & Provenance scenarios (Binary detection)
    det_substitution = substitution_result["status"] == "SUBSTITUTION_DETECTED"
    det_tamper = tamp_verif.status == "TAMPERED"
    det_replay = replay_res["is_replay"]
    det_shift = shift_analysis["overall_shift_detected"]

    scenario_evaluations = [
        ("Label Flip", m_lbl["precision"], m_lbl["recall"], m_lbl["f1"], m_lbl["fpr"]),
        ("Duplicate Flooding", m_dup["precision"], m_dup["recall"], m_dup["f1"], m_dup["fpr"]),
        ("OOD Injection", m_ood["precision"], m_ood["recall"], m_ood["f1"], m_ood["fpr"]),
        ("Trigger / Backdoor", m_trg["precision"], m_trg["recall"], m_trg["f1"], m_trg["fpr"]),
        ("Model Substitution", 1.0 if det_substitution else 0.0, 1.0 if det_substitution else 0.0, 1.0 if det_substitution else 0.0, 0.0),
        ("Inference Tampering", 1.0 if det_tamper else 0.0, 1.0 if det_tamper else 0.0, 1.0 if det_tamper else 0.0, 0.0),
        ("Inference Replay", 1.0 if det_replay else 0.0, 1.0 if det_replay else 0.0, 1.0 if det_replay else 0.0, 0.0),
        ("Distribution Shift", 1.0 if det_shift else 0.0, 1.0 if det_shift else 0.0, 1.0 if det_shift else 0.0, 0.0),
    ]

    total_scenarios = len(scenario_evaluations)
    detected_scenarios = sum(1 for s in scenario_evaluations if s[2] > 0.5)
    missed_scenarios = total_scenarios - detected_scenarios

    print(f"\nDEMO COMPLETED SUCCESSFULLY")
    print(f"Attack scenarios evaluated: {total_scenarios}")
    print(f"Successfully Detected:     {detected_scenarios}")
    print(f"Missed Scenarios:          {missed_scenarios}")
    print("-" * 80)
    print(f"{'Attack Scenario':<26} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'FPR':<10}")
    print("-" * 80)
    for name, p, r, f, fpr in scenario_evaluations:
        print(f"{name:<26} {p:<12.3f} {r:<12.3f} {f:<12.3f} {fpr:<10.3f}")
    print("-" * 80)

    # Persist experiment results
    repo.create_experiment({
        "scenario_name": "Full_8_Attack_Assurance_Benchmark",
        "config_json": json.dumps({"dataset_samples": ground_truth["total_samples"], "models": 4}),
        "ground_truth_json": json.dumps(ground_truth),
        "results_json": json.dumps({
            "scenarios": [
                {"name": s[0], "precision": s[1], "recall": s[2], "f1": s[3], "fpr": s[4]}
                for s in scenario_evaluations
            ]
        }),
        "precision": round(float(np.mean([s[1] for s in scenario_evaluations])), 3),
        "recall": round(float(np.mean([s[2] for s in scenario_evaluations])), 3),
        "f1": round(float(np.mean([s[3] for s in scenario_evaluations])), 3),
        "detection_rate": round(float(detected_scenarios / total_scenarios), 3)
    })

    return {
        "status": "COMPLETED",
        "total_scenarios": total_scenarios,
        "detected_scenarios": detected_scenarios,
        "missed_scenarios": missed_scenarios,
        "metrics_table": [
            {"scenario": s[0], "precision": s[1], "recall": s[2], "f1": s[3], "fpr": s[4]}
            for s in scenario_evaluations
        ],
        "report_id": report.report_id,
        "pdf_report": str(pdf_p)
    }

if __name__ == "__main__":
    run_full_assurance_demo()
