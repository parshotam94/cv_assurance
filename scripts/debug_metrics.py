"""Debug label flip and duplicate detection."""
import sys
sys.path.insert(0, '.')
from experiments.generate_poisoned_data import generate_poisoned_experiment_dataset
from backend.app.dataset.coco import parse_coco_dataset
from backend.app.dataset.label_analysis import analyze_labels_and_mislabelling
from backend.app.dataset.duplicates import detect_near_duplicates
from pathlib import Path
import json

info = generate_poisoned_experiment_dataset()
records, class_names, _ = parse_coco_dataset(Path(info['dataset_dir']))
gt_path = Path(info['ground_truth_path'])
with open(gt_path) as f:
    gt = json.load(f)

flip_ids = set(gt['label_flip_samples'])
clean_ids = set(gt['clean_samples'])
dup_ids = set(gt['duplicate_flooding_samples'])

print("=== LABEL FLIP ANALYSIS ===")
_, mislabelled = analyze_labels_and_mislabelling(records, class_names)
detected_flips = set(s['sample_id'] for s in mislabelled)
tp = detected_flips & flip_ids
fp = detected_flips & clean_ids
fn = flip_ids - detected_flips
print(f"TP={len(tp)} FP={len(fp)} FN={len(fn)}")
print(f"Detected flips: {sorted(detected_flips)}")
print(f"True flips:     {sorted(flip_ids)}")
print(f"Missed:         {sorted(fn)}")
print(f"False positives: {sorted(fp)}")
print()
for s in mislabelled:
    print(f"  {s['sample_id']}: label={s['assigned_label']}, consensus={s['consensus_neighbor_label']}, "
          f"disagree={s['disagreement_rate']}, source={s['source_id']}")

print()
print("=== DUPLICATE ANALYSIS ===")
clusters, sample_map = detect_near_duplicates(records, similarity_threshold=0.95)
detected_dups = set(sample_map.keys())
tp2 = detected_dups & dup_ids
fp2 = detected_dups & clean_ids
fn2 = dup_ids - detected_dups
print(f"TP={len(tp2)} FP={len(fp2)} FN={len(fn2)}")
print(f"Detected duplicates: {sorted(detected_dups)}")
print(f"True duplicates:     {sorted(dup_ids)}")
print(f"False positives: {sorted(fp2)}")
print()
for c in clusters:
    print(f"  Cluster {c['cluster_id']}: samples={c['sample_ids']}, sources={c['sources']}")
