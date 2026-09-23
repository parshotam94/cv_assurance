"""Debug trigger patch signatures and detection."""
import sys
sys.path.insert(0, '.')
from experiments.generate_poisoned_data import generate_poisoned_experiment_dataset
from backend.app.dataset.coco import parse_coco_dataset
from backend.app.dataset.poisoning import detect_trigger_backdoors, extract_patch_signatures
from pathlib import Path
from PIL import Image
import numpy as np
import json

# Generate fresh data
info = generate_poisoned_experiment_dataset()
records, class_names, _ = parse_coco_dataset(Path(info['dataset_dir']))

gt_path = Path(info['ground_truth_path'])
with open(gt_path) as f:
    gt = json.load(f)

trigger_ids = set(gt['trigger_samples'])
clean_ids = set(gt['clean_samples'])

print('=== TRIGGER SAMPLE TL PATCH ANALYSIS ===')
for r in records:
    if r.sample_id in trigger_ids:
        with Image.open(r.image_path) as im:
            arr = np.array(im.convert('RGB').resize((128, 128)), dtype=np.float32) / 255.0
            patch = arr[0:16, 0:16]
            print(f'  {r.sample_id} TL: std={np.std(patch):.4f}, mean={np.mean(patch):.4f}, min={patch.min():.4f}, max={patch.max():.4f}')
        sigs = extract_patch_signatures(r.image_path)
        print(f'    Signatures extracted for regions: {list(sigs.keys())}')

print()
print('=== CLEAN VEHICLE TL PATCH ANALYSIS (first 5) ===')
count = 0
for r in records:
    if r.sample_id in clean_ids and r.labels and r.labels[0] == 'vehicle':
        with Image.open(r.image_path) as im:
            arr = np.array(im.convert('RGB').resize((128, 128)), dtype=np.float32) / 255.0
            patch = arr[0:16, 0:16]
            print(f'  {r.sample_id} TL: std={np.std(patch):.4f}, mean={np.mean(patch):.4f}')
        sigs = extract_patch_signatures(r.image_path)
        print(f'    Signatures extracted: {list(sigs.keys())}')
        count += 1
        if count >= 5:
            break

print()
print('=== FULL TRIGGER DETECTION RESULTS ===')
findings, suspicious = detect_trigger_backdoors(records)
print(f'Total trigger clusters: {len(findings)}')
print(f'Total suspicious samples: {len(suspicious)}')

detected_trigger_ids = set(s['sample_id'] for s in suspicious)
tp = len(trigger_ids & detected_trigger_ids)
fp = len(clean_ids & detected_trigger_ids)
fn = len(trigger_ids - detected_trigger_ids)
tn = len(clean_ids - detected_trigger_ids)
precision = tp / max(1, tp + fp)
recall = tp / max(1, tp + fn)
fpr = fp / max(1, fp + tn)
print(f'TP={tp} FP={fp} FN={fn} TN={tn}')
print(f'Precision={precision:.3f} Recall={recall:.3f} FPR={fpr:.3f}')

if findings:
    print()
    for f in findings[:5]:
        cid = f['cluster_id']
        loc = f['location']
        size = f['cluster_size']
        assoc = f['associated_class']
        print(f'  Cluster {cid}: region={loc}, size={size}, class={assoc}')
