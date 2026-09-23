"""
Trigger and backdoor data detection using localized patch analysis and high-frequency pattern matching.
Key insight: Backdoor triggers form a SMALL cluster sharing an unusual patch — not a large majority-class cluster.
Normal class feature similarity creates large clusters; true triggers form anomalously small, highly correlated clusters.
"""
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from backend.app.dataset.loader import SampleRecord
from backend.app.config import settings


def extract_patch_signatures(img_path: str, patch_size: int = 16) -> Dict[str, np.ndarray]:
    """
    Extract fixed spatial corner/center patches and compute their normalized pixel signatures.
    Regions: Top-Left (TL), Top-Right (TR), Bottom-Left (BL), Bottom-Right (BR), Center (C).
    Skip patches that are too uniform (background noise) to avoid false matches.
    """
    signatures = {}
    try:
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB").resize((128, 128))
            arr = np.array(im_rgb, dtype=np.float32) / 255.0

            h, w, _ = arr.shape
            ps = patch_size

            # Corner and center coordinates
            regions = {
                "TL": arr[0:ps, 0:ps],
                "TR": arr[0:ps, w-ps:w],
                "BL": arr[h-ps:h, 0:ps],
                "BR": arr[h-ps:h, w-ps:w],
                "C": arr[h//2-ps//2:h//2+ps//2, w//2-ps//2:w//2+ps//2]
            }

            for loc, patch in regions.items():
                # Skip flat / uniform background patches (not anomalous triggers)
                # Threshold 0.10 filters dark background corners (std~0.06-0.08) while
                # preserving high-contrast trigger patches (std~0.40+)
                if np.std(patch) < 0.10:
                    continue
                # Normalized color + gradient signature of patch
                flat_patch = patch.flatten()
                norm = np.linalg.norm(flat_patch)
                if norm > 0:
                    flat_patch = flat_patch / norm
                signatures[loc] = flat_patch
    except Exception:
        pass
    return signatures


def detect_trigger_backdoors(
    records: List[SampleRecord],
    min_cluster_size: int = 3,
    patch_similarity_threshold: float = 0.98,
    max_cluster_fraction: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scan for repeated localized visual patterns (triggers/backdoors) across images,
    especially correlated with specific class targets.

    Key filtering rules to reduce FPR:
    1. Similarity threshold raised to 0.98 — only near-identical patches (as from a copy-paste trigger) cluster.
    2. max_cluster_fraction: clusters exceeding 25% of the total dataset are normal class structure, not triggers.
    3. Only report clusters where the cluster accounts for < max_cluster_fraction of ALL images (not just the class).
    """
    if len(records) < min_cluster_size:
        return [], []

    total_samples = len(records)
    # Maximum cluster size to report (ignore large majority-class clusters)
    max_cluster_size = int(total_samples * max_cluster_fraction)

    # Map: region -> list of (sample_id, record, signature)
    region_signatures: defaultdict = defaultdict(list)
    for r in records:
        sigs = extract_patch_signatures(r.image_path)
        for loc, sig in sigs.items():
            region_signatures[loc].append((r.sample_id, r, sig))

    trigger_findings: List[Dict[str, Any]] = []
    suspicious_samples_map: Dict[str, Dict[str, Any]] = {}

    for loc, items in region_signatures.items():
        if len(items) < min_cluster_size:
            continue

        n = len(items)
        # Compare pairwise patch similarity at this location
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            sid_i, rec_i, sig_i = items[i]
            cluster = [i]

            for j in range(i + 1, n):
                if j in visited:
                    continue
                _, _, sig_j = items[j]

                # Cosine similarity — raised to 0.98 for near-identical detection
                sim = float(np.dot(sig_i, sig_j))
                if sim >= patch_similarity_threshold:
                    cluster.append(j)

            if len(cluster) < min_cluster_size:
                continue

            # KEY FILTER: Skip clusters that are too large — they represent normal class structure
            # A backdoor cluster should be a small anomalous minority
            if len(cluster) > max_cluster_size:
                for idx in cluster:
                    visited.add(idx)
                continue

            for idx in cluster:
                visited.add(idx)

            cluster_records = [items[idx][1] for idx in cluster]
            cluster_sample_ids = [items[idx][0] for idx in cluster]

            # Check label correlation
            labels = [r.labels[0] if r.labels else "unlabelled" for r in cluster_records]
            label_counts: defaultdict = defaultdict(int)
            for lbl in labels:
                label_counts[lbl] += 1

            dominant_label, dominant_count = max(label_counts.items(), key=lambda x: x[1])
            class_correlation = round(dominant_count / len(cluster), 2)
            num_unique_labels = len(label_counts)

            # FILTER: Pure single-class center clusters are normal within-class visual similarity (NOT backdoors).
            # Real triggers cluster at corner regions (TL/TR/BL/BR) or show cross-class correlation.
            # Skip center-region clusters where all samples share the same label.
            if loc == "C" and class_correlation >= 0.90 and num_unique_labels == 1:
                continue

            # High class correlation + identical patch = strong indicator of trigger poisoning
            confidence = 0.92 if (class_correlation >= 0.75 and len(cluster) >= 4) else 0.78
            severity = "HIGH" if confidence > 0.85 else "MEDIUM"

            evidence = [
                f"{len(cluster)} images contain visually near-identical localized patch at region '{loc}' "
                f"(threshold={patch_similarity_threshold}, fraction={len(cluster)/total_samples:.1%} of dataset).",
                f"Near-identical patch cosine similarity >= {patch_similarity_threshold}.",
                f"Strong class correlation: {int(class_correlation*100)}% of affected samples share label '{dominant_label}'.",
            ]

            finding = {
                "cluster_id": f"TRIG-{loc}-{len(trigger_findings)+1:02d}",
                "location": loc,
                "affected_samples": cluster_sample_ids,
                "cluster_size": len(cluster),
                "associated_class": dominant_label,
                "class_correlation": class_correlation,
                "severity": severity,
                "confidence": confidence,
                "evidence": evidence,
                "recommendation": "REVIEW" if severity == "MEDIUM" else "QUARANTINE",
                "limitations": [
                    "Heuristic localized patch search; complex blended or steganographic triggers require gradient-based attribution."
                ]
            }
            trigger_findings.append(finding)

            for sid in cluster_sample_ids:
                suspicious_samples_map[sid] = {
                    "sample_id": sid,
                    "trigger_location": loc,
                    "associated_class": dominant_label,
                    "confidence": confidence,
                    "reason": (
                        f"Localized patch at region '{loc}' exhibits anomalous cross-image repetition "
                        f"correlated with label '{dominant_label}' ({len(cluster)} samples, "
                        f"{len(cluster)/total_samples:.1%} of dataset)."
                    )
                }

    return trigger_findings, list(suspicious_samples_map.values())
