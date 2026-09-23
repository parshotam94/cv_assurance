"""
Label integrity and systematic mislabelling detection
"""
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from backend.app.dataset.loader import SampleRecord

def extract_basic_visual_features(img_path: str) -> np.ndarray:
    """
    Extract offline statistical visual feature vector:
    - Global RGB channel statistics (means, standard deviations)
    - Center object crop statistics (means, standard deviations)
    - Chromatic channel interaction ratios (B/R, R/G)
    - Grayscale gradients (spatial variation / texture energy)
    """
    try:
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB").resize((64, 64))
            arr = np.array(im_rgb, dtype=np.float32) / 255.0
            
            # Global channel statistics
            g_means = arr.mean(axis=(0, 1))
            g_stds = arr.std(axis=(0, 1))
            
            # Center object crop (32x32 center region where objects are localized)
            center = arr[16:48, 16:48]
            c_means = center.mean(axis=(0, 1))
            c_stds = center.std(axis=(0, 1))
            
            # Chromatic interaction ratios
            r_c, g_c, b_c = c_means
            br_ratio = b_c / (r_c + 1e-4)
            rg_ratio = r_c / (g_c + 1e-4)
            
            # Grayscale texture & spatial gradients
            gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)
            grad_energy = np.mean(np.abs(dx)) + np.mean(np.abs(dy))

            feat = np.hstack([
                g_means, g_stds,
                c_means, c_stds,
                [br_ratio, rg_ratio, grad_energy]
            ])
            return feat.astype(np.float32)
    except Exception:
        return np.zeros(15, dtype=np.float32)

def analyze_labels_and_mislabelling(
    records: List[SampleRecord],
    class_names: List[str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Detect class imbalances, abnormal distributions, and suspicious mislabellings via k-NN feature neighborhood.
    """
    label_counts = Counter()
    sample_primary_label = {}
    feature_matrix = []
    valid_sample_records = []

    for r in records:
        lbl = r.labels[0] if r.labels else "unlabelled"
        label_counts[lbl] += 1
        sample_primary_label[r.sample_id] = lbl
        feat = extract_basic_visual_features(r.image_path)
        feature_matrix.append(feat)
        valid_sample_records.append(r)

    total_samples = len(records)
    imbalance_ratio = 1.0
    if label_counts:
        max_c = max(label_counts.values())
        min_c = min(label_counts.values())
        imbalance_ratio = round(max_c / max(1, min_c), 2)

    # Class distribution summary
    distribution_stats = {
        "class_counts": dict(label_counts),
        "total_samples": total_samples,
        "imbalance_ratio": imbalance_ratio,
        "is_heavily_imbalanced": imbalance_ratio > 8.0
    }

    suspicious_samples: List[Dict[str, Any]] = []

    # If we have enough samples, run k-NN neighborhood label consensus
    if len(valid_sample_records) >= 8:
        X = np.array(feature_matrix)
        # Normalize features
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        X_norm = (X - np.mean(X, axis=0)) / std

        k = min(5, len(valid_sample_records) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(X_norm)
        distances, indices = nn.kneighbors(X_norm)

        for i, rec in enumerate(valid_sample_records):
            current_label = sample_primary_label[rec.sample_id]
            if current_label == "unlabelled":
                continue

            neighbor_indices = indices[i][1:]  # exclude self
            neighbor_labels = [sample_primary_label[valid_sample_records[idx].sample_id] for idx in neighbor_indices]
            
            # Check neighbor consensus
            neighbor_counts = Counter(neighbor_labels)
            most_common_neighbor_label, top_count = neighbor_counts.most_common(1)[0]
            
            disagreement_rate = 1.0 - (neighbor_counts.get(current_label, 0) / k)

            source_id = getattr(rec, "source_id", "")
            is_suspect_source = (
                source_id and (
                    "rogue" in source_id.lower() or
                    "poison" in source_id.lower() or
                    "flood" in source_id.lower() or
                    "bad" in source_id.lower()
                )
            )

            # A sample is flagged if its k-NN feature neighborhood consensus disagrees with its assigned label
            if disagreement_rate >= 0.6 and most_common_neighbor_label != current_label:
                confidence = "HIGH" if disagreement_rate >= 0.8 else "MEDIUM"
                suspicious_samples.append({
                    "sample_id": rec.sample_id,
                    "image_path": rec.image_path,
                    "assigned_label": current_label,
                    "consensus_neighbor_label": most_common_neighbor_label,
                    "disagreement_rate": round(disagreement_rate, 2),
                    "confidence": confidence,
                    "source_id": source_id,
                    "batch_id": rec.batch_id,
                    "reason": (
                        f"Sample label '{current_label}' disagrees with {int(disagreement_rate*100)}% "
                        f"of feature-space nearest neighbors (consensus: '{most_common_neighbor_label}')."
                        f"{' Source flagged as high-risk.' if is_suspect_source else ''}"
                    )
                })

    return distribution_stats, suspicious_samples
