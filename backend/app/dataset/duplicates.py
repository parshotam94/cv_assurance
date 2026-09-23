"""
Near-duplicate detection and duplicate flooding analysis using perceptual hashing (pHash, dHash)
"""
from typing import List, Dict, Any, Tuple
from PIL import Image
import imagehash
from backend.app.dataset.loader import SampleRecord
from backend.app.config import settings

def compute_hashes_for_records(records: List[SampleRecord]) -> Dict[str, Dict[str, Any]]:
    """Compute pHash and dHash for all records."""
    hashes = {}
    for r in records:
        try:
            with Image.open(r.image_path) as im:
                ph = imagehash.phash(im)
                dh = imagehash.dhash(im)
                hashes[r.sample_id] = {
                    "phash": str(ph),
                    "dhash": str(dh),
                    "phash_obj": ph,
                    "dhash_obj": dh,
                    "record": r
                }
        except Exception:
            continue
    return hashes

def detect_near_duplicates(
    records: List[SampleRecord],
    similarity_threshold: float = 0.92
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Detect near-duplicate image clusters using pHash and dHash hamming distance.
    Threshold: similarity >= 0.92 means hamming distance <= round(64 * (1 - 0.92)) = <= 5 bits.
    Returns: (duplicate_clusters, sample_to_duplicate_map)
    """
    max_hamming_dist = int(round(64 * (1.0 - similarity_threshold)))
    hashes = compute_hashes_for_records(records)
    sample_ids = list(hashes.keys())

    visited = set()
    clusters: List[Dict[str, Any]] = []
    sample_to_dup_map: Dict[str, List[str]] = {}

    cluster_counter = 1

    for i in range(len(sample_ids)):
        sid_i = sample_ids[i]
        if sid_i in visited:
            continue

        current_cluster = [sid_i]
        ph_i = hashes[sid_i]["phash_obj"]
        dh_i = hashes[sid_i]["dhash_obj"]

        for j in range(i + 1, len(sample_ids)):
            sid_j = sample_ids[j]
            if sid_j in visited:
                continue

            ph_j = hashes[sid_j]["phash_obj"]
            dh_j = hashes[sid_j]["dhash_obj"]

            dist_ph = ph_i - ph_j
            dist_dh = dh_i - dh_j

            # Both perceptual hash AND difference hash must match closely (AND logic reduces FPR)
            if dist_ph <= max_hamming_dist and dist_dh <= max_hamming_dist:
                current_cluster.append(sid_j)

        if len(current_cluster) >= 3:  # Require at least 3 to avoid accidental near-duplicate pairs
            for s in current_cluster:
                visited.add(s)
                sample_to_dup_map[s] = [other for other in current_cluster if other != s]

            # Collect source information
            cluster_sources = [hashes[s]["record"].source_id for s in current_cluster]
            sources_set = sorted(list(set(cluster_sources)))

            # Estimate avg similarity across pair
            similarity_val = round(1.0 - (max_hamming_dist / 64.0), 3)

            is_flooding = len(current_cluster) >= 4 or (len(current_cluster) / max(1, len(records)) > 0.08)
            cluster_risk = "HIGH" if is_flooding else ("MEDIUM" if len(current_cluster) >= 3 else "LOW")

            clusters.append({
                "cluster_id": f"DUP-CLUSTER-{cluster_counter:03d}",
                "sample_ids": current_cluster,
                "cluster_size": len(current_cluster),
                "similarity": similarity_val,
                "sources": sources_set,
                "is_flooding": is_flooding,
                "risk": cluster_risk
            })
            cluster_counter += 1

    return clusters, sample_to_dup_map
