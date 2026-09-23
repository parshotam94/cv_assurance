"""
Statistical Drift and Distribution Shift Metrics: PSI, KS Test, Wasserstein Distance, Jensen-Shannon
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

def calculate_psi(reference: np.ndarray, candidate: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant distribution shift
    """
    if len(reference) == 0 or len(candidate) == 0:
        return 0.0

    # Define bins based on reference percentiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cand_counts, _ = np.histogram(candidate, bins=bin_edges)

    # Add epsilon to prevent log(0)
    eps = 1e-4
    ref_pct = (ref_counts + eps) / (np.sum(ref_counts) + eps * len(ref_counts))
    cand_pct = (cand_counts + eps) / (np.sum(cand_counts) + eps * len(cand_counts))

    psi_val = np.sum((cand_pct - ref_pct) * np.log(cand_pct / ref_pct))
    return float(np.clip(psi_val, 0.0, 10.0))

def compute_drift_metrics(reference_vals: List[float], candidate_vals: List[float]) -> Dict[str, Any]:
    """
    Compute comprehensive statistical divergence metrics between two sample populations.
    """
    ref_arr = np.array(reference_vals, dtype=np.float32)
    cand_arr = np.array(candidate_vals, dtype=np.float32)

    if len(ref_arr) < 2 or len(cand_arr) < 2:
        return {
            "psi": 0.0,
            "ks_statistic": 0.0,
            "ks_p_value": 1.0,
            "wasserstein_distance": 0.0,
            "jensen_shannon_divergence": 0.0,
            "is_significant_drift": False
        }

    # 1. Population Stability Index (PSI)
    psi = calculate_psi(ref_arr, cand_arr)

    # 2. Kolmogorov-Smirnov Test (2-sample)
    ks_res = stats.ks_2samp(ref_arr, cand_arr)
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    # 3. Wasserstein Distance
    wd = float(stats.wasserstein_distance(ref_arr, cand_arr))

    # 4. Jensen-Shannon Divergence
    # Bin both to probability distribution
    min_v = min(np.min(ref_arr), np.min(cand_arr))
    max_v = max(np.max(ref_arr), np.max(cand_arr))
    if max_v > min_v:
        bins = np.linspace(min_v, max_v, 15)
        p, _ = np.histogram(ref_arr, bins=bins, density=True)
        q, _ = np.histogram(cand_arr, bins=bins, density=True)
        p = (p + 1e-5) / np.sum(p + 1e-5)
        q = (q + 1e-5) / np.sum(q + 1e-5)
        js_div = float(jensenshannon(p, q))
    else:
        js_div = 0.0

    is_significant = (psi >= 0.2) or (ks_stat >= 0.35 and ks_pval < 0.01)

    return {
        "psi": round(psi, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_p_value": round(ks_pval, 6),
        "wasserstein_distance": round(wd, 4),
        "jensen_shannon_divergence": round(js_div, 4),
        "is_significant_drift": is_significant
    }
