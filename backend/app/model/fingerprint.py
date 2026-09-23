"""
Model Cryptographic Fingerprinting and Substitution Detection
"""
from pathlib import Path
from typing import Dict, Any, Tuple
from backend.app.core.hashing import hash_file, hash_canonical_json
from backend.app.model.loader import ModelAdapter

def generate_model_fingerprint(adapter: ModelAdapter) -> Dict[str, Any]:
    """
    Generate comprehensive model fingerprint combining SHA-256 identity,
    architecture metadata, weight tensor statistics, and access level.
    """
    model_path = adapter.model_path
    sha256 = hash_file(model_path)
    file_size = model_path.stat().st_size

    arch_info = adapter.inspect_architecture()
    weight_info = adapter.compute_weight_fingerprint()

    fingerprint = {
        "model_id": f"MOD-{sha256[:12].upper()}",
        "sha256": sha256,
        "file_name": model_path.name,
        "file_size_bytes": file_size,
        "framework": arch_info.get("framework", "UNKNOWN"),
        "parameter_count": arch_info.get("parameter_count", 0),
        "access_level": adapter.access_level,
        "architecture_summary": arch_info,
        "weight_fingerprint": weight_info,
        "fingerprint_hash": ""
    }

    # Cryptographic hash of canonical fingerprint
    fingerprint["fingerprint_hash"] = hash_canonical_json(fingerprint)
    return fingerprint

def compare_model_fingerprints(ref_fp: Dict[str, Any], cand_fp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare reference model fingerprint against candidate model fingerprint.
    Detects exact identity match, parameter mismatch, layer structural deviation,
    or complete model substitution.
    """
    is_exact_match = ref_fp.get("sha256") == cand_fp.get("sha256")
    param_ref = ref_fp.get("parameter_count", 0)
    param_cand = cand_fp.get("parameter_count", 0)

    param_diff = abs(param_ref - param_cand)
    param_diff_ratio = round(param_diff / max(1, param_ref), 4)

    weight_ref = ref_fp.get("weight_fingerprint", {}).get("mean_l2_norm", 0.0)
    weight_cand = cand_fp.get("weight_fingerprint", {}).get("mean_l2_norm", 0.0)
    weight_diff_ratio = round(abs(weight_ref - weight_cand) / max(1e-5, weight_ref), 4) if weight_ref else 0.0

    if is_exact_match:
        status = "MATCH"
        severity = "LOW"
        confidence = 1.0
        details = "Candidate model SHA-256 and structural fingerprint exactly match the reference model."
        recommendation = "ACCEPT"
    elif param_diff_ratio > 0.05 or ref_fp.get("framework") != cand_fp.get("framework"):
        status = "SUBSTITUTION_DETECTED"
        severity = "CRITICAL"
        confidence = 0.95
        details = (
            f"Severe model substitution detected: Candidate SHA-256 differs ({cand_fp.get('sha256')[:12]}... vs ref {ref_fp.get('sha256')[:12]}...), "
            f"parameter count differs by {int(param_diff_ratio*100)}% ({param_cand} vs {param_ref})."
        )
        recommendation = "QUARANTINE"
    elif weight_diff_ratio > 0.08:
        status = "WEIGHT_MODIFICATION_DETECTED"
        severity = "HIGH"
        confidence = 0.88
        details = (
            f"Model architecture matches, but weight tensor statistics have deviated by {int(weight_diff_ratio*100)}% "
            f"(Mean L2 norm: {weight_cand} vs reference {weight_ref}). Possible fine-tuning, trojan injection, or parameter tampering."
        )
        recommendation = "REVIEW"
    else:
        status = "MINOR_DEVIATION"
        severity = "MEDIUM"
        confidence = 0.75
        details = "SHA-256 differs but architecture and weight distributions closely align. Possible re-serialization or non-deterministic build."
        recommendation = "REVIEW"

    return {
        "status": status,
        "is_exact_match": is_exact_match,
        "severity": severity,
        "confidence": confidence,
        "parameter_difference_ratio": param_diff_ratio,
        "weight_norm_difference_ratio": weight_diff_ratio,
        "details": details,
        "recommendation": recommendation
    }
