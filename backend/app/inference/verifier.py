"""
Cryptographic Provenance Verifier and Tamper Detection Engine
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.app.core.hashing import hash_file, hash_canonical_json
from backend.app.core.signatures import verify_signature
from backend.app.inference.schema import ProvenanceRecord, VerificationResult

def verify_provenance_record(
    record: ProvenanceRecord,
    current_input_path: Optional[Path] = None,
    current_model_path: Optional[Path] = None,
    current_output_data: Optional[Dict[str, Any]] = None,
    current_preprocessing: Optional[Dict[str, Any]] = None,
    current_config: Optional[Dict[str, Any]] = None
) -> VerificationResult:
    """
    Validate the cryptographic binding, recompute record hashes, and verify Ed25519 signature.
    Identifies exact component where tampering occurred.
    """
    tamper_components = []

    # 1. Verify input image hash if current image provided
    if current_input_path and current_input_path.exists():
        actual_input_hash = hash_file(current_input_path)
        if actual_input_hash != record.input_sha256:
            tamper_components.append(f"INPUT_IMAGE (expected {record.input_sha256[:12]}..., got {actual_input_hash[:12]}...)")

    # 2. Verify model hash if model path provided
    if current_model_path and current_model_path.exists():
        actual_model_hash = hash_file(current_model_path)
        if actual_model_hash != record.model_sha256:
            tamper_components.append(f"MODEL_FILE (expected {record.model_sha256[:12]}..., got {actual_model_hash[:12]}...)")

    # 3. Verify output payload hash if current output provided
    if current_output_data is not None:
        actual_output_hash = hash_canonical_json(current_output_data)
        if actual_output_hash != record.output_sha256:
            tamper_components.append(f"OUTPUT_PAYLOAD (expected {record.output_sha256[:12]}..., got {actual_output_hash[:12]}...)")

    # 4. Verify preprocessing config hash
    if current_preprocessing is not None:
        actual_prep_hash = hash_canonical_json(current_preprocessing)
        if actual_prep_hash != record.preprocessing_sha256:
            tamper_components.append("PREPROCESSING_CONFIG")

    # 5. Verify inference config hash
    if current_config is not None:
        actual_conf_hash = hash_canonical_json(current_config)
        if actual_conf_hash != record.config_sha256:
            tamper_components.append("INFERENCE_CONFIG")

    # 6. Recompute canonical record hash
    reconstructed_payload = {
        "record_id": record.record_id,
        "input_sha256": record.input_sha256,
        "model_sha256": record.model_sha256,
        "preprocessing_sha256": record.preprocessing_sha256,
        "config_sha256": record.config_sha256,
        "output_sha256": record.output_sha256,
        "timestamp": record.timestamp,
        "nonce": record.nonce,
        "sequence": record.sequence
    }
    recomputed_hash = hash_canonical_json(reconstructed_payload)
    hash_match = (recomputed_hash == record.record_hash)

    if not hash_match:
        tamper_components.append("RECORD_METADATA_OR_HASH_MISMATCH")

    # 7. Verify Ed25519 digital signature
    sig_valid = verify_signature(
        data=record.record_hash.encode("utf-8"),
        signature_hex=record.signature,
        public_key_hex=record.public_key_hex
    )

    if not sig_valid:
        tamper_components.append("ED25519_DIGITAL_SIGNATURE")

    # Evaluate status
    if not sig_valid:
        status = "INVALID_SIGNATURE"
        disposition = "QUARANTINE"
        details = "Digital signature validation failed. Record was not signed by authorized keypair or has been altered."
    elif tamper_components:
        status = "TAMPERED"
        disposition = "QUARANTINE"
        details = f"Cryptographic integrity violation detected in component(s): {', '.join(tamper_components)}."
    else:
        status = "VERIFIED"
        disposition = "ACCEPT"
        details = "Cryptographic integrity verified. All input, model, configuration, output hashes and Ed25519 signature match."

    return VerificationResult(
        status=status,
        is_valid=(status == "VERIFIED"),
        record_id=record.record_id,
        tamper_detected_in=tamper_components,
        signature_valid=sig_valid,
        hash_match=hash_match,
        details=details,
        disposition=disposition
    )
