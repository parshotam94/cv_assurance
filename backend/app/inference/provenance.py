"""
Cryptographic Provenance Record Generator
Binds input image, model, preprocessing, configurations, output, timestamp, nonce, and sequence into an Ed25519 signed canonical record.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
from pathlib import Path
from backend.app.core.hashing import hash_bytes, hash_file, hash_canonical_json
from backend.app.core.signatures import sign_data, get_public_key_hex, generate_or_load_keypair
from backend.app.inference.schema import ProvenanceRecord, PreprocessingConfig, InferenceConfig

_SEQUENCE_COUNTER = 1000

def get_next_sequence() -> int:
    global _SEQUENCE_COUNTER
    _SEQUENCE_COUNTER += 1
    return _SEQUENCE_COUNTER

def create_provenance_record(
    input_image_path: Path,
    model_sha256: str,
    output_data: Dict[str, Any],
    preprocessing_config: PreprocessingConfig = None,
    inference_config: InferenceConfig = None,
    custom_nonce: str = None,
    custom_sequence: int = None
) -> ProvenanceRecord:
    """
    Generate canonical cryptographic inference provenance record.
    """
    if preprocessing_config is None:
        preprocessing_config = PreprocessingConfig()
    if inference_config is None:
        inference_config = InferenceConfig()

    input_sha256 = hash_file(input_image_path)
    preprocessing_sha256 = hash_canonical_json(preprocessing_config.model_dump())
    config_sha256 = hash_canonical_json(inference_config.model_dump())
    output_sha256 = hash_canonical_json(output_data)

    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = custom_nonce or uuid.uuid4().hex
    sequence = custom_sequence or get_next_sequence()
    record_id = f"REC-{uuid.uuid4().hex[:12].upper()}"

    canonical_payload = {
        "record_id": record_id,
        "input_sha256": input_sha256,
        "model_sha256": model_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "config_sha256": config_sha256,
        "output_sha256": output_sha256,
        "timestamp": timestamp,
        "nonce": nonce,
        "sequence": sequence
    }

    record_hash = hash_canonical_json(canonical_payload)
    
    # Sign canonical payload with local Ed25519 private key
    sig_hex = sign_data(record_hash.encode("utf-8"))
    pubkey_hex = get_public_key_hex()

    return ProvenanceRecord(
        record_id=record_id,
        input_sha256=input_sha256,
        model_sha256=model_sha256,
        preprocessing_sha256=preprocessing_sha256,
        config_sha256=config_sha256,
        output_sha256=output_sha256,
        timestamp=timestamp,
        nonce=nonce,
        sequence=sequence,
        record_hash=record_hash,
        signature=sig_hex,
        public_key_hex=pubkey_hex,
        raw_output=output_data
    )
