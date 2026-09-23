"""
Unit Tests for Cryptographic Inference Provenance and Replay Detection
"""
import pytest
from backend.app.inference.provenance import create_provenance_record
from backend.app.inference.verifier import verify_provenance_record
from backend.app.inference.replay import check_for_replay_attack
from backend.app.inference.schema import ProvenanceRecord, PreprocessingConfig, InferenceConfig

def test_create_and_verify_clean_provenance(test_image):
    output_data = {"class": 1, "confidence": 0.98}
    record = create_provenance_record(
        input_image_path=test_image,
        model_sha256="a" * 64,
        output_data=output_data
    )

    assert record.signature is not None
    assert len(record.signature) == 128
    assert len(record.input_sha256) == 64
    assert len(record.record_hash) == 64

    # Verify clean record
    result = verify_provenance_record(record, current_output_data=output_data)
    assert result.is_valid is True
    assert result.status == "VERIFIED"

def test_tampered_output_payload(test_image):
    output_data = {"class": 1, "confidence": 0.98}
    record = create_provenance_record(
        input_image_path=test_image,
        model_sha256="a" * 64,
        output_data=output_data
    )

    # Tamper with output
    tampered_output = {"class": 999, "confidence": 0.999}
    result = verify_provenance_record(record, current_output_data=tampered_output)
    assert result.is_valid is False
    assert result.status == "TAMPERED"

def test_tampered_signature(test_image):
    output_data = {"class": 1, "confidence": 0.98}
    record = create_provenance_record(
        input_image_path=test_image,
        model_sha256="a" * 64,
        output_data=output_data
    )

    # Corrupt signature string
    corrupted_dict = record.model_dump()
    corrupted_dict["signature"] = "0" * 128
    corrupted_record = ProvenanceRecord(**corrupted_dict)

    result = verify_provenance_record(corrupted_record, current_output_data=output_data)
    assert result.is_valid is False
    assert result.status == "INVALID_SIGNATURE"

def test_replay_attack_detection(test_image, db_session, test_repo):
    output_data = {"class": 1, "confidence": 0.98}
    record = create_provenance_record(
        input_image_path=test_image,
        model_sha256="a" * 64,
        output_data=output_data
    )

    # 1. Register the clean record in DB
    test_repo.create_inference_record({
        "record_id": record.record_id,
        "input_sha256": record.input_sha256,
        "model_sha256": record.model_sha256,
        "preprocessing_sha256": record.preprocessing_sha256,
        "config_sha256": record.config_sha256,
        "output_sha256": record.output_sha256,
        "timestamp": record.timestamp,
        "nonce": record.nonce,
        "sequence": record.sequence,
        "record_hash": record.record_hash,
        "signature": record.signature,
        "public_key_hex": record.public_key_hex,
        "verification_status": "VERIFIED"
    })

    # 2. Check replay on the exact same record
    replay_result = check_for_replay_attack(db_session, record)
    assert replay_result["is_replay"] is True
    assert len(replay_result["reasons"]) >= 1
