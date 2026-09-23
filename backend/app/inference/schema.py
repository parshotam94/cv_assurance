"""
Pydantic schemas for Inference Provenance and Verification
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class PreprocessingConfig(BaseModel):
    resize: List[int] = Field(default_factory=lambda: [64, 64])
    normalize_mean: List[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    color_mode: str = "RGB"

class InferenceConfig(BaseModel):
    confidence_threshold: float = 0.5
    device: str = "CPU"
    batch_size: int = 1
    postprocess_filter: str = "standard"

class ProvenanceRecord(BaseModel):
    record_id: str
    input_sha256: str
    model_sha256: str
    preprocessing_sha256: str
    config_sha256: str
    output_sha256: str
    timestamp: str
    nonce: str
    sequence: int
    record_hash: str
    signature: str
    public_key_hex: str
    raw_output: Optional[Dict[str, Any]] = None

class VerificationResult(BaseModel):
    status: str  # VERIFIED, TAMPERED, REPLAY_DETECTED, INVALID_SIGNATURE
    is_valid: bool
    record_id: str
    tamper_detected_in: List[str] = Field(default_factory=list)
    signature_valid: bool
    hash_match: bool
    details: str
    disposition: str  # ACCEPT, REVIEW, QUARANTINE
