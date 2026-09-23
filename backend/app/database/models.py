"""
SQLAlchemy ORM models for VisionTrust Assurance
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from backend.app.database.database import Base

def utcnow():
    return datetime.now(timezone.utc)

class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    format = Column(String(32), nullable=False)  # COCO, YOLO
    file_path = Column(String(512), nullable=False)
    sha256 = Column(String(64), nullable=True)
    status = Column(String(32), default="PENDING")  # PENDING, ANALYZING, COMPLETED, FAILED
    sample_count = Column(Integer, default=0)
    annotation_count = Column(Integer, default=0)
    classes = Column(Text, default="[]")  # JSON string
    metadata_json = Column(Text, default="{}")
    risk_score = Column(Float, default=0.0)
    disposition = Column(String(32), default="ACCEPT")  # ACCEPT, REVIEW, QUARANTINE
    analysis_summary = Column(Text, default="{}")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    samples = relationship("DatasetSampleModel", back_populates="dataset", cascade="all, delete-orphan")
    sources = relationship("DatasetSourceModel", back_populates="dataset", cascade="all, delete-orphan")


class DatasetSampleModel(Base):
    __tablename__ = "dataset_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(64), ForeignKey("datasets.id"), index=True, nullable=False)
    sample_id = Column(String(128), index=True, nullable=False)
    image_path = Column(String(512), nullable=False)
    source_id = Column(String(128), default="default", index=True)
    batch_id = Column(String(128), default="default")
    contributor = Column(String(128), default="anonymous")
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    labels = Column(Text, default="[]")  # JSON list of class labels
    annotations = Column(Text, default="[]")  # JSON list of bboxes / segmentation
    phash = Column(String(64), nullable=True)
    dhash = Column(String(64), nullable=True)
    is_suspicious = Column(Boolean, default=False)
    anomaly_reasons = Column(Text, default="[]")  # JSON list of reasons
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utcnow)

    dataset = relationship("DatasetModel", back_populates="samples")


class DatasetSourceModel(Base):
    __tablename__ = "dataset_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(64), ForeignKey("datasets.id"), index=True, nullable=False)
    source_id = Column(String(128), index=True, nullable=False)
    sample_count = Column(Integer, default=0)
    suspicious_count = Column(Integer, default=0)
    duplicate_clusters = Column(Integer, default=0)
    ood_count = Column(Integer, default=0)
    label_anomalies = Column(Integer, default=0)
    trigger_count = Column(Integer, default=0)
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String(32), default="LOW")  # LOW, MEDIUM, HIGH, CRITICAL
    disposition = Column(String(32), default="ACCEPT")
    metadata_json = Column(Text, default="{}")

    dataset = relationship("DatasetModel", back_populates="sources")


class ModelAssetModel(Base):
    __tablename__ = "models"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    framework = Column(String(32), nullable=False)  # ONNX, PyTorch, TorchScript
    file_path = Column(String(512), nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True, index=True)
    file_size_bytes = Column(Integer, default=0)
    parameter_count = Column(Integer, default=0)
    architecture = Column(String(128), default="unknown")
    input_shape = Column(String(128), default="unknown")
    output_shape = Column(String(128), default="unknown")
    access_level = Column(String(32), default="BLACK_BOX")  # WHITE_BOX, GRAY_BOX, BLACK_BOX
    fingerprint_json = Column(Text, default="{}")
    behavioural_summary = Column(Text, default="{}")
    trigger_summary = Column(Text, default="{}")
    limitations_json = Column(Text, default="[]")
    status = Column(String(32), default="REGISTERED")
    risk_score = Column(Float, default=0.0)
    disposition = Column(String(32), default="ACCEPT")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class InferenceRecordModel(Base):
    __tablename__ = "inference_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String(64), unique=True, index=True, nullable=False)
    input_sha256 = Column(String(64), nullable=False)
    model_sha256 = Column(String(64), nullable=False)
    preprocessing_sha256 = Column(String(64), nullable=False)
    config_sha256 = Column(String(64), nullable=False)
    output_sha256 = Column(String(64), nullable=False)
    timestamp = Column(String(64), nullable=False)
    nonce = Column(String(64), nullable=False, index=True)
    sequence = Column(Integer, nullable=False, index=True)
    record_hash = Column(String(64), nullable=False, index=True)
    signature = Column(Text, nullable=False)
    public_key_hex = Column(String(128), nullable=False)
    is_tampered = Column(Boolean, default=False)
    is_replayed = Column(Boolean, default=False)
    verification_status = Column(String(32), default="VERIFIED")  # VERIFIED, TAMPERED, REPLAY_DETECTED, INVALID_SIGNATURE
    verification_details = Column(Text, default="{}")
    payload_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utcnow)


class FindingModel(Base):
    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    finding_id = Column(String(32), unique=True, index=True, nullable=False)
    asset_type = Column(String(32), nullable=False)  # DATASET, MODEL, INFERENCE, DISTRIBUTION
    asset_id = Column(String(128), index=True, nullable=False)
    category = Column(String(64), nullable=False)  # NEAR_DUPLICATE, LABEL_FLIP, OOD, TRIGGER, MODEL_SUBSTITUTION, TAMPERING, REPLAY, etc.
    severity = Column(String(32), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    confidence = Column(Float, nullable=False)  # 0.0 - 1.0
    risk_score = Column(Float, nullable=False)  # 0.0 - 100.0
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    evidence = Column(Text, default="[]")  # JSON list
    affected_samples = Column(Text, default="[]")  # JSON list
    detection_method = Column(String(128), nullable=False)
    limitations = Column(Text, default="[]")  # JSON list
    recommendation = Column(String(32), default="REVIEW")  # ACCEPT, REVIEW, QUARANTINE
    status = Column(String(32), default="ACTIVE")
    created_at = Column(DateTime, default=utcnow)


class AuditEventModel(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(String(64), nullable=False)
    actor = Column(String(128), default="system")
    action = Column(String(128), nullable=False)
    asset = Column(String(255), nullable=False)
    previous_hash = Column(String(64), nullable=False)
    current_hash = Column(String(64), nullable=False)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utcnow)


class ReportModel(Base):
    __tablename__ = "assurance_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    overall_risk_score = Column(Float, default=0.0)
    overall_disposition = Column(String(32), default="ACCEPT")
    executive_summary = Column(Text, default="{}")
    findings_summary = Column(Text, default="{}")
    full_report_json = Column(Text, default="{}")
    pdf_path = Column(String(512), nullable=True)
    json_path = Column(String(512), nullable=True)
    csv_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=utcnow)


class ExperimentModel(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_name = Column(String(128), nullable=False)
    config_json = Column(Text, default="{}")
    ground_truth_json = Column(Text, default="{}")
    results_json = Column(Text, default="{}")
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    false_positive_rate = Column(Float, nullable=True)
    detection_rate = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utcnow)
