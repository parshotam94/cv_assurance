"""
Repository pattern helper methods for database interaction
"""
import json
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.database.models import (
    DatasetModel, DatasetSampleModel, DatasetSourceModel,
    ModelAssetModel, InferenceRecordModel, FindingModel,
    AuditEventModel, ReportModel, ExperimentModel
)

class Repository:
    def __init__(self, db: Session):
        self.db = db

    # Datasets
    def create_dataset(self, dataset_data: Dict[str, Any]) -> DatasetModel:
        db_obj = DatasetModel(**dataset_data)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

    def get_dataset(self, dataset_id: str) -> Optional[DatasetModel]:
        return self.db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()

    def list_datasets(self) -> List[DatasetModel]:
        return self.db.query(DatasetModel).order_by(DatasetModel.created_at.desc()).all()

    def update_dataset(self, dataset_id: str, updates: Dict[str, Any]) -> Optional[DatasetModel]:
        obj = self.get_dataset(dataset_id)
        if obj:
            for k, v in updates.items():
                setattr(obj, k, v)
            self.db.commit()
            self.db.refresh(obj)
        return obj

    # Dataset Samples
    def add_samples(self, samples_data: List[Dict[str, Any]]):
        objs = [DatasetSampleModel(**s) for s in samples_data]
        self.db.bulk_save_objects(objs)
        self.db.commit()

    def get_dataset_samples(self, dataset_id: str, limit: int = 100, offset: int = 0, suspicious_only: bool = False):
        q = self.db.query(DatasetSampleModel).filter(DatasetSampleModel.dataset_id == dataset_id)
        if suspicious_only:
            q = q.filter(DatasetSampleModel.is_suspicious == True)
        return q.offset(offset).limit(limit).all()

    # Sources
    def update_sources(self, dataset_id: str, sources_data: List[Dict[str, Any]]):
        self.db.query(DatasetSourceModel).filter(DatasetSourceModel.dataset_id == dataset_id).delete()
        valid_cols = {"source_id", "sample_count", "suspicious_count", "duplicate_clusters", "ood_count", "label_anomalies", "trigger_count", "risk_score", "risk_level", "disposition"}
        objs = []
        for s in sources_data:
            filtered = {k: v for k, v in s.items() if k in valid_cols}
            extra = {k: v for k, v in s.items() if k not in valid_cols}
            filtered["metadata_json"] = json.dumps(extra)
            objs.append(DatasetSourceModel(dataset_id=dataset_id, **filtered))
        self.db.bulk_save_objects(objs)
        self.db.commit()

    def get_sources(self, dataset_id: str) -> List[DatasetSourceModel]:
        return self.db.query(DatasetSourceModel).filter(DatasetSourceModel.dataset_id == dataset_id).all()

    # Models
    def create_model_asset(self, model_data: Dict[str, Any]) -> ModelAssetModel:
        obj = ModelAssetModel(**model_data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def get_model_asset(self, model_id: str) -> Optional[ModelAssetModel]:
        return self.db.query(ModelAssetModel).filter(ModelAssetModel.id == model_id).first()

    def get_model_by_sha256(self, sha256: str) -> Optional[ModelAssetModel]:
        return self.db.query(ModelAssetModel).filter(ModelAssetModel.sha256 == sha256).first()

    def list_models(self) -> List[ModelAssetModel]:
        return self.db.query(ModelAssetModel).order_by(ModelAssetModel.created_at.desc()).all()

    def update_model(self, model_id: str, updates: Dict[str, Any]) -> Optional[ModelAssetModel]:
        obj = self.get_model_asset(model_id)
        if obj:
            for k, v in updates.items():
                setattr(obj, k, v)
            self.db.commit()
            self.db.refresh(obj)
        return obj

    # Inference Records
    def create_inference_record(self, record_data: Dict[str, Any]) -> InferenceRecordModel:
        obj = InferenceRecordModel(**record_data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def get_inference_record(self, record_id: str) -> Optional[InferenceRecordModel]:
        return self.db.query(InferenceRecordModel).filter(InferenceRecordModel.record_id == record_id).first()

    def list_inference_records(self, limit: int = 100) -> List[InferenceRecordModel]:
        return self.db.query(InferenceRecordModel).order_by(InferenceRecordModel.id.desc()).limit(limit).all()

    # Findings
    def create_finding(self, finding_data: Dict[str, Any]) -> FindingModel:
        obj = FindingModel(**finding_data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def list_findings(self, asset_id: Optional[str] = None, category: Optional[str] = None, severity: Optional[str] = None) -> List[FindingModel]:
        q = self.db.query(FindingModel)
        if asset_id:
            q = q.filter(FindingModel.asset_id == asset_id)
        if category:
            q = q.filter(FindingModel.category == category)
        if severity:
            q = q.filter(FindingModel.severity == severity)
        return q.order_by(FindingModel.id.desc()).all()

    def get_finding(self, finding_id: str) -> Optional[FindingModel]:
        return self.db.query(FindingModel).filter(FindingModel.finding_id == finding_id).first()

    # Audit Events
    def get_latest_audit_event(self) -> Optional[AuditEventModel]:
        return self.db.query(AuditEventModel).order_by(AuditEventModel.id.desc()).first()

    def create_audit_event(self, event_data: Dict[str, Any]) -> AuditEventModel:
        obj = AuditEventModel(**event_data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def list_audit_events(self, limit: int = 100) -> List[AuditEventModel]:
        return self.db.query(AuditEventModel).order_by(AuditEventModel.id.asc()).limit(limit).all()

    # Reports
    def create_report(self, report_data: Dict[str, Any]) -> ReportModel:
        obj = ReportModel(**report_data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def get_report(self, report_id: str) -> Optional[ReportModel]:
        return self.db.query(ReportModel).filter(ReportModel.report_id == report_id).first()

    def list_reports(self) -> List[ReportModel]:
        return self.db.query(ReportModel).order_by(ReportModel.created_at.desc()).all()

    # Experiments
    def create_experiment(self, exp_data: Dict[str, Any]) -> ExperimentModel:
        obj = ExperimentModel(**exp_data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def list_experiments(self) -> List[ExperimentModel]:
        return self.db.query(ExperimentModel).order_by(ExperimentModel.created_at.desc()).all()
