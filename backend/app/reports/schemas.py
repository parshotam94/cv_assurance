"""
Assurance Report Schemas
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class FindingItem(BaseModel):
    finding_id: str
    asset_type: str
    asset_id: str
    category: str
    severity: str
    confidence: float
    risk_score: float
    title: str
    description: str
    evidence: List[str] = Field(default_factory=list)
    affected_samples: List[str] = Field(default_factory=list)
    detection_method: str
    limitations: List[str] = Field(default_factory=list)
    recommendation: str
    created_at: str

class CoverageItem(BaseModel):
    threat_id: str
    capability: str
    supported_status: str  # SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED
    access_requirement: str
    primary_method: str
    key_limitation: str

class AssuranceReportData(BaseModel):
    report_id: str
    title: str = "VisionTrust AI Assurance & Integrity Assessment"
    generated_at: str
    overall_risk_score: float
    overall_risk_level: str
    overall_disposition: str  # ACCEPT, REVIEW, QUARANTINE
    executive_summary: Dict[str, Any]
    dataset_assurance: Dict[str, Any]
    model_assurance: Dict[str, Any]
    inference_assurance: Dict[str, Any]
    distribution_assurance: Dict[str, Any]
    findings: List[FindingItem] = Field(default_factory=list)
    coverage_matrix: List[CoverageItem] = Field(default_factory=list)
    system_limitations: List[str] = Field(default_factory=list)
