"""
Integration Tests for FastAPI REST Endpoints
"""
import pytest

def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["air_gapped"] is True
    assert "ml_runtimes" in data

def test_api_datasets_list(client):
    res = client.get("/api/datasets")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_api_models_list(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_api_inference_records_list(client):
    res = client.get("/api/inference/records")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_api_findings_list(client):
    res = client.get("/api/findings")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_api_audit_list(client):
    res = client.get("/api/audit")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_api_audit_verify(client):
    res = client.post("/api/audit/verify")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data

def test_api_reports_list(client):
    res = client.get("/api/reports")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
