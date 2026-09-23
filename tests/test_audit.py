"""
Unit Tests for Hash-Chained Audit Logging and Cryptographic Chain Verification
"""
import pytest
from backend.app.core.audit import log_audit_event, verify_audit_chain
from backend.app.database.models import AuditEventModel

def test_audit_hash_chain_validity(db_session):
    # Log 3 events
    ev1 = log_audit_event(db_session, action="TEST_ACTION_1", asset="Asset_1")
    ev2 = log_audit_event(db_session, action="TEST_ACTION_2", asset="Asset_2")
    ev3 = log_audit_event(db_session, action="TEST_ACTION_3", asset="Asset_3")

    # Genesis check
    assert ev1.previous_hash == "0" * 64
    assert ev2.previous_hash == ev1.current_hash
    assert ev3.previous_hash == ev2.current_hash

    # Verify chain
    res = verify_audit_chain(db_session)
    assert res["status"] == "VALID"
    assert res["total_events"] == 3

def test_audit_chain_tamper_detection(db_session):
    # Log 3 events
    log_audit_event(db_session, action="ACTION_A", asset="Asset_A")
    ev2 = log_audit_event(db_session, action="ACTION_B", asset="Asset_B")
    log_audit_event(db_session, action="ACTION_C", asset="Asset_C")

    # Directly corrupt ev2 in the database
    ev2.current_hash = "deadbeef" * 8
    db_session.commit()

    # Verify chain detects tampering
    res = verify_audit_chain(db_session)
    assert res["status"] == "TAMPERED"
    assert "broken_event_id" in res
