"""
Demo and Benchmark Execution API Router
"""
from fastapi import APIRouter
from scripts.run_demo import run_full_assurance_demo

router = APIRouter(prefix="/demo", tags=["Demo Benchmark"])

@router.post("/run")
def trigger_full_demo():
    """
    Execute end-to-end assurance evaluation benchmark across all 8 attack scenarios:
    Label flip, duplicate flooding, OOD injection, patch backdoors, model substitution,
    inference tampering, nonce replay, and distribution drift.
    Computes and returns actual Precision, Recall, F1, and FPR metrics.
    """
    results = run_full_assurance_demo()
    return results
