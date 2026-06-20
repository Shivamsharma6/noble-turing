import pytest
from app.health import get_health_status, get_readiness, get_capabilities

def test_health_checks():
    h = get_health_status()
    assert h["status"] == "ok"
    assert h["service"] == "noble-turing"
    
    r = get_capabilities()
    assert "xgboost" in r["supported_tabular_model_families"]
    assert "pytorch_cnn" in r["supported_sequence_model_families"]
