import os
# Set test database path before importing app to avoid downloading FinBERT model
os.environ["DATABASE_PATH"] = "test_news_cache.db"

from app.config import get_settings
get_settings.cache_clear()

from fastapi.testclient import TestClient
import pytest
from app.main import app

client = TestClient(app)


def test_news_annotation_endpoint(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()

    news_data = [
        {
            "event_id": "123",
            "dedupe_hash": "d123",
            "title": "Stock market surges",
            "snippet": "Strong earnings drive market higher.",
            "matched_symbols": ["SPY"],
            "observed_at": "2026-06-20T12:00:00",
            "source": "news",
        }
    ]
    response = client.post(
        "/api/v1/annotate_news",
        headers={"X-API-Key": "my-secret"},
        json=news_data,
    )
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["dedupe_hash"] == "d123"


def test_unauthenticated_request(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()
    response = client.post("/api/v1/annotate_news", json=[])
    assert response.status_code == 401


def test_score_daily_mover_endpoint_404(monkeypatch):
    """Scoring a non-existent model should return 404."""
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()

    config = '{"model_id": "nonexistent", "candidate_rows": []}'
    response = client.post(
        "/api/v1/score_daily_mover_candidates",
        data={"config_str": config},
        headers={"X-API-Key": "my-secret"},
    )
    # Should fail because model doesn't exist
    assert response.status_code in (400, 404)


def test_score_time_series_endpoint_404(monkeypatch):
    """Scoring a non-existent model should return 404."""
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()

    config = '{"model_id": "nonexistent", "candidate_rows": []}'
    response = client.post(
        "/api/v1/score_time_series_candidates",
        data={"config_str": config},
        headers={"X-API-Key": "my-secret"},
    )
    assert response.status_code in (400, 404)


def test_export_onnx_endpoint_404(monkeypatch):
    """Export for a non-existent model should return 404."""
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()

    response = client.post(
        "/api/v1/export_onnx",
        data={"model_id": "nonexistent"},
        headers={"X-API-Key": "my-secret"},
    )
    assert response.status_code == 404


def test_export_onnx_returns_error_message(monkeypatch):
    """Export should return error_message field even on failure."""
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()

    response = client.post(
        "/api/v1/export_onnx",
        data={"model_id": "nonexistent"},
        headers={"X-API-Key": "my-secret"},
    )
    body = response.json()
    assert "error_message" in body


def test_model_status_returns_error_message(monkeypatch):
    """Model status should include error_message field."""
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()

    response = client.get(
        "/api/v1/model_status/nonexistent",
        headers={"X-API-Key": "my-secret"},
    )
    assert response.status_code == 404


def test_root_scoring_endpoints_404(monkeypatch):
    """Verifying root scoring endpoints return 404 when model doesn't exist."""
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()

    config = '{"model_id": "nonexistent", "candidate_rows": []}'
    
    response1 = client.post(
        "/score_daily_mover_candidates",
        data={"config_str": config},
        headers={"X-API-Key": "my-secret"},
    )
    assert response1.status_code == 404

    response2 = client.post(
        "/score_time_series_candidates",
        data={"config_str": config},
        headers={"X-API-Key": "my-secret"},
    )
    assert response2.status_code == 404


def test_health_check_unauthenticated():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "noble-turing"
    assert "commit" in response.json()

def test_readiness_authenticated():
    response = client.get("/api/v1/readiness", headers={"X-API-Key": "test-token"})
    assert response.status_code == 200
    assert "api_key_auth_enabled" in response.json()
    assert response.json()["api_key_auth_enabled"] is True

def test_capabilities_authenticated():
    response = client.get("/api/v1/capabilities", headers={"X-API-Key": "test-token"})
    assert response.status_code == 200
    assert "supported_endpoints" in response.json()
    assert "/health" in response.json()["supported_endpoints"]

def test_export_package_unauthenticated_blocked():
    response = client.post("/api/v1/export_artha_package")
    assert response.status_code == 401

@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    # Cleanup test database
    if os.path.exists("test_news_cache.db"):
        os.remove("test_news_cache.db")
