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
            "source": "news"
        }
    ]
    response = client.post(
        "/api/v1/annotate_news", 
        headers={"X-API-Key": "my-secret"},
        json=news_data
    )
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["dedupe_hash"] == "d123"

def test_unauthenticated_request(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
    get_settings.cache_clear()
    response = client.post("/api/v1/annotate_news", json=[])
    assert response.status_code == 401

@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    # Cleanup test database
    if os.path.exists("test_news_cache.db"):
        os.remove("test_news_cache.db")
