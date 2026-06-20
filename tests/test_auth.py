from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
import pytest
from app.auth import verify_api_key
from app.config import get_settings

app = FastAPI()


@app.get("/secure")
def secure_endpoint(api_key: str = Depends(verify_api_key)):
    return {"status": "authenticated"}


client = TestClient(app)


def test_auth_success(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "correct-key")
    get_settings.cache_clear()
    response = client.get("/secure", headers={"X-API-Key": "correct-key"})
    assert response.status_code == 200
    assert response.json() == {"status": "authenticated"}


def test_auth_failure(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "correct-key")
    get_settings.cache_clear()
    response = client.get("/secure", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]


def test_auth_missing_header():
    response = client.get("/secure")
    assert response.status_code == 401  # FastAPI raises 401 for missing APIKeyHeader


def test_verify_api_key_signature():
    import inspect

    sig = inspect.signature(verify_api_key)
    param = sig.parameters["api_key"]
    assert param.annotation == str


def test_auth_no_env_var_fails(monkeypatch):
    """When MACBOOK_API_KEY is not set, Settings() raises a validation error."""
    monkeypatch.delenv("MACBOOK_API_KEY", raising=False)
    get_settings.cache_clear()
    # Without the env var, Settings() will raise a validation error
    # because macbook_api_key is now required
    with pytest.raises(Exception):
        get_settings()
