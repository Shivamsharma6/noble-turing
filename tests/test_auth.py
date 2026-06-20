from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
import pytest
from app.auth import verify_api_key

app = FastAPI()

@app.get("/secure")
def secure_endpoint(api_key: str = Depends(verify_api_key)):
    return {"status": "authenticated"}

client = TestClient(app)

def test_auth_success(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "correct-key")
    response = client.get("/secure", headers={"X-API-Key": "correct-key"})
    assert response.status_code == 200
    assert response.json() == {"status": "authenticated"}

def test_auth_failure(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "correct-key")
    response = client.get("/secure", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
