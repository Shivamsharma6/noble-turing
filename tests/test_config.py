import pytest
from app.config import get_settings

def test_settings_load(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "test-token")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.macbook_api_key == "test-token"
    assert settings.database_path == "news_cache.db"

def test_settings_default(monkeypatch):
    monkeypatch.delenv("MACBOOK_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.macbook_api_key == "default-secret-token"

