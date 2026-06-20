import pytest
from app.config import get_settings


def test_settings_load(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "test-token")
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.macbook_api_key == "test-token"
    assert settings.database_path == "news_cache.db"


def test_settings_default_database_path(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "some-key")
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.database_path == "news_cache.db"


def test_settings_default_onnx_tolerance(monkeypatch):
    monkeypatch.setenv("MACBOOK_API_KEY", "some-key")
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.onnx_parity_tolerance == 1e-3


def test_settings_no_api_key_raises(monkeypatch):
    """When MACBOOK_API_KEY is not set, Settings() raises a validation error."""
    monkeypatch.delenv("MACBOOK_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()
