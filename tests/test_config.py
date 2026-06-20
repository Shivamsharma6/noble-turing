import os
import pytest
from app.config import get_settings

def test_settings_load():
    os.environ["MACBOOK_API_KEY"] = "test-token"
    settings = get_settings()
    assert settings.macbook_api_key == "test-token"
    assert settings.database_path == "news_cache.db"
