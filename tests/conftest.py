import os
import pytest
from app.config import get_settings
from app.database import init_db

# Ensure MACBOOK_API_KEY is always set during testing to prevent pydantic validation errors
os.environ["MACBOOK_API_KEY"] = "test-token"
if not os.environ.get("DATABASE_PATH"):
    os.environ["DATABASE_PATH"] = "test_news_cache.db"

@pytest.fixture(autouse=True)
def init_test_db():
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings.database_path)
