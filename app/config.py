import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    macbook_api_key: str = os.getenv("MACBOOK_API_KEY", "default-secret-token")
    database_path: str = "news_cache.db"
    models_dir: str = "models"
    data_dir: str = "data"

def get_settings() -> Settings:
    return Settings()
