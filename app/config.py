from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    macbook_api_key: str = "default-secret-token"
    database_path: str = "news_cache.db"
    models_dir: str = "models"
    data_dir: str = "data"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

