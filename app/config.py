import os
# Prevent OpenMP/MKL multi-threading deadlocks between PyTorch and XGBoost on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

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

