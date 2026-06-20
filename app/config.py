import os
# Prevent OpenMP/MKL multi-threading deadlocks between PyTorch and XGBoost on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from functools import cache
from pydantic_settings import BaseSettings

ONNX_PARITY_TOLERANCE = 1e-3


class Settings(BaseSettings):
    macbook_api_key: str
    database_path: str = "news_cache.db"
    models_dir: str = "models"
    data_dir: str = "data"
    onnx_parity_tolerance: float = ONNX_PARITY_TOLERANCE


@cache
def get_settings() -> Settings:
    return Settings()
