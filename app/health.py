import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List

def get_commit_sha() -> str:
    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"

def get_health_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "noble-turing",
        "version": "0.1.0",
        "commit": get_commit_sha(),
        "time": datetime.now(timezone.utc).isoformat()
    }

def get_readiness(db_path: str, models_dir: str, data_dir: str) -> Dict[str, Any]:
    import torch
    from app.models_lab import finbert
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    # FinBERT status
    finbert_status = "loaded" if finbert._pipeline is not None else "not_loaded"
    
    # Query active / failed jobs
    active_jobs = 0
    last_failed_job = None
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('pending', 'training', 'scoring')"
            )
            active_jobs = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT model_id, error_message FROM jobs WHERE status = 'failed' ORDER BY completed_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                last_failed_job = {
                    "model_id": row["model_id"],
                    "error_message": row["error_message"]
                }
            conn.close()
        except Exception:
            pass
            
    return {
        "api_key_auth_enabled": True,
        "database_path": db_path,
        "model_directory": models_dir,
        "data_directory": data_dir,
        "finbert_load_status": finbert_status,
        "device": device,
        "active_jobs": active_jobs,
        "last_failed_job": last_failed_job
    }

def get_capabilities() -> Dict[str, Any]:
    return {
        "supported_endpoints": [
            "/health",
            "/api/v1/readiness",
            "/api/v1/capabilities",
            "/api/v1/train_tabular_model",
            "/api/v1/train_time_series_model",
            "/api/v1/score_daily_mover_candidates",
            "/api/v1/score_time_series_candidates",
            "/api/v1/annotate_news",
            "/api/v1/export_onnx",
            "/api/v1/validate_onnx_parity",
            "/api/v1/export_artha_package"
        ],
        "supported_tabular_model_families": ["xgboost", "catboost", "autogluon", "tabm", "tabpfn", "lightgbm"],
        "supported_sequence_model_families": ["pytorch_cnn", "minirocket", "inceptiontime", "tcn", "resnet"],
        "supported_export_formats": ["pkl", "pt", "onnx"],
        "package_formats": ["artha_onnx", "artha_mac_api"],
        "scoring_support_status": True
    }
