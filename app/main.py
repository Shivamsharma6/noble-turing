import os
import uuid
import json
import sqlite3
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("macbook_lab")

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from app.auth import verify_api_key
from app.config import get_settings
from app.database import init_db, get_db_connection
from app.models_lab.finbert import annotate_news_batch
from app.models_lab.tabular import train_tabular_pipeline
from app.models_lab.sequence import train_sequence_pipeline
from app.models_lab.onnx_utils import export_and_verify_onnx

# Import new Artha compatible modules
from app.audit import log_audit_record, redact_secrets
from app.health import get_health_status, get_readiness, get_capabilities
from app.scoring.daily_mover import score_daily_mover
from app.scoring.time_series import score_time_series
from app.packaging.artha_package import export_artha_package

# ---------------------------------------------------------------------------
# Lifespan: create and shut down the training thread pool
# ---------------------------------------------------------------------------

training_pool = ThreadPoolExecutor(max_workers=2)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(settings.database_path)
    os.makedirs(settings.models_dir, exist_ok=True)
    os.makedirs(settings.data_dir, exist_ok=True)
    yield
    global training_pool
    if training_pool is not None:
        training_pool.shutdown(wait=True)

app = FastAPI(title="MacBook Model Lab Service", lifespan=lifespan)

# Helper function to update training job status
def _update_job_status(
    cursor: sqlite3.Cursor,
    model_id: str,
    status: str,
    metrics: Dict[str, Any],
    error_message: str | None = None,
):
    cursor.execute(
        """
        UPDATE jobs
        SET status = ?, completed_at = CURRENT_TIMESTAMP,
            metrics_json = ?, error_message = ?
        WHERE model_id = ?
    """,
        (status, json.dumps(metrics), error_message, model_id),
    )

# ---------------------------------------------------------------------------
# Safety Audit Logging Decorator
# ---------------------------------------------------------------------------
def audit_endpoint(endpoint_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            started_at = datetime.now(timezone.utc)
            
            # Find request
            request = kwargs.get("request") or next((a for a in args if isinstance(a, Request)), None)
            
            input_data = {}
            model_id = None
            
            if request:
                try:
                    # Read body for JSON payload
                    body_bytes = await request.body()
                    if body_bytes:
                        input_data = json.loads(body_bytes)
                        model_id = input_data.get("model_id")
                except Exception:
                    pass
                
                input_data = {
                    "body": input_data,
                    "query_params": dict(request.query_params),
                    "headers": dict(request.headers)
                }
            
            if "config_str" in kwargs and kwargs["config_str"]:
                try:
                    parsed = json.loads(kwargs["config_str"])
                    input_data["config_str"] = parsed
                    if not model_id:
                        model_id = parsed.get("model_id")
                except Exception:
                    input_data["config_str"] = kwargs["config_str"]
            if "model_id" in kwargs:
                model_id = kwargs["model_id"]
                input_data["model_id"] = model_id
                
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Retrieve JSON representation of Response if applicable
                output_data = result
                if isinstance(result, JSONResponse):
                    try:
                        output_data = json.loads(result.body.decode("utf-8"))
                    except Exception:
                        pass
                
                log_audit_record(
                    db_path=settings.database_path,
                    request_id=request_id,
                    endpoint=endpoint_name,
                    model_id=model_id,
                    input_data=input_data,
                    output_data=output_data,
                    status="completed",
                    started_at=started_at
                )
                return result
            except Exception as e:
                log_audit_record(
                    db_path=settings.database_path,
                    request_id=request_id,
                    endpoint=endpoint_name,
                    model_id=model_id,
                    input_data=input_data,
                    output_data=None,
                    status="failed",
                    error_message=str(e),
                    started_at=started_at
                )
                raise e
        return wrapper
    return decorator

# ---------------------------------------------------------------------------
# Background Training Workers
# ---------------------------------------------------------------------------
def _run_async_tabular_training(
    model_id: str, config: Dict[str, Any], db_path: str, models_dir: str
):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE jobs SET status = 'training' WHERE model_id = ?", (model_id,)
        )
        conn.commit()

        metrics, file_paths = train_tabular_pipeline(model_id, config, models_dir)

        # ONNX export (if requested)
        onnx_error: str | None = None
        if config.get("requested_export_format") == "onnx":
            onnx_status, parity_status, onnx_error = export_and_verify_onnx(
                model_id, models_dir
            )
            metrics["onnx_export_status"] = onnx_status
            metrics["onnx_parity_status"] = parity_status
            metrics["onnx_error_message"] = onnx_error

        _update_job_status(
            cursor, model_id, "completed", metrics, onnx_error
        )
    except Exception as e:
        logger.error(
            f"Error in async tabular training for model {model_id}: {e}",
            exc_info=True,
        )
        _update_job_status(cursor, model_id, "failed", {}, str(e))
    finally:
        conn.commit()
        conn.close()

def _run_async_sequence_training(
    model_id: str, config: Dict[str, Any], db_path: str, models_dir: str
):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE jobs SET status = 'training' WHERE model_id = ?", (model_id,)
        )
        conn.commit()

        metrics, file_paths = train_sequence_pipeline(model_id, config, models_dir)

        # ONNX export (if requested)
        onnx_error: str | None = None
        if config.get("requested_export_format") == "onnx":
            onnx_status, parity_status, onnx_error = export_and_verify_onnx(
                model_id, models_dir
            )
            metrics["onnx_export_status"] = onnx_status
            metrics["onnx_parity_status"] = parity_status
            metrics["onnx_error_message"] = onnx_error

        _update_job_status(
            cursor, model_id, "completed", metrics, onnx_error
        )
    except Exception as e:
        logger.error(
            f"Error in async sequence training for model {model_id}: {e}",
            exc_info=True,
        )
        _update_job_status(cursor, model_id, "failed", {}, str(e))
    finally:
        conn.commit()
        conn.close()

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_endpoint():
    """Unauthenticated health status of the service."""
    return get_health_status()

@app.get("/api/v1/readiness", dependencies=[Depends(verify_api_key)])
def readiness_endpoint():
    """System configuration parameters and job queue status."""
    return get_readiness(settings.database_path, settings.models_dir, settings.data_dir)

@app.get("/api/v1/capabilities", dependencies=[Depends(verify_api_key)])
def capabilities_endpoint():
    """Supported model families and packaging layouts."""
    return get_capabilities()

@app.post("/annotate_news", dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/annotate_news", dependencies=[Depends(verify_api_key)])
@audit_endpoint("/api/v1/annotate_news")
async def annotate_news(request: Request):
    news_items = await request.json()
    use_mock = "test" in settings.database_path
    return annotate_news_batch(news_items, settings.database_path, use_mock=use_mock)

@app.post("/train_tabular_model", dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/train_tabular_model", dependencies=[Depends(verify_api_key)])
@audit_endpoint("/api/v1/train_tabular_model")
async def train_tabular_model(
    request: Request,
    config_str: str = Form(...), 
    file: UploadFile = File(None)
):
    config = json.loads(config_str)
    model_id = str(uuid.uuid4())

    if file:
        save_path = os.path.join(settings.data_dir, f"{model_id}_{file.filename}")
        with open(save_path, "wb") as buffer:
            buffer.write(file.file.read())
        config["dataset_uri"] = save_path

    conn = get_db_connection(settings.database_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO jobs (model_id, status, model_family, model_type)
            VALUES (?, 'pending', ?, 'tabular')
        """,
            (model_id, config.get("model_family", "xgboost")),
        )
        conn.commit()
    finally:
        conn.close()

    training_pool.submit(
        _run_async_tabular_training,
        model_id,
        config,
        settings.database_path,
        settings.models_dir,
    )

    return {"model_id": model_id, "status": "pending", "message": "Job submitted."}

@app.post("/train_time_series_model", dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/train_time_series_model", dependencies=[Depends(verify_api_key)])
@audit_endpoint("/api/v1/train_time_series_model")
async def train_time_series_model(
    request: Request,
    config_str: str = Form(...), 
    file: UploadFile = File(None)
):
    config = json.loads(config_str)
    model_id = str(uuid.uuid4())

    if file:
        save_path = os.path.join(settings.data_dir, f"{model_id}_{file.filename}")
        with open(save_path, "wb") as buffer:
            buffer.write(file.file.read())
        config["dataset_uri"] = save_path

    conn = get_db_connection(settings.database_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO jobs (model_id, status, model_family, model_type)
            VALUES (?, 'pending', ?, 'sequence')
        """,
            (model_id, config.get("model_family", "pytorch_cnn")),
        )
        conn.commit()
    finally:
        conn.close()

    training_pool.submit(
        _run_async_sequence_training,
        model_id,
        config,
        settings.database_path,
        settings.models_dir,
    )

    return {"model_id": model_id, "status": "pending", "message": "Job submitted."}

# ---------------------------------------------------------------------------
# Synchronous Scoring Endpoints
# ---------------------------------------------------------------------------

@app.post("/score_daily_mover_candidates", dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/score_daily_mover_candidates", dependencies=[Depends(verify_api_key)])
@audit_endpoint("/api/v1/score_daily_mover_candidates")
async def score_daily_mover_candidates_endpoint(
    request: Request,
    config_str: str = Form(None)
):
    # Support form parameter config_str or JSON body
    payload = {}
    if config_str:
        try:
            payload = json.loads(config_str)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config_str JSON: {e}")
    else:
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    model_id = payload.get("model_id")
    if not model_id:
        return JSONResponse(
            status_code=404,
            content={"status": "blocked", "errors": ["model_missing"], "scores": []}
        )

    res = score_daily_mover(model_id, settings.database_path, settings.models_dir, payload)
    
    if res["status"] == "blocked":
        status_code = 404 if "model_missing" in res["errors"] else 400
        return JSONResponse(status_code=status_code, content=res)
        
    return res

@app.post("/score_time_series_candidates", dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/score_time_series_candidates", dependencies=[Depends(verify_api_key)])
@audit_endpoint("/api/v1/score_time_series_candidates")
async def score_time_series_candidates_endpoint(
    request: Request,
    config_str: str = Form(None)
):
    # Support form parameter config_str or JSON body
    payload = {}
    if config_str:
        try:
            payload = json.loads(config_str)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config_str JSON: {e}")
    else:
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    model_id = payload.get("model_id")
    if not model_id:
        return JSONResponse(
            status_code=404,
            content={"status": "blocked", "errors": ["sequence_model_missing"], "scores": []}
        )

    res = score_time_series(model_id, settings.models_dir, payload)
    
    if res["status"] == "blocked":
        status_code = 404 if "sequence_model_missing" in res["errors"] else 400
        return JSONResponse(status_code=status_code, content=res)
        
    return res

# ---------------------------------------------------------------------------
# ONNX Mutators & Parity
# ---------------------------------------------------------------------------

@app.post("/export_onnx", dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/export_onnx", dependencies=[Depends(verify_api_key)])
@audit_endpoint("/api/v1/export_onnx")
async def export_onnx_endpoint(
    request: Request,
    model_id: str = Form(...)
):
    save_dir = os.path.join(settings.models_dir, model_id)
    metadata_path = os.path.join(save_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return JSONResponse(
            status_code=404,
            content={
                "model_id": model_id,
                "onnx_export_status": "failed",
                "onnx_parity_status": "unchecked",
                "error_message": "metadata.json not found",
            }
        )

    onnx_status, parity_status, error_msg = export_and_verify_onnx(
        model_id, settings.models_dir
    )

    return {
        "model_id": model_id,
        "onnx_export_status": onnx_status,
        "onnx_parity_status": parity_status,
        "error_message": error_msg,
    }

@app.post("/validate_onnx_parity", dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/validate_onnx_parity", dependencies=[Depends(verify_api_key)])
@audit_endpoint("/api/v1/validate_onnx_parity")
async def validate_onnx_parity_endpoint(
    request: Request,
    model_id: str = Form(...)
):
    return await export_onnx_endpoint(request, model_id)

# ---------------------------------------------------------------------------
# Artha Compatible Packages Export
# ---------------------------------------------------------------------------

@app.post("/export_artha_package", dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/export_artha_package", dependencies=[Depends(verify_api_key)])
@audit_endpoint("/api/v1/export_artha_package")
async def export_artha_package_endpoint(
    request: Request,
    model_id: str = Form(...),
    package_type: str = Form(...)
):
    try:
        exported = export_artha_package(model_id, package_type, settings.models_dir)
        return {
            "status": "success",
            "model_id": model_id,
            "package_type": package_type,
            "exported_files": exported
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------------------------------------------------------------------------
# General Model Status and Downloader Aliases
# ---------------------------------------------------------------------------

@app.get("/model_status/{model_id}", dependencies=[Depends(verify_api_key)])
@app.get("/api/v1/model_status/{model_id}", dependencies=[Depends(verify_api_key)])
def get_model_status(model_id: str):
    conn = get_db_connection(settings.database_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM jobs WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}

    return {
        "model_id": row["model_id"],
        "status": row["status"],
        "model_family": row["model_family"],
        "model_type": row["model_type"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "error_message": row["error_message"],
        "metrics": metrics,
    }

@app.get("/model_artifact/{model_id}", dependencies=[Depends(verify_api_key)])
@app.get("/api/v1/model_artifact/{model_id}", dependencies=[Depends(verify_api_key)])
def get_model_artifacts(model_id: str):
    save_dir = os.path.join(settings.models_dir, model_id)
    if not os.path.exists(save_dir):
        raise HTTPException(status_code=404, detail="Model artifacts not found")
    files = os.listdir(save_dir)
    return {"model_id": model_id, "artifacts": files}

@app.get("/model_artifact/{model_id}/{filename}", dependencies=[Depends(verify_api_key)])
@app.get("/api/v1/model_artifact/{model_id}/{filename}", dependencies=[Depends(verify_api_key)])
def download_model_artifact(model_id: str, filename: str):
    file_path = os.path.join(settings.models_dir, model_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(file_path)
