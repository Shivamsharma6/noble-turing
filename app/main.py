import os
import uuid
import json
import sqlite3
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse
from app.auth import verify_api_key
from app.config import get_settings
from app.database import init_db, get_db_connection
from app.models_lab.finbert import annotate_news_batch
from app.models_lab.tabular import train_tabular_pipeline
from app.models_lab.sequence import train_sequence_pipeline
from app.models_lab.onnx_utils import export_and_verify_onnx

app = FastAPI(title="MacBook Model Lab Service")
settings = get_settings()

# Initialize directories & database
init_db(settings.database_path)
os.makedirs(settings.models_dir, exist_ok=True)
os.makedirs(settings.data_dir, exist_ok=True)

# ThreadPoolExecutor for heavy background training jobs (CPU/GPU bound)
training_pool = ThreadPoolExecutor(max_workers=2)

def run_async_tabular_training(model_id: str, config: Dict[str, Any], db_path: str, models_dir: str):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE jobs SET status = 'training' WHERE model_id = ?", (model_id,))
        conn.commit()
        
        metrics, file_paths = train_tabular_pipeline(model_id, config, models_dir)
        
        # ONNX export
        if config.get("requested_export_format") == "onnx":
            onnx_status, parity_status = export_and_verify_onnx(model_id, models_dir)
            metrics["onnx_export_status"] = onnx_status
            metrics["onnx_parity_status"] = parity_status
            
        cursor.execute("""
            UPDATE jobs 
            SET status = 'completed', completed_at = CURRENT_TIMESTAMP, metrics_json = ? 
            WHERE model_id = ?
        """, (json.dumps(metrics), model_id))
    except Exception as e:
        cursor.execute("""
            UPDATE jobs 
            SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error_message = ? 
            WHERE model_id = ?
        """, (str(e), model_id))
    finally:
        conn.commit()
        conn.close()

def run_async_sequence_training(model_id: str, config: Dict[str, Any], db_path: str, models_dir: str):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE jobs SET status = 'training' WHERE model_id = ?", (model_id,))
        conn.commit()
        
        metrics, file_paths = train_sequence_pipeline(model_id, config, models_dir)
        
        # ONNX export
        if config.get("requested_export_format") == "onnx":
            onnx_status, parity_status = export_and_verify_onnx(model_id, models_dir)
            metrics["onnx_export_status"] = onnx_status
            metrics["onnx_parity_status"] = parity_status
            
        cursor.execute("""
            UPDATE jobs 
            SET status = 'completed', completed_at = CURRENT_TIMESTAMP, metrics_json = ? 
            WHERE model_id = ?
        """, (json.dumps(metrics), model_id))
    except Exception as e:
        cursor.execute("""
            UPDATE jobs 
            SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error_message = ? 
            WHERE model_id = ?
        """, (str(e), model_id))
    finally:
        conn.commit()
        conn.close()

@app.post("/api/v1/annotate_news", dependencies=[Depends(verify_api_key)])
def annotate_news(news_items: List[Dict[str, Any]]):
    # Auto-mock in test databases to prevent downloading model during test runs
    use_mock = "test" in settings.database_path
    return annotate_news_batch(news_items, settings.database_path, use_mock=use_mock)

@app.post("/api/v1/train_tabular_model", dependencies=[Depends(verify_api_key)])
def train_tabular_model(config_str: str = Form(...), file: UploadFile = File(None)):
    config = json.loads(config_str)
    model_id = str(uuid.uuid4())
    
    # Handle multipart dataset upload
    if file:
        save_path = os.path.join(settings.data_dir, f"{model_id}_{file.filename}")
        with open(save_path, "wb") as buffer:
            buffer.write(file.file.read())
        config["dataset_uri"] = save_path
        
    conn = get_db_connection(settings.database_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO jobs (model_id, status, model_family, model_type)
            VALUES (?, 'pending', ?, 'tabular')
        """, (model_id, config.get("model_family", "xgboost")))
        conn.commit()
    finally:
        conn.close()
        
    training_pool.submit(
        run_async_tabular_training, 
        model_id, config, settings.database_path, settings.models_dir
    )
    
    return {"model_id": model_id, "status": "pending", "message": "Job submitted."}

@app.post("/api/v1/train_time_series_model", dependencies=[Depends(verify_api_key)])
def train_time_series_model(config_str: str = Form(...), file: UploadFile = File(None)):
    config = json.loads(config_str)
    model_id = str(uuid.uuid4())
    
    # Handle multipart dataset upload
    if file:
        save_path = os.path.join(settings.data_dir, f"{model_id}_{file.filename}")
        with open(save_path, "wb") as buffer:
            buffer.write(file.file.read())
        config["dataset_uri"] = save_path
        
    conn = get_db_connection(settings.database_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO jobs (model_id, status, model_family, model_type)
            VALUES (?, 'pending', ?, 'sequence')
        """, (model_id, config.get("model_family", "pytorch_cnn")))
        conn.commit()
    finally:
        conn.close()
        
    training_pool.submit(
        run_async_sequence_training, 
        model_id, config, settings.database_path, settings.models_dir
    )
    
    return {"model_id": model_id, "status": "pending", "message": "Job submitted."}

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
        "metrics": metrics
    }

@app.get("/api/v1/model_artifact/{model_id}", dependencies=[Depends(verify_api_key)])
def get_model_artifacts(model_id: str):
    save_dir = os.path.join(settings.models_dir, model_id)
    if not os.path.exists(save_dir):
        raise HTTPException(status_code=404, detail="Model artifacts not found")
    files = os.listdir(save_dir)
    return {"model_id": model_id, "artifacts": files}

@app.get("/api/v1/model_artifact/{model_id}/{filename}", dependencies=[Depends(verify_api_key)])
def download_model_artifact(model_id: str, filename: str):
    file_path = os.path.join(settings.models_dir, model_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(file_path)
