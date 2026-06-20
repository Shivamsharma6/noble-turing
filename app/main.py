import os
import uuid
import json
import sqlite3
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("macbook_lab")

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from app.auth import verify_api_key
from app.config import get_settings
from app.database import init_db, get_db_connection
from app.models_lab.finbert import annotate_news_batch
from app.models_lab.tabular import train_tabular_pipeline
from app.models_lab.sequence import train_sequence_pipeline
from app.models_lab.onnx_utils import export_and_verify_onnx

# ---------------------------------------------------------------------------
# Lifespan: create and shut down the training thread pool
# ---------------------------------------------------------------------------

training_pool = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db(settings.database_path)
    os.makedirs(settings.models_dir, exist_ok=True)
    os.makedirs(settings.data_dir, exist_ok=True)
    yield
    global training_pool
    if training_pool is not None:
        training_pool.shutdown(wait=True)


app = FastAPI(title="MacBook Model Lab Service", lifespan=lifespan)
settings = get_settings()


def _update_job_status(
    cursor: sqlite3.Cursor,
    model_id: str,
    status: str,
    metrics: Dict[str, Any],
    error_message: str | None = None,
):
    """Persist job completion (success or failure) to the database."""
    cursor.execute(
        """
        UPDATE jobs
        SET status = ?, completed_at = CURRENT_TIMESTAMP,
            metrics_json = ?, error_message = ?
        WHERE model_id = ?
    """,
        (status, json.dumps(metrics), error_message, model_id),
    )


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


def _run_async_scoring(
    model_id: str, config: Dict[str, Any], db_path: str, models_dir: str
):
    """Score candidates using a trained model."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE jobs SET status = 'scoring' WHERE model_id = ?", (model_id,)
        )
        conn.commit()

        scoring_results = score_candidates(model_id, config, models_dir)

        _update_job_status(cursor, model_id, "completed", scoring_results)
    except Exception as e:
        logger.error(
            f"Error in async scoring for model {model_id}: {e}",
            exc_info=True,
        )
        _update_job_status(cursor, model_id, "failed", {}, str(e))
    finally:
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Scoring helpers (called from background workers)
# ---------------------------------------------------------------------------

def score_candidates(
    model_id: str, config: Dict[str, Any], models_dir: str
) -> Dict[str, Any]:
    """Score candidate rows using a trained model.

    Supports both tabular and sequence models.
    """
    save_dir = os.path.join(models_dir, model_id)
    metadata_path = os.path.join(save_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model {model_id} not found")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model_type = metadata["model_type"]
    candidate_rows = config.get("candidate_rows", [])

    if model_type == "tabular":
        return _score_tabular(model_id, save_dir, metadata, candidate_rows)
    elif model_type == "sequence":
        return _score_sequence(model_id, save_dir, metadata, candidate_rows)
    else:
        raise ValueError(f"Unsupported model_type for scoring: {model_type}")


def _score_tabular(
    model_id: str,
    save_dir: str,
    metadata: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    import pickle
    import numpy as np

    with open(os.path.join(save_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    # Load threshold
    thresholds_path = os.path.join(save_dir, "thresholds.json")
    threshold = 0.5
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            threshold = json.load(f).get("threshold", 0.5)

    # Build feature matrix from candidate rows
    feature_names = list(getattr(model, "feature_names_in_", None) or [])
    if not feature_names:
        # Try to infer from metadata
        feature_names = [f"f{i}" for i in range(len(candidate_rows[0].keys()) - 1)]

    X_candidates = np.array(
        [[row.get(fn, 0.0) for fn in feature_names] for row in candidate_rows],
        dtype=np.float32,
    )

    probabilities = model.predict_proba(X_candidates)[:, 1].tolist()
    predictions = (np.array(probabilities) >= threshold).astype(int).tolist()

    return {
        "model_id": model_id,
        "model_family": metadata.get("model_family", "xgboost"),
        "model_type": "tabular",
        "threshold": threshold,
        "predictions": [
            {
                "row_index": i,
                "probability": float(prob),
                "prediction": int(pred),
            }
            for i, (prob, pred) in enumerate(zip(probabilities, predictions))
        ],
    }


def _score_sequence(
    model_id: str,
    save_dir: str,
    metadata: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    import torch
    import numpy as np

    seq_len = metadata["sequence_length"]
    num_features = metadata["num_features"]

    # Try all known architectures
    from app.models_lab.sequence import (
        Simple1DCNN,
        MiniRocketFeatureExtractor,
        InceptionTime,
        TCN,
        ResNetTS,
    )

    model_classes = [
        ("Simple1DCNN", Simple1DCNN),
        ("MiniRocketFeatureExtractor", MiniRocketFeatureExtractor),
        ("InceptionTime", InceptionTime),
        ("TCN", TCN),
        ("ResNetTS", ResNetTS),
    ]

    model = None
    for _, cls in model_classes:
        try:
            model = cls(seq_len, num_features)
            model.load_state_dict(
                torch.load(
                    os.path.join(save_dir, "model.pt"), weights_only=True
                ),
            )
            model.eval()
            break
        except (KeyError, RuntimeError, AttributeError):
            model = None
            continue

    if model is None:
        raise FileNotFoundError(
            f"Could not load any known PyTorch model for {model_id}"
        )

    # Load threshold
    thresholds_path = os.path.join(save_dir, "thresholds.json")
    threshold = 0.5
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            threshold = json.load(f).get("threshold", 0.5)

    # Build candidate tensors
    # candidate_rows is a list of dicts; each dict maps column names to values
    # We need to reshape into (N, seq_len, num_features)
    all_values = []
    for row in candidate_rows:
        vals = [row.get(f"t{t}_f{f}", 0.0) for t in range(seq_len) for f in range(num_features)]
        all_values.append(vals)

    X_candidates = np.array(all_values, dtype=np.float32).reshape(
        -1, seq_len, num_features
    )

    with torch.no_grad():
        probabilities = model(
            torch.tensor(X_candidates, dtype=torch.float32)
        ).numpy().flatten().tolist()

    predictions = (np.array(probabilities) >= threshold).astype(int).tolist()

    return {
        "model_id": model_id,
        "model_family": metadata.get("model_family", "pytorch_cnn"),
        "model_type": "sequence",
        "threshold": threshold,
        "predictions": [
            {
                "row_index": i,
                "probability": float(prob),
                "prediction": int(pred),
            }
            for i, (prob, pred) in enumerate(zip(probabilities, predictions))
        ],
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/annotate_news", dependencies=[Depends(verify_api_key)])
def annotate_news(news_items: List[Dict[str, Any]]):
    # Auto-mock in test databases to prevent downloading model during test runs
    use_mock = "test" in settings.database_path
    return annotate_news_batch(news_items, settings.database_path, use_mock=use_mock)


@app.post("/api/v1/train_tabular_model", dependencies=[Depends(verify_api_key)])
def train_tabular_model(
    config_str: str = Form(...), file: UploadFile = File(None)
):
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


@app.post("/api/v1/train_time_series_model", dependencies=[Depends(verify_api_key)])
def train_time_series_model(
    config_str: str = Form(...), file: UploadFile = File(None)
):
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


@app.post(
    "/score_daily_mover_candidates",
    dependencies=[Depends(verify_api_key)],
)
@app.post(
    "/api/v1/score_daily_mover_candidates",
    dependencies=[Depends(verify_api_key)],
)
def score_daily_mover_candidates(
    config_str: str = Form(...),
):
    """Score daily mover candidates using a trained tabular model."""
    config = json.loads(config_str)
    model_id = config.get("model_id")

    if not model_id:
        raise HTTPException(
            status_code=400, detail="model_id is required in config"
        )

    # Validate model exists and schema hash match
    save_dir = os.path.join(settings.models_dir, model_id)
    metadata_path = os.path.join(save_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404, detail=f"Model {model_id} not found"
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    requested_schema = config.get("feature_schema_hash")
    if requested_schema and metadata.get("feature_schema_hash") != requested_schema:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Schema hash mismatch: requested {requested_schema}, "
                f"model has {metadata.get('feature_schema_hash')}"
            ),
        )

    scoring_model_id = str(uuid.uuid4())

    conn = get_db_connection(settings.database_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO jobs (model_id, status, model_family, model_type)
            VALUES (?, 'pending', ?, 'tabular')
        """,
            (scoring_model_id, "scoring"),
        )
        conn.commit()
    finally:
        conn.close()

    training_pool.submit(
        _run_async_scoring,
        scoring_model_id,
        config,
        settings.database_path,
        settings.models_dir,
    )

    return {
        "model_id": scoring_model_id,
        "status": "pending",
        "message": "Scoring job submitted.",
    }


@app.post(
    "/score_time_series_candidates",
    dependencies=[Depends(verify_api_key)],
)
@app.post(
    "/api/v1/score_time_series_candidates",
    dependencies=[Depends(verify_api_key)],
)
def score_time_series_candidates(
    config_str: str = Form(...),
):
    """Score time-series candidates using a trained sequence model."""
    config = json.loads(config_str)
    model_id = config.get("model_id")

    if not model_id:
        raise HTTPException(
            status_code=400, detail="model_id is required in config"
        )

    # Validate model exists and schema hash match
    save_dir = os.path.join(settings.models_dir, model_id)
    metadata_path = os.path.join(save_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404, detail=f"Model {model_id} not found"
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    requested_schema = config.get("sequence_schema_hash")
    if requested_schema and metadata.get("sequence_schema_hash") != requested_schema:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Schema hash mismatch: requested {requested_schema}, "
                f"model has {metadata.get('sequence_schema_hash')}"
            ),
        )

    scoring_model_id = str(uuid.uuid4())

    conn = get_db_connection(settings.database_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO jobs (model_id, status, model_family, model_type)
            VALUES (?, 'pending', ?, 'sequence')
        """,
            (scoring_model_id, "scoring"),
        )
        conn.commit()
    finally:
        conn.close()

    training_pool.submit(
        _run_async_scoring,
        scoring_model_id,
        config,
        settings.database_path,
        settings.models_dir,
    )

    return {
        "model_id": scoring_model_id,
        "status": "pending",
        "message": "Scoring job submitted.",
    }


@app.post(
    "/api/v1/export_onnx",
    dependencies=[Depends(verify_api_key)],
)
def export_onnx_endpoint(model_id: str = Form(...)):
    """Manually trigger ONNX export and parity check for an existing model."""
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


@app.post(
    "/api/v1/validate_onnx_parity",
    dependencies=[Depends(verify_api_key)],
)
def validate_onnx_parity(model_id: str = Form(...)):
    """Re-run ONNX parity check for an existing model."""
    return export_onnx_endpoint(model_id)


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


@app.get(
    "/api/v1/model_artifact/{model_id}", dependencies=[Depends(verify_api_key)]
)
def get_model_artifacts(model_id: str):
    save_dir = os.path.join(settings.models_dir, model_id)
    if not os.path.exists(save_dir):
        raise HTTPException(status_code=404, detail="Model artifacts not found")
    files = os.listdir(save_dir)
    return {"model_id": model_id, "artifacts": files}


@app.get(
    "/api/v1/model_artifact/{model_id}/{filename}",
    dependencies=[Depends(verify_api_key)],
)
def download_model_artifact(model_id: str, filename: str):
    file_path = os.path.join(settings.models_dir, model_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(file_path)
