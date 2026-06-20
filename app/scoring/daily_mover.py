import os
import json
import pickle
import sqlite3
import numpy as np
from typing import Dict, Any, List

def get_news_sentiment_score(db_path: str, news_event_ids: List[str]) -> float:
    if not news_event_ids:
        return 0.0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    scores = []
    try:
        for eid in news_event_ids:
            cursor.execute(
                "SELECT positive_score, negative_score FROM news_annotations WHERE dedupe_hash = ?",
                (eid,)
            )
            row = cursor.fetchone()
            if row:
                pos = row["positive_score"]
                neg = row["negative_score"]
                # Normalize to [0, 1]
                scores.append((pos - neg + 1.0) / 2.0)
    except Exception:
        pass
    finally:
        conn.close()
    return float(np.mean(scores)) if scores else 0.0

def score_daily_mover(
    model_id: str,
    db_path: str,
    models_dir: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    save_dir = os.path.join(models_dir, model_id)
    metadata_path = os.path.join(save_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return {
            "status": "blocked",
            "errors": ["model_missing"],
            "scores": []
        }
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    # Schema check
    req_schema = payload.get("feature_schema_hash")
    if req_schema and metadata.get("feature_schema_hash") != req_schema:
        return {
            "status": "blocked",
            "errors": ["feature_schema_mismatch"],
            "scores": []
        }
        
    # Load model
    model_pkl = os.path.join(save_dir, "model.pkl")
    if not os.path.exists(model_pkl):
        return {
            "status": "blocked",
            "errors": ["model_missing"],
            "scores": []
        }
        
    with open(model_pkl, "rb") as f:
        model = pickle.load(f)
        
    # Load threshold
    thresholds_path = os.path.join(save_dir, "thresholds.json")
    threshold = 0.5
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            threshold = json.load(f).get("threshold", 0.5)
            
    feature_names = metadata.get("feature_columns", [])
    if not feature_names:
        feature_names = list(getattr(model, "feature_names_in_", None) or [])
        
    candidates = payload.get("candidates", [])
    scores_list = []
    missing_candidates = []
    
    for cand in candidates:
        cand_id = cand.get("candidate_id")
        feats = cand.get("features")
        if feats is None:
            missing_candidates.append(cand_id)
            continue
            
        # Format features
        X_row = []
        for fname in feature_names:
            val = feats.get(fname)
            X_row.append(float(val) if val is not None else 0.0)
            
        X_arr = np.array([X_row], dtype=np.float32)
        tabular_score = float(model.predict_proba(X_arr)[0, 1])
        
        # Compute news score
        news_score = get_news_sentiment_score(db_path, cand.get("news_event_ids", []))
        
        # Default timeseries score
        ts_score = 0.0
        
        final_score = 0.9 * tabular_score + 0.1 * news_score
        
        scores_list.append({
            "candidate_id": cand_id,
            "tabular_score": tabular_score,
            "time_series_score": ts_score,
            "news_score": news_score,
            "final_score": final_score,
            "score_source": "mac_api",
            "model_id": model_id,
            "metadata": {
                "feature_hash": cand.get("feature_hash"),
                "sequence_hash": cand.get("sequence_hash"),
                "paper_only": True,
                "broker_routed": False,
                "live_eligible": False,
                "scoring_formula": "0.9 * tabular_score + 0.1 * news_score",
                "weights": {
                    "tabular_score": 0.9,
                    "news_score": 0.1,
                    "time_series_score": 0.0
                },
                "threshold_source": "thresholds.json"
            }
        })
        
    return {
        "status": "completed",
        "model_id": model_id,
        "model_package_id": payload.get("model_package_id"),
        "scores": scores_list,
        "missing_candidates": missing_candidates,
        "errors": []
    }
