import os
import json
import torch
import numpy as np
from typing import Dict, Any, List

def score_time_series(
    model_id: str,
    models_dir: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    save_dir = os.path.join(models_dir, model_id)
    metadata_path = os.path.join(save_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return {
            "status": "blocked",
            "errors": ["sequence_model_missing"],
            "scores": []
        }
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    # Schema check
    req_schema = payload.get("sequence_schema_hash")
    if req_schema and metadata.get("sequence_schema_hash") != req_schema:
        return {
            "status": "blocked",
            "errors": ["sequence_schema_mismatch"],
            "scores": []
        }
        
    model_pt = os.path.join(save_dir, "model.pt")
    if not os.path.exists(model_pt):
        return {
            "status": "blocked",
            "errors": ["sequence_model_missing"],
            "scores": []
        }
        
    seq_len = metadata.get("sequence_length", 10)
    num_features = metadata.get("num_features", 2)
    
    # Reconstruct PyTorch model
    from app.models_lab.sequence import (
        Simple1DCNN, MiniRocketFeatureExtractor, InceptionTime, TCN, ResNetTS
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
            model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
            model.eval()
            break
        except Exception:
            model = None
            
    if model is None:
        return {
            "status": "blocked",
            "errors": ["sequence_model_missing"],
            "scores": []
        }
        
    candidates = payload.get("candidates", [])
    scores_list = []
    missing_candidates = []
    
    for cand in candidates:
        cand_id = cand.get("candidate_id")
        seq_feats = cand.get("sequence_features")
        
        if seq_feats is None:
            missing_candidates.append(cand_id)
            continue
            
        # Parse sequence_features
        try:
            if isinstance(seq_feats, list):
                # list of dicts or list of lists
                if len(seq_feats) == 0:
                    X_row = np.zeros((seq_len, num_features), dtype=np.float32)
                elif isinstance(seq_feats[0], dict):
                    # list of dicts
                    feature_names = metadata.get("feature_columns", [])
                    X_row_list = []
                    for t in range(seq_len):
                        step_row = []
                        step_dict = seq_feats[t] if t < len(seq_feats) else {}
                        for f in range(num_features):
                            # try different lookup options
                            keys_to_try = [
                                f"t{t}_f{f}",
                                f"f{f}",
                            ]
                            if feature_names:
                                feat_idx = t * num_features + f
                                if feat_idx < len(feature_names):
                                    keys_to_try.append(feature_names[feat_idx])
                            val = None
                            for k in keys_to_try:
                                if k in step_dict:
                                    val = step_dict[k]
                                    break
                            step_row.append(float(val) if val is not None else 0.0)
                        X_row_list.append(step_row)
                    X_row = np.array(X_row_list, dtype=np.float32)
                else:
                    # list of lists
                    X_row_list = []
                    for t in range(seq_len):
                        step_list = seq_feats[t] if t < len(seq_feats) else []
                        step_row = []
                        for f in range(num_features):
                            val = step_list[f] if f < len(step_list) else 0.0
                            step_row.append(float(val))
                        X_row_list.append(step_row)
                    X_row = np.array(X_row_list, dtype=np.float32)
            elif isinstance(seq_feats, dict):
                # flat dict
                feature_names = metadata.get("feature_columns", [])
                X_row_list = []
                for t in range(seq_len):
                    step_row = []
                    for f in range(num_features):
                        keys_to_try = [
                            f"t{t}_f{f}",
                            f"f{f}"
                        ]
                        if feature_names:
                            feat_idx = t * num_features + f
                            if feat_idx < len(feature_names):
                                keys_to_try.append(feature_names[feat_idx])
                        val = None
                        for k in keys_to_try:
                            if k in seq_feats:
                                val = seq_feats[k]
                                break
                        step_row.append(float(val) if val is not None else 0.0)
                    X_row_list.append(step_row)
                X_row = np.array(X_row_list, dtype=np.float32)
            else:
                X_row = np.zeros((seq_len, num_features), dtype=np.float32)
        except Exception:
            X_row = np.zeros((seq_len, num_features), dtype=np.float32)
            
        X_tens = torch.tensor(X_row, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ts_score = float(model(X_tens).numpy().flatten()[0])
            
        scores_list.append({
            "candidate_id": cand_id,
            "tabular_score": 0.0,
            "time_series_score": ts_score,
            "news_score": 0.0,
            "final_score": ts_score,
            "score_source": "mac_api",
            "model_id": model_id,
            "metadata": {
                "feature_hash": cand.get("feature_hash"),
                "sequence_hash": cand.get("sequence_hash"),
                "paper_only": True,
                "broker_routed": False,
                "live_eligible": False,
                "scoring_formula": "1.0 * time_series_score",
                "weights": {
                    "tabular_score": 0.0,
                    "news_score": 0.0,
                    "time_series_score": 1.0
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
