import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score, log_loss, f1_score
import xgboost as xgb

# CatBoost is optional / fallback if installed
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

def train_tabular_pipeline(model_id: str, config: Dict[str, Any], models_dir: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    dataset_uri = config["dataset_uri"]
    label_col = config["label_definition"]
    model_family = config.get("model_family", "xgboost").lower()
    
    # Load dataset
    df = pd.read_csv(dataset_uri)
    
    # Parse train/holdout splits
    train_split = config["train_split"]
    holdout_split = config["holdout_split"]
    
    if train_split["type"] == "indices":
        train_df = df.iloc[train_split["values"]]
    else:
        train_df = df.iloc[:int(len(df) * 0.8)]
        
    if holdout_split["type"] == "indices":
        holdout_df = df.iloc[holdout_split["values"]]
    else:
        holdout_df = df.iloc[int(len(df) * 0.8):]
        
    features = [col for col in df.columns if col != label_col]
    
    X_train, y_train = train_df[features], train_df[label_col]
    X_hold, y_hold = holdout_df[features], holdout_df[label_col]
    
    # Select and train model
    blockers = []
    if model_family == "xgboost":
        model = xgb.XGBClassifier(eval_metric="logloss")
        model.fit(X_train, y_train)
    elif model_family == "catboost":
        if CATBOOST_AVAILABLE:
            model = CatBoostClassifier(verbose=0)
            model.fit(X_train, y_train)
        else:
            blockers.append("CatBoost not installed. Falling back to xgboost.")
            model = xgb.XGBClassifier(eval_metric="logloss")
            model.fit(X_train, y_train)
            model_family = "xgboost"
    else:
        # Fallback to XGBoost
        blockers.append(f"Model family '{model_family}' is unsupported. Falling back to xgboost.")
        model = xgb.XGBClassifier(eval_metric="logloss")
        model.fit(X_train, y_train)
        model_family = "xgboost"
        
    # Predict and evaluate
    p_train = model.predict_proba(X_train)[:, 1]
    p_hold = model.predict_proba(X_hold)[:, 1]
    
    metrics = {
        "training_metrics": {
            "roc_auc": float(roc_auc_score(y_train, p_train)),
            "log_loss": float(log_loss(y_train, p_train)),
            "f1": float(f1_score(y_train, p_train > 0.5))
        },
        "holdout_metrics": {
            "roc_auc": float(roc_auc_score(y_hold, p_hold)),
            "log_loss": float(log_loss(y_hold, p_hold)),
            "f1": float(f1_score(y_hold, p_hold > 0.5))
        },
        "onnx_export_status": "pending",
        "onnx_parity_status": "unchecked",
        "blockers": blockers
    }
    
    # Save artifacts
    save_dir = os.path.join(models_dir, model_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model pickle
    model_path = os.path.join(save_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    # Save reports
    with open(os.path.join(save_dir, "training_report.json"), "w") as f:
        json.dump(metrics["training_metrics"], f, indent=2)
    with open(os.path.join(save_dir, "holdout_report.json"), "w") as f:
        json.dump(metrics["holdout_metrics"], f, indent=2)
        
    metadata = {
        "model_id": model_id,
        "experiment_id": config["experiment_id"],
        "model_family": model_family,
        "model_type": "tabular",
        "feature_schema_hash": config["feature_schema_hash"],
        "label_definition": label_col
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    file_paths = {
        "model": model_path,
        "metadata": os.path.join(save_dir, "metadata.json")
    }
    
    return metrics, file_paths
