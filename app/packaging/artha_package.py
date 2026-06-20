import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

def export_artha_package(
    model_id: str,
    package_type: str, # "onnx" or "mac_api"
    models_dir: str
) -> List[str]:
    save_dir = os.path.join(models_dir, model_id)
    metadata_path = os.path.join(save_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata {metadata_path} not found")
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    # Determine if parity passed
    parity_failed = False
    parity_report_path = os.path.join(save_dir, "onnx_parity_report.json")
    if os.path.exists(parity_report_path):
        with open(parity_report_path, "r") as f:
            parity = json.load(f)
            if not parity.get("passed", False) and not parity.get("parity_passed", False):
                parity_failed = True
                
    exported_files = []
    
    # 1. approval.json
    approval_path = os.path.join(save_dir, "approval.json")
    approval_data = {
        "status": "blocked" if parity_failed else "not_approved",
        "paper_only": True,
        "broker_routed": False,
        "live_eligible": False
    }
    if parity_failed:
        approval_data["errors"] = ["onnx_parity_failed"]
    with open(approval_path, "w") as f:
        json.dump(approval_data, f, indent=2)
    exported_files.append("approval.json")
    
    # 2. model_package.json
    pkg_path = os.path.join(save_dir, "model_package.json")
    pkg_data = {
        "model_package_id": f"pkg_{model_id}",
        "model_id": model_id,
        "package_type": package_type,
        "status": "blocked" if parity_failed else "completed",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    with open(pkg_path, "w") as f:
        json.dump(pkg_data, f, indent=2)
    exported_files.append("model_package.json")
    
    # 3. feature_schema.json
    feat_path = os.path.join(save_dir, "feature_schema.json")
    feat_cols = metadata.get("feature_columns", [])
    feat_data = {
        "feature_schema_hash": metadata.get("feature_schema_hash"),
        "features": [{"name": fn, "type": "float"} for fn in feat_cols]
    }
    with open(feat_path, "w") as f:
        json.dump(feat_data, f, indent=2)
    exported_files.append("feature_schema.json")
    
    # 4. validation_report.json
    val_path = os.path.join(save_dir, "validation_report.json")
    metrics = {}
    for rep in ("training_report.json", "holdout_report.json"):
        p = os.path.join(save_dir, rep)
        if os.path.exists(p):
            with open(p, "r") as f:
                metrics[rep.replace(".json", "")] = json.load(f)
    val_data = {
        "experiment_id": metadata.get("experiment_id"),
        "dataset_id": metadata.get("dataset_id"),
        "metrics": metrics,
        "splits": {
            "train": metadata.get("train_split"),
            "holdout": metadata.get("holdout_split")
        },
        "assumptions": {
            "target_stop_assumptions": metadata.get("target_stop_assumptions"),
            "cost_assumptions": metadata.get("cost_assumptions")
        },
        "broker_limits": {
            "limits": metadata.get("broker_limits"),
            "policy": metadata.get("broker_limit_policy")
        }
    }
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)
    exported_files.append("validation_report.json")
    
    # 5. thresholds.json
    thresh_path = os.path.join(save_dir, "thresholds.json")
    if not os.path.exists(thresh_path):
        with open(thresh_path, "w") as f:
            json.dump({"threshold": 0.5}, f, indent=2)
    exported_files.append("thresholds.json")
    
    if package_type == "onnx":
        # 6. sequence_schema.json
        seq_path = os.path.join(save_dir, "sequence_schema.json")
        seq_data = {
            "sequence_schema_hash": metadata.get("sequence_schema_hash"),
            "sequence_length": metadata.get("sequence_length"),
            "num_features": metadata.get("num_features"),
            "features": [{"name": fn, "type": "float"} for fn in feat_cols]
        }
        with open(seq_path, "w") as f:
            json.dump(seq_data, f, indent=2)
        exported_files.append("sequence_schema.json")
        
        # 7. preprocessing.json
        pre_path = os.path.join(save_dir, "preprocessing.json")
        pre_data = {
            "impute_missing_with_zero": True
        }
        with open(pre_path, "w") as f:
            json.dump(pre_data, f, indent=2)
        exported_files.append("preprocessing.json")
        
        # 8. calibration.json
        cal_path = os.path.join(save_dir, "calibration.json")
        if not os.path.exists(cal_path):
            with open(cal_path, "w") as f:
                json.dump({}, f, indent=2)
        exported_files.append("calibration.json")
        
        # Ensure model.onnx exists (or just register the export requirement)
        exported_files.append("model.onnx")
        exported_files.append("onnx_parity_report.json")
        
    else: # mac_api
        # 6. mac_api.json
        mac_path = os.path.join(save_dir, "mac_api.json")
        mac_data = {
            "scoring_endpoints": {
                "daily_mover": "/api/v1/score_daily_mover_candidates",
                "time_series": "/api/v1/score_time_series_candidates"
            },
            "model_id": model_id
        }
        with open(mac_path, "w") as f:
            json.dump(mac_data, f, indent=2)
        exported_files.append("mac_api.json")
        
    return exported_files
