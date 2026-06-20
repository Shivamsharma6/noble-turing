import os
import json
import pytest
from app.packaging.artha_package import export_artha_package

def test_export_package_missing_model():
    with pytest.raises(FileNotFoundError):
        export_artha_package("nonexistent", "mac_api", "models_dir")

def test_export_package_mac_api_success(tmp_path):
    model_dir = str(tmp_path / "models")
    save_dir = os.path.join(model_dir, "m_mac")
    os.makedirs(save_dir, exist_ok=True)
    
    metadata = {
        "model_id": "m_mac",
        "experiment_id": "exp1",
        "dataset_id": "ds1",
        "model_family": "xgboost",
        "feature_schema_hash": "schema123",
        "feature_columns": ["f0", "f1"]
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    exported = export_artha_package("m_mac", "mac_api", model_dir)
    
    assert "approval.json" in exported
    assert "mac_api.json" in exported
    assert "feature_schema.json" in exported
    
    with open(os.path.join(save_dir, "approval.json"), "r") as f:
        appr = json.load(f)
    assert appr["status"] == "not_approved"
    assert appr["paper_only"] is True

def test_export_package_onnx_blocked_on_parity_failure(tmp_path):
    model_dir = str(tmp_path / "models")
    save_dir = os.path.join(model_dir, "m_onnx")
    os.makedirs(save_dir, exist_ok=True)
    
    metadata = {
        "model_id": "m_onnx",
        "experiment_id": "exp2",
        "dataset_id": "ds2",
        "model_family": "xgboost",
        "feature_schema_hash": "schema123",
        "feature_columns": ["f0", "f1"],
        "sequence_schema_hash": "seq123",
        "sequence_length": 5,
        "num_features": 2
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    # Mock a failed parity report
    with open(os.path.join(save_dir, "onnx_parity_report.json"), "w") as f:
        json.dump({"passed": False}, f)
        
    exported = export_artha_package("m_onnx", "onnx", model_dir)
    
    with open(os.path.join(save_dir, "approval.json"), "r") as f:
        appr = json.load(f)
    assert appr["status"] == "blocked"
    assert "onnx_parity_failed" in appr["errors"]

def test_package_metadata_compliance(tmp_path):
    model_dir = str(tmp_path / "models")
    model_id = "m_test_compliance"
    save_dir = os.path.join(model_dir, model_id)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "model_family": "xgboost",
            "model_type": "tabular",
            "feature_schema_hash": "feat_h",
            "feature_columns": ["f0"]
        }, f)
        
    with open(os.path.join(save_dir, "thresholds.json"), "w") as f:
        json.dump({"threshold": 0.55}, f)
        
    export_artha_package(model_id, "mac_api", model_dir)
    
    with open(os.path.join(save_dir, "model_package.json"), "r") as f:
        pkg = json.load(f)
    assert pkg["backend"] == "mac_api"
    
    with open(os.path.join(save_dir, "thresholds.json"), "r") as f:
        thresh = json.load(f)
    assert thresh["threshold"] == 0.55
    assert thresh["tabular_score"] == 0.55
    assert thresh["time_series_score"] == 0.55
    assert thresh["final_score"] == 0.55
    assert "scoring_formula" in thresh
