import os
import shutil
import numpy as np
import pandas as pd
import pytest
from app.models_lab.tabular import train_tabular_pipeline
from app.models_lab.onnx_utils import export_and_verify_onnx


def test_onnx_export_parity(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(42)
    df = pd.DataFrame(
        np.random.rand(100, 3), columns=["feat_0", "feat_1", "feat_2"]
    )
    df["target"] = np.random.choice([0, 1], size=100)
    dataset_path = str(tmp_path / "test_onnx_ds.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp_onnx",
        "dataset_id": "ds_onnx",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "hash_onnx",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(80))},
        "holdout_split": {"type": "indices", "values": list(range(80, 100))},
        "model_family": "xgboost",
    }

    metrics, file_paths = train_tabular_pipeline("m_onnx", config, model_dir)
    status, parity, error_msg = export_and_verify_onnx("m_onnx", model_dir)

    assert status == "success"
    assert parity == "success"
    assert error_msg is None
    assert os.path.exists(os.path.join(model_dir, "m_onnx", "model.onnx"))
    report_path = os.path.join(model_dir, "m_onnx", "onnx_parity_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        rep = json.load(f)
    assert "sample_count" in rep
    assert "max_abs_delta" in rep
    assert "mean_abs_delta" in rep
    assert "passed" in rep
    assert "tolerance" in rep


def test_onnx_sequence_export_parity(tmp_path):
    from app.models_lab.sequence import train_sequence_pipeline

    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    cols = [f"t{t}_f{f}" for t in range(10) for f in range(2)]
    df = pd.DataFrame(np.random.rand(100, 20), columns=cols)
    df["target"] = np.random.choice([0, 1], size=100)
    dataset_path = str(tmp_path / "test_seq_onnx_ds.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp_seq_onnx",
        "dataset_id": "ds_seq_onnx",
        "dataset_uri": dataset_path,
        "sequence_schema_hash": "seq_hash_onnx",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(80))},
        "holdout_split": {"type": "indices", "values": list(range(80, 100))},
        "model_family": "pytorch_cnn",
        "sequence_length": 10,
        "num_features": 2,
    }

    metrics, file_paths = train_sequence_pipeline(
        "m_seq_onnx", config, model_dir
    )
    status, parity, error_msg = export_and_verify_onnx("m_seq_onnx", model_dir)

    assert status == "success"
    assert parity == "success"
    assert error_msg is None
    assert os.path.exists(
        os.path.join(model_dir, "m_seq_onnx", "model.onnx")
    )
    assert os.path.exists(
        os.path.join(model_dir, "m_seq_onnx", "onnx_parity_report.json")
    )


def test_onnx_unknown_model_type(tmp_path):
    """Export should return 'unsupported' for unknown model types."""
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    # Create a metadata file with unknown model_type
    os.makedirs(os.path.join(model_dir, "m_unknown"), exist_ok=True)
    with open(
        os.path.join(model_dir, "m_unknown", "metadata.json"), "w"
    ) as f:
        json.dump({"model_type": "unknown_type"}, f)

    status, parity, error_msg = export_and_verify_onnx(
        "m_unknown", model_dir
    )

    assert status == "unsupported"
    assert parity == "unchecked"
    assert error_msg is not None


def test_onnx_missing_metadata(tmp_path):
    """Export should fail gracefully when metadata.json is missing."""
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    status, parity, error_msg = export_and_verify_onnx(
        "m_missing", model_dir
    )

    assert status == "failed"
    assert parity == "unchecked"
    assert error_msg is not None


import json

def test_onnx_parity_missing_validation_samples(tmp_path):
    from app.models_lab.onnx_utils import export_and_verify_onnx
    import json
    
    model_id = "m_missing_val_samples"
    save_dir = os.path.join(str(tmp_path), model_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # create dummy metadata with no validation_samples.npy
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "model_type": "tabular",
            "model_family": "xgboost",
            "feature_schema_hash": "feat_h",
            "feature_columns": ["f0"]
        }, f)
        
    # Create dummy model
    import pickle
    import xgboost as xgb
    import numpy as np
    m = xgb.XGBClassifier()
    m.fit(np.random.rand(10, 1), np.array([0,1,0,1,0,1,0,1,0,1]))
    with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
        pickle.dump(m, f)
        
    export_status, parity_status, error_msg = export_and_verify_onnx(model_id, str(tmp_path))
    assert export_status == "success"
    assert parity_status == "failed"
    
    with open(os.path.join(save_dir, "onnx_parity_report.json"), "r") as f:
        report = json.load(f)
    assert report["passed"] is False
    assert report["real_validation_samples"] is False

def test_verify_existing_onnx_missing_file(tmp_path):
    from app.models_lab.onnx_utils import verify_existing_onnx
    import json
    
    model_id = "m_no_onnx"
    save_dir = os.path.join(str(tmp_path), model_id)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "model_type": "tabular",
            "model_family": "xgboost",
            "feature_schema_hash": "feat_h",
            "feature_columns": ["f0"]
        }, f)
        
    status, parity, error_msg = verify_existing_onnx(model_id, str(tmp_path))
    assert status == "failed"
    assert parity == "unchecked"
    assert error_msg == "ONNX model file not found"
