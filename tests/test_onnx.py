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
    df = pd.DataFrame(np.random.rand(100, 3), columns=["f1", "f2", "f3"])
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
        "model_family": "xgboost"
    }
    
    metrics, file_paths = train_tabular_pipeline("m_onnx", config, model_dir)
    status, parity = export_and_verify_onnx("m_onnx", model_dir)
    
    assert status == "success"
    assert parity == "success"
    assert os.path.exists(os.path.join(model_dir, "m_onnx", "model.onnx"))
    assert os.path.exists(os.path.join(model_dir, "m_onnx", "onnx_parity_report.json"))

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
        "num_features": 2
    }
    
    metrics, file_paths = train_sequence_pipeline("m_seq_onnx", config, model_dir)
    status, parity = export_and_verify_onnx("m_seq_onnx", model_dir)
    
    assert status == "success"
    assert parity == "success"
    assert os.path.exists(os.path.join(model_dir, "m_seq_onnx", "model.onnx"))
    assert os.path.exists(os.path.join(model_dir, "m_seq_onnx", "onnx_parity_report.json"))

