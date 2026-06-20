import os
import shutil
import pandas as pd
import numpy as np
import pytest
from app.models_lab.tabular import train_tabular_pipeline

def test_tabular_training_flow(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate synthetic dataset
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    df["target"] = np.random.choice([0, 1], size=100)
    dataset_path = str(tmp_path / "test_dataset.csv")
    df.to_csv(dataset_path, index=False)
    
    config = {
        "experiment_id": "exp1",
        "dataset_id": "ds1",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "hash123",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(80))},
        "holdout_split": {"type": "indices", "values": list(range(80, 100))},
        "model_family": "xgboost"
    }
    
    metrics, file_paths = train_tabular_pipeline("m1", config, model_dir)
    
    assert "training_metrics" in metrics
    assert "holdout_metrics" in metrics
    assert os.path.exists(os.path.join(model_dir, "m1", "model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "m1", "metadata.json"))
