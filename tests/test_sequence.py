import os
import shutil
import numpy as np
import pandas as pd
import pytest
from app.models_lab.sequence import train_sequence_pipeline

def test_sequence_training(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate synthetic 3D timeseries data as CSV (flattened for CSV loading)
    # 100 samples, 10 timestamps, 2 features = 20 columns
    cols = [f"t{t}_f{f}" for t in range(10) for f in range(2)]
    df = pd.DataFrame(np.random.rand(100, 20), columns=cols)
    df["target"] = np.random.choice([0, 1], size=100)
    dataset_path = str(tmp_path / "test_seq_dataset.csv")
    df.to_csv(dataset_path, index=False)
    
    config = {
        "experiment_id": "exp_seq1",
        "dataset_id": "ds_seq1",
        "dataset_uri": dataset_path,
        "sequence_schema_hash": "seq_hash123",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(80))},
        "holdout_split": {"type": "indices", "values": list(range(80, 100))},
        "model_family": "pytorch_cnn",
        "sequence_length": 10,
        "num_features": 2
    }
    
    metrics, file_paths = train_sequence_pipeline("m_seq1", config, model_dir)
    
    assert "training_metrics" in metrics
    assert "holdout_metrics" in metrics
    assert os.path.exists(os.path.join(model_dir, "m_seq1", "model.pt"))
    assert os.path.exists(os.path.join(model_dir, "m_seq1", "metadata.json"))
