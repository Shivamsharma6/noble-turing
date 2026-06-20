import os
import shutil
import numpy as np
import pandas as pd
import pytest
from app.models_lab.sequence import (
    train_sequence_pipeline,
    Simple1DCNN,
    MiniRocketFeatureExtractor,
    InceptionTime,
    TCN,
    ResNetTS,
)


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
        "num_features": 2,
    }

    metrics, file_paths = train_sequence_pipeline("m_seq1", config, model_dir)

    assert "training_metrics" in metrics
    assert "holdout_metrics" in metrics
    assert "threshold" in metrics
    assert "calibration_metrics" in metrics
    assert os.path.exists(os.path.join(model_dir, "m_seq1", "model.pt"))
    assert os.path.exists(os.path.join(model_dir, "m_seq1", "metadata.json"))
    assert os.path.exists(os.path.join(model_dir, "m_seq1", "thresholds.json"))
    assert os.path.exists(
        os.path.join(model_dir, "m_seq1", "calibration.json")
    )
    assert os.path.exists(os.path.join(model_dir, "m_seq1", "validation_samples.npy"))


def test_sequence_unknown_split_rejected(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    cols = [f"t{t}_f{f}" for t in range(10) for f in range(2)]
    df = pd.DataFrame(np.random.rand(50, 20), columns=cols)
    df["target"] = np.random.choice([0, 1], size=50)
    dataset_path = str(tmp_path / "test_seq_split.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp_seq2",
        "dataset_id": "ds_seq2",
        "dataset_uri": dataset_path,
        "sequence_schema_hash": "seq_hash2",
        "label_definition": "target",
        "train_split": {"type": "random", "values": 0.7},
        "holdout_split": {"type": "indices", "values": list(range(35, 50))},
        "model_family": "pytorch_cnn",
        "sequence_length": 10,
        "num_features": 2,
    }

    with pytest.raises(ValueError, match="Unknown train_split type"):
        train_sequence_pipeline("m_seq2", config, model_dir)


def test_sequence_calibration_metrics(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    cols = [f"t{t}_f{f}" for t in range(10) for f in range(2)]
    df = pd.DataFrame(np.random.rand(100, 20), columns=cols)
    df["target"] = np.random.choice([0, 1], size=100)
    dataset_path = str(tmp_path / "test_seq_cal.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp_seq3",
        "dataset_id": "ds_seq3",
        "dataset_uri": dataset_path,
        "sequence_schema_hash": "seq_hash3",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(80))},
        "holdout_split": {"type": "indices", "values": list(range(80, 100))},
        "model_family": "pytorch_cnn",
        "sequence_length": 10,
        "num_features": 2,
    }

    metrics, _ = train_sequence_pipeline("m_seq3", config, model_dir)

    cal = metrics["calibration_metrics"]
    assert "expected_calibration_error" in cal
    assert "brier_score" in cal
    assert "calibration_curve" in cal


def test_sequence_model_architectures(tmp_path):
    """Verify all supported model architectures can be instantiated."""
    seq_len = 20
    num_features = 3

    # Simple1DCNN
    m1 = Simple1DCNN(seq_len, num_features)
    x = torch.randn(10, seq_len, num_features)
    out = m1(x)
    assert out.shape == (10, 1)

    # MiniRocketFeatureExtractor
    m2 = MiniRocketFeatureExtractor(seq_len, num_features)
    out2 = m2(x)
    assert out2.shape[0] == 10

    # InceptionTime
    m3 = InceptionTime(seq_len, num_features)
    out3 = m3(x)
    assert out3.shape == (10, 1)

    # TCN
    m4 = TCN(seq_len, num_features)
    out4 = m4(x)
    assert out4.shape == (10, 1)

    # ResNetTS
    m5 = ResNetTS(seq_len, num_features)
    out5 = m5(x)
    assert out5.shape == (10, 1)


import torch
