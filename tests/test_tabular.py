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
    df = pd.DataFrame(
        np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)]
    )
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
        "model_family": "xgboost",
    }

    metrics, file_paths = train_tabular_pipeline("m1", config, model_dir)

    assert "training_metrics" in metrics
    assert "holdout_metrics" in metrics
    assert "threshold" in metrics
    assert "feature_importance" in metrics
    assert "calibration_metrics" in metrics
    assert "blockers" in metrics
    assert os.path.exists(os.path.join(model_dir, "m1", "model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "m1", "metadata.json"))
    assert os.path.exists(os.path.join(model_dir, "m1", "thresholds.json"))
    assert os.path.exists(
        os.path.join(model_dir, "m1", "feature_importance.json")
    )
    assert os.path.exists(os.path.join(model_dir, "m1", "calibration.json"))


def test_tabular_unknown_split_type_rejected(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(42)
    df = pd.DataFrame(
        np.random.rand(50, 3), columns=["a", "b", "c"]
    )
    df["target"] = np.random.choice([0, 1], size=50)
    dataset_path = str(tmp_path / "test_split.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp2",
        "dataset_id": "ds2",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "hash456",
        "label_definition": "target",
        "train_split": {"type": "random", "values": 0.7},  # unknown type
        "holdout_split": {"type": "indices", "values": list(range(35, 50))},
        "model_family": "xgboost",
    }

    with pytest.raises(ValueError, match="Unknown train_split type"):
        train_tabular_pipeline("m2", config, model_dir)


def test_tabular_unknown_holdout_split_rejected(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(42)
    df = pd.DataFrame(
        np.random.rand(50, 3), columns=["a", "b", "c"]
    )
    df["target"] = np.random.choice([0, 1], size=50)
    dataset_path = str(tmp_path / "test_split2.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp3",
        "dataset_id": "ds3",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "hash789",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(35))},
        "holdout_split": {"type": "random", "values": 0.3},  # unknown type
        "model_family": "xgboost",
    }

    with pytest.raises(ValueError, match="Unknown holdout_split type"):
        train_tabular_pipeline("m3", config, model_dir)


def test_tabular_calibration_metrics(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(42)
    df = pd.DataFrame(
        np.random.rand(100, 4), columns=[f"f{i}" for i in range(4)]
    )
    df["target"] = np.random.choice([0, 1], size=100)
    dataset_path = str(tmp_path / "test_cal.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp4",
        "dataset_id": "ds4",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "hash_cal",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(80))},
        "holdout_split": {"type": "indices", "values": list(range(80, 100))},
        "model_family": "xgboost",
    }

    metrics, _ = train_tabular_pipeline("m4", config, model_dir)

    cal = metrics["calibration_metrics"]
    assert "expected_calibration_error" in cal
    assert "brier_score" in cal
    assert "calibration_curve" in cal
    assert isinstance(cal["calibration_curve"], list)
    assert len(cal["calibration_curve"]) == 10  # 10 bins


def test_tabular_threshold_in_response(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(42)
    df = pd.DataFrame(
        np.random.rand(100, 3), columns=["a", "b", "c"]
    )
    df["target"] = np.random.choice([0, 1], size=100)
    dataset_path = str(tmp_path / "test_thresh.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp5",
        "dataset_id": "ds5",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "hash_thresh",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(80))},
        "holdout_split": {"type": "indices", "values": list(range(80, 100))},
        "model_family": "xgboost",
    }

    metrics, _ = train_tabular_pipeline("m5", config, model_dir)

    assert "threshold" in metrics
    assert 0.0 <= metrics["threshold"] <= 1.0


def test_tabular_lightgbm_fallback(tmp_path, monkeypatch):
    import json
    from app.models_lab import tabular
    monkeypatch.setattr(tabular, "LIGHTGBM_AVAILABLE", False)

    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.DataFrame(np.random.rand(100, 3), columns=["feat_0", "feat_1", "feat_2"])
    df["target"] = np.random.choice([0, 1], size=100)
    dataset_path = str(tmp_path / "test_lgb.csv")
    df.to_csv(dataset_path, index=False)

    config = {
        "experiment_id": "exp_lgb",
        "dataset_id": "ds_lgb",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "hash_lgb",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(80))},
        "holdout_split": {"type": "indices", "values": list(range(80, 100))},
        "model_family": "lightgbm",
    }

    metrics, file_paths = train_tabular_pipeline("m_lgb", config, model_dir)
    assert "LightGBM not installed. Falling back to xgboost." in metrics["blockers"]

    # Verify metadata saved "xgboost" as the family since it fell back
    with open(file_paths["metadata"], "r") as f:
        meta = json.load(f)
    assert meta["model_family"] == "xgboost"
