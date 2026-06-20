import os
import json
import pytest
import numpy as np
import pandas as pd
from app.models_lab.sequence import train_sequence_pipeline
from app.scoring.time_series import score_time_series

def test_score_time_series_missing_model(tmp_path):
    res = score_time_series("nonexistent", "models_dir", {})
    assert res["status"] == "blocked"
    assert "sequence_model_missing" in res["errors"]

def test_score_time_series_success(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Train a sequence model
    cols = [f"t{t}_f{f}" for t in range(10) for f in range(2)]
    df = pd.DataFrame(np.random.rand(50, 20), columns=cols)
    df["target"] = np.random.choice([0, 1], size=50)
    dataset_path = str(tmp_path / "ds_seq.csv")
    df.to_csv(dataset_path, index=False)
    
    config = {
        "experiment_id": "exp_seq",
        "dataset_id": "ds_seq",
        "dataset_uri": dataset_path,
        "sequence_schema_hash": "seq_hash_123",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(40))},
        "holdout_split": {"type": "indices", "values": list(range(40, 50))},
        "model_family": "pytorch_cnn",
        "sequence_length": 10,
        "num_features": 2,
    }
    
    train_sequence_pipeline("m_seq", config, model_dir)
    
    # Score candidates
    payload = {
        "model_package_id": "pkg_456",
        "model_id": "m_seq",
        "sequence_schema_hash": "seq_hash_123",
        "candidates": [
            {
                "candidate_id": "c1",
                "snapshot_id": "s1",
                "dataset_id": "ds_seq",
                "symbol": "AAPL",
                "feature_hash": "cf_hash_1",
                "sequence_hash": "cs_hash_1",
                "sequence_features": [[0.1, 0.2] for _ in range(10)]
            },
            {
                "candidate_id": "c2",
                "snapshot_id": "s1",
                "dataset_id": "ds_seq",
                "symbol": "MSFT",
                "feature_hash": "cf_hash_2",
                "sequence_hash": "cs_hash_2",
                "sequence_features": [{"t0_f0": 0.5, "t0_f1": 0.3} for _ in range(10)]
            }
        ]
    }
    
    res = score_time_series("m_seq", model_dir, payload)
    
    assert res["status"] == "completed"
    assert len(res["scores"]) == 2
    
    c1_score = res["scores"][0]
    assert c1_score["candidate_id"] == "c1"
    assert c1_score["score_source"] == "mac_api"
    assert c1_score["metadata"]["paper_only"] is True
    assert c1_score["metadata"]["broker_routed"] is False
    assert c1_score["metadata"]["live_eligible"] is False
    assert c1_score["metadata"]["feature_hash"] == "cf_hash_1"
    assert c1_score["metadata"]["sequence_hash"] == "cs_hash_1"
    assert c1_score["metadata"]["scoring_formula"] == "1.0 * time_series_score"
    assert c1_score["metadata"]["weights"]["time_series_score"] == 1.0
    assert 0.0 <= c1_score["time_series_score"] <= 1.0

def test_score_time_series_schema_mismatch(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Train a sequence model
    cols = [f"t{t}_f{f}" for t in range(10) for f in range(2)]
    df = pd.DataFrame(np.random.rand(50, 20), columns=cols)
    df["target"] = np.random.choice([0, 1], size=50)
    dataset_path = str(tmp_path / "ds_seq2.csv")
    df.to_csv(dataset_path, index=False)
    
    config = {
        "experiment_id": "exp_seq2",
        "dataset_id": "ds_seq2",
        "dataset_uri": dataset_path,
        "sequence_schema_hash": "seq_hash_123",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(40))},
        "holdout_split": {"type": "indices", "values": list(range(40, 50))},
        "model_family": "pytorch_cnn",
        "sequence_length": 10,
        "num_features": 2,
    }
    
    train_sequence_pipeline("m_seq2", config, model_dir)
    
    payload = {
        "model_id": "m_seq2",
        "sequence_schema_hash": "wrong_hash",
        "candidates": []
    }
    
    res = score_time_series("m_seq2", model_dir, payload)
    assert res["status"] == "blocked"
    assert "sequence_schema_mismatch" in res["errors"]
