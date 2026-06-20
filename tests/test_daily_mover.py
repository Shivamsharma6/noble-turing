import os
import json
import pytest
import numpy as np
import pandas as pd
from app.models_lab.tabular import train_tabular_pipeline
from app.scoring.daily_mover import score_daily_mover

def test_score_daily_mover_missing_model(tmp_path):
    res = score_daily_mover("nonexistent", "db.sqlite", "models_dir", {})
    assert res["status"] == "blocked"
    assert "model_missing" in res["errors"]

def test_score_daily_mover_success(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    db_path = str(tmp_path / "news.db")
    
    # Train a tabular model
    df = pd.DataFrame(np.random.rand(50, 4), columns=["f0", "f1", "f2", "target"])
    df["target"] = np.random.choice([0, 1], size=50)
    dataset_path = str(tmp_path / "ds.csv")
    df.to_csv(dataset_path, index=False)
    
    config = {
        "experiment_id": "exp_tab",
        "dataset_id": "ds_tab",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "schema_hash_123",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(40))},
        "holdout_split": {"type": "indices", "values": list(range(40, 50))},
        "model_family": "xgboost",
    }
    
    train_tabular_pipeline("m_tab", config, model_dir)
    
    # Score candidates
    payload = {
        "model_package_id": "pkg_123",
        "model_id": "m_tab",
        "feature_schema_hash": "schema_hash_123",
        "candidates": [
            {
                "candidate_id": "c1",
                "snapshot_id": "s1",
                "dataset_id": "ds_tab",
                "symbol": "AAPL",
                "feature_hash": "cf_hash_1",
                "sequence_hash": "cs_hash_1",
                "features": {"f0": 0.5, "f1": 0.2, "f2": 0.1}
            },
            {
                "candidate_id": "c2",
                "snapshot_id": "s1",
                "dataset_id": "ds_tab",
                "symbol": "MSFT",
                "feature_hash": "cf_hash_2",
                "sequence_hash": "cs_hash_2",
                "features": {"f0": 0.8, "f1": None, "f2": 0.9} # missing/None f1
            }
        ]
    }
    
    res = score_daily_mover("m_tab", db_path, model_dir, payload)
    
    assert res["status"] == "completed"
    assert len(res["scores"]) == 2
    
    # Check details of c1
    c1_score = res["scores"][0]
    assert c1_score["candidate_id"] == "c1"
    assert c1_score["score_source"] == "mac_api"
    assert c1_score["metadata"]["paper_only"] is True
    assert c1_score["metadata"]["broker_routed"] is False
    assert c1_score["metadata"]["live_eligible"] is False
    assert c1_score["metadata"]["feature_hash"] == "cf_hash_1"
    assert c1_score["metadata"]["sequence_hash"] == "cs_hash_1"
    assert c1_score["metadata"]["scoring_formula"] == "0.9 * tabular_score + 0.1 * news_score"
    assert c1_score["metadata"]["weights"]["tabular_score"] == 0.9
    assert c1_score["metadata"]["weights"]["news_score"] == 0.1
    assert 0.0 <= c1_score["tabular_score"] <= 1.0

def test_score_daily_mover_schema_mismatch(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Train a tabular model
    df = pd.DataFrame(np.random.rand(50, 4), columns=["f0", "f1", "f2", "target"])
    df["target"] = np.random.choice([0, 1], size=50)
    dataset_path = str(tmp_path / "ds2.csv")
    df.to_csv(dataset_path, index=False)
    
    config = {
        "experiment_id": "exp_tab2",
        "dataset_id": "ds_tab2",
        "dataset_uri": dataset_path,
        "feature_schema_hash": "schema_hash_123",
        "label_definition": "target",
        "train_split": {"type": "indices", "values": list(range(40))},
        "holdout_split": {"type": "indices", "values": list(range(40, 50))},
        "model_family": "xgboost",
    }
    
    train_tabular_pipeline("m_tab2", config, model_dir)
    
    payload = {
        "model_id": "m_tab2",
        "feature_schema_hash": "wrong_hash",
        "candidates": []
    }
    
    res = score_daily_mover("m_tab2", "db.sqlite", model_dir, payload)
    assert res["status"] == "blocked"
    assert "feature_schema_mismatch" in res["errors"]
