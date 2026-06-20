# MacBook Model Lab & ONNX Export Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a secure, standalone FastAPI-based MacBook Model Lab service that trains tabular and sequence models, runs FinBERT news annotation, manages caching using SQLite, and packages/verifies ONNX exports.

**Architecture:** A FastAPI service backed by a SQLite database (`news_cache.db`) for news sentiment caching and background job tracking. A Python `ThreadPoolExecutor` processes model training tasks asynchronously.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, SQLite, PyTorch, XGBoost, CatBoost, ONNX, ONNX Runtime, Transformers, PyTest.

---

### Task 1: Environment and Dependency Setup

**Files:**
- Create: `pyproject.toml`
- Create: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test for configuration**
  
  Create `tests/test_config.py`:
  ```python
  import os
  import pytest
  from app.config import get_settings

  def test_settings_load():
      os.environ["MACBOOK_API_KEY"] = "test-token"
      settings = get_settings()
      assert settings.macbook_api_key == "test-token"
      assert settings.database_path == "news_cache.db"
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `pytest tests/test_config.py`
  Expected: FAIL with `ModuleNotFoundError: No module named 'app'` or similar.

- [ ] **Step 3: Write minimal implementation**
  
  Create `pyproject.toml`:
  ```toml
  [project]
  name = "noble-turing"
  version = "0.1.0"
  description = "MacBook Model Lab and ONNX Export Service"
  dependencies = [
      "fastapi>=0.100.0",
      "uvicorn>=0.22.0",
      "pydantic-settings>=2.0.0",
      "pydantic>=2.0.0",
      "xgboost>=1.7.5",
      "catboost>=1.2.0",
      "onnx>=1.14.0",
      "onnxruntime>=1.15.0",
      "onnxmltools>=1.11.2",
      "skl2onnx>=1.14.1",
      "torch>=2.0.0",
      "transformers>=4.30.0",
      "pandas>=2.0.0",
      "numpy>=1.24.0",
      "python-multipart>=0.0.6",
      "scikit-learn>=1.2.0"
  ]

  [tool.pytest.ini_options]
  pythonpath = ["."]
  ```
  
  Create `app/config.py`:
  ```python
  import os
  from pydantic_settings import BaseSettings

  class Settings(BaseSettings):
      macbook_api_key: str = os.getenv("MACBOOK_API_KEY", "default-secret-token")
      database_path: str = "news_cache.db"
      models_dir: str = "models"
      data_dir: str = "data"

  def get_settings() -> Settings:
      return Settings()
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_config.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add pyproject.toml app/config.py tests/test_config.py
  git commit -m "feat: configure dependencies and settings module"
  ```

---

### Task 2: Database Initialization and Job Tracking

**Files:**
- Create: `app/database.py`
- Test: `tests/test_database.py`

- [ ] **Step 1: Write the failing database connection and schema initialization test**
  
  Create `tests/test_database.py`:
  ```python
  import os
  import sqlite3
  from app.database import get_db_connection, init_db

  def test_db_init_and_tables():
      db_path = "test_news_cache.db"
      if os.path.exists(db_path):
          os.remove(db_path)
      
      init_db(db_path)
      conn = get_db_connection(db_path)
      cursor = conn.cursor()
      
      cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
      tables = {row[0] for row in cursor.fetchall()}
      assert "news_annotations" in tables
      assert "jobs" in tables
      conn.close()
      
      if os.path.exists(db_path):
          os.remove(db_path)
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_database.py`
  Expected: FAIL with `ImportError` or `AttributeError` for `init_db`.

- [ ] **Step 3: Write minimal database implementation**
  
  Create `app/database.py`:
  ```python
  import sqlite3

  def get_db_connection(db_path: str) -> sqlite3.Connection:
      conn = sqlite3.connect(db_path)
      conn.row_factory = sqlite3.Row
      return conn

  def init_db(db_path: str):
      conn = get_db_connection(db_path)
      cursor = conn.cursor()
      
      # News sentiment annotation cache table
      cursor.execute("""
      CREATE TABLE IF NOT EXISTS news_annotations (
          dedupe_hash TEXT PRIMARY KEY,
          model_id TEXT,
          sentiment_label TEXT,
          sentiment_score REAL,
          positive_score REAL,
          negative_score REAL,
          neutral_score REAL,
          annotated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
      """)
      
      # Background training job tracking table
      cursor.execute("""
      CREATE TABLE IF NOT EXISTS jobs (
          model_id TEXT PRIMARY KEY,
          status TEXT,
          model_family TEXT,
          model_type TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          completed_at TIMESTAMP,
          metrics_json TEXT,
          error_message TEXT
      );
      """)
      conn.commit()
      conn.close()
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_database.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add app/database.py tests/test_database.py
  git commit -m "feat: add SQLite database and table definitions"
  ```

---

### Task 3: API Key Security Middleware

**Files:**
- Create: `app/auth.py`
- Test: `tests/test_auth.py`

- [ ] **Step 1: Write the authentication middleware test**
  
  Create `tests/test_auth.py`:
  ```python
  from fastapi import FastAPI, Depends
  from fastapi.testclient import TestClient
  import pytest
  from app.auth import verify_api_key
  from app.config import Settings

  app = FastAPI()

  @app.get("/secure")
  def secure_endpoint(api_key: str = Depends(verify_api_key)):
      return {"status": "authenticated"}

  client = TestClient(app)

  def test_auth_success(monkeypatch):
      monkeypatch.setenv("MACBOOK_API_KEY", "correct-key")
      response = client.get("/secure", headers={"X-API-Key": "correct-key"})
      assert response.status_code == 200
      assert response.json() == {"status": "authenticated"}

  def test_auth_failure(monkeypatch):
      monkeypatch.setenv("MACBOOK_API_KEY", "correct-key")
      response = client.get("/secure", headers={"X-API-Key": "wrong-key"})
      assert response.status_code == 401
      assert "Invalid API Key" in response.json()["detail"]
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_auth.py`
  Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal auth verification logic**
  
  Create `app/auth.py`:
  ```python
  from fastapi import HTTPException, Security, status
  from fastapi.security.api_key import APIKeyHeader
  from app.config import get_settings

  API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

  def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
      settings = get_settings()
      if api_key != settings.macbook_api_key:
          raise HTTPException(
              status_code=status.HTTP_401_UNAUTHORIZED,
              detail="Invalid API Key."
          )
      return api_key
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_auth.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add app/auth.py tests/test_auth.py
  git commit -m "feat: implement header API Key authentication middleware"
  ```

---

### Task 4: FinBERT Annotation with SQLite Cache

**Files:**
- Create: `app/models_lab/finbert.py`
- Test: `tests/test_finbert.py`

- [ ] **Step 1: Write news sentiment annotation and caching test**
  
  Create `tests/test_finbert.py`:
  ```python
  import os
  import sqlite3
  import pytest
  from app.database import init_db
  from app.models_lab.finbert import annotate_news_batch

  def test_finbert_mock_annotation():
      db_path = "test_news_cache_finbert.db"
      if os.path.exists(db_path):
          os.remove(db_path)
      init_db(db_path)
      
      news_items = [
          {
              "event_id": "e1",
              "dedupe_hash": "hash_1",
              "title": "Stock market surges",
              "snippet": "Strong earnings drive market higher.",
              "matched_symbols": ["SPY"],
              "observed_at": "2026-06-20T12:00:00",
              "source": "news"
          }
      ]
      
      # Mock the transformers pipeline in the function to test routing/caching
      results = annotate_news_batch(news_items, db_path, use_mock=True)
      assert len(results) == 1
      assert results[0]["dedupe_hash"] == "hash_1"
      assert results[0]["sentiment_label"] in ["positive", "negative", "neutral"]
      
      # Check if cached in SQLite
      conn = sqlite3.connect(db_path)
      cursor = conn.cursor()
      cursor.execute("SELECT sentiment_label FROM news_annotations WHERE dedupe_hash='hash_1'")
      row = cursor.fetchone()
      assert row is not None
      conn.close()
      
      if os.path.exists(db_path):
          os.remove(db_path)
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_finbert.py`
  Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write FinBERT annotation & caching implementation**
  
  Create `app/models_lab/finbert.py`:
  ```python
  import sqlite3
  from datetime import datetime
  from typing import List, Dict, Any
  import torch

  # Global cache for pipeline to prevent reload
  _pipeline = None

  def get_finbert_pipeline():
      global _pipeline
      if _pipeline is None:
          from transformers import pipeline
          device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else "mps")
          # Fallback if MPS device index is not directly accepted as integer by older pipeline versions
          if device == "mps":
              device = "mps"
          _pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
      return _pipeline

  def annotate_news_batch(news_items: List[Dict[str, Any]], db_path: str, use_mock: bool = False) -> List[Dict[str, Any]]:
      results = []
      to_compute = []
      to_compute_indices = []
      
      conn = sqlite3.connect(db_path)
      conn.row_factory = sqlite3.Row
      cursor = conn.cursor()
      
      # Step 1: Check cache
      for idx, item in enumerate(news_items):
          h = item["dedupe_hash"]
          cursor.execute(
              "SELECT sentiment_label, sentiment_score, positive_score, negative_score, neutral_score FROM news_annotations WHERE dedupe_hash = ?", 
              (h,)
          )
          row = cursor.fetchone()
          if row:
              results.append({
                  "dedupe_hash": h,
                  "model_id": "cached",
                  "sentiment_label": row["sentiment_label"],
                  "sentiment_score": row["sentiment_score"],
                  "positive_score": row["positive_score"],
                  "negative_score": row["negative_score"],
                  "neutral_score": row["neutral_score"],
                  "annotated_at": datetime.utcnow().isoformat()
              })
          else:
              # Placeholder to keep ordering
              results.append(None)
              to_compute.append(f"{item['title']}. {item['snippet']}")
              to_compute_indices.append(idx)
              
      # Step 2: Run inference on misses
      if to_compute:
          if use_mock:
              # Mock outputs for testing
              computed_results = [
                  {"label": "positive", "score": 0.95} for _ in to_compute
              ]
          else:
              pipe = get_finbert_pipeline()
              # FinBERT model outputs labels: positive, negative, neutral
              computed_results = pipe(to_compute)
              
          for idx_in_batch, out in enumerate(computed_results):
              orig_idx = to_compute_indices[idx_in_batch]
              item = news_items[orig_idx]
              h = item["dedupe_hash"]
              
              # Map scores
              label = out["label"]
              score = out["score"]
              
              pos, neg, neu = 0.0, 0.0, 0.0
              if label == "positive":
                  pos = score
                  neg = (1 - score) * 0.4
                  neu = (1 - score) * 0.6
              elif label == "negative":
                  neg = score
                  pos = (1 - score) * 0.3
                  neu = (1 - score) * 0.7
              else:
                  neu = score
                  pos = (1 - score) * 0.5
                  neg = (1 - score) * 0.5
                  
              res = {
                  "dedupe_hash": h,
                  "model_id": "ProsusAI/finbert",
                  "sentiment_label": label,
                  "sentiment_score": score,
                  "positive_score": pos,
                  "negative_score": neg,
                  "neutral_score": neu,
                  "annotated_at": datetime.utcnow().isoformat()
              }
              
              # Save to DB cache
              cursor.execute("""
                  INSERT OR REPLACE INTO news_annotations 
                  (dedupe_hash, model_id, sentiment_label, sentiment_score, positive_score, negative_score, neutral_score, annotated_at)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?)
              """, (h, res["model_id"], res["sentiment_label"], res["sentiment_score"], res["positive_score"], res["negative_score"], res["neutral_score"], res["annotated_at"]))
              
              results[orig_idx] = res
              
      conn.commit()
      conn.close()
      return results
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_finbert.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add app/models_lab/finbert.py tests/test_finbert.py
  git commit -m "feat: implement news annotation with SQLite cache"
  ```

---

### Task 5: Tabular Model Training and Evaluation

**Files:**
- Create: `app/models_lab/tabular.py`
- Test: `tests/test_tabular.py`

- [ ] **Step 1: Write tabular model training and validation test**
  
  Create `tests/test_tabular.py`:
  ```python
  import os
  import shutil
  import pandas as pd
  import numpy as np
  import pytest
  from app.models_lab.tabular import train_tabular_pipeline

  def test_tabular_training_flow():
      model_dir = "test_models"
      if os.path.exists(model_dir):
          shutil.rmtree(model_dir)
      os.makedirs(model_dir)
      
      # Generate synthetic dataset
      np.random.seed(42)
      df = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
      df["target"] = np.random.choice([0, 1], size=100)
      df.to_csv("test_dataset.csv", index=False)
      
      config = {
          "experiment_id": "exp1",
          "dataset_id": "ds1",
          "dataset_uri": "test_dataset.csv",
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
      
      if os.path.exists("test_dataset.csv"):
          os.remove("test_dataset.csv")
      if os.path.exists(model_dir):
          shutil.rmtree(model_dir)
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_tabular.py`
  Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write tabular training and evaluation logic**
  
  Create `app/models_lab/tabular.py`:
  ```python
  import os
  import pickle
  import json
  import pandas as pd
  import numpy as np
  from typing import Dict, Any, Tuple
  from sklearn.metrics import roc_auc_score, log_loss, f1_score
  import xgboost as xgb
  from catboost import CatBoostClassifier

  def train_tabular_pipeline(model_id: str, config: Dict[str, Any], models_dir: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
      dataset_uri = config["dataset_uri"]
      label_col = config["label_definition"]
      model_family = config.get("model_family", "xgboost").lower()
      
      # Load dataset
      df = pd.read_csv(dataset_uri)
      
      # Parse train/holdout splits
      train_split = config["train_split"]
      holdout_split = config["holdout_split"]
      
      if train_split["type"] == "indices":
          train_df = df.iloc[train_split["values"]]
      else:
          train_df = df.iloc[:int(len(df) * 0.8)]
          
      if holdout_split["type"] == "indices":
          holdout_df = df.iloc[holdout_split["values"]]
      else:
          holdout_df = df.iloc[int(len(df) * 0.8):]
          
      features = [col for col in df.columns if col != label_col]
      
      X_train, y_train = train_df[features], train_df[label_col]
      X_hold, y_hold = holdout_df[features], holdout_df[label_col]
      
      # Select and train model
      blockers = []
      if model_family == "xgboost":
          model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
          model.fit(X_train, y_train)
      elif model_family == "catboost":
          model = CatBoostClassifier(verbose=0)
          model.fit(X_train, y_train)
      else:
          # Fallback to XGBoost
          blockers.append(f"Model family '{model_family}' is unsupported. Falling back to xgboost.")
          model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
          model.fit(X_train, y_train)
          model_family = "xgboost"
          
      # Predict and evaluate
      p_train = model.predict_proba(X_train)[:, 1]
      p_hold = model.predict_proba(X_hold)[:, 1]
      
      metrics = {
          "training_metrics": {
              "roc_auc": float(roc_auc_score(y_train, p_train)),
              "log_loss": float(log_loss(y_train, p_train)),
              "f1": float(f1_score(y_train, p_train > 0.5))
          },
          "holdout_metrics": {
              "roc_auc": float(roc_auc_score(y_hold, p_hold)),
              "log_loss": float(log_loss(y_hold, p_hold)),
              "f1": float(f1_score(y_hold, p_hold > 0.5))
          },
          "onnx_export_status": "pending",
          "onnx_parity_status": "unchecked",
          "blockers": blockers
      }
      
      # Save artifacts
      save_dir = os.path.join(models_dir, model_id)
      os.makedirs(save_dir, exist_ok=True)
      
      # Save model pickle
      model_path = os.path.join(save_dir, "model.pkl")
      with open(model_path, "wb") as f:
          pickle.dump(model, f)
          
      # Save reports
      with open(os.path.join(save_dir, "training_report.json"), "w") as f:
          json.dump(metrics["training_metrics"], f, indent=2)
      with open(os.path.join(save_dir, "holdout_report.json"), "w") as f:
          json.dump(metrics["holdout_metrics"], f, indent=2)
          
      metadata = {
          "model_id": model_id,
          "experiment_id": config["experiment_id"],
          "model_family": model_family,
          "model_type": "tabular",
          "feature_schema_hash": config["feature_schema_hash"],
          "label_definition": label_col
      }
      with open(os.path.join(save_dir, "metadata.json"), "w") as f:
          json.dump(metadata, f, indent=2)
          
      file_paths = {
          "model": model_path,
          "metadata": os.path.join(save_dir, "metadata.json")
      }
      
      return metrics, file_paths
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_tabular.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add app/models_lab/tabular.py tests/test_tabular.py
  git commit -m "feat: implement tabular model training and metrics reporting"
  ```

---

### Task 6: Time-Series sequence Model Training

**Files:**
- Create: `app/models_lab/sequence.py`
- Test: `tests/test_sequence.py`

- [ ] **Step 1: Write time-series sequence model training test**
  
  Create `tests/test_sequence.py`:
  ```python
  import os
  import shutil
  import numpy as np
  import pandas as pd
  import pytest
  from app.models_lab.sequence import train_sequence_pipeline

  def test_sequence_training():
      model_dir = "test_models_seq"
      if os.path.exists(model_dir):
          shutil.rmtree(model_dir)
      os.makedirs(model_dir)
      
      # Generate synthetic 3D timeseries data as CSV (flattened for CSV loading)
      # 100 samples, 10 timestamps, 2 features = 20 columns
      cols = [f"t{t}_f{f}" for t in range(10) for f in range(2)]
      df = pd.DataFrame(np.random.rand(100, 20), columns=cols)
      df["target"] = np.random.choice([0, 1], size=100)
      df.to_csv("test_seq_dataset.csv", index=False)
      
      config = {
          "experiment_id": "exp_seq1",
          "dataset_id": "ds_seq1",
          "dataset_uri": "test_seq_dataset.csv",
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
      
      if os.path.exists("test_seq_dataset.csv"):
          os.remove("test_seq_dataset.csv")
      if os.path.exists(model_dir):
          shutil.rmtree(model_dir)
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_sequence.py`
  Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write PyTorch 1D CNN training and evaluation logic**
  
  Create `app/models_lab/sequence.py`:
  ```python
  import os
  import json
  import pandas as pd
  import numpy as np
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from typing import Dict, Any, Tuple
  from sklearn.metrics import roc_auc_score, log_loss, f1_score

  class Simple1DCNN(nn.Module):
      def __init__(self, sequence_length: int, num_features: int):
          super().__init__()
          self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=8, kernel_size=3, padding=1)
          self.pool = nn.MaxPool1d(2)
          self.fc = nn.Linear(8 * (sequence_length // 2), 1)
          
      def forward(self, x):
          # x shape: (batch_size, seq_len, features) -> transpose to (batch_size, features, seq_len)
          x = x.transpose(1, 2)
          x = torch.relu(self.conv1(x))
          x = self.pool(x)
          x = x.view(x.size(0), -1)
          x = torch.sigmoid(self.fc(x))
          return x

  def train_sequence_pipeline(model_id: str, config: Dict[str, Any], models_dir: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
      dataset_uri = config["dataset_uri"]
      label_col = config["label_definition"]
      seq_len = config.get("sequence_length", 10)
      num_features = config.get("num_features", 2)
      
      df = pd.read_csv(dataset_uri)
      
      # Parse splits
      train_split = config["train_split"]
      holdout_split = config["holdout_split"]
      
      if train_split["type"] == "indices":
          train_df = df.iloc[train_split["values"]]
      else:
          train_df = df.iloc[:int(len(df) * 0.8)]
          
      if holdout_split["type"] == "indices":
          holdout_df = df.iloc[holdout_split["values"]]
      else:
          holdout_df = df.iloc[int(len(df) * 0.8):]
          
      feature_cols = [col for col in df.columns if col != label_col]
      
      # Parse into 3D tensors: (batch_size, seq_len, features)
      def to_tensor_dataset(df_part):
          X_raw = df_part[feature_cols].values
          # Reshape to (N, seq_len, num_features)
          X_reshaped = X_raw.reshape(-1, seq_len, num_features)
          y_raw = df_part[label_col].values.astype(np.float32)
          return torch.tensor(X_reshaped, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)
          
      X_train, y_train = to_tensor_dataset(train_df)
      X_hold, y_hold = to_tensor_dataset(holdout_df)
      
      model = Simple1DCNN(seq_len, num_features)
      criterion = nn.BCELoss()
      optimizer = optim.Adam(model.parameters(), lr=0.01)
      
      # Train for 5 epochs
      model.train()
      for epoch in range(5):
          optimizer.zero_grad()
          outputs = model(X_train)
          loss = criterion(outputs, y_train)
          loss.backward()
          optimizer.step()
          
      # Evaluate
      model.eval()
      with torch.no_grad():
          p_train = model(X_train).numpy().flatten()
          p_hold = model(X_hold).numpy().flatten()
          
      y_train_np = y_train.numpy().flatten()
      y_hold_np = y_hold.numpy().flatten()
      
      metrics = {
          "training_metrics": {
              "roc_auc": float(roc_auc_score(y_train_np, p_train)),
              "log_loss": float(log_loss(y_train_np, p_train)),
              "f1": float(f1_score(y_train_np, p_train > 0.5))
          },
          "holdout_metrics": {
              "roc_auc": float(roc_auc_score(y_hold_np, p_hold)),
              "log_loss": float(log_loss(y_hold_np, p_hold)),
              "f1": float(f1_score(y_hold_np, p_hold > 0.5))
          },
          "onnx_export_status": "pending",
          "onnx_parity_status": "unchecked",
          "blockers": []
      }
      
      save_dir = os.path.join(models_dir, model_id)
      os.makedirs(save_dir, exist_ok=True)
      
      model_path = os.path.join(save_dir, "model.pt")
      torch.save(model.state_dict(), model_path)
      
      with open(os.path.join(save_dir, "training_report.json"), "w") as f:
          json.dump(metrics["training_metrics"], f, indent=2)
      with open(os.path.join(save_dir, "holdout_report.json"), "w") as f:
          json.dump(metrics["holdout_metrics"], f, indent=2)
          
      metadata = {
          "model_id": model_id,
          "experiment_id": config["experiment_id"],
          "model_family": "pytorch_cnn",
          "model_type": "sequence",
          "sequence_schema_hash": config["sequence_schema_hash"],
          "label_definition": label_col,
          "sequence_length": seq_len,
          "num_features": num_features
      }
      with open(os.path.join(save_dir, "metadata.json"), "w") as f:
          json.dump(metadata, f, indent=2)
          
      file_paths = {
          "model": model_path,
          "metadata": os.path.join(save_dir, "metadata.json")
      }
      
      return metrics, file_paths
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_sequence.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add app/models_lab/sequence.py tests/test_sequence.py
  git commit -m "feat: implement PyTorch 1D CNN sequence model training"
  ```

---

### Task 7: ONNX Export and Parity Verification

**Files:**
- Create: `app/models_lab/onnx_utils.py`
- Test: `tests/test_onnx.py`

- [ ] **Step 1: Write ONNX export and parity validation test**
  
  Create `tests/test_onnx.py`:
  ```python
  import os
  import shutil
  import numpy as np
  import pandas as pd
  import pytest
  from app.models_lab.tabular import train_tabular_pipeline
  from app.models_lab.onnx_utils import export_and_verify_onnx

  def test_onnx_export_parity():
      model_dir = "test_models_onnx"
      if os.path.exists(model_dir):
          shutil.rmtree(model_dir)
      os.makedirs(model_dir)
      
      np.random.seed(42)
      df = pd.DataFrame(np.random.rand(100, 3), columns=["f1", "f2", "f3"])
      df["target"] = np.random.choice([0, 1], size=100)
      df.to_csv("test_onnx_ds.csv", index=False)
      
      config = {
          "experiment_id": "exp_onnx",
          "dataset_id": "ds_onnx",
          "dataset_uri": "test_onnx_ds.csv",
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
      
      if os.path.exists("test_onnx_ds.csv"):
          os.remove("test_onnx_ds.csv")
      if os.path.exists(model_dir):
          shutil.rmtree(model_dir)
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_onnx.py`
  Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write ONNX export and parity checks implementation**
  
  Create `app/models_lab/onnx_utils.py`:
  ```python
  import os
  import json
  import pickle
  import numpy as np
  import pandas as pd
  import torch
  import onnxruntime as ort
  from typing import Tuple
  from app.models_lab.sequence import Simple1DCNN

  def export_and_verify_onnx(model_id: str, models_dir: str) -> Tuple[str, str]:
      save_dir = os.path.join(models_dir, model_id)
      
      # Read metadata
      with open(os.path.join(save_dir, "metadata.json"), "r") as f:
          metadata = json.load(f)
          
      model_type = metadata["model_type"]
      onnx_path = os.path.join(save_dir, "model.onnx")
      
      try:
          if model_type == "tabular":
              # Load python model
              model_pkl_path = os.path.join(save_dir, "model.pkl")
              with open(model_pkl_path, "rb") as f:
                  model = pickle.load(f)
              
              # XGBoost ONNX Export
              from reraise_check import convert_xgboost if False else None
              from skl2onnx import update_registered_converter
              from skl2onnx.common.data_types import FloatTensorType
              from onnxmltools import convert_xgboost
              
              n_features = len(model.feature_names_in_)
              initial_type = [('input', FloatTensorType([None, n_features]))]
              onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=13)
              
              with open(onnx_path, "wb") as f:
                  f.write(onnx_model.SerializeToString())
                  
              # Parity Check Input
              dummy_input = np.random.rand(100, n_features).astype(np.float32)
              py_pred = model.predict_proba(dummy_input)[:, 1]
              
              # Run ONNX prediction
              ort_sess = ort.InferenceSession(onnx_path)
              onnx_outputs = ort_sess.run(None, {'input': dummy_input})
              # onnxmltools returns predictions and list of probabilities maps
              # probabilities map lists dictionary values of probability per class
              if len(onnx_outputs) > 1 and isinstance(onnx_outputs[1], list):
                  onnx_pred = np.array([d[1] for d in onnx_outputs[1]])
              else:
                  # depending on skl2onnx version it returns tensor or dict
                  probs_output = onnx_outputs[1]
                  if isinstance(probs_output, np.ndarray):
                      onnx_pred = probs_output[:, 1]
                  else:
                      onnx_pred = np.array([d[1] for d in probs_output])
                      
          elif model_type == "sequence":
              seq_len = metadata["sequence_length"]
              n_features = metadata["num_features"]
              
              # Load PyTorch model
              model = Simple1DCNN(seq_len, n_features)
              model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
              model.eval()
              
              dummy_input_torch = torch.randn(100, seq_len, n_features)
              torch.onnx.export(
                  model,
                  dummy_input_torch[0:1], # single batch dummy input
                  onnx_path,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  opset_version=13
              )
              
              # Parity Check Input
              dummy_input = dummy_input_torch.numpy().astype(np.float32)
              with torch.no_grad():
                  py_pred = model(dummy_input_torch).numpy().flatten()
                  
              # Run ONNX prediction
              ort_sess = ort.InferenceSession(onnx_path)
              onnx_pred = ort_sess.run(None, {'input': dummy_input})[0].flatten()
              
          else:
              return "unsupported", "unchecked"
              
          # Parity math checks
          diff = np.abs(py_pred - onnx_pred)
          max_diff = float(np.max(diff))
          parity_status = "success" if max_diff < 1e-5 else "failed"
          
          # Write report
          report = {
              "max_absolute_difference": max_diff,
              "parity_passed": max_diff < 1e-5,
              "onnx_output_sample": [float(x) for x in onnx_pred[:5]],
              "py_output_sample": [float(x) for x in py_pred[:5]]
          }
          with open(os.path.join(save_dir, "onnx_parity_report.json"), "w") as f:
              json.dump(report, f, indent=2)
              
          return "success", parity_status
          
      except Exception as e:
          # Log failure to report
          err_msg = str(e)
          report = {
              "error": err_msg,
              "parity_passed": False
          }
          with open(os.path.join(save_dir, "onnx_parity_report.json"), "w") as f:
              json.dump(report, f, indent=2)
          return "failed", "unchecked"
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_onnx.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add app/models_lab/onnx_utils.py tests/test_onnx.py
  git commit -m "feat: implement ONNX export and parity check logic"
  ```

---

### Task 8: FastAPI API Gateway and Background Workers

**Files:**
- Create: `app/main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write integration tests for API routes**
  
  Create `tests/test_main.py`:
  ```python
  import os
  import json
  import sqlite3
  from fastapi.testclient import TestClient
  import pytest
  from app.main import app
  from app.config import get_settings

  client = TestClient(app)

  def test_news_annotation_endpoint(monkeypatch):
      monkeypatch.setenv("MACBOOK_API_KEY", "my-secret")
      news_data = [
          {
              "event_id": "123",
              "dedupe_hash": "d123",
              "title": "Stock market surges",
              "snippet": "Strong earnings drive market higher.",
              "matched_symbols": ["SPY"],
              "observed_at": "2026-06-20T12:00:00",
              "source": "news"
          }
      ]
      response = client.post(
          "/api/v1/annotate_news", 
          headers={"X-API-Key": "my-secret"},
          json=news_data
      )
      assert response.status_code == 200
      assert len(response.json()) == 1
      assert response.json()[0]["dedupe_hash"] == "d123"

  def test_unauthenticated_request():
      response = client.post("/api/v1/annotate_news", json=[])
      assert response.status_code == 401
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `uv run pytest tests/test_main.py`
  Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write main FastAPI application and routers**
  
  Create `app/main.py`:
  ```python
  import os
  import uuid
  import json
  import sqlite3
  from typing import List, Dict, Any
  from concurrent.futures import ThreadPoolExecutor
  from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form
  from fastapi.responses import FileResponse
  from app.auth import verify_api_key
  from app.config import get_settings
  from app.database import init_db, get_db_connection
  from app.models_lab.finbert import annotate_news_batch
  from app.models_lab.tabular import train_tabular_pipeline
  from app.models_lab.sequence import train_sequence_pipeline
  from app.models_lab.onnx_utils import export_and_verify_onnx

  app = FastAPI(title="MacBook Model Lab Service")
  settings = get_settings()

  # Ensure databases & folders exist
  init_db(settings.database_path)
  os.makedirs(settings.models_dir, exist_ok=True)
  os.makedirs(settings.data_dir, exist_ok=True)

  # Thread pool for CPU/GPU heavy model training background jobs
  training_pool = ThreadPoolExecutor(max_workers=2)

  def run_async_tabular_training(model_id: str, config: Dict[str, Any], db_path: str, models_dir: str):
      conn = get_db_connection(db_path)
      cursor = conn.cursor()
      try:
          # Update status to training
          cursor.execute("UPDATE jobs SET status = 'training' WHERE model_id = ?", (model_id,))
          conn.commit()
          
          # Train model
          metrics, file_paths = train_tabular_pipeline(model_id, config, models_dir)
          
          # ONNX conversion
          if config.get("requested_export_format") == "onnx":
              onnx_status, parity_status = export_and_verify_onnx(model_id, models_dir)
              metrics["onnx_export_status"] = onnx_status
              metrics["onnx_parity_status"] = parity_status
              
          cursor.execute("""
              UPDATE jobs 
              SET status = 'completed', completed_at = CURRENT_TIMESTAMP, metrics_json = ? 
              WHERE model_id = ?
          """, (json.dumps(metrics), model_id))
      except Exception as e:
          cursor.execute("""
              UPDATE jobs 
              SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error_message = ? 
              WHERE model_id = ?
          """, (str(e), model_id))
      finally:
          conn.commit()
          conn.close()

  def run_async_sequence_training(model_id: str, config: Dict[str, Any], db_path: str, models_dir: str):
      conn = get_db_connection(db_path)
      cursor = conn.cursor()
      try:
          # Update status to training
          cursor.execute("UPDATE jobs SET status = 'training' WHERE model_id = ?", (model_id,))
          conn.commit()
          
          # Train model
          metrics, file_paths = train_sequence_pipeline(model_id, config, models_dir)
          
          # ONNX conversion
          if config.get("requested_export_format") == "onnx":
              onnx_status, parity_status = export_and_verify_onnx(model_id, models_dir)
              metrics["onnx_export_status"] = onnx_status
              metrics["onnx_parity_status"] = parity_status
              
          cursor.execute("""
              UPDATE jobs 
              SET status = 'completed', completed_at = CURRENT_TIMESTAMP, metrics_json = ? 
              WHERE model_id = ?
          """, (json.dumps(metrics), model_id))
      except Exception as e:
          cursor.execute("""
              UPDATE jobs 
              SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error_message = ? 
              WHERE model_id = ?
          """, (str(e), model_id))
      finally:
          conn.commit()
          conn.close()

  @app.post("/api/v1/annotate_news", dependencies=[Depends(verify_api_key)])
  def annotate_news(news_items: List[Dict[str, Any]]):
      return annotate_news_batch(news_items, settings.database_path)

  @app.post("/api/v1/train_tabular_model", dependencies=[Depends(verify_api_key)])
  def train_tabular_model(config_str: str = Form(...), file: UploadFile = File(None)):
      config = json.loads(config_str)
      model_id = str(uuid.uuid4())
      
      # Handle dataset upload
      if file:
          save_path = os.path.join(settings.data_dir, f"{model_id}_{file.filename}")
          with open(save_path, "wb") as buffer:
              buffer.write(file.file.read())
          config["dataset_uri"] = save_path
          
      # Log job into SQLite
      conn = get_db_connection(settings.database_path)
      cursor = conn.cursor()
      cursor.execute("""
          INSERT INTO jobs (model_id, status, model_family, model_type)
          VALUES (?, 'pending', ?, 'tabular')
      """, (model_id, config.get("model_family", "xgboost")))
      conn.commit()
      conn.close()
      
      # Dispatch to background thread
      training_pool.submit(
          run_async_tabular_training, 
          model_id, config, settings.database_path, settings.models_dir
      )
      
      return {"model_id": model_id, "status": "pending", "message": "Job submitted."}

  @app.post("/api/v1/train_time_series_model", dependencies=[Depends(verify_api_key)])
  def train_time_series_model(config_str: str = Form(...), file: UploadFile = File(None)):
      config = json.loads(config_str)
      model_id = str(uuid.uuid4())
      
      # Handle dataset upload
      if file:
          save_path = os.path.join(settings.data_dir, f"{model_id}_{file.filename}")
          with open(save_path, "wb") as buffer:
              buffer.write(file.file.read())
          config["dataset_uri"] = save_path
          
      # Log job into SQLite
      conn = get_db_connection(settings.database_path)
      cursor = conn.cursor()
      cursor.execute("""
          INSERT INTO jobs (model_id, status, model_family, model_type)
          VALUES (?, 'pending', ?, 'sequence')
      """, (model_id, config.get("model_family", "pytorch_cnn")))
      conn.commit()
      conn.close()
      
      # Dispatch to background thread
      training_pool.submit(
          run_async_sequence_training, 
          model_id, config, settings.database_path, settings.models_dir
      )
      
      return {"model_id": model_id, "status": "pending", "message": "Job submitted."}

  @app.get("/api/v1/model_status/{model_id}", dependencies=[Depends(verify_api_key)])
  def get_model_status(model_id: str):
      conn = get_db_connection(settings.database_path)
      cursor = conn.cursor()
      cursor.execute("SELECT * FROM jobs WHERE model_id = ?", (model_id,))
      row = cursor.fetchone()
      conn.close()
      
      if not row:
          raise HTTPException(status_code=404, detail="Model not found")
          
      metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
      
      return {
          "model_id": row["model_id"],
          "status": row["status"],
          "model_family": row["model_family"],
          "model_type": row["model_type"],
          "created_at": row["created_at"],
          "completed_at": row["completed_at"],
          "error_message": row["error_message"],
          "metrics": metrics
      }

  @app.get("/api/v1/model_artifact/{model_id}", dependencies=[Depends(verify_api_key)])
  def get_model_artifacts(model_id: str):
      save_dir = os.path.join(settings.models_dir, model_id)
      if not os.path.exists(save_dir):
          raise HTTPException(status_code=404, detail="Model artifacts not found")
      files = os.listdir(save_dir)
      return {"model_id": model_id, "artifacts": files}

  @app.get("/api/v1/model_artifact/{model_id}/{filename}", dependencies=[Depends(verify_api_key)])
  def download_model_artifact(model_id: str, filename: str):
      file_path = os.path.join(settings.models_dir, model_id, filename)
      if not os.path.exists(file_path):
          raise HTTPException(status_code=404, detail="Artifact file not found")
      return FileResponse(file_path)
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `uv run pytest tests/test_main.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add app/main.py tests/test_main.py
  git commit -m "feat: implement FastAPI router integrations and background workers"
  ```
