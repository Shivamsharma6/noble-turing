# Noble-Turing Artha Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `noble-turing` from a generic model lab into an Artha-compatible external model service that performs synchronous candidate scoring, standard packaging, FinBERT caching annotations, safety checks, and database auditing.

**Architecture:** Split route handling from scoring/packaging logic into dedicated sub-packages (`app/scoring/`, `app/packaging/`, `app/health.py`, `app/audit.py`). Run scoring synchronously on the calling thread, and intercept all actions with a redact-aware audit decorator.

**Tech Stack:** Python 3.11, FastAPI, SQLite, PyTorch, ONNX, ONNX Runtime, pandas, numpy.

---

## User Review Required

- We will require `X-API-Key` on all endpoints except the new `/health` endpoint.
- Scoring requests will execute synchronously, directly returning scores or a structured blocker JSON instead of returning a background job ID.

## Open Questions

None. The spec has been reviewed and approved.

---

## Proposed Changes

### Task 1: Audit Logging Table and Database Upgrade

**Files:**
- Modify: `app/database.py`
- Create: `app/audit.py`
- Test: `tests/test_audit.py`

- [ ] **Step 1: Write failing test for audit logging**
  Create `tests/test_audit.py` with this test:
  ```python
  import sqlite3
  from app.database import init_db
  from app.audit import log_audit_record, redact_secrets

  def test_audit_records_creation(tmp_path):
      db_path = str(tmp_path / "test_audit.db")
      init_db(db_path)
      
      log_audit_record(
          db_path=db_path,
          request_id="req-123",
          endpoint="/score_daily_mover_candidates",
          model_id="m-123",
          input_data={"api_key": "secret-key", "model_id": "m-123"},
          output_data={"status": "completed"},
          status="completed"
      )
      
      conn = sqlite3.connect(db_path)
      conn.row_factory = sqlite3.Row
      cursor = conn.cursor()
      cursor.execute("SELECT * FROM audit_records WHERE request_id = 'req-123'")
      row = cursor.fetchone()
      assert row is not None
      assert row["endpoint"] == "/score_daily_mover_candidates"
      assert "secret-key" not in row["input_hash"]
      conn.close()

  def test_redact_secrets():
      data = {"api_key": "my-secret", "headers": {"X-API-Key": "my-secret"}, "other": "ok"}
      redacted = redact_secrets(data)
      assert redacted["api_key"] == "[REDACTED]"
      assert redacted["headers"]["X-API-Key"] == "[REDACTED]"
      assert redacted["other"] == "ok"
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_audit.py -v`
  Expected: FAIL (ImportError or ModuleNotFoundError)
- [ ] **Step 3: Modify database.py and implement audit.py**
  Add the `audit_records` table to `app/database.py` in `init_db`:
  ```python
          cursor.execute("""
          CREATE TABLE IF NOT EXISTS audit_records (
              request_id TEXT PRIMARY KEY,
              endpoint TEXT,
              model_id TEXT,
              input_hash TEXT,
              output_hash TEXT,
              started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              completed_at TIMESTAMP,
              status TEXT,
              error_message TEXT
          );
          """)
  ```
  Create `app/audit.py`:
  ```python
  import json
  import hashlib
  import sqlite3
  import copy
  from datetime import datetime, timezone
  from typing import Any, Dict

  def redact_secrets(data: Any) -> Any:
      if isinstance(data, dict):
          copy_dict = {}
          for k, v in data.items():
              if k.lower() in ("api_key", "x-api-key", "token", "secret", "password"):
                  copy_dict[k] = "[REDACTED]"
              else:
                  copy_dict[k] = redact_secrets(v)
          return copy_dict
      elif isinstance(data, list):
          return [redact_secrets(item) for item in data]
      return data

  def hash_payload(data: Any) -> str:
      redacted = redact_secrets(data)
      dump = json.dumps(redacted, sort_keys=True)
      return hashlib.sha256(dump.encode("utf-8")).hexdigest()

  def log_audit_record(
      db_path: str,
      request_id: str,
      endpoint: str,
      model_id: str | None,
      input_data: Any,
      output_data: Any,
      status: str,
      error_message: str | None = None,
      started_at: datetime | None = None,
  ):
      if started_at is None:
          started_at = datetime.now(timezone.utc)
      completed_at = datetime.now(timezone.utc)
      
      in_hash = hash_payload(input_data)
      out_hash = hash_payload(output_data) if output_data is not None else None
      
      conn = sqlite3.connect(db_path)
      try:
          cursor = conn.cursor()
          cursor.execute("""
              INSERT OR REPLACE INTO audit_records 
              (request_id, endpoint, model_id, input_hash, output_hash, started_at, completed_at, status, error_message)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
          """, (
              request_id,
              endpoint,
              model_id,
              in_hash,
              out_hash,
              started_at.isoformat(),
              completed_at.isoformat(),
              status,
              error_message
          ))
          conn.commit()
      finally:
          conn.close()
  ```
- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv/bin/pytest tests/test_audit.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/database.py app/audit.py tests/test_audit.py && git commit -m "feat: add safety audit logging and database migration"`

---

### Task 2: Health, Readiness, and Capabilities Endpoints

**Files:**
- Create: `app/health.py`
- Test: `tests/test_health.py`

- [ ] **Step 1: Write failing test for health checks**
  Create `tests/test_health.py`:
  ```python
  from app.health import get_health_status, get_readiness, get_capabilities

  def test_health_checks():
      h = get_health_status()
      assert h["status"] == "ok"
      assert h["service"] == "noble-turing"
      
      r = get_capabilities()
      assert "xgboost" in r["supported_tabular_model_families"]
      assert "pytorch_cnn" in r["supported_sequence_model_families"]
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_health.py -v`
  Expected: FAIL
- [ ] **Step 3: Implement health.py**
  Create `app/health.py`:
  ```python
  import os
  import sqlite3
  from datetime import datetime, timezone
  from typing import Dict, Any, List

  def get_commit_sha() -> str:
      try:
          import subprocess
          res = subprocess.run(
              ["git", "rev-parse", "HEAD"],
              stdout=subprocess.PIPE,
              stderr=subprocess.PIPE,
              text=True,
              timeout=2.0
          )
          if res.returncode == 0:
              return res.stdout.strip()
      except Exception:
          pass
      return "unknown"

  def get_health_status() -> Dict[str, Any]:
      return {
          "status": "ok",
          "service": "noble-turing",
          "version": "0.1.0",
          "commit": get_commit_sha(),
          "time": datetime.now(timezone.utc).isoformat()
      }

  def get_readiness(db_path: str, models_dir: str, data_dir: str) -> Dict[str, Any]:
      import torch
      from app.models_lab import finbert
      
      # Determine device
      if torch.cuda.is_available():
          device = "cuda"
      elif torch.backends.mps.is_available():
          device = "mps"
      else:
          device = "cpu"
          
      # FinBERT status
      finbert_status = "loaded" if finbert._pipeline is not None else "not_loaded"
      
      # Query active / failed jobs
      active_jobs = 0
      last_failed_job = None
      
      if os.path.exists(db_path):
          try:
              conn = sqlite3.connect(db_path)
              conn.row_factory = sqlite3.Row
              cursor = conn.cursor()
              
              cursor.execute(
                  "SELECT COUNT(*) FROM jobs WHERE status IN ('pending', 'training', 'scoring')"
              )
              active_jobs = cursor.fetchone()[0]
              
              cursor.execute(
                  "SELECT model_id, error_message FROM jobs WHERE status = 'failed' ORDER BY completed_at DESC LIMIT 1"
              )
              row = cursor.fetchone()
              if row:
                  last_failed_job = {
                      "model_id": row["model_id"],
                      "error_message": row["error_message"]
                  }
              conn.close()
          except Exception:
              pass
              
      return {
          "api_key_auth_enabled": True,
          "database_path": db_path,
          "model_directory": models_dir,
          "data_directory": data_dir,
          "finbert_load_status": finbert_status,
          "device": device,
          "active_jobs": active_jobs,
          "last_failed_job": last_failed_job
      }

  def get_capabilities() -> Dict[str, Any]:
      return {
          "supported_endpoints": [
              "/health",
              "/api/v1/readiness",
              "/api/v1/capabilities",
              "/api/v1/train_tabular_model",
              "/api/v1/train_time_series_model",
              "/api/v1/score_daily_mover_candidates",
              "/api/v1/score_time_series_candidates",
              "/api/v1/annotate_news",
              "/api/v1/export_onnx",
              "/api/v1/validate_onnx_parity",
              "/api/v1/export_artha_package"
          ],
          "supported_tabular_model_families": ["xgboost", "catboost", "autogluon", "tabm", "tabpfn", "lightgbm"],
          "supported_sequence_model_families": ["pytorch_cnn", "minirocket", "inceptiontime", "tcn", "resnet"],
          "supported_export_formats": ["pkl", "pt", "onnx"],
          "package_formats": ["artha_onnx", "artha_mac_api"],
          "scoring_support_status": True
      }
  ```
- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv/bin/pytest tests/test_health.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/health.py tests/test_health.py && git commit -m "feat: add health status backend functions"`

---

### Task 3: Artha News Annotation Response and Caching Upgrade

**Files:**
- Modify: `app/models_lab/finbert.py`
- Test: `tests/test_finbert.py`

- [ ] **Step 1: Write failing test for Artha FinBERT annotation format**
  Modify `tests/test_finbert.py` to ensure the response fields exactly match Artha requirements:
  ```python
      # Run mock annotation (since downloading transformer model takes time)
      results = annotate_news_batch(news_items, db_path, use_mock=True)
      assert len(results) == 1
      assert "cache_hit" in results[0]
      assert results[0]["cache_hit"] is False
      
      # Second hit should cache
      results2 = annotate_news_batch(news_items, db_path, use_mock=True)
      assert results2[0]["cache_hit"] is True
      assert "title" not in results2[0]  # Ensure full article details are not stored/returned
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_finbert.py -v`
  Expected: FAIL
- [ ] **Step 3: Modify finbert.py**
  Update `annotate_news_batch` to include `cache_hit` flag and return the exact Artha output schema (removing keys like `event_id` or description fields from output):
  In `app/models_lab/finbert.py`:
  Replace the database fetching loop:
  ```python
          for idx, item in enumerate(news_items):
              h = item["dedupe_hash"]
              cursor.execute(
                  "SELECT sentiment_label, sentiment_score, positive_score, negative_score, neutral_score FROM news_annotations WHERE dedupe_hash = ?",
                  (h,),
              )
              row = cursor.fetchone()
              if row:
                  results.append(
                      {
                          "dedupe_hash": h,
                          "model_id": "cached",
                          "sentiment_label": row["sentiment_label"],
                          "sentiment_score": row["sentiment_score"],
                          "positive_score": row["positive_score"],
                          "negative_score": row["negative_score"],
                          "neutral_score": row["neutral_score"],
                          "cache_hit": True,
                          "annotated_at": datetime.now(timezone.utc).isoformat(),
                      }
                  )
              else:
                  results.append(None)
                  to_compute.append(f"{item.get('title', '')}. {item.get('snippet', '')}")
                  to_compute_indices.append(idx)
  ```
  And when parsing computed results:
  ```python
                  res = {
                      "dedupe_hash": h,
                      "model_id": "ProsusAI/finbert",
                      "sentiment_label": label,
                      "sentiment_score": sentiment_score,
                      "positive_score": positive_score,
                      "negative_score": negative_score,
                      "neutral_score": neutral_score,
                      "cache_hit": False,
                      "annotated_at": now,
                  }
  ```
- [ ] **Step 4: Run tests and make sure they pass**
  Run: `.venv/bin/pytest tests/test_finbert.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/models_lab/finbert.py && git commit -m "feat: align FinBERT outputs with Artha schema and cache hit tracking"`

---

### Task 4: Tabular Validation Data Preservation & LightGBM Blocker

**Files:**
- Modify: `app/models_lab/tabular.py`
- Test: `tests/test_tabular.py`

- [ ] **Step 1: Write failing test for LightGBM blocker and validation samples**
  Modify `tests/test_tabular.py` to add tests for:
  1. Saving `validation_samples.npy` to the model directory.
  2. Training LightGBM when it is not installed raising an error/blocker instead of silently falling back.
  ```python
  def test_lightgbm_missing_blocker(tmp_path, monkeypatch):
      model_dir = str(tmp_path / "models")
      os.makedirs(model_dir, exist_ok=True)
      df = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(4)] + ["target"])
      df["target"] = np.random.choice([0, 1], size=50)
      dataset_path = str(tmp_path / "dataset.csv")
      df.to_csv(dataset_path, index=False)
      
      config = {
          "experiment_id": "exp_lgb_block",
          "dataset_id": "ds_lgb",
          "dataset_uri": dataset_path,
          "feature_schema_hash": "hash_lgb",
          "label_definition": "target",
          "train_split": {"type": "indices", "values": list(range(35))},
          "holdout_split": {"type": "indices", "values": list(range(35, 50))},
          "model_family": "lightgbm"
      }
      
      # Mock LIGHTGBM_AVAILABLE = False
      import app.models_lab.tabular
      monkeypatch.setattr(app.models_lab.tabular, "LIGHTGBM_AVAILABLE", False)
      
      with pytest.raises(ValueError, match="LightGBM not installed. Cannot train."):
          train_tabular_pipeline("m_lgb_block", config, model_dir)
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_tabular.py -v`
  Expected: FAIL
- [ ] **Step 3: Modify tabular.py**
  In `app/models_lab/tabular.py`:
  1. Record exact column order and Artha metadata in `metadata.json`.
  2. Save first 100 rows of validation samples to `validation_samples.npy`.
  3. Fail training if LightGBM requested but not available.
  
  ```python
      # Add exact column order to metadata
      metadata = {
          "model_id": model_id,
          "experiment_id": config["experiment_id"],
          "model_family": actual_family,
          "model_type": "tabular",
          "feature_schema_hash": config["feature_schema_hash"],
          "label_definition": label_col,
          "feature_columns": features, # Exact column order
          # Artha metadata fields:
          "dataset_id": config.get("dataset_id"),
          "train_split": train_split,
          "holdout_split": holdout_split,
          "target_stop_assumptions": config.get("target_stop_assumptions"),
          "cost_assumptions": config.get("cost_assumptions"),
          "broker_limits": config.get("broker_limits"),
          "broker_limit_policy": config.get("broker_limit_policy")
      }
  ```
  Save validation samples:
  ```python
      # Save validation samples
      val_samples_path = os.path.join(save_dir, "validation_samples.npy")
      X_hold_np = X_hold[features].values.astype(np.float32)
      np.save(val_samples_path, X_hold_np)
  ```
  Modify LightGBM branch:
  ```python
      elif model_family == "lightgbm":
          if LIGHTGBM_AVAILABLE:
              trained_model = lgb.LGBMClassifier(verbose=-1)
              trained_model.fit(X_train, y_train)
              actual_family = "lightgbm"
          else:
              raise ValueError("LightGBM not installed. Cannot train.")
  ```
- [ ] **Step 4: Run tests and make sure they pass**
  Run: `.venv/bin/pytest tests/test_tabular.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/models_lab/tabular.py tests/test_tabular.py && git commit -m "feat: save tabular validation samples and implement LightGBM blocker"`

---

### Task 5: Sequence Validation Data Preservation

**Files:**
- Modify: `app/models_lab/sequence.py`
- Test: `tests/test_sequence.py`

- [ ] **Step 1: Write failing test for sequence validation samples**
  Modify `tests/test_sequence.py` to assert that `validation_samples.npy` is created in sequence model directories:
  ```python
      # inside test_sequence_training
      assert os.path.exists(os.path.join(model_dir, "m_seq1", "validation_samples.npy"))
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_sequence.py -v`
  Expected: FAIL
- [ ] **Step 3: Modify sequence.py**
  In `app/models_lab/sequence.py`, record column order, Artha metadata, and save validation samples.
  ```python
      # Save validation samples
      val_samples_path = os.path.join(save_dir, "validation_samples.npy")
      X_hold_np = X_hold.numpy().astype(np.float32)
      np.save(val_samples_path, X_hold_np)
  ```
  ```python
      # Record metadata
      metadata = {
          "model_id": model_id,
          "experiment_id": config["experiment_id"],
          "model_family": actual_family,
          "model_type": "sequence",
          "sequence_schema_hash": config["sequence_schema_hash"],
          "label_definition": label_col,
          "sequence_length": seq_len,
          "num_features": num_features,
          "feature_columns": feature_cols, # Exact column order
          # Artha metadata fields:
          "dataset_id": config.get("dataset_id"),
          "train_split": train_split,
          "holdout_split": holdout_split,
          "target_stop_assumptions": config.get("target_stop_assumptions"),
          "cost_assumptions": config.get("cost_assumptions"),
          "broker_limits": config.get("broker_limits"),
          "broker_limit_policy": config.get("broker_limit_policy")
      }
  ```
- [ ] **Step 4: Run tests and make sure they pass**
  Run: `.venv/bin/pytest tests/test_sequence.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/models_lab/sequence.py && git commit -m "feat: save sequence validation samples during training"`

---

### Task 6: ONNX Parity Real Validation Rows & Tolerance Upgrades

**Files:**
- Modify: `app/models_lab/onnx_utils.py`
- Test: `tests/test_onnx.py`

- [ ] **Step 1: Write failing test for validation-samples parity checks**
  Modify `tests/test_onnx.py` to ensure ONNX parity checks read `validation_samples.npy` and produce the updated keys:
  ```python
      # Validate parity report keys
      report_path = os.path.join(model_dir, "m_seq1", "onnx_parity_report.json")
      with open(report_path, "r") as f:
          report = json.load(f)
      assert "sample_count" in report
      assert "max_abs_delta" in report
      assert "passed" in report
      assert "tolerance" in report
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_onnx.py -v`
  Expected: FAIL
- [ ] **Step 3: Modify onnx_utils.py**
  In `app/models_lab/onnx_utils.py`:
  1. Default tolerance to `0.0001` (1e-4).
  2. Load `validation_samples.npy` for the input of check parity.
  3. Save the report structure containing `sample_count`, `max_abs_delta`, `mean_abs_delta`, `passed`, `tolerance`.
  
  ```python
  # Update tolerance default
  if tolerance is None:
      tolerance = 0.0001
  ```
  ```python
  # Loading samples in check parity
  def _check_parity(
      py_pred: np.ndarray,
      onnx_pred: np.ndarray,
      save_dir: str,
      tolerance: float,
  ) -> Tuple[str, str, Optional[str]]:
      diff = np.abs(py_pred - onnx_pred)
      max_diff = float(np.max(diff))
      mean_diff = float(np.mean(diff))
      parity_passed = max_diff < tolerance
      
      report = {
          "sample_count": len(py_pred),
          "max_abs_delta": max_diff,
          "mean_abs_delta": mean_diff,
          "passed": parity_passed,
          "tolerance": tolerance,
          # Backward compatibility
          "max_absolute_difference": max_diff,
          "parity_passed": parity_passed,
          "onnx_output_sample": [float(x) for x in onnx_pred[:5]],
          "py_output_sample": [float(x) for x in py_pred[:5]],
      }
      with open(os.path.join(save_dir, "onnx_parity_report.json"), "w") as f:
          json.dump(report, f, indent=2)
          
      if parity_passed:
          return "success", "success", None
      return "success", "failed", f"Parity mismatch max {max_diff:.2e} >= {tolerance:.2e}"
  ```
- [ ] **Step 4: Run tests and make sure they pass**
  Run: `.venv/bin/pytest tests/test_onnx.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/models_lab/onnx_utils.py tests/test_onnx.py && git commit -m "feat: upgrade ONNX parity check to use validation rows and 1e-4 tolerance"`

---

### Task 7: Daily Mover Candidate Scoring Module

**Files:**
- Create: `app/scoring/daily_mover.py`
- Test: `tests/test_daily_mover.py`

- [ ] **Step 1: Write failing test for Daily Mover scoring**
  Create `tests/test_daily_mover.py` with mock tabular prediction and schema hashing verification:
  ```python
  import pytest
  import os
  import json
  from app.scoring.daily_mover import score_daily_mover

  def test_score_daily_mover_missing_model():
      res = score_daily_mover("nonexistent", "db.sqlite", "models_dir", {})
      assert res["status"] == "blocked"
      assert "model_missing" in res["errors"]
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_daily_mover.py -v`
  Expected: FAIL
- [ ] **Step 3: Implement daily_mover.py**
  Create `app/scoring/daily_mover.py`:
  ```python
  import os
  import json
  import pickle
  import sqlite3
  import numpy as np
  from typing import Dict, Any, List

  def get_news_sentiment_score(db_path: str, news_event_ids: List[str]) -> float:
      if not news_event_ids:
          return 0.0
      conn = sqlite3.connect(db_path)
      conn.row_factory = sqlite3.Row
      cursor = conn.cursor()
      scores = []
      try:
          for eid in news_event_ids:
              cursor.execute(
                  "SELECT positive_score, negative_score FROM news_annotations WHERE dedupe_hash = ?",
                  (eid,)
              )
              row = cursor.fetchone()
              if row:
                  pos = row["positive_score"]
                  neg = row["negative_score"]
                  # Normalize to [0, 1]
                  scores.append((pos - neg + 1.0) / 2.0)
      except Exception:
          pass
      finally:
          conn.close()
      return float(np.mean(scores)) if scores else 0.0

  def score_daily_mover(
      model_id: str,
      db_path: str,
      models_dir: str,
      payload: Dict[str, Any]
  ) -> Dict[str, Any]:
      save_dir = os.path.join(models_dir, model_id)
      metadata_path = os.path.join(save_dir, "metadata.json")
      
      if not os.path.exists(metadata_path):
          return {
              "status": "blocked",
              "errors": ["model_missing"],
              "scores": []
          }
          
      with open(metadata_path, "r") as f:
          metadata = json.load(f)
          
      # Schema check
      req_schema = payload.get("feature_schema_hash")
      if req_schema and metadata.get("feature_schema_hash") != req_schema:
          return {
              "status": "blocked",
              "errors": ["feature_schema_mismatch"],
              "scores": []
          }
          
      # Load model
      model_pkl = os.path.join(save_dir, "model.pkl")
      if not os.path.exists(model_pkl):
          return {
              "status": "blocked",
              "errors": ["model_missing"],
              "scores": []
          }
          
      with open(model_pkl, "rb") as f:
          model = pickle.load(f)
          
      # Load threshold
      thresholds_path = os.path.join(save_dir, "thresholds.json")
      threshold = 0.5
      if os.path.exists(thresholds_path):
          with open(thresholds_path, "r") as f:
              threshold = json.load(f).get("threshold", 0.5)
              
      feature_names = metadata.get("feature_columns", [])
      if not feature_names:
          feature_names = list(getattr(model, "feature_names_in_", None) or [])
          
      candidates = payload.get("candidates", [])
      scores_list = []
      missing_candidates = []
      
      for cand in candidates:
          cand_id = cand.get("candidate_id")
          feats = cand.get("features")
          if feats is None:
              missing_candidates.append(cand_id)
              continue
              
          # Format features
          X_row = []
          for fname in feature_names:
              val = feats.get(fname)
              X_row.append(float(val) if val is not None else 0.0)
              
          X_arr = np.array([X_row], dtype=np.float32)
          tabular_score = float(model.predict_proba(X_arr)[0, 1])
          
          # Compute news score
          news_score = get_news_sentiment_score(db_path, cand.get("news_event_ids", []))
          
          # Default timeseries score
          ts_score = 0.0
          
          final_score = 0.9 * tabular_score + 0.1 * news_score
          
          scores_list.append({
              "candidate_id": cand_id,
              "tabular_score": tabular_score,
              "time_series_score": ts_score,
              "news_score": news_score,
              "final_score": final_score,
              "score_source": "mac_api",
              "model_id": model_id,
              "metadata": {
                  "feature_hash": metadata.get("feature_schema_hash"),
                  "sequence_hash": metadata.get("sequence_schema_hash"),
                  "paper_only": True,
                  "broker_routed": False,
                  "live_eligible": False
              }
          })
          
      return {
          "status": "completed",
          "model_id": model_id,
          "model_package_id": payload.get("model_package_id"),
          "scores": scores_list,
          "missing_candidates": missing_candidates,
          "errors": []
      }
  ```
- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv/bin/pytest tests/test_daily_mover.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/scoring/daily_mover.py tests/test_daily_mover.py && git commit -m "feat: implement synchronous daily mover candidate scoring"`

---

### Task 8: Time Series Candidate Scoring Module

**Files:**
- Create: `app/scoring/time_series.py`
- Test: `tests/test_time_series.py`

- [ ] **Step 1: Write failing test for Time Series scoring**
  Create `tests/test_time_series.py`:
  ```python
  import pytest
  from app.scoring.time_series import score_time_series

  def test_score_time_series_missing_model():
      res = score_time_series("nonexistent", "models_dir", {})
      assert res["status"] == "blocked"
      assert "sequence_model_missing" in res["errors"]
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_time_series.py -v`
  Expected: FAIL
- [ ] **Step 3: Implement time_series.py**
  Create `app/scoring/time_series.py`:
  ```python
  import os
  import json
  import torch
  import numpy as np
  from typing import Dict, Any, List

  def score_time_series(
      model_id: str,
      models_dir: str,
      payload: Dict[str, Any]
  ) -> Dict[str, Any]:
      save_dir = os.path.join(models_dir, model_id)
      metadata_path = os.path.join(save_dir, "metadata.json")
      
      if not os.path.exists(metadata_path):
          return {
              "status": "blocked",
              "errors": ["sequence_model_missing"],
              "scores": []
          }
          
      with open(metadata_path, "r") as f:
          metadata = json.load(f)
          
      # Schema check
      req_schema = payload.get("sequence_schema_hash")
      if req_schema and metadata.get("sequence_schema_hash") != req_schema:
          return {
              "status": "blocked",
              "errors": ["sequence_schema_mismatch"],
              "scores": []
          }
          
      model_pt = os.path.join(save_dir, "model.pt")
      if not os.path.exists(model_pt):
          return {
              "status": "blocked",
              "errors": ["sequence_model_missing"],
              "scores": []
          }
          
      seq_len = metadata.get("sequence_length", 10)
      num_features = metadata.get("num_features", 2)
      
      # Reconstruct PyTorch model
      from app.models_lab.sequence import (
          Simple1DCNN, MiniRocketFeatureExtractor, InceptionTime, TCN, ResNetTS
      )
      
      model_classes = [
          ("Simple1DCNN", Simple1DCNN),
          ("MiniRocketFeatureExtractor", MiniRocketFeatureExtractor),
          ("InceptionTime", InceptionTime),
          ("TCN", TCN),
          ("ResNetTS", ResNetTS),
      ]
      
      model = None
      for _, cls in model_classes:
          try:
              model = cls(seq_len, num_features)
              model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
              model.eval()
              break
          except Exception:
              model = None
              
      if model is None:
          return {
              "status": "blocked",
              "errors": ["sequence_model_missing"],
              "scores": []
          }
          
      candidates = payload.get("candidates", [])
      scores_list = []
      missing_candidates = []
      
      for cand in candidates:
          cand_id = cand.get("candidate_id")
          seq_feats = cand.get("sequence_features")
          
          if seq_feats is None:
              missing_candidates.append(cand_id)
              continue
              
          # Parse sequence_features
          try:
              if isinstance(seq_feats, list):
                  # list of dicts or list of lists
                  if len(seq_feats) == 0:
                      X_row = np.zeros((seq_len, num_features), dtype=np.float32)
                  elif isinstance(seq_feats[0], dict):
                      # list of dicts
                      feature_names = metadata.get("feature_columns", [])
                      X_row_list = []
                      for t in range(seq_len):
                          step_row = []
                          step_dict = seq_feats[t] if t < len(seq_feats) else {}
                          for f in range(num_features):
                              # try different lookup options
                              keys_to_try = [
                                  f"t{t}_f{f}",
                                  f"f{f}",
                              ]
                              if feature_names:
                                  feat_idx = t * num_features + f
                                  if feat_idx < len(feature_names):
                                      keys_to_try.append(feature_names[feat_idx])
                              val = None
                              for k in keys_to_try:
                                  if k in step_dict:
                                      val = step_dict[k]
                                      break
                              step_row.append(float(val) if val is not None else 0.0)
                          X_row_list.append(step_row)
                      X_row = np.array(X_row_list, dtype=np.float32)
                  else:
                      # list of lists
                      X_row_list = []
                      for t in range(seq_len):
                          step_list = seq_feats[t] if t < len(seq_feats) else []
                          step_row = []
                          for f in range(num_features):
                              val = step_list[f] if f < len(step_list) else 0.0
                              step_row.append(float(val))
                          X_row_list.append(step_row)
                      X_row = np.array(X_row_list, dtype=np.float32)
              elif isinstance(seq_feats, dict):
                  # flat dict
                  feature_names = metadata.get("feature_columns", [])
                  X_row_list = []
                  for t in range(seq_len):
                      step_row = []
                      for f in range(num_features):
                          keys_to_try = [
                              f"t{t}_f{f}",
                              f"f{f}"
                          ]
                          if feature_names:
                              feat_idx = t * num_features + f
                              if feat_idx < len(feature_names):
                                  keys_to_try.append(feature_names[feat_idx])
                          val = None
                          for k in keys_to_try:
                              if k in seq_feats:
                                  val = seq_feats[k]
                                  break
                          step_row.append(float(val) if val is not None else 0.0)
                      X_row_list.append(step_row)
                  X_row = np.array(X_row_list, dtype=np.float32)
              else:
                  X_row = np.zeros((seq_len, num_features), dtype=np.float32)
          except Exception:
              X_row = np.zeros((seq_len, num_features), dtype=np.float32)
              
          X_tens = torch.tensor(X_row, dtype=torch.float32).unsqueeze(0)
          with torch.no_grad():
              ts_score = float(model(X_tens).numpy().flatten()[0])
              
          scores_list.append({
              "candidate_id": cand_id,
              "tabular_score": 0.0,
              "time_series_score": ts_score,
              "news_score": 0.0,
              "final_score": ts_score,
              "score_source": "mac_api",
              "model_id": model_id,
              "metadata": {
                  "feature_hash": metadata.get("feature_schema_hash"),
                  "sequence_hash": metadata.get("sequence_schema_hash"),
                  "paper_only": True,
                  "broker_routed": False,
                  "live_eligible": False
              }
          })
          
      return {
          "status": "completed",
          "model_id": model_id,
          "model_package_id": payload.get("model_package_id"),
          "scores": scores_list,
          "missing_candidates": missing_candidates,
          "errors": []
      }
  ```
- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv/bin/pytest tests/test_time_series.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/scoring/time_series.py tests/test_time_series.py && git commit -m "feat: implement synchronous time series sequence candidate scoring"`

---

### Task 9: Model Package Exporter Module

**Files:**
- Create: `app/packaging/artha_package.py`
- Test: `tests/test_artha_package.py`

- [ ] **Step 1: Write failing test for Artha package export**
  Create `tests/test_artha_package.py`:
  ```python
  import pytest
  import os
  import json
  from app.packaging.artha_package import export_artha_package

  def test_export_package_missing_model():
      with pytest.raises(FileNotFoundError):
          export_artha_package("nonexistent", "mac_api", "models_dir")
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_artha_package.py -v`
  Expected: FAIL
- [ ] **Step 3: Implement artha_package.py**
  Create `app/packaging/artha_package.py`:
  ```python
  import os
  import json
  from datetime import datetime, timezone
  from typing import Dict, Any, List

  def export_artha_package(
      model_id: str,
      package_type: str, # "onnx" or "mac_api"
      models_dir: str
  ) -> List[str]:
      save_dir = os.path.join(models_dir, model_id)
      metadata_path = os.path.join(save_dir, "metadata.json")
      
      if not os.path.exists(metadata_path):
          raise FileNotFoundError(f"Model metadata {metadata_path} not found")
          
      with open(metadata_path, "r") as f:
          metadata = json.load(f)
          
      # Determine if parity passed
      parity_failed = False
      parity_report_path = os.path.join(save_dir, "onnx_parity_report.json")
      if os.path.exists(parity_report_path):
          with open(parity_report_path, "r") as f:
              parity = json.load(f)
              if not parity.get("passed", False):
                  parity_failed = True
                  
      exported_files = []
      
      # 1. approval.json
      approval_path = os.path.join(save_dir, "approval.json")
      approval_data = {
          "status": "blocked" if parity_failed else "not_approved",
          "paper_only": True,
          "broker_routed": False,
          "live_eligible": False
      }
      if parity_failed:
          approval_data["errors"] = ["onnx_parity_failed"]
      with open(approval_path, "w") as f:
          json.dump(approval_data, f, indent=2)
      exported_files.append("approval.json")
      
      # 2. model_package.json
      pkg_path = os.path.join(save_dir, "model_package.json")
      pkg_data = {
          "model_package_id": f"pkg_{model_id}",
          "model_id": model_id,
          "package_type": package_type,
          "status": "blocked" if parity_failed else "completed",
          "created_at": datetime.now(timezone.utc).isoformat()
      }
      with open(pkg_path, "w") as f:
          json.dump(pkg_data, f, indent=2)
      exported_files.append("model_package.json")
      
      # 3. feature_schema.json
      feat_path = os.path.join(save_dir, "feature_schema.json")
      feat_cols = metadata.get("feature_columns", [])
      feat_data = {
          "feature_schema_hash": metadata.get("feature_schema_hash"),
          "features": [{"name": fn, "type": "float"} for fn in feat_cols]
      }
      with open(feat_path, "w") as f:
          json.dump(feat_data, f, indent=2)
      exported_files.append("feature_schema.json")
      
      # 4. validation_report.json
      val_path = os.path.join(save_dir, "validation_report.json")
      metrics = {}
      for rep in ("training_report.json", "holdout_report.json"):
          p = os.path.join(save_dir, rep)
          if os.path.exists(p):
              with open(p, "r") as f:
                  metrics[rep.replace(".json", "")] = json.load(f)
      val_data = {
          "experiment_id": metadata.get("experiment_id"),
          "dataset_id": metadata.get("dataset_id"),
          "metrics": metrics,
          "splits": {
              "train": metadata.get("train_split"),
              "holdout": metadata.get("holdout_split")
          },
          "assumptions": {
              "target_stop_assumptions": metadata.get("target_stop_assumptions"),
              "cost_assumptions": metadata.get("cost_assumptions")
          },
          "broker_limits": {
              "limits": metadata.get("broker_limits"),
              "policy": metadata.get("broker_limit_policy")
          }
      }
      with open(val_path, "w") as f:
          json.dump(val_data, f, indent=2)
      exported_files.append("validation_report.json")
      
      # 5. thresholds.json
      thresh_path = os.path.join(save_dir, "thresholds.json")
      if not os.path.exists(thresh_path):
          with open(thresh_path, "w") as f:
              json.dump({"threshold": 0.5}, f, indent=2)
      exported_files.append("thresholds.json")
      
      if package_type == "onnx":
          # 6. sequence_schema.json
          seq_path = os.path.join(save_dir, "sequence_schema.json")
          seq_data = {
              "sequence_schema_hash": metadata.get("sequence_schema_hash"),
              "sequence_length": metadata.get("sequence_length"),
              "num_features": metadata.get("num_features"),
              "features": [{"name": fn, "type": "float"} for fn in feat_cols]
          }
          with open(seq_path, "w") as f:
              json.dump(seq_data, f, indent=2)
          exported_files.append("sequence_schema.json")
          
          # 7. preprocessing.json
          pre_path = os.path.join(save_dir, "preprocessing.json")
          pre_data = {
              "impute_missing_with_zero": True
          }
          with open(pre_path, "w") as f:
              json.dump(pre_data, f, indent=2)
          exported_files.append("preprocessing.json")
          
          # 8. calibration.json
          cal_path = os.path.join(save_dir, "calibration.json")
          if not os.path.exists(cal_path):
              with open(cal_path, "w") as f:
                  json.dump({}, f, indent=2)
          exported_files.append("calibration.json")
          
          # Ensure model.onnx exists (or just register the export requirement)
          exported_files.append("model.onnx")
          exported_files.append("onnx_parity_report.json")
          
      else: # mac_api
          # 6. mac_api.json
          mac_path = os.path.join(save_dir, "mac_api.json")
          mac_data = {
              "scoring_endpoints": {
                  "daily_mover": "/api/v1/score_daily_mover_candidates",
                  "time_series": "/api/v1/score_time_series_candidates"
              },
              "model_id": model_id
          }
          with open(mac_path, "w") as f:
              json.dump(mac_data, f, indent=2)
          exported_files.append("mac_api.json")
          
      return exported_files
  ```
- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv/bin/pytest tests/test_artha_package.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/packaging/artha_package.py tests/test_artha_package.py && git commit -m "feat: implement Artha package exporter"`

---

### Task 10: Register aliases/routes in `app/main.py`

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write failing integration tests**
  Modify `tests/test_main.py` to add tests for `/health`, `/api/v1/readiness`, `/api/v1/capabilities`, `/api/v1/export_artha_package` and check that scoring returns synchronous predictions:
  ```python
  def test_health_check_unauthenticated():
      response = client.get("/health")
      assert response.status_code == 200
      assert response.json()["service"] == "noble-turing"
      
  def test_readiness_authenticated():
      response = client.get("/api/v1/readiness", headers={"X-API-Key": "test-token"})
      assert response.status_code == 200
      assert "api_key_auth_enabled" in response.json()
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_main.py -v`
  Expected: FAIL
- [ ] **Step 3: Modify app/main.py**
  Register routes, wire up score logic synchronously, add database audit logs using decorator/middleware.
  Let's replace routes in `app/main.py` to route scoring to `score_daily_mover` and `score_time_series`, news annotation to updated finbert, health/readiness/capabilities to the health module, and add audit logging.
- [ ] **Step 4: Run tests and make sure they pass**
  Run: `.venv/bin/pytest tests/test_main.py -v`
  Expected: PASS
- [ ] **Step 5: Commit**
  Run: `git add app/main.py && git commit -m "feat: rewire FastAPI app to use Artha modules and add audit logging"`
