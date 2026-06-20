# Design Spec: Artha Integration and Upgrades

## Summary
Upgrade the `noble-turing` service from a local model lab into an Artha-compatible external model service. The service will support synchronous Daily-Mover and Time-Series scoring, standardized Artha packaging layouts, real-validation ONNX parity checks, metadata alignment, audit logging, and readiness/capabilities endpoints.

## API Contract Compatibility
All non-health endpoints require the `X-API-Key` header verified against `verify_api_key`.
Endpoints are exposed under both the `/api/v1/` prefix and their bare-path equivalents:
- `POST /train_tabular_model` and `POST /api/v1/train_tabular_model`
- `POST /train_time_series_model` and `POST /api/v1/train_time_series_model`
- `POST /score_daily_mover_candidates` and `POST /api/v1/score_daily_mover_candidates`
- `POST /score_time_series_candidates` and `POST /api/v1/score_time_series_candidates`
- `POST /annotate_news` and `POST /api/v1/annotate_news`
- `POST /export_onnx` and `POST /api/v1/export_onnx`
- `POST /validate_onnx_parity` and `POST /api/v1/validate_onnx_parity`
- `GET /model_status/{model_id}` and `GET /api/v1/model_status/{model_id}`
- `GET /model_artifact/{model_id}` and `GET /api/v1/model_artifact/{model_id}`
- `GET /model_artifact/{model_id}/{filename}` and `GET /api/v1/model_artifact/{model_id}/{filename}`
- `POST /export_artha_package` and `POST /api/v1/export_artha_package`

Status and health endpoints:
- `GET /health` (Unauthenticated, returns general service metadata)
- `GET /api/v1/readiness` (Authenticated, details device, load state, database config, active/failed jobs)
- `GET /api/v1/capabilities` (Authenticated, details supported algorithms and export packages)

---

## 1. Candidate Scoring & Schema Verification

### Daily Mover Candidate Scoring (`POST /score_daily_mover_candidates`)
Scores tabular candidates synchronously.
- **Request Body**:
  - `model_package_id`: str (optional)
  - `model_id`: str (required)
  - `advisor_config_id`: str (optional)
  - `thresholds`: dict | float (optional)
  - `candidates`: list of Candidate objects
- **Candidate Object**:
  - `candidate_id`: str
  - `snapshot_id`: str
  - `dataset_id`: str
  - `symbol`: str
  - `exchange`: str (optional)
  - `direction`: str (optional)
  - `mover_type`: str (optional)
  - `decision_timestamp`: str (optional)
  - `features`: dict (optional)
  - `sequence_features`: list | dict (optional)
  - `feature_hash`: str (optional)
  - `sequence_hash`: str (optional)
  - `factor_event_ids`: list (optional)
  - `news_event_ids`: list (optional)
- **Validation**:
  - If model doesn't exist, return structured blocker: `{"status": "blocked", "errors": ["model_missing"], "scores": []}`.
  - If `feature_schema_hash` is specified in request and mismatches metadata, return: `{"status": "blocked", "errors": ["feature_schema_mismatch"], "scores": []}`.
- **Scoring**:
  - Missing feature values are filled with `0.0`.
  - Predictions are generated using the stored feature order list in `metadata.json`.
  - Calculate `news_score` from `news_annotations` cache table by lookup on `news_event_ids` (mapped to `dedupe_hash`). Normalized sentiment score: `(positive_score - negative_score + 1) / 2` averaged.
  - Compute `final_score`: `0.7 * tabular_score + 0.2 * time_series_score + 0.1 * news_score` (if sequence model exists) or `0.9 * tabular_score + 0.1 * news_score`.
  - Output contains exact candidate score rows matching request with safety metadata (`paper_only: true`, `broker_routed: false`, `live_eligible: false`).

### Time Series Candidate Scoring (`POST /score_time_series_candidates`)
Scores sequential/sequence candidates synchronously.
- **Validation**:
  - If sequence model (`model.pt`) or metadata is missing, return: `{"status": "blocked", "errors": ["sequence_model_missing"], "scores": []}`.
  - If `sequence_schema_hash` is specified in request and mismatches metadata, return schema mismatch blocker.
- **Scoring**:
  - Sequence features are processed into the 3D tensor shape `(N, seq_len, num_features)`.
  - Support flat dictionary `t{t}_f{f}` naming, list of timesteps, or list of lists, filling missing with `0.0`.
  - Perform forward pass to generate `time_series_score`.
  - Return score rows with safety metadata.

---

## 2. Artha-Compatible Model Packages

### Export Artha Package (`POST /export_artha_package`)
Exports standard configuration files to `models/<model_id>/` depending on `package_type`:
1. **ONNX Package**:
   - `model_package.json`: Main registry entry
   - `model.onnx`: Exported ONNX model graph
   - `feature_schema.json`: Input feature list and hash
   - `sequence_schema.json`: Input sequence list, seq_len, features count, and sequence hash
   - `preprocessing.json`: Missing value imputation config
   - `calibration.json`: Saved calibration curve
   - `thresholds.json`: Saved F1 threshold
   - `validation_report.json`: Validation split metrics & config
   - `onnx_parity_report.json`: Max delta and sample size
   - `approval.json`: Defaulting to `"status": "not_approved"`.
2. **Mac API Package**:
   - `model_package.json`, `feature_schema.json`, `validation_report.json`, `thresholds.json`, `approval.json`
   - `mac_api.json`: Endpoints configuration pointing back to the MacBook scoring APIs.

- **Approval Control**:
  - `approval.json` defaults to `not_approved` with `paper_only: true`.
  - If ONNX parity failed during checking, `approval.json` status will be written as `"blocked"` with error detail.
  - Never set `approved_for_paper` or enable live trading.

---

## 3. ONNX Parity Upgrades

- **Real Validation Data**:
  - During training, the first 100 rows of validation split feature values are saved as `validation_samples.npy` in the model directory.
  - Parity check (`validate_onnx_parity`) loads these real rows instead of generating random dummy tensors.
- **Parity Report**:
  - Parity tolerance is `0.0001` (1e-4).
  - Parity results are stored in `onnx_parity_report.json` with keys: `sample_count`, `max_abs_delta`, `mean_abs_delta`, `passed`, `tolerance`.
  - If parity fails, package export is marked blocked but the files are still written.

---

## 4. Metadata Alignment & Dependencies

- **Training Config**:
  - Tabs/seq pipelines will store Artha training parameters: `experiment_id`, `dataset_id`, `feature_schema_hash`, `sequence_schema_hash`, `label_definition`, `train_split`, `holdout_split`, `target_stop_assumptions`, `cost_assumptions`, `broker_limits`, `broker_limit_policy` inside `metadata.json` and `validation_report.json`.
  - Keep XGBoost/CatBoost support.
  - LightGBM: If LightGBM is missing but requested, fail training job with blocker error instead of silently falling back to XGBoost.
- **Column Order**:
  - Record the exact order of features used in training inside `metadata.json`.

---

## 5. FinBERT News Annotation

- **Format alignment**:
  - Accepts `POST /api/v1/annotate_news` with standard Artha news items.
  - Returns Artha-format JSON responses.
  - Logs `cache_hit` flag as `true` on SQLite deduplication hits.
  - Do not save full article bodies in SQLite.

---

## 6. Audit & Safety

- **Safety Constraints**:
  - No endpoints for routing orders, placing trades, or executing transactions.
- **Safety Flags**:
  - Always include safety parameters in scoring/package outputs: `paper_only: true`, `broker_routed: false`, `live_eligible: false`.
- **Database Auditing**:
  - Create `audit_records` SQLite table.
  - Log `request_id` (UUID), `endpoint`, `model_id`, SHA-256 hash of redacted request input, SHA-256 hash of redacted response output, start/completed timestamps, status, and error messages.
- **Secret Redaction**:
  - API keys or secrets are completely scrubbed from terminal logging and SQLite audit records (replaced with `[REDACTED]`).
