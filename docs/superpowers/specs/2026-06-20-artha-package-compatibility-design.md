# Design Spec: Artha Package Compatibility, Scoring Formulations, and Strict Validation

## Goal
Upgrade the Artha external model integration on `noble-turing` to support strictly defined `backend` fields in `model_package.json`, complete advisor-style thresholds, exact echoing of candidate hashes, explicit scoring formulation metadata, strict validation row parity checks, a standalone `validate_onnx_parity` endpoint, and comprehensive deployment documentation.

---

## Component Designs

### 1. Artha Package Compatibility & Threshold Updates

#### `model_package.json` backend field
- **Files**: `app/packaging/artha_package.py`
- **Design**:
  - Update `pkg_data` to map `"backend": package_type`. This will output `"backend": "mac_api"` or `"backend": "onnx"`, satisfying Artha's validator.

#### Artha-style thresholds
- **Files**: `app/models_lab/tabular.py`, `app/models_lab/sequence.py`, `app/packaging/artha_package.py`
- **Design**:
  - When writing `thresholds.json` in both training pipelines, save the following format:
    ```json
    {
      "threshold": 0.5,
      "tabular_score": 0.5,
      "time_series_score": 0.5,
      "final_score": 0.5,
      "scoring_formula": "0.9 * tabular_score + 0.1 * news_score",
      "weights": {
        "tabular_score": 0.9,
        "news_score": 0.1,
        "time_series_score": 0.0
      },
      "threshold_source": "thresholds.json"
    }
    ```
    *(Note: For sequence models, the weights default to time-series only: `time_series_score: 1.0` and formula `"1.0 * time_series_score"`).*
  - In `app/packaging/artha_package.py`, if `thresholds.json` exists in the model directory, load it and fill in any missing fields (e.g. `tabular_score`, `time_series_score`, `final_score`, `scoring_formula`, `weights`, `threshold_source`), defaulting missing values based on the root `threshold` key.

---

### 2. Candidate Hash Echoing & Formula Logging

- **Files**: `app/scoring/daily_mover.py`, `app/scoring/time_series.py`
- **Design**:
  - In the scores loop, construct the candidate metadata block.
  - Set `feature_hash` to `cand.get("feature_hash")` and `sequence_hash` to `cand.get("sequence_hash")` instead of returning the model's schema hashes.
  - Append scoring formulation details:
    ```python
    "scoring_formula": "0.9 * tabular_score + 0.1 * news_score",  # or "1.0 * time_series_score"
    "weights": { ... },
    "threshold_source": "thresholds.json"
    ```

---

### 3. Strict Validation & Parity Endpoint Refactoring

- **Files**: `app/models_lab/onnx_utils.py`, `app/main.py`
- **Design**:
  - Modify `_get_validation_samples` to return `Tuple[np.ndarray, bool]`, where the boolean is `real_samples_used`.
  - In `_check_parity`, accept `real_samples_used`. If `real_samples_used` is `False`, set `passed = False` and add warning information to `onnx_parity_report.json`. Return `"failed"` status.
  - Implement `verify_existing_onnx(model_id, models_dir, tolerance) -> Tuple[str, str, Optional[str]]` in `onnx_utils.py` to check parity of an already exported ONNX file without rewriting it.
  - In `export_and_verify_onnx`, if the ONNX file already exists, directly return `verify_existing_onnx(...)`.
  - In `app/main.py`, rewrite `/validate_onnx_parity` to call `verify_existing_onnx` instead of `export_and_verify_onnx`.

---

### 4. Documentation Changes

- **Files**: `DEPLOYMENT.md`, `README.md`
- **Design**:
  - Create `DEPLOYMENT.md` detailing:
    - Environment variables
    - Startup commands
    - Example config
    - Exposing to LAN
    - Testing health and readiness endpoints
  - Reference `DEPLOYMENT.md` in `README.md`.

---

## Verification Plan

### Automated Tests
- Run `pytest` on the test suite to verify no regressions.
- Add unit tests verifying:
  - Backend and thresholds are correctly structured in exported package files.
  - Validation endpoint checks parity of existing files and rejects missing ONNX models.
  - Parity is marked as failed/weak when validation samples are missing.
  - Candidate scoring endpoints correctly echo candidate feature/sequence hashes and formulation metadata.
