import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from typing import Tuple, Dict, Any, Optional
from app.models_lab.sequence import Simple1DCNN
from app.config import get_settings, ONNX_PARITY_TOLERANCE

def _get_validation_samples(save_dir: str, n_features: int, seq_len: Optional[int] = None) -> np.ndarray:
    val_samples_path = os.path.join(save_dir, "validation_samples.npy")
    if os.path.exists(val_samples_path):
        try:
            arr = np.load(val_samples_path)
            if seq_len is not None:
                if len(arr.shape) == 3 and arr.shape[1] == seq_len and arr.shape[2] == n_features:
                    return arr
            else:
                if len(arr.shape) == 2 and arr.shape[1] == n_features:
                    return arr
        except Exception:
            pass
    if seq_len is not None:
        return np.random.rand(100, seq_len, n_features).astype(np.float32)
    return np.random.rand(100, n_features).astype(np.float32)


def export_and_verify_onnx(
    model_id: str,
    models_dir: str,
    tolerance: Optional[float] = None,
) -> Tuple[str, str, Optional[str]]:
    """Export a trained model to ONNX and verify parity with Python predictions.

    Returns:
        (onnx_export_status, parity_status, error_message)
        - onnx_export_status: "success", "failed", or "unsupported"
        - parity_status: "success", "failed", or "unchecked"
        - error_message: None on success, or a human-readable error string
    """
    if tolerance is None:
        try:
            settings = get_settings()
            tolerance = settings.onnx_parity_tolerance
        except Exception:
            tolerance = ONNX_PARITY_TOLERANCE

    save_dir = os.path.join(models_dir, model_id)

    # Read metadata
    metadata_path = os.path.join(save_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return "failed", "unchecked", "metadata.json not found"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model_type = metadata["model_type"]
    onnx_path = os.path.join(save_dir, "model.onnx")

    try:
        if model_type == "tabular":
            return _export_tabular(model_id, save_dir, onnx_path, metadata, tolerance)
        elif model_type == "sequence":
            return _export_sequence(model_id, save_dir, onnx_path, metadata, tolerance)
        else:
            return "unsupported", "unchecked", f"Unsupported model_type: {model_type}"

    except Exception as e:
        err_msg = str(e)
        report = {"error": err_msg, "parity_passed": False}
        with open(os.path.join(save_dir, "onnx_parity_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return "failed", "unchecked", err_msg


def _export_tabular(
    model_id: str,
    save_dir: str,
    onnx_path: str,
    metadata: Dict[str, Any],
    tolerance: float,
) -> Tuple[str, str, Optional[str]]:
    """Export a tabular model (XGBoost / CatBoost) to ONNX."""
    model_pkl_path = os.path.join(save_dir, "model.pkl")
    with open(model_pkl_path, "rb") as f:
        model = pickle.load(f)

    model_family = metadata.get("model_family", "xgboost")

    if model_family == "xgboost":
        return _export_xgboost(model, save_dir, onnx_path, metadata, tolerance)
    elif model_family == "catboost":
        return _export_catboost(model, save_dir, onnx_path, metadata, tolerance)
    elif model_family == "lightgbm":
        return _export_lightgbm(model, save_dir, onnx_path, metadata, tolerance)
    else:
        return "unsupported", "unchecked", f"Unsupported model_family for ONNX: {model_family}"



def _export_xgboost(
    model, save_dir: str, onnx_path: str, metadata: Dict[str, Any], tolerance: float
) -> Tuple[str, str, Optional[str]]:
    from onnxmltools import convert_xgboost

    try:
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        from skl2onnx.common.data_types import FloatTensorType

    # Preserve original feature names instead of overwriting with generic f"f{i}"
    feature_names_in = getattr(model, "feature_names_in_", None)
    if feature_names_in is not None:
        feature_names = list(feature_names_in)
    else:
        feature_names = []

    if not feature_names:
        feature_names = [f"f{i}" for i in range(len(model.feature_importances_))]

    n_features = len(feature_names)
    initial_type = [("input", FloatTensorType([None, n_features]))]

    # Temporarily set booster's feature names to generic f{i} for the converter,
    # then restore them to avoid corrupting the model's feature names.
    original_booster_features = getattr(model.get_booster(), "feature_names", None)
    try:
        model.get_booster().feature_names = [f"f{i}" for i in range(n_features)]
        onnx_model = convert_xgboost(
            model, initial_types=initial_type, target_opset=13,
            name="xgboost_model",
        )
    finally:
        if original_booster_features is not None:
            model.get_booster().feature_names = original_booster_features

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Parity Check
    dummy_input = _get_validation_samples(save_dir, n_features)
    py_pred = model.predict_proba(dummy_input)[:, 1]

    ort_sess = ort.InferenceSession(onnx_path)
    onnx_outputs = ort_sess.run(None, {"input": dummy_input})

    # onnxmltools returns predictions (index 0) and list of probabilities maps (index 1)
    probs_output = onnx_outputs[1]
    if isinstance(probs_output, np.ndarray):
        onnx_pred = probs_output[:, 1]
    elif isinstance(probs_output, list) and len(probs_output) > 0 and isinstance(probs_output[0], dict):
        onnx_pred = np.array([d[1] for d in probs_output])
    else:
        onnx_pred = np.array([list(d.values())[1] for d in probs_output])

    return _check_parity(py_pred, onnx_pred, save_dir, tolerance)


def _export_catboost(
    model, save_dir: str, onnx_path: str, metadata: Dict[str, Any], tolerance: float
) -> Tuple[str, str, Optional[str]]:
    # CatBoost native ONNX export
    model.save_model(onnx_path, format="onnx")

    n_features = model.n_features_in_
    dummy_input = _get_validation_samples(save_dir, n_features)
    py_pred = model.predict_proba(dummy_input)[:, 1]

    ort_sess = ort.InferenceSession(onnx_path)
    input_name = ort_sess.get_inputs()[0].name
    onnx_outputs = ort_sess.run(None, {input_name: dummy_input})

    probs_output = onnx_outputs[1]
    onnx_pred = probs_output[:, 1]

    return _check_parity(py_pred, onnx_pred, save_dir, tolerance)


def _export_lightgbm(
    model, save_dir: str, onnx_path: str, metadata: Dict[str, Any], tolerance: float
) -> Tuple[str, str, Optional[str]]:
    from onnxmltools import convert_lightgbm

    try:
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        from skl2onnx.common.data_types import FloatTensorType

    feature_name_attr = getattr(model, "feature_name_", None)
    if feature_name_attr is not None:
        feature_names = list(feature_name_attr)
    else:
        feature_names_in = getattr(model, "feature_names_in_", None)
        if feature_names_in is not None:
            feature_names = list(feature_names_in)
        else:
            feature_names = []

    if not feature_names:
        feature_names = [f"f{i}" for i in range(model.n_features_in_)]

    n_features = len(feature_names)
    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_lightgbm(
        model, initial_types=initial_type, target_opset=13,
        name="lightgbm_model",
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Parity Check
    dummy_input = _get_validation_samples(save_dir, n_features)
    py_pred = model.predict_proba(dummy_input)[:, 1]

    ort_sess = ort.InferenceSession(onnx_path)
    onnx_outputs = ort_sess.run(None, {"input": dummy_input})

    probs_output = onnx_outputs[1]
    if isinstance(probs_output, np.ndarray):
        onnx_pred = probs_output[:, 1]
    elif isinstance(probs_output, list) and len(probs_output) > 0 and isinstance(probs_output[0], dict):
        onnx_pred = np.array([d[1] for d in probs_output])
    else:
        onnx_pred = np.array([list(d.values())[1] for d in probs_output])

    return _check_parity(py_pred, onnx_pred, save_dir, tolerance)



def _export_sequence(
    model_id: str,
    save_dir: str,
    onnx_path: str,
    metadata: Dict[str, Any],
    tolerance: float,
) -> Tuple[str, str, Optional[str]]:
    seq_len = metadata["sequence_length"]
    n_features = metadata["num_features"]

    # Load PyTorch model — try all known architectures
    from app.models_lab.sequence import (
        Simple1DCNN,
        MiniRocketFeatureExtractor,
        InceptionTime,
        TCN,
        ResNetTS,
    )

    model_classes = [
        ("Simple1DCNN", Simple1DCNN),
        ("MiniRocketFeatureExtractor", MiniRocketFeatureExtractor),
        ("InceptionTime", InceptionTime),
        ("TCN", TCN),
        ("ResNetTS", ResNetTS),
    ]

    model = None
    for name, cls in model_classes:
        try:
            model = cls(seq_len, n_features)
            model.load_state_dict(
                torch.load(os.path.join(save_dir, "model.pt"), weights_only=True)
            )
            model.eval()
            break
        except (KeyError, RuntimeError, AttributeError):
            model = None
            continue

    if model is None:
        return "failed", "unchecked", "Could not load any known PyTorch model architecture"

    dummy_input = _get_validation_samples(save_dir, n_features, seq_len=seq_len)
    dummy_input_torch = torch.tensor(dummy_input, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input_torch[0:1],
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )

    # Parity Check
    with torch.no_grad():
        py_pred = model(dummy_input_torch).numpy().flatten()

    ort_sess = ort.InferenceSession(onnx_path)
    onnx_pred = ort_sess.run(None, {"input": dummy_input})[0].flatten()

    return _check_parity(py_pred, onnx_pred, save_dir, tolerance)


def _check_parity(
    py_pred: np.ndarray,
    onnx_pred: np.ndarray,
    save_dir: str,
    tolerance: float,
) -> Tuple[str, str, Optional[str]]:
    """Compare Python and ONNX predictions and write the parity report."""
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
        # backward compatibility:
        "max_absolute_difference": max_diff,
        "parity_passed": parity_passed,
        "onnx_output_sample": [float(x) for x in onnx_pred[:5]],
        "py_output_sample": [float(x) for x in py_pred[:5]],
    }
    with open(os.path.join(save_dir, "onnx_parity_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if parity_passed:
        return "success", "success", None
    else:
        return (
            "success",
            "failed",
            f"ONNX parity check failed: max diff {max_diff:.2e} exceeds tolerance {tolerance:.2e}",
        )
