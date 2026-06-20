import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import roc_auc_score, log_loss, f1_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

# CatBoost is optional / fallback if installed
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# AutoGluon is optional
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

# TabM is optional
try:
    import tabm
    TABM_AVAILABLE = True
except ImportError:
    TABM_AVAILABLE = False

# TabPFN is optional
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

# LightGBM is optional
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False



def _compute_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Compute calibration metrics: ECE, Brier score, and isotonic calibration curve."""
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    ece_numer = 0.0
    ece_denom = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob == bin_edges[i + 1])  # include right edge
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_conf = y_prob[mask].mean()
            avg_acc = y_true[mask].mean()
            ece_numer += n_in_bin * abs(avg_conf - avg_acc)

    ece = ece_numer / ece_denom if ece_denom > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    # Isotonic calibration curve (up to 20 points)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    cal_x = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cal_y = iso.predict(cal_x).tolist()

    return {
        "expected_calibration_error": float(ece),
        "brier_score": brier,
        "calibration_curve": [
            {"predicted": float(cal_x[i]), "calibrated": float(cal_y[i])}
            for i in range(len(cal_x))
        ],
    }


def _compute_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximizes F1 score."""
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1 = -1.0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh


def train_tabular_pipeline(
    model_id: str,
    config: Dict[str, Any],
    models_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    dataset_uri = config["dataset_uri"]
    label_col = config["label_definition"]
    model_family = config.get("model_family", "xgboost").lower()

    # Load dataset
    df = pd.read_csv(dataset_uri)

    # Parse train/holdout splits — reject unknown types per plan
    train_split = config["train_split"]
    holdout_split = config["holdout_split"]

    if train_split["type"] == "indices":
        train_df = df.iloc[train_split["values"]]
    else:
        raise ValueError(
            f"Unknown train_split type '{train_split['type']}'. "
            "Must be 'indices'. The plan requires Artha-provided splits "
            "and forbids creating hidden splits."
        )

    if holdout_split["type"] == "indices":
        holdout_df = df.iloc[holdout_split["values"]]
    else:
        raise ValueError(
            f"Unknown holdout_split type '{holdout_split['type']}'. "
            "Must be 'indices'. The plan requires Artha-provided splits "
            "and forbids creating hidden splits."
        )

    features = [col for col in df.columns if col != label_col]

    X_train = train_df[features]
    y_train = train_df[label_col]
    X_hold = holdout_df[features]
    y_hold = holdout_df[label_col]

    # Select and train model
    blockers: List[str] = []
    trained_model = None
    actual_family = model_family

    if model_family == "xgboost":
        trained_model = xgb.XGBClassifier(eval_metric="logloss")
        trained_model.fit(X_train, y_train)
    elif model_family == "catboost":
        if CATBOOST_AVAILABLE:
            trained_model = CatBoostClassifier(verbose=0)
            trained_model.fit(X_train, y_train)
        else:
            blockers.append("CatBoost not installed. Falling back to xgboost.")
            trained_model = xgb.XGBClassifier(eval_metric="logloss")
            trained_model.fit(X_train, y_train)
            actual_family = "xgboost"
    elif model_family == "autogluon":
        if AUTOGLUON_AVAILABLE:
            trained_model = TabularPredictor(
                label=label_col, eval_metric="logloss"
            ).fit(
                train_df,
                presets="best_quality",
                time_limit=300,  # 5 min lab time limit
            )
            actual_family = "autogluon"
        else:
            blockers.append(
                "AutoGluon not installed. Falling back to xgboost."
            )
            trained_model = xgb.XGBClassifier(eval_metric="logloss")
            trained_model.fit(X_train, y_train)
            actual_family = "xgboost"
    elif model_family == "tabm":
        if TABM_AVAILABLE:
            # TabM expects a dict with 'X' and 'y' keys
            trained_model = tabm.TabMModel()
            trained_model.fit({"X": X_train, "y": y_train})
            actual_family = "tabm"
        else:
            blockers.append("TabM not installed. Falling back to xgboost.")
            trained_model = xgb.XGBClassifier(eval_metric="logloss")
            trained_model.fit(X_train, y_train)
            actual_family = "xgboost"
    elif model_family == "tabpfn":
        if TABPFN_AVAILABLE:
            trained_model = TabPFNClassifier()
            trained_model.fit(X_train, y_train)
            actual_family = "tabpfn"
        else:
            blockers.append("TabPFN not installed. Falling back to xgboost.")
            trained_model = xgb.XGBClassifier(eval_metric="logloss")
            trained_model.fit(X_train, y_train)
            actual_family = "xgboost"
    elif model_family == "lightgbm":
        if LIGHTGBM_AVAILABLE:
            trained_model = lgb.LGBMClassifier(verbose=-1)
            trained_model.fit(X_train, y_train)
            actual_family = "lightgbm"
        else:
            blockers.append("LightGBM not installed. Falling back to xgboost.")
            trained_model = xgb.XGBClassifier(eval_metric="logloss")
            trained_model.fit(X_train, y_train)
            actual_family = "xgboost"
    else:
        blockers.append(
            f"Model family '{model_family}' is unsupported. "
            f"Supported: xgboost, catboost, autogluon, tabm, tabpfn, lightgbm. "
            "Falling back to xgboost."
        )
        trained_model = xgb.XGBClassifier(eval_metric="logloss")
        trained_model.fit(X_train, y_train)
        actual_family = "xgboost"

    # Predict and evaluate
    p_train = trained_model.predict_proba(X_train)[:, 1]
    p_hold = trained_model.predict_proba(X_hold)[:, 1]

    # Compute threshold that maximizes F1 on holdout
    threshold = float(_compute_threshold(y_hold.values, p_hold))

    # Compute feature importance (XGBoost/CatBoost native, otherwise from tree_ feature_importances_)
    feature_importance: Dict[str, float] = {}
    if hasattr(trained_model, "feature_importances_"):
        importances = trained_model.feature_importances_
        for fname, imp in zip(features, importances):
            feature_importance[fname] = float(imp)
    elif hasattr(trained_model, "get_booster") and hasattr(
        trained_model.get_booster(), "get_score"
    ):
        scores = trained_model.get_booster().get_score(importance_type="gain")
        feature_importance = {k: float(v) for k, v in scores.items()}

    # Compute calibration metrics
    calibration = _compute_calibration(y_hold.values, p_hold)

    metrics = {
        "training_metrics": {
            "roc_auc": float(roc_auc_score(y_train, p_train)),
            "log_loss": float(log_loss(y_train, p_train)),
            "f1": float(f1_score(y_train, p_train > threshold)),
        },
        "holdout_metrics": {
            "roc_auc": float(roc_auc_score(y_hold, p_hold)),
            "log_loss": float(log_loss(y_hold, p_hold)),
            "f1": float(f1_score(y_hold, (p_hold >= threshold).astype(int))),
        },
        "threshold": threshold,
        "feature_importance": feature_importance,
        "calibration_metrics": calibration,
        "onnx_export_status": "pending",
        "onnx_parity_status": "unchecked",
        "blockers": blockers,
    }

    # Save artifacts
    save_dir = os.path.join(models_dir, model_id)
    os.makedirs(save_dir, exist_ok=True)

    # Save model pickle
    model_path = os.path.join(save_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(trained_model, f)

    # Save reports
    with open(os.path.join(save_dir, "training_report.json"), "w") as f:
        json.dump(metrics["training_metrics"], f, indent=2)
    with open(os.path.join(save_dir, "holdout_report.json"), "w") as f:
        json.dump(metrics["holdout_metrics"], f, indent=2)

    # Save threshold and feature importance
    with open(os.path.join(save_dir, "thresholds.json"), "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    with open(
        os.path.join(save_dir, "feature_importance.json"), "w"
    ) as f:
        json.dump(feature_importance, f, indent=2)

    # Save calibration report
    with open(os.path.join(save_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)

    metadata = {
        "model_id": model_id,
        "experiment_id": config["experiment_id"],
        "model_family": actual_family,
        "model_type": "tabular",
        "feature_schema_hash": config["feature_schema_hash"],
        "label_definition": label_col,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    file_paths = {
        "model": model_path,
        "metadata": os.path.join(save_dir, "metadata.json"),
    }

    return metrics, file_paths
