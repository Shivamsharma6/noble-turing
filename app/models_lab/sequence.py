import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List
from sklearn.metrics import roc_auc_score, log_loss, f1_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# Time-series model architectures
# ---------------------------------------------------------------------------

class MiniRocketFeatureExtractor(nn.Module):
    """MiniRocket-style random convolutional feature extractor.

    Generates a fixed set of random convolutional kernels and returns
    summary statistics (max, mean, etc.) as features for classification.
    """

    def __init__(self, sequence_length: int, num_features: int, num_kernels: int = 84):
        super().__init__()
        if sequence_length < 4:
            raise ValueError("sequence_length must be at least 4 for MiniRocket.")
        self.num_features = num_features
        self.num_kernels = num_kernels

        # Generate random kernels at shape (num_kernels, num_features, k_len)
        self.register_buffer("kernels", self._generate_kernels(sequence_length, num_features, num_kernels))

    def _generate_kernels(self, seq_len: int, num_features: int, num_kernels: int) -> torch.Tensor:
        k_len = min(9, seq_len)
        kernels = torch.randn(num_kernels, num_features, k_len)
        for i in range(num_kernels):
            kernels[i] = kernels[i] / (kernels[i].norm(p=2) + 1e-8)
        return kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, num_features) -> (batch, num_kernels * stats_per_kernel)"""
        # Transpose to (batch, num_features, seq_len) for Conv1d
        x_transposed = x.transpose(1, 2)
        
        # Convolve using PyTorch's native batched conv1d
        conv_out = torch.nn.functional.conv1d(
            x_transposed,
            self.kernels,
            padding=self.kernels.size(2) // 2,
        )  # (batch, num_kernels, seq_len)

        # Compute summary statistics along the sequence dimension (dim=2)
        cp = conv_out.max(dim=2).values  # (batch, num_kernels)
        mean = conv_out.mean(dim=2)  # (batch, num_kernels)
        
        diff = conv_out - mean.unsqueeze(2)
        skewness = (diff ** 4).mean(dim=2).sqrt()  # (batch, num_kernels)
        spread = (diff ** 2).mean(dim=2).sqrt()  # (batch, num_kernels)

        return torch.cat([cp, mean, skewness, spread], dim=1)  # (batch, num_kernels * 4)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c = out_channels // 4
        self.conv3 = nn.Conv1d(in_channels, c, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, c, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, c, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels - 3 * c, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out_pool = self.conv_pool(self.pool(x))
        out = torch.cat([out3, out5, out7, out_pool], dim=1)
        return torch.relu(self.bn(out))


class InceptionTime(nn.Module):
    """InceptionTime classifier for time series.

    Uses multiple inception modules with residual connections.
    """

    def __init__(self, sequence_length: int, num_features: int, num_classes: int = 1):
        super().__init__()
        if sequence_length < 4:
            raise ValueError("sequence_length must be at least 4 for InceptionTime.")

        self.initial_conv = nn.Conv1d(num_features, 64, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(64)

        self.inception_blocks = nn.ModuleList([
            InceptionBlock(64, 64) for _ in range(3)
        ])

        self.residual_proj = nn.Conv1d(num_features, 64, kernel_size=1)

        self.final_bn = nn.BatchNorm1d(64)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, num_features) -> (batch, num_classes)"""
        x = x.transpose(1, 2)

        out = torch.relu(self.initial_bn(self.initial_conv(x)))
        res = self.residual_proj(x)

        for i, block in enumerate(self.inception_blocks):
            out = block(out)
            if i % 2 == 1:
                out = out + res
                out = torch.relu(out)

        out = torch.relu(self.final_bn(out))
        out = out.mean(dim=2)
        out = self.classifier(out)
        return torch.sigmoid(out)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network dilated convolution block preserving shape."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding='same', dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding='same', dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class TCN(nn.Module):
    """Temporal Convolutional Network for time series classification."""

    def __init__(self, sequence_length: int, num_features: int, num_channels: List[int] = None, num_classes: int = 1):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 256]
        if sequence_length < 4:
            raise ValueError("sequence_length must be at least 4 for TCN.")

        self.tcn_layers = nn.ModuleList([
            TCNBlock(
                in_channels=num_channels[i - 1] if i > 0 else num_features,
                out_channels=c,
                dilation=2 ** i,
            )
            for i, c in enumerate(num_channels)
        ])

        self.final_bn = nn.BatchNorm1d(num_channels[-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, num_features) -> (batch, num_classes)"""
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        for layer in self.tcn_layers:
            x = layer(x)

        x = torch.relu(self.final_bn(x))
        x = x.mean(dim=2)  # global average pooling
        return torch.sigmoid(self.classifier(x))


class ResNetBlock(nn.Module):
    """ResNet-style block for time series."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class ResNetTS(nn.Module):
    """ResNet-style time-series classifier (from Hill et al. 2019)."""

    def __init__(self, sequence_length: int, num_features: int, num_blocks: int = 3, num_channels: int = 128, num_classes: int = 1):
        super().__init__()
        if sequence_length < 4:
            raise ValueError("sequence_length must be at least 4 for ResNetTS.")

        self.initial_conv = nn.Conv1d(num_features, num_channels, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(num_channels)

        self.blocks = nn.ModuleList([
            ResNetBlock(num_channels, kernel_size=7)
            for _ in range(num_blocks)
        ])

        self.final_bn = nn.BatchNorm1d(num_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, num_features) -> (batch, num_classes)"""
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        x = torch.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.blocks:
            x = block(x)

        x = torch.relu(self.final_bn(x))
        x = x.mean(dim=2)  # global average pooling
        return torch.sigmoid(self.classifier(x))


# Keep Simple1DCNN for backward compatibility
class Simple1DCNN(nn.Module):
    def __init__(self, sequence_length: int, num_features: int):
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        reduced_len = sequence_length // 2
        self.fc = nn.Linear(8 * reduced_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Tuple] = {
    "pytorch_cnn": (Simple1DCNN, {}),
    "minirocket": (MiniRocketFeatureExtractor, {}),
    "inceptiontime": (InceptionTime, {}),
    "tcn": (TCN, {}),
    "resnet": (ResNetTS, {}),
}



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
            mask = mask | (y_prob == bin_edges[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_conf = y_prob[mask].mean()
            avg_acc = y_true[mask].mean()
            ece_numer += n_in_bin * abs(avg_conf - avg_acc)

    ece = ece_numer / ece_denom if ece_denom > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

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





def train_sequence_pipeline(
    model_id: str,
    config: Dict[str, Any],
    models_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    dataset_uri = config["dataset_uri"]
    label_col = config["label_definition"]
    seq_len = config.get("sequence_length", 10)
    num_features = config.get("num_features", 2)
    model_family = config.get("model_family", "pytorch_cnn").lower()

    if seq_len < 2:
        raise ValueError("sequence_length must be at least 2 to prevent shape errors in the pooling layer.")

    df = pd.read_csv(dataset_uri)

    # Parse splits — reject unknown types
    train_split = config["train_split"]
    holdout_split = config["holdout_split"]

    if train_split["type"] == "indices":
        train_df = df.iloc[train_split["values"]]
    else:
        raise ValueError(
            f"Unknown train_split type '{train_split['type']}'. "
            "Must be 'indices'. The plan requires Artha-provided splits."
        )

    if holdout_split["type"] == "indices":
        holdout_df = df.iloc[holdout_split["values"]]
    else:
        raise ValueError(
            f"Unknown holdout_split type '{holdout_split['type']}'. "
            "Must be 'indices'. The plan requires Artha-provided splits."
        )

    feature_cols = [col for col in df.columns if col != label_col]

    # Parse into 3D tensors: (batch_size, seq_len, features)
    def to_tensor_dataset(df_part: pd.DataFrame):
        X_raw = df_part[feature_cols].values
        X_reshaped = X_raw.reshape(-1, seq_len, num_features)
        y_raw = df_part[label_col].values.astype(np.float32)
        return (
            torch.tensor(X_reshaped, dtype=torch.float32),
            torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1),
        )

    X_train, y_train = to_tensor_dataset(train_df)
    X_hold, y_hold = to_tensor_dataset(holdout_df)

    # Select and train model
    blockers: List[str] = []
    actual_family = model_family

    if model_family not in _MODEL_REGISTRY:
        blockers.append(
            f"Model family '{model_family}' is unsupported. "
            f"Supported: {', '.join(_MODEL_REGISTRY.keys())}. "
            "Falling back to pytorch_cnn."
        )
        model_family = "pytorch_cnn"
        actual_family = "pytorch_cnn"

    model_cls, model_kwargs = _MODEL_REGISTRY[model_family]

    if model_family == "minirocket":
        # MiniRocket: train feature extractor, then fit a linear classifier
        feature_extractor = MiniRocketFeatureExtractor(seq_len, num_features)
        # Extract features
        with torch.no_grad():
            train_feats = feature_extractor(X_train)
            hold_feats = feature_extractor(X_hold)
        # Simple logistic regression on extracted features
        n_feat = train_feats.size(1)
        lin = nn.Linear(n_feat, 1)
        opt = optim.Adam(lin.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        for epoch in range(20):
            opt.zero_grad()
            loss = criterion(lin(train_feats), y_train)
            loss.backward()
            opt.step()
        model = nn.Sequential(feature_extractor, lin)
    elif model_family == "inceptiontime":
        model = InceptionTime(seq_len, num_features)
    elif model_family == "tcn":
        model = TCN(seq_len, num_features)
    elif model_family == "resnet":
        model = ResNetTS(seq_len, num_features)
    else:
        model = Simple1DCNN(seq_len, num_features)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train for 20 epochs (was 5, increased for better convergence)
    model.train()
    for epoch in range(20):
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

    # Compute threshold
    threshold = float(_compute_threshold(y_hold_np, p_hold))

    # Compute calibration
    calibration = _compute_calibration(y_hold_np, p_hold)

    metrics = {
        "training_metrics": {
            "roc_auc": float(roc_auc_score(y_train_np, p_train)),
            "log_loss": float(log_loss(y_train_np, p_train)),
            "f1": float(f1_score(y_train_np, (p_train >= threshold).astype(int))),
        },
        "holdout_metrics": {
            "roc_auc": float(roc_auc_score(y_hold_np, p_hold)),
            "log_loss": float(log_loss(y_hold_np, p_hold)),
            "f1": float(f1_score(y_hold_np, (p_hold >= threshold).astype(int))),
        },
        "threshold": threshold,
        "feature_importance": {},
        "calibration_metrics": calibration,
        "onnx_export_status": "pending",
        "onnx_parity_status": "unchecked",
        "blockers": blockers,
    }

    save_dir = os.path.join(models_dir, model_id)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    with open(os.path.join(save_dir, "training_report.json"), "w") as f:
        json.dump(metrics["training_metrics"], f, indent=2)
    with open(os.path.join(save_dir, "holdout_report.json"), "w") as f:
        json.dump(metrics["holdout_metrics"], f, indent=2)

    # Save threshold, feature importance and calibration
    with open(os.path.join(save_dir, "thresholds.json"), "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    with open(os.path.join(save_dir, "feature_importance.json"), "w") as f:
        json.dump({}, f, indent=2)
    with open(os.path.join(save_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)

    # Save validation samples
    val_samples_path = os.path.join(save_dir, "validation_samples.npy")
    X_hold_np = X_hold.numpy().astype(np.float32)
    np.save(val_samples_path, X_hold_np)

    metadata = {
        "model_id": model_id,
        "experiment_id": config.get("experiment_id"),
        "dataset_id": config.get("dataset_id"),
        "model_family": actual_family,
        "model_type": "sequence",
        "feature_schema_hash": config.get("feature_schema_hash"),
        "sequence_schema_hash": config.get("sequence_schema_hash"),
        "label_definition": label_col,
        "sequence_length": seq_len,
        "num_features": num_features,
        "feature_columns": feature_cols, # Exact column order
        "train_split": train_split,
        "holdout_split": holdout_split,
        "target_stop_assumptions": config.get("target_stop_assumptions"),
        "cost_assumptions": config.get("cost_assumptions"),
        "broker_limits": config.get("broker_limits"),
        "broker_limit_policy": config.get("broker_limit_policy")
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    file_paths = {
        "model": model_path,
        "metadata": os.path.join(save_dir, "metadata.json"),
        "validation_samples": val_samples_path
    }

    return metrics, file_paths
