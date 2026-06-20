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
        # Sequence length division after maxpool 1d of kernel size 2 and stride 2
        reduced_len = sequence_length // 2
        self.fc = nn.Linear(8 * reduced_len, 1)
        
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
