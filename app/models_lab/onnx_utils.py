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
            
            # XGBoost / CatBoost ONNX Export
            model_family = metadata.get("model_family", "xgboost")
            
            if model_family == "xgboost":
                from onnxmltools import convert_xgboost
                try:
                    from onnxmltools.convert.common.data_types import FloatTensorType
                except ImportError:
                    from skl2onnx.common.data_types import FloatTensorType
                    
                n_features = len(model.feature_names_in_)
                model.get_booster().feature_names = [f"f{i}" for i in range(n_features)]
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
                # onnxmltools returns predictions (index 0) and list of probabilities maps (index 1)
                # Depending on skl2onnx version, probabilities output is a sequence of dicts, or a float array.
                probs_output = onnx_outputs[1]
                if isinstance(probs_output, np.ndarray):
                    onnx_pred = probs_output[:, 1]
                elif isinstance(probs_output, list) and len(probs_output) > 0 and isinstance(probs_output[0], dict):
                    onnx_pred = np.array([d[1] for d in probs_output])
                else:
                    # fallback if output format differs
                    onnx_pred = np.array([list(d.values())[1] for d in probs_output])
                    
            elif model_family == "catboost":
                # CatBoost has native ONNX export
                model.save_model(onnx_path, format="onnx")
                
                # Get number of features
                # In CatBoost it's model.n_features_in_
                n_features = model.n_features_in_
                dummy_input = np.random.rand(100, n_features).astype(np.float32)
                py_pred = model.predict_proba(dummy_input)[:, 1]
                
                # Run ONNX
                ort_sess = ort.InferenceSession(onnx_path)
                # CatBoost ONNX has input name 'features' or similar, let's query the input name dynamically
                input_name = ort_sess.get_inputs()[0].name
                onnx_outputs = ort_sess.run(None, {input_name: dummy_input})
                # CatBoost ONNX outputs probabilities at index 1 or index 0 depending on format
                # Usually it has output names: ['label', 'probabilities']
                probs_output = onnx_outputs[1]
                onnx_pred = probs_output[:, 1]
            else:
                return "unsupported", "unchecked"
                
        elif model_type == "sequence":
            seq_len = metadata["sequence_length"]
            n_features = metadata["num_features"]
            
            # Load PyTorch model
            model = Simple1DCNN(seq_len, n_features)
            model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
            model.eval()
            
            dummy_input_torch = torch.randn(100, seq_len, n_features)
            # Standard single batch dummy input for export
            torch.onnx.export(
                model,
                dummy_input_torch[0:1],
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
