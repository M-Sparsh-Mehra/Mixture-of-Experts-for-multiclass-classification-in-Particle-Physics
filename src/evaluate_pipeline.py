import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import FEATURES, DATA_PATHS, FROCC_WEIGHTS_DIR, EXPERTS, EXPERT_CONFIG
from src.sorter_wrapper import FROCCWrapper
from src.experts import MLPExpert
from src.pipeline import LHCDynamicMoE

def load_trained_expert(filepath, input_dim):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expert not found at {filepath}")
    
    checkpoint = torch.load(filepath, weights_only=False)
    
    expert = MLPExpert(input_dim=input_dim, signal_name=checkpoint["signal_name"])
    
    expert.load_state_dict(checkpoint["model_state_dict"])
    expert.temperature = checkpoint["temperature"]
    expert.eval()
    
    return expert

def main():
    print("==================================================")
    print("    LHC MoE Pipeline: Validation Evaluation")
    print("==================================================")

    # 1. Load the Sorter
    print("[1/4] Loading Sorter...")
    sorter = FROCCWrapper(model_path=os.path.join(PROJECT_ROOT, FROCC_WEIGHTS_DIR, "sorter.pkl"))
    
    # 2. Dynamically Load ALL Experts in the Registry
    print(f"[2/4] Loading {len(EXPERTS)} Stage II Experts...")
    loaded_experts = []
    for expert_info in EXPERTS:
        expert_path = os.path.join(PROJECT_ROOT, expert_info["model_path"])
        loaded_experts.append(load_trained_expert(expert_path, input_dim=EXPERT_CONFIG["input_dim"]))    
        
    pipeline = LHCDynamicMoE(sorter=sorter, experts=loaded_experts, bg_logit=0.0)
    class_names = pipeline.class_names
    print(f" -> Pipeline classes configured: {class_names}")

    # 3. Dynamically Build the Validation Dataset
    print("\n[3/4] Building Unified Validation Dataset...")
    df_list = []
    
    # A. Add Background
    df_bg = pd.read_csv(os.path.join(PROJECT_ROOT, DATA_PATHS["background_qcd_val"]))[FEATURES]
    df_bg['true_label'] = class_names.index("Background")
    df_list.append(df_bg)
    
    # B. Add All Signals from the Registry
    for i, expert_info in enumerate(EXPERTS):
        df_sig = pd.read_csv(os.path.join(PROJECT_ROOT, expert_info["val_data"]))[FEATURES]
        # Match the ground truth label to the exact name the PyTorch model has internally
        actual_signal_name = loaded_experts[i].signal_name
        df_sig['true_label'] = class_names.index(actual_signal_name)
        df_list.append(df_sig)

    # Combine into one massive tensor
    df_all = pd.concat(df_list, ignore_index=True)
    x_tensor = torch.tensor(df_all[FEATURES].values, dtype=torch.float32)
    y_true = df_all['true_label'].values

    # 4. Run Pipeline & Evaluate
    print(f"\n[4/4] Running Pipeline on {len(df_all)} events...")
    with torch.no_grad():
        results = pipeline.process_batch(x_tensor)
        
    y_pred = torch.argmax(results["probabilities"], dim=1).cpu().numpy()

    print("\n==================================================")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in class_names], columns=[f"Pred {c}" for c in class_names])
    print(cm_df.to_string())

if __name__ == "__main__":
    main()