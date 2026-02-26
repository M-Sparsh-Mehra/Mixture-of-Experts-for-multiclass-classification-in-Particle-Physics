"""

Evaluates the full LHC MoE pipeline on unseen validation datasets.
Calculates system-wide accuracy, precision, and recall by combining 
Background and multiple Signal validation sets.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.sorter_wrapper import FROCCWrapper
from src.experts import MLPExpert
from src.pipeline import LHCDynamicMoE

def load_trained_expert(filepath, input_dim=6):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expert not found at {filepath}")
    checkpoint = torch.load(filepath, weights_only=False)
    expert = MLPExpert(input_dim=input_dim, signal_name=checkpoint['signal_name'])
    expert.load_state_dict(checkpoint['model_state_dict'])
    expert.temperature = checkpoint['temperature']
    expert.eval()
    return expert

def main():
    print("==================================================")
    print("    LHC MoE Pipeline: Validation Evaluation")
    print("==================================================")

    # 1. Load the Models
    print("[1/4] Loading Sorter and Experts...")
    sorter = FROCCWrapper(model_path=os.path.join(PROJECT_ROOT, "models/frocc_weights/sorter.pkl"))
    
    # --- UPDATE THESE FILENAMES IF NEEDED ---
    expert_tau = load_trained_expert(os.path.join(PROJECT_ROOT, "models/expert_weights/expert_Tau_signal.pt"))
    expert_electron = load_trained_expert(os.path.join(PROJECT_ROOT, "models/expert_weights/expert_electron_signal.pt")) 
    
    pipeline = LHCDynamicMoE(sorter=sorter, experts=[expert_tau, expert_electron], bg_logit=0.0)
    class_names = pipeline.class_names # e.g., ['Tau_signal', 'electron_signal', 'Background']
    print(f" -> Pipeline classes: {class_names}")

    # 2. Load Validation Data & Create Ground Truth Labels
    print("\n[2/4] Loading Validation Datasets...")
    feature_cols = ["pt", "px", "py", "pz", "E", "mass"]
    
    # Load CSVs (Update paths to match your actual val files)
    df_bg = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/background_val.csv"))[feature_cols]
    df_tau = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/signalA_val.csv"))[feature_cols]
    df_el = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/signalB_val.csv"))[feature_cols] # Update this!

    # Assign Ground Truth Integers based on pipeline.class_names index
    # If class_names = ['Tau_signal', 'electron_signal', 'Background']
    # Then Tau=0, Signal_B=1, Background=2
    idx_tau = class_names.index("Tau_signal")
    idx_el = class_names.index(expert_electron.signal_name)
    idx_bg = class_names.index("Background")

    df_bg['true_label'] = idx_bg
    df_tau['true_label'] = idx_tau
    df_el['true_label'] = idx_el

    # Combine into one massive validation set
    df_all = pd.concat([df_bg, df_tau, df_el], ignore_index=True)
    
    x_tensor = torch.tensor(df_all[feature_cols].values, dtype=torch.float32)
    y_true = df_all['true_label'].values

    #  Run Pipeline
    print(f"\n[3/4] Running Pipeline on {len(df_all)} events... (This is the real test!)")
    with torch.no_grad():
        results = pipeline.process_batch(x_tensor)
        
    probs = results["probabilities"]
    
    # The final prediction is the class with the highest probability
    y_pred = torch.argmax(probs, dim=1).cpu().numpy()

    #  Print Metrics
    print("\n[4/4] Evaluation Results:")
    print("==================================================")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in class_names], columns=[f"Pred {c}" for c in class_names])
    print(cm_df.to_string())

if __name__ == "__main__":
    main()