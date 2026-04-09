"""
Runs the full LHC Mixture of Experts pipeline on real data using trained models.
Loads the Scikit-Learn Sorter, the PyTorch Experts, and outputs the final classifications.
"""

import os
import sys
import torch
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import FEATURES, EXPERT_CONFIG, FROCC_WEIGHTS_DIR, EXPERTS
from src.sorter_wrapper import FROCCWrapper
from src.experts import MLPExpert
from src.pipeline import LHCDynamicMoE

def load_trained_expert(filepath, input_dim):
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
    print("      LHC MoE Pipeline: Real Data Inference")
    print("==================================================")

    print("[1/4] Loading Stage I Sorter...")
    sorter = FROCCWrapper(model_path=os.path.join(PROJECT_ROOT, FROCC_WEIGHTS_DIR, "sorter.pkl"))

    print(f"[2/4] Dynamically Loading {len(EXPERTS)} Stage II Experts...")
    loaded_experts = []
    for expert_info in EXPERTS:
        expert_path = os.path.join(PROJECT_ROOT, expert_info["model_path"])
        loaded_experts.append(load_trained_expert(expert_path, input_dim=EXPERT_CONFIG["input_dim"]))
    
    print("[3/4] Assembling the Orchestrator...")
    pipeline = LHCDynamicMoE(sorter=sorter, experts=loaded_experts, bg_logit=0.0)
    
    print(f"\n[4/4] Loading 10 rows of real test data (from {EXPERTS[0]['id']})...")
    test_csv = os.path.join(PROJECT_ROOT, EXPERTS[0]["test_data"]) 
    df = pd.read_csv(test_csv).head(10)
    x_tensor = torch.tensor(df[FEATURES].values, dtype=torch.float32)

    # --- INFERENCE ---
    with torch.no_grad():
        results = pipeline.process_batch(x_tensor)

    # --- DISPLAY RESULTS ---
    print("\n==================================================")
    print("                PIPELINE OUTPUT")
    print("==================================================")
    
    probs = results["probabilities"]
    keep_mask = results["kept_mask"]
    classes = results["classes"]
    
    header = f"{'Row':<5} | {'Sorter Passed?':<15} | " + " | ".join([f"{c:<12}" for c in classes])
    print(header)
    print("-" * len(header))
    
    for i in range(len(df)):
        passed = "YES (Signal)" if keep_mask[i].item() else "NO (Bg)"
        p_str = " | ".join([f"{probs[i, j].item():<12.3f}" for j in range(len(classes))])
        print(f"{i:<5} | {passed:<15} | {p_str}")

if __name__ == "__main__":
    main()