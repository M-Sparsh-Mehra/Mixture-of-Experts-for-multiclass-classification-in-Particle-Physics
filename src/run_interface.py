"""
Runs the full LHC Mixture of Experts pipeline on real data using trained models.
Loads the Scikit-Learn Sorter, the PyTorch Experts, and outputs the final classifications.
"""

import os
import sys
import torch
import pandas as pd

# Point Python to the root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.sorter_wrapper import FROCCWrapper
from src.experts import MLPExpert
from src.pipeline import LHCDynamicMoE

def load_trained_expert(filepath, input_dim=6):
    """Helper to load the PyTorch weights and the custom temperature."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expert not found at {filepath}")
        
    checkpoint = torch.load(filepath, weights_only=False)
    
    # Initialize the architecture
    expert = MLPExpert(input_dim=input_dim, signal_name=checkpoint['signal_name'])
    
    # Load the trained weights
    expert.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply the calibrated temperature
    expert.temperature = checkpoint['temperature']
    expert.eval() # Set to inference mode
    
    return expert

def main():
    print("==================================================")
    print("      LHC MoE Pipeline: Real Data Inference")
    print("==================================================")

    # 1. Load the Sorter
    print("[1/4] Loading Stage I Sorter...")
    sorter = FROCCWrapper(model_path=os.path.join(PROJECT_ROOT, "models/frocc_weights/sorter.pkl"))

    # 2. Load the Experts
    print("[2/4] Loading Stage II Trained Experts...")
    # NOTE: Change the filenames below if your second signal is named something else!
    expert_tau = load_trained_expert(os.path.join(PROJECT_ROOT, "models/expert_weights/expert_Tau_signal.pt"))
    
    # REPLACE "expert_Signal_B.pt" with whatever you named your second signal's file
    expert_electron = load_trained_expert(os.path.join(PROJECT_ROOT, "models/expert_weights/expert_electron_signal.pt")) 
    
    # 3. Assemble Pipeline
    print("[3/4] Assembling the Orchestrator...")
    pipeline = LHCDynamicMoE(sorter=sorter, experts=[expert_tau,expert_electron], bg_logit=0.0)
    
    # 4. Load Real Test Data
    print("\n[4/4] Loading 10 rows of real test data...")
    feature_cols = ["pt", "px", "py", "pz", "E", "mass"]
    
    # Let's load a mix of data just to see how it reacts
    # Assuming you have some signal data to test it on
    test_csv = os.path.join(PROJECT_ROOT, "data/raw/signalA_train.csv") 
    df = pd.read_csv(test_csv).head(10) # Grab the first 10 rows
    
    # Convert to PyTorch Tensor
    x_tensor = torch.tensor(df[feature_cols].values, dtype=torch.float32)

    # --- INFERENCE ---
    with torch.no_grad():
        results = pipeline.process_batch(x_tensor)

    # --- DISPLAY RESULTS ---
    print("\n==================================================")
    print("                 PIPELINE OUTPUT")
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