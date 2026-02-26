"""
Integration test for the LHC Dynamic Mixture of Experts pipeline.
Validates that Stage I (Scikit-Learn/FROCC) and Stage II (PyTorch/MLPs) 
can pass tensors to each other and compute the final Softmax probabilities without crashing.
"""

import torch
import sys
import os

# Ensure the src module is accessible
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.sorter_wrapper import FROCCWrapper
from src.experts import MLPExpert
from src.pipeline import LHCDynamicMoE

def main():
    print("==================================================")
    print("      LHC MoE Pipeline: End-to-End Test")
    print("==================================================")

    # 1. Load the real Stage I Sorter (The Bouncer)
    print("[1/4] Loading Stage I: FROCC Sorter...")
    try:
        sorter = FROCCWrapper(model_path="models/frocc_weights/sorter.pkl")
    except Exception as e:
        print(f"❌ Failed to load Sorter: {e}")
        return

    # 2. Create Dummy Stage II Experts (The VIP Room)
    # We use 6 features to match your CSV (pt, px, py, pz, E, mass)
    print("[2/4] Initializing Stage II: Untrained PyTorch Experts...")
    
    # Expert 1: Higgs Boson
    expert_higgs = MLPExpert(input_dim=6, hidden_dim=16, signal_name="Higgs")
    expert_higgs.temperature = 1.5 # Simulating a calibrated temperature
    expert_higgs.eval() # Set to evaluation mode
    
    # Expert 2: Dark Matter
    expert_dm = MLPExpert(input_dim=6, hidden_dim=16, signal_name="Dark_Matter")
    expert_dm.temperature = 0.8
    expert_dm.eval()

    # 3. Assemble the Pipeline
    print("[3/4] Assembling the Orchestrator...")
    # bg_logit=0.0 means if the experts output negative logits, Background wins.
    pipeline = LHCDynamicMoE(sorter=sorter, experts=[expert_higgs, expert_dm], bg_logit=0.0)
    
    print(f" -> Active Classes: {pipeline.class_names}")

    # 4. Generate Dummy LHC Data
    print("\n[4/4] Generating 10 Dummy LHC Events and pushing through pipeline...")
    # Shape: (10 events, 6 features)
    dummy_batch = torch.randn(10, 6, dtype=torch.float32)

    # --- THE MAIN EVENT ---
    # Run the batch through the pipeline
    with torch.no_grad(): # No gradients needed for inference
        results = pipeline.process_batch(dummy_batch)

    # --- DISPLAY RESULTS ---
    print("\n==================================================")
    print("                 PIPELINE OUTPUT")
    print("==================================================")
    
    probs = results["probabilities"]
    keep_mask = results["kept_mask"]
    classes = results["classes"]
    
    # Print a clean table
    header = f"{'Event':<7} | {'Sorter Passed?':<15} | " + " | ".join([f"{c:<12}" for c in classes])
    print(header)
    print("-" * len(header))
    
    for i in range(10):
        passed = "YES (Signal)" if keep_mask[i].item() else "NO (Bg)"
        
        # Format probabilities to 3 decimal places
        p_str = " | ".join([f"{probs[i, j].item():<12.3f}" for j in range(len(classes))])
        
        print(f"Row {i:<3} | {passed:<15} | {p_str}")

if __name__ == "__main__":
    main()