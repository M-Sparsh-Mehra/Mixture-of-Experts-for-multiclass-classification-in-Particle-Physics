"""
Integration test for the LHC Dynamic Mixture of Experts pipeline.
Validates that Stage I (Scikit-Learn/FROCC) and Stage II (PyTorch/MLPs) 
can pass tensors to each other and compute the final Softmax probabilities without crashing.
"""

import torch
import sys
import os

# Ensure the project root is accessible
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Pull dimensions and paths strictly from the config!
from config import EXPERT_CONFIG, FROCC_WEIGHTS_DIR
from src.sorter_wrapper import FROCCWrapper
from src.experts import MLPExpert
from src.pipeline import LHCDynamicMoE

def main():
    print("==================================================")
    print("      LHC MoE Pipeline: End-to-End Test")
    print("==================================================")

    # 1. Load the real Stage I Sorter (The Bouncer)
    print("[1/4] Loading Stage I: FROCC Sorter...")
    sorter_path = os.path.join(PROJECT_ROOT, FROCC_WEIGHTS_DIR, "sorter.pkl")
    try:
        sorter = FROCCWrapper(model_path=sorter_path)
    except Exception as e:
        print(f"❌ Failed to load Sorter. Have you trained it yet?\nError: {e}")
        return

    # 2. Create Dummy Stage II Experts (The VIP Room)
    # We dynamically pull the input dimension (e.g., 1360) from config
    input_dim = EXPERT_CONFIG["input_dim"]
    print(f"[2/4] Initializing Stage II: Untrained PyTorch Experts ({input_dim}D)...")
    
    # Expert 1: Test Signal A
    # (hidden_dim is removed because experts.py now pulls layers automatically from config!)
    expert_a = MLPExpert(input_dim=input_dim, signal_name="Test_Signal_A")
    expert_a.temperature = 1.5 # Simulating a calibrated temperature
    expert_a.eval() # Set to evaluation mode
    
    # Expert 2: Test Signal B
    expert_b = MLPExpert(input_dim=input_dim, signal_name="Test_Signal_B")
    expert_b.temperature = 0.8
    expert_b.eval()

    # 3. Assemble the Pipeline
    print("[3/4] Assembling the Orchestrator...")
    # bg_logit=0.0 means if the experts output negative logits, Background wins.
    pipeline = LHCDynamicMoE(sorter=sorter, experts=[expert_a, expert_b], bg_logit=0.0)
    
    print(f" -> Active Classes: {pipeline.class_names}")

    # 4. Generate Dummy LHC Data
    print(f"\n[4/4] Generating 10 Dummy LHC Events and pushing through pipeline...")
    # Shape dynamically matches config: (10 events, 1360 features)
    dummy_batch = torch.randn(10, input_dim, dtype=torch.float32)

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