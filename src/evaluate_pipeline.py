import os
import sys
import json
import torch
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Make sure RESULTS_DIR is imported from config!
from config import FEATURES, DATA_PATHS, FROCC_WEIGHTS_DIR, EXPERTS, EXPERT_CONFIG, RESULTS_DIR
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
    print("    LHC MoE Pipeline: Unseen Test Evaluation")
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

    # 3. Dynamically Build the TEST Dataset
    print("\n[3/4] Building Unified UNSEEN TEST Dataset...")
    df_list = []
    
    # A. Add Background (Using TEST data)
    df_bg = pd.read_csv(os.path.join(PROJECT_ROOT, DATA_PATHS["background_qcd_test"]))[FEATURES]
    df_bg['true_label'] = class_names.index("Background")
    df_list.append(df_bg)
    
    # B. Add All Signals from the Registry (Using TEST data)
    for i, expert_info in enumerate(EXPERTS):
        df_sig = pd.read_csv(os.path.join(PROJECT_ROOT, expert_info["test_data"]))[FEATURES]
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
    keep_mask = results["kept_mask"].cpu().numpy()

    # ==========================================================
    # METRICS & JSON EXPORT
    # ==========================================================
    print("\n==================================================")
    print("      STAGE I: PER-CLASS REJECTION STATS")
    print("==================================================")
    
    rejection_stats = {}
    
    for class_name in class_names:
        class_idx = class_names.index(class_name)
        is_class = (y_true == class_idx)
        
        total_class_events = np.sum(is_class)
        kept_class_events = np.sum(keep_mask & is_class)
        rejected_class_events = total_class_events - kept_class_events
        
        # Avoid division by zero
        rejection_rate = (rejected_class_events / total_class_events) if total_class_events > 0 else 0.0
        
        rejection_stats[class_name] = {
            "total_events": int(total_class_events),
            "passed_stage_1": int(kept_class_events),
            "rejected_stage_1": int(rejected_class_events),
            "rejection_rate": float(rejection_rate)
        }
        
        # Print it nicely to the console
        print(f"[{class_name}]")
        print(f"  Total: {total_class_events} | Passed: {kept_class_events} | Rejected: {rejected_class_events}")
        print(f"  Rejection Rate: {rejection_rate:.4%}\n")

    print("==================================================")
    print("      STAGE II: CLASSIFICATION REPORT")
    print("==================================================")
    
    # Print string version for the terminal
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Get dict version for the JSON export
    report_dict = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in class_names], columns=[f"Pred {c}" for c in class_names])
    print(cm_df.to_string())

    # --- COMPILE AND SAVE JSON ---
    final_output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_config": {
            "num_features": len(FEATURES),
            "sorter_threshold": float(sorter.threshold)
        },
        "stage_1_rejection_stats": rejection_stats,
        "stage_2_classification_report": report_dict,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist() # JSON can't serialize numpy arrays directly
        }
    }
    
    # Ensure results directory exists
    os.makedirs(os.path.join(PROJECT_ROOT, RESULTS_DIR), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, RESULTS_DIR, "pipeline_evaluation_results.json")
    
    with open(save_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n✅ All metrics successfully exported to: {save_path}")

if __name__ == "__main__":
    main()