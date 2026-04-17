import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import recall_score
import config

# --- Import strictly DFROCC ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dfrocc import DFROCC 

# ==========================================
# DATA PREPARATION
# ==========================================
# Instead of hardcoding the signal combining step, we check if it exists.
# If not, we build it dynamically using the paths from config.py!
if not os.path.exists(config.DATA_PATHS["val_signal"]):
    print("Combined validation signal not found. Generating it now...")
    sigA = pd.read_csv(config.DATA_PATHS["expert_1_val"])
    sigB = pd.read_csv(config.DATA_PATHS["expert_2_val"])

    # 20% holdout
    valA = sigA.sample(frac=0.2, random_state=42)
    valB = sigB.sample(frac=0.2, random_state=42)

    val_signal = pd.concat([valA, valB], ignore_index=True)
    val_signal.to_csv(config.DATA_PATHS["val_signal"], index=False)

# ==========================================
#  LOCAL CONFIGURATION MAPPING
# ==========================================
#map the imported config to a local dictionary 
LOCAL_CONFIG = {
    "path_train_bg": config.DATA_PATHS["background_qcd_train"],  
    "path_val_bg": config.DATA_PATHS["background_qcd_val"],      
    "path_val_signal": config.DATA_PATHS["val_signal"],      
    
    # Output paths
    "model_save_path": os.path.join(config.FROCC_WEIGHTS_DIR, "sorter.pkl"),
    "config_save_path": "config/sorter_config.json",
    
    # Physics Constraints & Features pulled dynamically
    "target_recall": config.SORTER_CONFIG["target_recall"],  
    "feature_cols": config.FEATURES,  # <--- ALL 1,360 FEATURES IN ONE LINE!
    
    "frocc_params": {
        "num_clf_dim": config.SORTER_CONFIG["num_clf_dim"],     
        "epsilon": config.SORTER_CONFIG["epsilon"],       
        "bin_factor": config.SORTER_CONFIG["bin_factor"],
        "threshold": config.SORTER_CONFIG["threshold_default"],
        "kernel": linear_kernel # Linear kernel function passed directly
    }
}

def load_data(path, cols):
    if not os.path.exists(path):
        print(f"!! Error: Data file not found at {path}")
        return None
    df = pd.read_csv(path)
    # DFROCC precision defaults to float32
    return df[cols].values.astype(np.float32)

def main():
    print("==================================================")
    print("   LHC Sorter Automation: Train & Tune Protocol")
    print("==================================================")

    # --------------------------------------------------------
    #  LOAD DATA
    # --------------------------------------------------------
    print(f"\n[1/4] Loading Data (Features: {len(LOCAL_CONFIG['feature_cols'])}D)...")
    X_train_bg = load_data(LOCAL_CONFIG["path_train_bg"], LOCAL_CONFIG["feature_cols"])
    X_val_bg = load_data(LOCAL_CONFIG["path_val_bg"], LOCAL_CONFIG["feature_cols"])
    X_val_signal = load_data(LOCAL_CONFIG["path_val_signal"], LOCAL_CONFIG["feature_cols"])

    if X_train_bg is None: return

    print(f" -> Training Background Events: {X_train_bg.shape[0]}")
    if X_val_signal is not None:
        print(f" -> Validation Signal Events:   {X_val_signal.shape[0]}")
    
    # --------------------------------------------------------
    # 2. TRAIN DFROCC
    # --------------------------------------------------------
    print(f"\n[2/4] Training Scaler + DFROCC Pipeline...")
    clf = Pipeline([
        ('scaler', RobustScaler()), # Protects zero-padded sparsity
        ('frocc', DFROCC(**LOCAL_CONFIG["frocc_params"]))
    ])
    clf.fit(X_train_bg)
    print(" -> Model fitted successfully.")

    # --------------------------------------------------------
    # TUNE THRESHOLD
    # --------------------------------------------------------
    print(f"\n[3/4] Tuning Threshold for Target Recall: {LOCAL_CONFIG['target_recall']*100}%...")
    
    scores_bg = clf.decision_function(X_val_bg)
    scores_sig = clf.decision_function(X_val_signal)


    ###############################################33
    y_true = np.concatenate([np.zeros(len(scores_bg)), np.ones(len(scores_sig))])
    y_scores = np.concatenate([scores_bg, scores_sig])
    
    best_threshold = 0.0
    best_recall = 0.0
    best_fpr = 1.0 
    
    # --- THE FIX: DYNAMIC SEARCH GRID ---
    # Find the actual min and max anomaly scores produced by the 1360D model
    min_score = np.min(y_scores)
    max_score = np.max(y_scores)
    
    print(f"    -> Dynamic Score Range Detected: [{min_score:.4f} to {max_score:.4f}]")
    
    # Scan 1,000 steps between the actual minimum and maximum scores
    thresholds_to_test = np.linspace(min_score, max_score, 1000)
    

    ##########################
    # Prevent the scanner from ever testing exactly 1.0 (which breaks the logic)
    # Use 5000 steps to handle the microscopic differences in 1360D space
    thresholds_to_test = np.linspace(min_score, max_score - 1e-6, 5000)
    
    best_threshold = max_score - 1e-6 # Safe fallback


    for t in thresholds_to_test:
        # If agreement score < t, it is classified as Signal (1)
        preds = (y_scores <= t).astype(int)
        current_recall = recall_score(y_true, preds)
        
        if current_recall >= LOCAL_CONFIG["target_recall"]:
            bg_kept = (scores_bg <= t).sum()
            current_fpr = bg_kept / len(scores_bg)
            
            best_threshold = t
            best_recall = current_recall
            best_fpr = current_fpr
            break 
            
    # Calculate the fraction rejected
    bg_removed_fraction = 1.0 - best_fpr
    
    print(f" -> Calibration complete.")
    print(f"    Selected Threshold: {best_threshold:.5f}")
    print(f"    Actual Signal Recall: {best_recall:.4f}")
    print(f"    Background Pass Rate: {best_fpr:.2%}") 
    print(f"    Background REMOVED:   {bg_removed_fraction:.2%}") 

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------
    print(f"\n[4/4] Saving System State...")
    
    clf.threshold = best_threshold
    
    os.makedirs(os.path.dirname(LOCAL_CONFIG["model_save_path"]), exist_ok=True)
    joblib.dump(clf, LOCAL_CONFIG["model_save_path"])
    
    final_config = {
        "threshold": float(best_threshold),
        "target_recall": LOCAL_CONFIG["target_recall"],
        "achieved_recall": float(best_recall),
        "fpr": float(best_fpr),
        "bg_removed_fraction": float(bg_removed_fraction),
        "features": LOCAL_CONFIG["feature_cols"]
    }
    
    os.makedirs(os.path.dirname(LOCAL_CONFIG["config_save_path"]), exist_ok=True)
    with open(LOCAL_CONFIG["config_save_path"], 'w') as f:
        json.dump(final_config, f, indent=4)
        
    print(f" -> Model saved to: {LOCAL_CONFIG['model_save_path']}")
    print(f" -> Config saved to: {LOCAL_CONFIG['config_save_path']}")
    print("\n[SUCCESS] Sorter is ready for the pipeline.")

if __name__ == "__main__":
    main()