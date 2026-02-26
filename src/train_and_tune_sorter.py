import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

# --- Import strictly DFROCC ---
# Ensure dfrocc.py is saved inside the src/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dfrocc import DFROCC 

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Paths to your data
    "path_train_bg": "data/raw/background_train.csv",  
    "path_val_bg": "data/raw/background_val.csv",      
    "path_val_signal": "data/raw/signal_val.csv",      
    
    # Output paths
    "model_save_path": "models/frocc_weights/sorter.pkl",
    "config_save_path": "config/sorter_config.json",
    
    # Physics Constraints
    "target_recall": 0.995,  
    "feature_cols": ["pt", "px", "py", "pz", "E", "mass"], 
    
    "frocc_params": {
        "num_clf_dim": 50,     # Increased from 10 to look harder
        "epsilon": 0.05,       # Tightened from 0.1 to catch smaller deviations
        "bin_factor": 2,
        "threshold": 1.0,
        "kernel": linear_kernel
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
    print(f"\n[1/4] Loading Data...")
    X_train_bg = load_data(CONFIG["path_train_bg"], CONFIG["feature_cols"])
    X_val_bg = load_data(CONFIG["path_val_bg"], CONFIG["feature_cols"])
    X_val_signal = load_data(CONFIG["path_val_signal"], CONFIG["feature_cols"])

    if X_train_bg is None: return

    print(f" -> Training Background Events: {X_train_bg.shape[0]}")
    if X_val_signal is not None:
        print(f" -> Validation Signal Events:   {X_val_signal.shape[0]}")
    
    # --------------------------------------------------------
    # 2. TRAIN DFROCC
    # --------------------------------------------------------
    print(f"\n[2/4] Training Standard DFROCC on Pure Background...")
    
    
    # clf = DFROCC(**CONFIG["frocc_params"])
    
    print(f"\n[2/4] Training Scaler + DFROCC Pipeline...")
    clf = Pipeline([
        ('scaler', StandardScaler()), # Forces all kinematics to N(0,1)
        ('frocc', DFROCC(**CONFIG["frocc_params"]))
    ])
    clf.fit(X_train_bg)
    print(" -> Model fitted successfully.")

    # --------------------------------------------------------
    # TUNE THRESHOLD
    # --------------------------------------------------------
    print(f"\n[3/4] Tuning Threshold for Target Recall: {CONFIG['target_recall']*100}%...")
    
    scores_bg = clf.decision_function(X_val_bg)
    scores_sig = clf.decision_function(X_val_signal)
    
    y_true = np.concatenate([np.zeros(len(scores_bg)), np.ones(len(scores_sig))])
    y_scores = np.concatenate([scores_bg, scores_sig])
    
    best_threshold = 0.0
    best_recall = 0.0
    best_fpr = 1.0 
    
    thresholds_to_test = np.linspace(0.0, 1.0, 200)
    
    for t in thresholds_to_test:
        # If agreement score < t, it is classified as Signal (1)
        preds = (y_scores <= t).astype(int)
        current_recall = recall_score(y_true, preds)
        
        if current_recall >= CONFIG["target_recall"]:
            bg_kept = (scores_bg < t).sum()
            current_fpr = bg_kept / len(scores_bg)
            
            best_threshold = t
            best_recall = current_recall
            best_fpr = current_fpr
            break 
            
    print(f" -> Calibration complete.")
    print(f"    Selected Threshold: {best_threshold:.4f}")
    print(f"    Actual Recall:      {best_recall:.4f}")
    print(f"    Background Pass Rate: {best_fpr:.2%}") 

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------
    print(f"\n[4/4] Saving System State...")
    
    clf.threshold = best_threshold
    
    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
    joblib.dump(clf, CONFIG["model_save_path"])
    
    final_config = {
        "threshold": float(best_threshold),
        "target_recall": CONFIG["target_recall"],
        "achieved_recall": float(best_recall),
        "fpr": float(best_fpr),
        "features": CONFIG["feature_cols"]
    }
    
    os.makedirs(os.path.dirname(CONFIG["config_save_path"]), exist_ok=True)
    with open(CONFIG["config_save_path"], 'w') as f:
        json.dump(final_config, f, indent=4)
        
    print(f" -> Model saved to: {CONFIG['model_save_path']}")
    print(f" -> Config saved to: {CONFIG['config_save_path']}")
    print("\n[SUCCESS] Sorter is ready for the pipeline.")

if __name__ == "__main__":
    main()