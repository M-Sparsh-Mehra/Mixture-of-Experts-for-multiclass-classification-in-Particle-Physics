import joblib
import pandas as pd
import numpy as np

# --- Configurations ---
MODEL_PATH = "models/frocc_weights/sorter.pkl"
BG_DATA_PATH = "data/raw/background_val.csv"
SIG_DATA_PATH = "data/raw/signal_val.csv"
FEATURES = ["pt", "px", "py", "pz", "E", "mass"]

def verify():
    print("--- Sorter Verification ---")
    
    # 1. Load the Model
    try:
        clf = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded successfully from {MODEL_PATH}")
        print(f"     Loaded Threshold: {clf.threshold:.4f}")
    except Exception as e:
        print(f"[FAIL] Could not load model: {e}")
        return

    # 2. Load a tiny sample of data (5 rows each)
    try:
        df_bg = pd.read_csv(BG_DATA_PATH).head(5)
        df_sig = pd.read_csv(SIG_DATA_PATH).head(5)
        
        X_bg = df_bg[FEATURES].values.astype(np.float32)
        X_sig = df_sig[FEATURES].values.astype(np.float32)
        print("[OK] Loaded 5 Background and 5 Signal test events.")
    except Exception as e:
        print(f"[FAIL] Could not load data: {e}")
        return

    # 3. Test the Decision Function (Raw Scores)
    print("\n--- Raw Anomaly Scores (Lower = More Anomalous) ---")
    scores_bg = clf.decision_function(X_bg)
    scores_sig = clf.decision_function(X_sig)
    
    print(f"Background Scores: {np.round(scores_bg, 4)}")
    print(f"Signal Scores:     {np.round(scores_sig, 4)}")

    # 4. Test the Predict Logic (Boolean Mask)
    # Remember: predict() returns True if it's an INLIER (Background)
    # So we invert it (~) to see what gets passed to the Experts.
    print("\n--- Pipeline Routing Logic ---")
    
    keep_bg = ~clf.predict(X_bg)
    keep_sig = ~clf.predict(X_sig)
    
    print(f"Backgrounds passed to Experts: {keep_bg} (Ideally mostly False)")
    print(f"Signals passed to Experts:     {keep_sig} (Ideally ALL True)")
    print("\n[SUCCESS] Verification complete.")

if __name__ == "__main__":
    verify()