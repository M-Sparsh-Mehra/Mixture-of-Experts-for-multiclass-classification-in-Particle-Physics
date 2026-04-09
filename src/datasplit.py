import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================
# SYSTEM SETUP
# ==========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import config

# ==========================
# CONFIG
# ==========================

RAW_DIR = config.BASE_DATA_DIR 
TRAIN_RATIO = 0.8
SEED = 42

# ==========================
# MAIN
# ==========================
def main():
    print("==================================================")
    print("   LHC Data Splitting Protocol")
    print("==================================================")

    # Find all CSVs that haven't been split yet
    files = [
        f for f in os.listdir(RAW_DIR)
        if os.path.isfile(os.path.join(RAW_DIR, f))
        and f.endswith(".csv")
        and not f.startswith(("train_", "val_"))  # avoid re-splitting
    ]

    if not files:
        print("No original files found to split. (Already split?)")
        return

    for file in files:
        file_path = os.path.join(RAW_DIR, file)
        
        print(f"Reading: {file}...")
        df = pd.read_csv(file_path)

        train_df, val_df = train_test_split(
            df,
            train_size=TRAIN_RATIO,
            random_state=SEED,
            shuffle=True
        )

        train_path = os.path.join(RAW_DIR, f"train_{file}")
        val_path = os.path.join(RAW_DIR, f"val_{file}")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        
        print(
            f" -> SUCCESS: {file} \n"
            f"    Train: {len(train_df)} rows \n"
            f"    Val:   {len(val_df)} rows\n"
        )

if __name__ == "__main__":
    main()