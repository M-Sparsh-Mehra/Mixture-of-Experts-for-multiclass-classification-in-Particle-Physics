import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================
# CONFIG
# ==========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "new raw")
TRAIN_RATIO = 0.8
SEED = 42

# ==========================
# MAIN
# ==========================

files = [
    f for f in os.listdir(RAW_DIR)
    if os.path.isfile(os.path.join(RAW_DIR, f))
    and not f.startswith(("train_", "val_"))  # avoid re-splitting
]

if not files:
    raise ValueError("No original files found to split")

for file in files:
    file_path = os.path.join(RAW_DIR, file)
    name, ext = os.path.splitext(file)

    if ext.lower() != ".csv":
        print(f"Skipping {file} (not CSV)")
        continue

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

    #  REMOVE ORIGINAL FILE
    os.remove(file_path)

    print(
        f"{file} → "
        f"train_{file} ({len(train_df)} rows), "
        f"val_{file} ({len(val_df)} rows)"
    )