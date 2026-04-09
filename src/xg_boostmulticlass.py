import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD AND PREPARE DATA 
# =========================================================
print("Loading datasets...")

# TODO: Replace these with your actual file paths (e.g., '../data/qcd.csv' or '.parquet')
PATH_QCD = r"D:\SPARSH\particle\MOE\Mixture-of-Experts-for-multiclass-classification-in-Particle-Physics\data\raw\background_full.csv"
PATH_TAUTAU = r"D:\SPARSH\particle\MOE\Mixture-of-Experts-for-multiclass-classification-in-Particle-Physics\data\raw\signalA_train.csv"
PATH_EE = r"D:\SPARSH\particle\MOE\Mixture-of-Experts-for-multiclass-classification-in-Particle-Physics\data\raw\signalB_train.csv"

# Load the files
df_qcd = pd.read_csv(PATH_QCD)
df_tautau = pd.read_csv(PATH_TAUTAU)
df_ee = pd.read_csv(PATH_EE)

# Assign multiclass labels: 
# 0 = QCD Background, 1 = Higgs TauTau, 2 = Signal ee
df_qcd['label'] = 0
df_tautau['label'] = 1
df_ee['label'] = 2

# Concatenate all three into the single master dataframe 'df'
df = pd.concat([df_qcd, df_tautau, df_ee], ignore_index=True)

# Shuffle the combined dataset to prevent sequential bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define your exact 6D kinematic feature vector
features = ['pt', 'px', 'py', 'pz', 'e', 'mass'] 

X = df[features].values
y = df['label'].values 

# =========================================================
# 2. PRE-PROCESSING (Matching MoE Standards)
# =========================================================
print("Splitting and scaling data...")

# Split the data (80% train, 20% test - matching standard MoE splits)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler (crucial for a 1-to-1 comparison with MoE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 3. TRAIN THE XGBOOST MODEL
# =========================================================
print("Training XGBoost Multiclass Model (this might take a minute)...")

# Initialize the model
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    max_depth=6,           
    learning_rate=0.1,     
    n_estimators=200,      
    random_state=42
)

# Fit the model
xgb_model.fit(X_train_scaled, y_train)

# =========================================================
# 4. EVALUATION & METRICS
# =========================================================
print("Predicting on Test Set...")
y_pred = xgb_model.predict(X_test_scaled)

target_names = ['QCD Background', 'Higgs TauTau', 'Signal ee']

# Print Classification Report
print("\n" + "="*50)
print(" XGBOOST CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=target_names))

# Print Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n" + "="*50)
print(" XGBOOST CONFUSION MATRIX")
print("="*50)
print(cm)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
plt.title('XGBoost Baseline Confusion Matrix')
plt.tight_layout()
plt.show()