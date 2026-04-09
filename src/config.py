# config.py
import os

# ==========================================
# DIRECTORY PATHS
# ==========================================
# relative paths from the project root for data, models, and results
BASE_DATA_DIR = r"D:\SPARSH\particle\MOE\Mixture-of-Experts-for-multiclass-classification-in-Particle-Physics\data\new raw"
FROCC_WEIGHTS_DIR = r"D:\SPARSH\particle\MOE\Mixture-of-Experts-for-multiclass-classification-in-Particle-Physics\models\frocc_weights"   
EXPERT_WEIGHTS_DIR = r"D:\SPARSH\particle\MOE\Mixture-of-Experts-for-multiclass-classification-in-Particle-Physics\models\expert_weights" 
RESULTS_DIR = r"D:\SPARSH\particle\MOE\Mixture-of-Experts-for-multiclass-classification-in-Particle-Physics\results"

# ==========================================
# DATASET FILES
# ==========================================
DATA_PATHS = {
    "background_qcd_train": os.path.join(BASE_DATA_DIR, "train_bg_QCD.csv"),
    "background_qcd_val": os.path.join(BASE_DATA_DIR, "val_bg_QCD.csv"),
    "expert_1_train": os.path.join(BASE_DATA_DIR, "train_TTBarLep_120.csv"),
    "expert_1_val": os.path.join(BASE_DATA_DIR, "val_TTBarLep_120.csv"),
    "expert_2_train": os.path.join(BASE_DATA_DIR, "train_WtoQQ_120.csv"),
    "expert_2_val": os.path.join(BASE_DATA_DIR, "val_WToQQ_120.csv"),
    "val_signal": os.path.join(BASE_DATA_DIR, "val_signal.csv")
}

# ==========================================
# FEATURE ENGINEERING 
# ==========================================
NUM_PARTICLES = 80

# The 17 core properties per particle
# ==========================================
# FEATURE ENGINEERING 
# ==========================================
NUM_PARTICLES = 80

# The 17 core properties per particle
BASE_FEATURES = [
    "deta", "dphi", "log_pt", "log_E", 
    "log_pt_rel", "log_E_rel", "delta_R", "charge",
    "isElectron", "isMuon", "isPhoton", "isCH", "isNH",
    "tanh_d0", "tanh_dz", "sigma_d0", "sigma_dz"
]


FEATURES = []
for feature in BASE_FEATURES:
    for i in range(1, NUM_PARTICLES + 1):
        FEATURES.append(f"{feature}_part{i}")



# ==========================================
#  STAGE I: DFROCC HYPERPARAMETERS
# ==========================================
SORTER_CONFIG = {
    "target_recall": 0.995,
    "num_clf_dim": 50,
    "epsilon": 0.05,
    "bin_factor": 2,
    "threshold_default": 1.0 
}

# ==========================================
# STAGE II: EXPERT NEURAL NETWORK PARAMS
# ==========================================
EXPERT_CONFIG = {
    "input_dim": len(FEATURES),
    "hidden_layers": [512, 256, 128],
    "dropout_rate": 0.3,
    "leaky_relu_alpha": 0.01,
    "batch_size": 256,
    "learning_rate": 0.001,
    "epochs": 50
}

# ==========================================
# EXPERTS 
# ==========================================
# Add new signals here to automatically inject them into the entire pipeline
EXPERTS = [
    {
        "id": "ttlep",
        "model_path": os.path.join(EXPERT_WEIGHTS_DIR, "expert_ttlep.pt"),
        "val_data": os.path.join(BASE_DATA_DIR, "val_TTBarLep_120.csv"),
        "test_data": os.path.join(BASE_DATA_DIR, "val_TTBarLep_120.csv") # Used for the inference.py sample
    },
    {
        "id": "htogg",
        "model_path": os.path.join(EXPERT_WEIGHTS_DIR, "expert_htogg.pt"),
        "val_data": os.path.join(BASE_DATA_DIR, "val_HToGG.csv"),
        "test_data": os.path.join(BASE_DATA_DIR, "val_HToGG.csv")
    }
]