"""
Trains a single PyTorch MLP Expert to distinguish ONE specific physics signal 
from the Standard Model background. 

This script ensures every expert is trained in isolation (mutually exclusive) 
and automatically applies Temperature Scaling at the end. This calibration step 
guarantees that when the expert joins the MoE pipeline, its confidence scores 
are statistically aligned with all other experts.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
# Ensure the src module is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.experts import MLPExpert

# --- Custom Dataset for LHC CSVs ---
class LHCDataset(Dataset):
    def __init__(self, bg_path, sig_path, feature_cols):
        # Load data
        df_bg = pd.read_csv(bg_path)[feature_cols].copy()
        df_sig = pd.read_csv(sig_path)[feature_cols].copy()
        
        # Labels: Background = 0.0, Signal = 1.0
        df_bg['label'] = 0.0
        df_sig['label'] = 1.0
        
        # Combine and shuffle
        df_combined = pd.concat([df_bg, df_sig]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(df_combined[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df_combined['label'].values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Temperature Calibration Logic ---
def calibrate_temperature(model, val_loader, device):
    """
    Learns a single scalar (Temperature) on the validation set to smooth out 
    overconfident logits.
    """
    print("\n--- Starting Temperature Calibration ---")
    model.eval()
    
    # We create a single parameter to optimize
    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()
    
    # Collect all validation logits and labels
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch) # Use raw forward pass
            all_logits.append(logits)
            all_labels.append(y_batch)
            
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Optimization closure for LBFGS
    def eval_loss():
        optimizer.zero_grad()
        # Apply temperature scaling: Logits / T
        scaled_logits = all_logits / temperature
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    
    optimal_t = temperature.item()
    print(f"✅ Calibration complete. Optimal Temperature: {optimal_t:.4f}")
    return optimal_t

# --- Main Training Script ---
def main():
    parser = argparse.ArgumentParser(description="Train a single LHC Signal Expert")
    parser.add_argument("--signal_name", type=str, required=True, help="Name of the signal (e.g., Higgs)")
    parser.add_argument("--bg_csv", type=str, required=True, help="Path to Background CSV")
    parser.add_argument("--sig_csv", type=str, required=True, help="Path to Signal CSV")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    feature_cols = [
    "deta_part1","deta_part2","deta_part3","deta_part4","deta_part5","deta_part6","deta_part7","deta_part8","deta_part9","deta_part10",
    "deta_part11","deta_part12","deta_part13","deta_part14","deta_part15","deta_part16","deta_part17","deta_part18","deta_part19","deta_part20",
    "deta_part21","deta_part22","deta_part23","deta_part24","deta_part25","deta_part26","deta_part27","deta_part28","deta_part29","deta_part30",
    "deta_part31","deta_part32","deta_part33","deta_part34","deta_part35","deta_part36","deta_part37","deta_part38","deta_part39","deta_part40",
    "deta_part41","deta_part42","deta_part43","deta_part44","deta_part45","deta_part46","deta_part47","deta_part48","deta_part49","deta_part50",
    "deta_part51","deta_part52","deta_part53","deta_part54","deta_part55","deta_part56","deta_part57","deta_part58","deta_part59","deta_part60",
    "deta_part61","deta_part62","deta_part63","deta_part64","deta_part65","deta_part66","deta_part67","deta_part68","deta_part69","deta_part70",
    "deta_part71","deta_part72","deta_part73","deta_part74","deta_part75","deta_part76","deta_part77","deta_part78","deta_part79","deta_part80",

    "dphi_part1","dphi_part2","dphi_part3","dphi_part4","dphi_part5","dphi_part6","dphi_part7","dphi_part8","dphi_part9","dphi_part10",
    "dphi_part11","dphi_part12","dphi_part13","dphi_part14","dphi_part15","dphi_part16","dphi_part17","dphi_part18","dphi_part19","dphi_part20",
    "dphi_part21","dphi_part22","dphi_part23","dphi_part24","dphi_part25","dphi_part26","dphi_part27","dphi_part28","dphi_part29","dphi_part30",
    "dphi_part31","dphi_part32","dphi_part33","dphi_part34","dphi_part35","dphi_part36","dphi_part37","dphi_part38","dphi_part39","dphi_part40",
    "dphi_part41","dphi_part42","dphi_part43","dphi_part44","dphi_part45","dphi_part46","dphi_part47","dphi_part48","dphi_part49","dphi_part50",
    "dphi_part51","dphi_part52","dphi_part53","dphi_part54","dphi_part55","dphi_part56","dphi_part57","dphi_part58","dphi_part59","dphi_part60",
    "dphi_part61","dphi_part62","dphi_part63","dphi_part64","dphi_part65","dphi_part66","dphi_part67","dphi_part68","dphi_part69","dphi_part70",
    "dphi_part71","dphi_part72","dphi_part73","dphi_part74","dphi_part75","dphi_part76","dphi_part77","dphi_part78","dphi_part79","dphi_part80",

    "log_pt_part1","log_pt_part2","log_pt_part3","log_pt_part4","log_pt_part5","log_pt_part6","log_pt_part7","log_pt_part8","log_pt_part9","log_pt_part10",
    "log_pt_part11","log_pt_part12","log_pt_part13","log_pt_part14","log_pt_part15","log_pt_part16","log_pt_part17","log_pt_part18","log_pt_part19","log_pt_part20",
    "log_pt_part21","log_pt_part22","log_pt_part23","log_pt_part24","log_pt_part25","log_pt_part26","log_pt_part27","log_pt_part28","log_pt_part29","log_pt_part30",
    "log_pt_part31","log_pt_part32","log_pt_part33","log_pt_part34","log_pt_part35","log_pt_part36","log_pt_part37","log_pt_part38","log_pt_part39","log_pt_part40",
    "log_pt_part41","log_pt_part42","log_pt_part43","log_pt_part44","log_pt_part45","log_pt_part46","log_pt_part47","log_pt_part48","log_pt_part49","log_pt_part50",
    "log_pt_part51","log_pt_part52","log_pt_part53","log_pt_part54","log_pt_part55","log_pt_part56","log_pt_part57","log_pt_part58","log_pt_part59","log_pt_part60",
    "log_pt_part61","log_pt_part62","log_pt_part63","log_pt_part64","log_pt_part65","log_pt_part66","log_pt_part67","log_pt_part68","log_pt_part69","log_pt_part70",
    "log_pt_part71","log_pt_part72","log_pt_part73","log_pt_part74","log_pt_part75","log_pt_part76","log_pt_part77","log_pt_part78","log_pt_part79","log_pt_part80",

    "log_E_part1","log_E_part2","log_E_part3","log_E_part4","log_E_part5","log_E_part6","log_E_part7","log_E_part8","log_E_part9","log_E_part10",
    "log_E_part11","log_E_part12","log_E_part13","log_E_part14","log_E_part15","log_E_part16","log_E_part17","log_E_part18","log_E_part19","log_E_part20",
    "log_E_part21","log_E_part22","log_E_part23","log_E_part24","log_E_part25","log_E_part26","log_E_part27","log_E_part28","log_E_part29","log_E_part30",
    "log_E_part31","log_E_part32","log_E_part33","log_E_part34","log_E_part35","log_E_part36","log_E_part37","log_E_part38","log_E_part39","log_E_part40",
    "log_E_part41","log_E_part42","log_E_part43","log_E_part44","log_E_part45","log_E_part46","log_E_part47","log_E_part48","log_E_part49","log_E_part50",
    "log_E_part51","log_E_part52","log_E_part53","log_E_part54","log_E_part55","log_E_part56","log_E_part57","log_E_part58","log_E_part59","log_E_part60",
    "log_E_part61","log_E_part62","log_E_part63","log_E_part64","log_E_part65","log_E_part66","log_E_part67","log_E_part68","log_E_part69","log_E_part70",
    "log_E_part71","log_E_part72","log_E_part73","log_E_part74","log_E_part75","log_E_part76","log_E_part77","log_E_part78","log_E_part79","log_E_part80",

    "log_pt_rel_part1","log_pt_rel_part2","log_pt_rel_part3","log_pt_rel_part4","log_pt_rel_part5","log_pt_rel_part6","log_pt_rel_part7","log_pt_rel_part8","log_pt_rel_part9","log_pt_rel_part10",
    "log_pt_rel_part11","log_pt_rel_part12","log_pt_rel_part13","log_pt_rel_part14","log_pt_rel_part15","log_pt_rel_part16","log_pt_rel_part17","log_pt_rel_part18","log_pt_rel_part19","log_pt_rel_part20",
    "log_pt_rel_part21","log_pt_rel_part22","log_pt_rel_part23","log_pt_rel_part24","log_pt_rel_part25","log_pt_rel_part26","log_pt_rel_part27","log_pt_rel_part28","log_pt_rel_part29","log_pt_rel_part30",
    "log_pt_rel_part31","log_pt_rel_part32","log_pt_rel_part33","log_pt_rel_part34","log_pt_rel_part35","log_pt_rel_part36","log_pt_rel_part37","log_pt_rel_part38","log_pt_rel_part39","log_pt_rel_part40",
    "log_pt_rel_part41","log_pt_rel_part42","log_pt_rel_part43","log_pt_rel_part44","log_pt_rel_part45","log_pt_rel_part46","log_pt_rel_part47","log_pt_rel_part48","log_pt_rel_part49","log_pt_rel_part50",
    "log_pt_rel_part51","log_pt_rel_part52","log_pt_rel_part53","log_pt_rel_part54","log_pt_rel_part55","log_pt_rel_part56","log_pt_rel_part57","log_pt_rel_part58","log_pt_rel_part59","log_pt_rel_part60",
    "log_pt_rel_part61","log_pt_rel_part62","log_pt_rel_part63","log_pt_rel_part64","log_pt_rel_part65","log_pt_rel_part66","log_pt_rel_part67","log_pt_rel_part68","log_pt_rel_part69","log_pt_rel_part70",
    "log_pt_rel_part71","log_pt_rel_part72","log_pt_rel_part73","log_pt_rel_part74","log_pt_rel_part75","log_pt_rel_part76","log_pt_rel_part77","log_pt_rel_part78","log_pt_rel_part79","log_pt_rel_part80",

    "log_E_rel_part1","log_E_rel_part2","log_E_rel_part3","log_E_rel_part4","log_E_rel_part5","log_E_rel_part6","log_E_rel_part7","log_E_rel_part8","log_E_rel_part9","log_E_rel_part10",
    "log_E_rel_part11","log_E_rel_part12","log_E_rel_part13","log_E_rel_part14","log_E_rel_part15","log_E_rel_part16","log_E_rel_part17","log_E_rel_part18","log_E_rel_part19","log_E_rel_part20",
    "log_E_rel_part21","log_E_rel_part22","log_E_rel_part23","log_E_rel_part24","log_E_rel_part25","log_E_rel_part26","log_E_rel_part27","log_E_rel_part28","log_E_rel_part29","log_E_rel_part30",
    "log_E_rel_part31","log_E_rel_part32","log_E_rel_part33","log_E_rel_part34","log_E_rel_part35","log_E_rel_part36","log_E_rel_part37","log_E_rel_part38","log_E_rel_part39","log_E_rel_part40",
    "log_E_rel_part41","log_E_rel_part42","log_E_rel_part43","log_E_rel_part44","log_E_rel_part45","log_E_rel_part46","log_E_rel_part47","log_E_rel_part48","log_E_rel_part49","log_E_rel_part50",
    "log_E_rel_part51","log_E_rel_part52","log_E_rel_part53","log_E_rel_part54","log_E_rel_part55","log_E_rel_part56","log_E_rel_part57","log_E_rel_part58","log_E_rel_part59","log_E_rel_part60",
    "log_E_rel_part61","log_E_rel_part62","log_E_rel_part63","log_E_rel_part64","log_E_rel_part65","log_E_rel_part66","log_E_rel_part67","log_E_rel_part68","log_E_rel_part69","log_E_rel_part70",
    "log_E_rel_part71","log_E_rel_part72","log_E_rel_part73","log_E_rel_part74","log_E_rel_part75","log_E_rel_part76","log_E_rel_part77","log_E_rel_part78","log_E_rel_part79","log_E_rel_part80",

    "delta_R_part1","delta_R_part2","delta_R_part3","delta_R_part4","delta_R_part5","delta_R_part6","delta_R_part7","delta_R_part8","delta_R_part9","delta_R_part10",
    "delta_R_part11","delta_R_part12","delta_R_part13","delta_R_part14","delta_R_part15","delta_R_part16","delta_R_part17","delta_R_part18","delta_R_part19","delta_R_part20",
    "delta_R_part21","delta_R_part22","delta_R_part23","delta_R_part24","delta_R_part25","delta_R_part26","delta_R_part27","delta_R_part28","delta_R_part29","delta_R_part30",
    "delta_R_part31","delta_R_part32","delta_R_part33","delta_R_part34","delta_R_part35","delta_R_part36","delta_R_part37","delta_R_part38","delta_R_part39","delta_R_part40",
    "delta_R_part41","delta_R_part42","delta_R_part43","delta_R_part44","delta_R_part45","delta_R_part46","delta_R_part47","delta_R_part48","delta_R_part49","delta_R_part50",
    "delta_R_part51","delta_R_part52","delta_R_part53","delta_R_part54","delta_R_part55","delta_R_part56","delta_R_part57","delta_R_part58","delta_R_part59","delta_R_part60",
    "delta_R_part61","delta_R_part62","delta_R_part63","delta_R_part64","delta_R_part65","delta_R_part66","delta_R_part67","delta_R_part68","delta_R_part69","delta_R_part70",
    "delta_R_part71","delta_R_part72","delta_R_part73","delta_R_part74","delta_R_part75","delta_R_part76","delta_R_part77","delta_R_part78","delta_R_part79","delta_R_part80",

    "charge_part1","charge_part2","charge_part3","charge_part4","charge_part5","charge_part6","charge_part7","charge_part8","charge_part9","charge_part10",
    "charge_part11","charge_part12","charge_part13","charge_part14","charge_part15","charge_part16","charge_part17","charge_part18","charge_part19","charge_part20",
    "charge_part21","charge_part22","charge_part23","charge_part24","charge_part25","charge_part26","charge_part27","charge_part28","charge_part29","charge_part30",
    "charge_part31","charge_part32","charge_part33","charge_part34","charge_part35","charge_part36","charge_part37","charge_part38","charge_part39","charge_part40",
    "charge_part41","charge_part42","charge_part43","charge_part44","charge_part45","charge_part46","charge_part47","charge_part48","charge_part49","charge_part50",
    "charge_part51","charge_part52","charge_part53","charge_part54","charge_part55","charge_part56","charge_part57","charge_part58","charge_part59","charge_part60",
    "charge_part61","charge_part62","charge_part63","charge_part64","charge_part65","charge_part66","charge_part67","charge_part68","charge_part69","charge_part70",
    "charge_part71","charge_part72","charge_part73","charge_part74","charge_part75","charge_part76","charge_part77","charge_part78","charge_part79","charge_part80",

    "isElectron_part1","isElectron_part2","isElectron_part3","isElectron_part4","isElectron_part5","isElectron_part6","isElectron_part7","isElectron_part8","isElectron_part9","isElectron_part10",
    "isElectron_part11","isElectron_part12","isElectron_part13","isElectron_part14","isElectron_part15","isElectron_part16","isElectron_part17","isElectron_part18","isElectron_part19","isElectron_part20",
    "isElectron_part21","isElectron_part22","isElectron_part23","isElectron_part24","isElectron_part25","isElectron_part26","isElectron_part27","isElectron_part28","isElectron_part29","isElectron_part30",
    "isElectron_part31","isElectron_part32","isElectron_part33","isElectron_part34","isElectron_part35","isElectron_part36","isElectron_part37","isElectron_part38","isElectron_part39","isElectron_part40",
    "isElectron_part41","isElectron_part42","isElectron_part43","isElectron_part44","isElectron_part45","isElectron_part46","isElectron_part47","isElectron_part48","isElectron_part49","isElectron_part50",
    "isElectron_part51","isElectron_part52","isElectron_part53","isElectron_part54","isElectron_part55","isElectron_part56","isElectron_part57","isElectron_part58","isElectron_part59","isElectron_part60",
    "isElectron_part61","isElectron_part62","isElectron_part63","isElectron_part64","isElectron_part65","isElectron_part66","isElectron_part67","isElectron_part68","isElectron_part69","isElectron_part70",
    "isElectron_part71","isElectron_part72","isElectron_part73","isElectron_part74","isElectron_part75","isElectron_part76","isElectron_part77","isElectron_part78","isElectron_part79","isElectron_part80",

    "isMuon_part1","isMuon_part2","isMuon_part3","isMuon_part4","isMuon_part5","isMuon_part6","isMuon_part7","isMuon_part8","isMuon_part9","isMuon_part10",
    "isMuon_part11","isMuon_part12","isMuon_part13","isMuon_part14","isMuon_part15","isMuon_part16","isMuon_part17","isMuon_part18","isMuon_part19","isMuon_part20",
    "isMuon_part21","isMuon_part22","isMuon_part23","isMuon_part24","isMuon_part25","isMuon_part26","isMuon_part27","isMuon_part28","isMuon_part29","isMuon_part30",
    "isMuon_part31","isMuon_part32","isMuon_part33","isMuon_part34","isMuon_part35","isMuon_part36","isMuon_part37","isMuon_part38","isMuon_part39","isMuon_part40",
    "isMuon_part41","isMuon_part42","isMuon_part43","isMuon_part44","isMuon_part45","isMuon_part46","isMuon_part47","isMuon_part48","isMuon_part49","isMuon_part50",
    "isMuon_part51","isMuon_part52","isMuon_part53","isMuon_part54","isMuon_part55","isMuon_part56","isMuon_part57","isMuon_part58","isMuon_part59","isMuon_part60",
    "isMuon_part61","isMuon_part62","isMuon_part63","isMuon_part64","isMuon_part65","isMuon_part66","isMuon_part67","isMuon_part68","isMuon_part69","isMuon_part70",
    "isMuon_part71","isMuon_part72","isMuon_part73","isMuon_part74","isMuon_part75","isMuon_part76","isMuon_part77","isMuon_part78","isMuon_part79","isMuon_part80",

    "isPhoton_part1","isPhoton_part2","isPhoton_part3","isPhoton_part4","isPhoton_part5","isPhoton_part6","isPhoton_part7","isPhoton_part8","isPhoton_part9","isPhoton_part10",
    "isPhoton_part11","isPhoton_part12","isPhoton_part13","isPhoton_part14","isPhoton_part15","isPhoton_part16","isPhoton_part17","isPhoton_part18","isPhoton_part19","isPhoton_part20",
    "isPhoton_part21","isPhoton_part22","isPhoton_part23","isPhoton_part24","isPhoton_part25","isPhoton_part26","isPhoton_part27","isPhoton_part28","isPhoton_part29","isPhoton_part30",
    "isPhoton_part31","isPhoton_part32","isPhoton_part33","isPhoton_part34","isPhoton_part35","isPhoton_part36","isPhoton_part37","isPhoton_part38","isPhoton_part39","isPhoton_part40",
    "isPhoton_part41","isPhoton_part42","isPhoton_part43","isPhoton_part44","isPhoton_part45","isPhoton_part46","isPhoton_part47","isPhoton_part48","isPhoton_part49","isPhoton_part50",
    "isPhoton_part51","isPhoton_part52","isPhoton_part53","isPhoton_part54","isPhoton_part55","isPhoton_part56","isPhoton_part57","isPhoton_part58","isPhoton_part59","isPhoton_part60",
    "isPhoton_part61","isPhoton_part62","isPhoton_part63","isPhoton_part64","isPhoton_part65","isPhoton_part66","isPhoton_part67","isPhoton_part68","isPhoton_part69","isPhoton_part70",
    "isPhoton_part71","isPhoton_part72","isPhoton_part73","isPhoton_part74","isPhoton_part75","isPhoton_part76","isPhoton_part77","isPhoton_part78","isPhoton_part79","isPhoton_part80",

    "isCH_part1","isCH_part2","isCH_part3","isCH_part4","isCH_part5","isCH_part6","isCH_part7","isCH_part8","isCH_part9","isCH_part10",
    "isCH_part11","isCH_part12","isCH_part13","isCH_part14","isCH_part15","isCH_part16","isCH_part17","isCH_part18","isCH_part19","isCH_part20",
    "isCH_part21","isCH_part22","isCH_part23","isCH_part24","isCH_part25","isCH_part26","isCH_part27","isCH_part28","isCH_part29","isCH_part30",
    "isCH_part31","isCH_part32","isCH_part33","isCH_part34","isCH_part35","isCH_part36","isCH_part37","isCH_part38","isCH_part39","isCH_part40",
    "isCH_part41","isCH_part42","isCH_part43","isCH_part44","isCH_part45","isCH_part46","isCH_part47","isCH_part48","isCH_part49","isCH_part50",
    "isCH_part51","isCH_part52","isCH_part53","isCH_part54","isCH_part55","isCH_part56","isCH_part57","isCH_part58","isCH_part59","isCH_part60",
    "isCH_part61","isCH_part62","isCH_part63","isCH_part64","isCH_part65","isCH_part66","isCH_part67","isCH_part68","isCH_part69","isCH_part70",
    "isCH_part71","isCH_part72","isCH_part73","isCH_part74","isCH_part75","isCH_part76","isCH_part77","isCH_part78","isCH_part79","isCH_part80",

    "isNH_part1","isNH_part2","isNH_part3","isNH_part4","isNH_part5","isNH_part6","isNH_part7","isNH_part8","isNH_part9","isNH_part10",
    "isNH_part11","isNH_part12","isNH_part13","isNH_part14","isNH_part15","isNH_part16","isNH_part17","isNH_part18","isNH_part19","isNH_part20",
    "isNH_part21","isNH_part22","isNH_part23","isNH_part24","isNH_part25","isNH_part26","isNH_part27","isNH_part28","isNH_part29","isNH_part30",
    "isNH_part31","isNH_part32","isNH_part33","isNH_part34","isNH_part35","isNH_part36","isNH_part37","isNH_part38","isNH_part39","isNH_part40",
    "isNH_part41","isNH_part42","isNH_part43","isNH_part44","isNH_part45","isNH_part46","isNH_part47","isNH_part48","isNH_part49","isNH_part50",
    "isNH_part51","isNH_part52","isNH_part53","isNH_part54","isNH_part55","isNH_part56","isNH_part57","isNH_part58","isNH_part59","isNH_part60",
    "isNH_part61","isNH_part62","isNH_part63","isNH_part64","isNH_part65","isNH_part66","isNH_part67","isNH_part68","isNH_part69","isNH_part70",
    "isNH_part71","isNH_part72","isNH_part73","isNH_part74","isNH_part75","isNH_part76","isNH_part77","isNH_part78","isNH_part79","isNH_part80",

    "tanh_d0_part1","tanh_d0_part2","tanh_d0_part3","tanh_d0_part4","tanh_d0_part5","tanh_d0_part6","tanh_d0_part7","tanh_d0_part8","tanh_d0_part9","tanh_d0_part10",
    "tanh_d0_part11","tanh_d0_part12","tanh_d0_part13","tanh_d0_part14","tanh_d0_part15","tanh_d0_part16","tanh_d0_part17","tanh_d0_part18","tanh_d0_part19","tanh_d0_part20",
    "tanh_d0_part21","tanh_d0_part22","tanh_d0_part23","tanh_d0_part24","tanh_d0_part25","tanh_d0_part26","tanh_d0_part27","tanh_d0_part28","tanh_d0_part29","tanh_d0_part30",
    "tanh_d0_part31","tanh_d0_part32","tanh_d0_part33","tanh_d0_part34","tanh_d0_part35","tanh_d0_part36","tanh_d0_part37","tanh_d0_part38","tanh_d0_part39","tanh_d0_part40",
    "tanh_d0_part41","tanh_d0_part42","tanh_d0_part43","tanh_d0_part44","tanh_d0_part45","tanh_d0_part46","tanh_d0_part47","tanh_d0_part48","tanh_d0_part49","tanh_d0_part50",
    "tanh_d0_part51","tanh_d0_part52","tanh_d0_part53","tanh_d0_part54","tanh_d0_part55","tanh_d0_part56","tanh_d0_part57","tanh_d0_part58","tanh_d0_part59","tanh_d0_part60",
    "tanh_d0_part61","tanh_d0_part62","tanh_d0_part63","tanh_d0_part64","tanh_d0_part65","tanh_d0_part66","tanh_d0_part67","tanh_d0_part68","tanh_d0_part69","tanh_d0_part70",
    "tanh_d0_part71","tanh_d0_part72","tanh_d0_part73","tanh_d0_part74","tanh_d0_part75","tanh_d0_part76","tanh_d0_part77","tanh_d0_part78","tanh_d0_part79","tanh_d0_part80",

    "tanh_dz_part1","tanh_dz_part2","tanh_dz_part3","tanh_dz_part4","tanh_dz_part5","tanh_dz_part6","tanh_dz_part7","tanh_dz_part8","tanh_dz_part9","tanh_dz_part10",
    "tanh_dz_part11","tanh_dz_part12","tanh_dz_part13","tanh_dz_part14","tanh_dz_part15","tanh_dz_part16","tanh_dz_part17","tanh_dz_part18","tanh_dz_part19","tanh_dz_part20",
    "tanh_dz_part21","tanh_dz_part22","tanh_dz_part23","tanh_dz_part24","tanh_dz_part25","tanh_dz_part26","tanh_dz_part27","tanh_dz_part28","tanh_dz_part29","tanh_dz_part30",
    "tanh_dz_part31","tanh_dz_part32","tanh_dz_part33","tanh_dz_part34","tanh_dz_part35","tanh_dz_part36","tanh_dz_part37","tanh_dz_part38","tanh_dz_part39","tanh_dz_part40",
    "tanh_dz_part41","tanh_dz_part42","tanh_dz_part43","tanh_dz_part44","tanh_dz_part45","tanh_dz_part46","tanh_dz_part47","tanh_dz_part48","tanh_dz_part49","tanh_dz_part50",
    "tanh_dz_part51","tanh_dz_part52","tanh_dz_part53","tanh_dz_part54","tanh_dz_part55","tanh_dz_part56","tanh_dz_part57","tanh_dz_part58","tanh_dz_part59","tanh_dz_part60",
    "tanh_dz_part61","tanh_dz_part62","tanh_dz_part63","tanh_dz_part64","tanh_dz_part65","tanh_dz_part66","tanh_dz_part67","tanh_dz_part68","tanh_dz_part69","tanh_dz_part70",
    "tanh_dz_part71","tanh_dz_part72","tanh_dz_part73","tanh_dz_part74","tanh_dz_part75","tanh_dz_part76","tanh_dz_part77","tanh_dz_part78","tanh_dz_part79","tanh_dz_part80",

    "sigma_d0_part1","sigma_d0_part2","sigma_d0_part3","sigma_d0_part4","sigma_d0_part5","sigma_d0_part6","sigma_d0_part7","sigma_d0_part8","sigma_d0_part9","sigma_d0_part10",
    "sigma_d0_part11","sigma_d0_part12","sigma_d0_part13","sigma_d0_part14","sigma_d0_part15","sigma_d0_part16","sigma_d0_part17","sigma_d0_part18","sigma_d0_part19","sigma_d0_part20",
    "sigma_d0_part21","sigma_d0_part22","sigma_d0_part23","sigma_d0_part24","sigma_d0_part25","sigma_d0_part26","sigma_d0_part27","sigma_d0_part28","sigma_d0_part29","sigma_d0_part30",
    "sigma_d0_part31","sigma_d0_part32","sigma_d0_part33","sigma_d0_part34","sigma_d0_part35","sigma_d0_part36","sigma_d0_part37","sigma_d0_part38","sigma_d0_part39","sigma_d0_part40",
    "sigma_d0_part41","sigma_d0_part42","sigma_d0_part43","sigma_d0_part44","sigma_d0_part45","sigma_d0_part46","sigma_d0_part47","sigma_d0_part48","sigma_d0_part49","sigma_d0_part50",
    "sigma_d0_part51","sigma_d0_part52","sigma_d0_part53","sigma_d0_part54","sigma_d0_part55","sigma_d0_part56","sigma_d0_part57","sigma_d0_part58","sigma_d0_part59","sigma_d0_part60",
    "sigma_d0_part61","sigma_d0_part62","sigma_d0_part63","sigma_d0_part64","sigma_d0_part65","sigma_d0_part66","sigma_d0_part67","sigma_d0_part68","sigma_d0_part69","sigma_d0_part70",
    "sigma_d0_part71","sigma_d0_part72","sigma_d0_part73","sigma_d0_part74","sigma_d0_part75","sigma_d0_part76","sigma_d0_part77","sigma_d0_part78","sigma_d0_part79","sigma_d0_part80",

    "sigma_dz_part1","sigma_dz_part2","sigma_dz_part3","sigma_dz_part4","sigma_dz_part5","sigma_dz_part6","sigma_dz_part7","sigma_dz_part8","sigma_dz_part9","sigma_dz_part10",
    "sigma_dz_part11","sigma_dz_part12","sigma_dz_part13","sigma_dz_part14","sigma_dz_part15","sigma_dz_part16","sigma_dz_part17","sigma_dz_part18","sigma_dz_part19","sigma_dz_part20",
    "sigma_dz_part21","sigma_dz_part22","sigma_dz_part23","sigma_dz_part24","sigma_dz_part25","sigma_dz_part26","sigma_dz_part27","sigma_dz_part28","sigma_dz_part29","sigma_dz_part30",
    "sigma_dz_part31","sigma_dz_part32","sigma_dz_part33","sigma_dz_part34","sigma_dz_part35","sigma_dz_part36","sigma_dz_part37","sigma_dz_part38","sigma_dz_part39","sigma_dz_part40",
    "sigma_dz_part41","sigma_dz_part42","sigma_dz_part43","sigma_dz_part44","sigma_dz_part45","sigma_dz_part46","sigma_dz_part47","sigma_dz_part48","sigma_dz_part49","sigma_dz_part50",
    "sigma_dz_part51","sigma_dz_part52","sigma_dz_part53","sigma_dz_part54","sigma_dz_part55","sigma_dz_part56","sigma_dz_part57","sigma_dz_part58","sigma_dz_part59","sigma_dz_part60",
    "sigma_dz_part61","sigma_dz_part62","sigma_dz_part63","sigma_dz_part64","sigma_dz_part65","sigma_dz_part66","sigma_dz_part67","sigma_dz_part68","sigma_dz_part69","sigma_dz_part70",
    "sigma_dz_part71","sigma_dz_part72","sigma_dz_part73","sigma_dz_part74","sigma_dz_part75","sigma_dz_part76","sigma_dz_part77","sigma_dz_part78","sigma_dz_part79","sigma_dz_part80"
]
    
    #  Prepare Data
    print(f"\n[1/4] Loading Data for {args.signal_name} Expert...")
    dataset = LHCDataset(args.bg_csv, args.sig_csv, feature_cols)
    
    # Simple 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    #  Initialize Model
    print(f"\n[2/4] Initializing MLP Expert...")
    model = MLPExpert(input_dim=len(feature_cols), hidden_dim=64, signal_name=args.signal_name).to(device)
    
    # BCEWithLogitsLoss is required because our model outputs raw logits, not probabilities
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #  Training Loop
    print(f"\n[3/4] Training Model for {args.epochs} Epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

    # Calibration & Save
    print(f"\n[4/4] Finalizing and Saving...")
    optimal_temp = calibrate_temperature(model, val_loader, device)
    model.temperature = optimal_temp

    # Ensure save directory exists
    os.makedirs("models/expert_weights", exist_ok=True)
    save_path = f"models/expert_weights/expert_{args.signal_name}.pt"
    
    # Save the PyTorch state dictionary PLUS our custom temperature
    torch.save({
        'model_state_dict': model.state_dict(),
        'temperature': optimal_temp,
        'signal_name': args.signal_name
    }, save_path)
    
    print(f"✅ Successfully saved expert to {save_path}")

if __name__ == "__main__":
    main()