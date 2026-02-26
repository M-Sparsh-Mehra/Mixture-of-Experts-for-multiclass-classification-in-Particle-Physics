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

    feature_cols = ["pt", "px", "py", "pz", "E", "mass"]
    
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