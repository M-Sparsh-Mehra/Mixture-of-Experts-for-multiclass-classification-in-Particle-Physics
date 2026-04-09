"""
This file defines the PyTorch architecture for Stage II of the pipeline: The Signal Experts.
Each expert is a Multi-Layer Perceptron (MLP) trained independently to identify ONE specific physics signal.

By keeping the model definition here, we ensure that every new signal we add to the system 
(Signal A, Signal B, etc.) uses the exact same underlying structure. 
It also enforces the Temperature Scaling logic required by the `BaseExpert` contract, 
ensuring no single model unfairly dominates the final Softmax decision.
"""

import torch
import torch.nn as nn
from .base import BaseExpert

# Pull our network hyperparameters directly from the Single Source of Truth
from config import EXPERT_CONFIG

class MLPExpert(BaseExpert, nn.Module):
    def __init__(self, input_dim: int, signal_name: str = "Unknown"):
        """
        Initializes the PyTorch Neural Network dynamically based on config.py.
        """
        # Initialize both parent classes (The Contract + PyTorch Module)
        BaseExpert.__init__(self, signal_name=signal_name)
        nn.Module.__init__(self)
        
        # Load hyperparameters from config
        hidden_dims = EXPERT_CONFIG["hidden_layers"]
        dropout_rate = EXPERT_CONFIG["dropout_rate"]
        alpha = EXPERT_CONFIG["leaky_relu_alpha"]

        # Dynamically build the network layers
        layers = []
        
        # 1. Input Normalization (CRITICAL for 1360D zero-padded data)
        # This replaces StandardScaler and learns the best scaling per-batch
        layers.append(nn.BatchNorm1d(input_dim))
        
        # 2. Build the hidden layers dynamically
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LeakyReLU(alpha))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim  # Update the input dimension for the next layer
            
        # 3. Final Output Node
        layers.append(nn.Linear(current_dim, 1)) # Crucial: NO Sigmoid here.

        # Unpack the list into a PyTorch Sequential container
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard PyTorch forward pass. 
        Returns the raw, uncalibrated logit.
        """
        return self.net(x)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fulfills the BaseExpert contract.
        Takes the raw output and applies Temperature Scaling to ensure 
        this expert's confidence is mathematically calibrated with all other experts.
        """
        raw_logits = self.forward(x)
        
        # Temperature scaling: dampens or boosts the logit based on how "overconfident" the model is
        calibrated_logits = raw_logits / self.temperature
        
        return calibrated_logits