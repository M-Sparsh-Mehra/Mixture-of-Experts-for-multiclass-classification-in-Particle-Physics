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

class MLPExpert(BaseExpert, nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, signal_name: str = "Unknown"):
        """
        Initializes the PyTorch Neural Network for a specific signal.
        """
        # Initialize both parent classes (The Contract + PyTorch Module)
        BaseExpert.__init__(self, signal_name=signal_name)
        nn.Module.__init__(self)
        
        # Define the Neural Network Architecture
        # This is a standard feed-forward network optimized for tabular kinematics
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),       # LeakyReLU often works well for continuous physics features
            nn.Dropout(0.2),         # Prevents overfitting on rare signal data
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1) # Crucial: Output is a single raw logit. NO Sigmoid here.
        )

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