# src/base.py
from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseSorter(ABC):
    """
    Contract for Stage I: The Anomaly Detector (e.g., FROCC).
    It must take in physics features and output a boolean mask.
    """
    @abstractmethod
    def predict_mask(self, x: torch.Tensor) -> torch.BoolTensor:
        """
        Input: Batch of events (PyTorch Tensor, shape: [batch_size, num_features])
        Output: Boolean Tensor (True = Keep/Pass to Experts, False = Drop/Background)
        """
        pass

class BaseExpert(ABC):
    """
    Contract for Stage II: The Specific Signal Expert (e.g., an MLP).
    It must take in physics features and output raw, unnormalized logits.
    """
    def __init__(self, signal_name: str, temperature: float = 1.0):
        self.signal_name = signal_name
        self.temperature = temperature # The calibration factor for the Softmax

    @abstractmethod
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Batch of events (PyTorch Tensor, shape: [batch_size, num_features])
        Output: Raw Logits (PyTorch Tensor, shape: [batch_size, 1])
        MUST apply Temperature Scaling before returning.
        """
        pass
        
    def calibrate(self, val_loader):
        """
        Optional but recommended: Logic to learn self.temperature 
        to ensure this expert plays nicely with others in the pipeline.
        """
        pass