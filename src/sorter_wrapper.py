'''
This is the bridge that takes your Scikit-Learn .pkl model 
and makes it speak natively to the PyTorch tensors that the Experts will be using.
'''


import torch
import joblib
import numpy as np
import os
from src.base import BaseSorter

class FROCCWrapper(BaseSorter):
    def __init__(self, model_path: str = "models/frocc_weights/sorter.pkl"):
        """
        Loads the pre-trained StandardScaler + DFROCC pipeline.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Sorter model not found at {model_path}. Run training script first.")
        
        # Load the pickled sklearn pipeline (Scaler + DFROCC)
        self.model = joblib.load(model_path)
        
        # Retrieve the custom threshold we calibrated during training
        self.threshold = self.model.threshold
        print(f"✅ FROCCWrapper initialized with threshold: {self.threshold:.4f}")

    def predict_mask(self, x: torch.Tensor) -> torch.BoolTensor:
        """
        Input: PyTorch Tensor batch of LHC events.
        Output: Boolean Tensor mask (True = Keep/Signal, False = Drop/Background).
        """
        # Store the original device (e.g., 'cuda:0' or 'cpu') so we can return the mask there
        original_device = x.device 
        
        # 1. Translate PyTorch to NumPy
        # (Detach from compute graph and move to CPU since sklearn requires numpy)
        x_np = x.detach().cpu().numpy()
        
        # 2. Get anomaly scores from the pipeline
        # The pipeline automatically applies the StandardScaler before feeding to DFROCC
        scores = self.model.decision_function(x_np)
        
        # 3. Apply the threshold logic
        # DFROCC outputs an "Agreement" fraction.
        # If score <= threshold, it deviates from the background manifold (Anomaly/Signal).
        keep_mask_np = (scores <= self.threshold)
        
        # 4. Translate NumPy back to PyTorch
        # Convert to a boolean tensor and push it back to the original device
        keep_mask = torch.from_numpy(keep_mask_np).to(torch.bool).to(original_device)
        
        return keep_mask