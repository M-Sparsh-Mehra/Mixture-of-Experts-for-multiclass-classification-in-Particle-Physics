"""
This file is the  the main brain of the system. 
It connects Stage I (The Sorter) and Stage II (The Experts) into a single, cohesive workflow.

Instead of manually passing data from the anomaly detector to the MLPs in a notebook, 
this class handles the routing automatically. It takes raw LHC events, filters out the obvious 
background using FROCC, asks all active experts to evaluate the remaining events simultaneously, 
and computes the final mutually exclusive probabilities using a Modular Softmax.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any

class LHCDynamicMoE:
    def __init__(self, sorter, experts: List, bg_logit: float = 0.0):
        """
        sorter: Initialized instance of FROCCWrapper
        experts: List of initialized MLPExpert instances
        bg_logit: The baseline "confidence" score for residual background. 
                  If no expert scores higher than this, the event defaults to Background.
        """
        self.sorter = sorter
        self.experts = experts
        self.bg_logit = bg_logit

        # Extract names for clean output mapping
        self.expert_names = [expert.signal_name for expert in self.experts]
        self.class_names = self.expert_names + ["Background"]

    def process_batch(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        The main forward pass for a batch of LHC events.
        Input: PyTorch Tensor of shape [Batch_Size, Num_Features]
        """
        batch_size = x.size(0)
        device = x.device

        # --- Stage I: The Sieve ---
        # keep_mask is True for anomalies (Signals), False for normal (Background)
        keep_mask = self.sorter.predict_mask(x)
        
        # Find the actual row indices of events that passed the sorter
        kept_indices = torch.nonzero(keep_mask, as_tuple=True)[0]
        
        # Initialize the final probability tensor (Shape: [Batch_Size, Num_Classes])
        # Default every event in the batch to 100% Background (the last column)
        final_probs = torch.zeros((batch_size, len(self.class_names)), device=device)
        final_probs[:, -1] = 1.0 

        # If the Sorter rejected everything, we save compute and return immediately!
        if len(kept_indices) == 0:
            return {
                "probabilities": final_probs,
                "kept_mask": keep_mask,
                "classes": self.class_names
            }

        # --- Stage II: The Expert Pool ---
        # Only process the events that survived Stage I
        x_filtered = x[kept_indices]
        
        expert_logits_list = []
        
        for expert in self.experts:
            # get_logits automatically applies the temperature scaling calibration!
            logits = expert.get_logits(x_filtered)
            expert_logits_list.append(logits)

        # Stack into shape: [Filtered_Batch_Size, Num_Experts]
        if len(expert_logits_list) > 0:
            expert_stack = torch.cat(expert_logits_list, dim=1)
        else:
            # Edge case handling: Pipeline running with 0 experts loaded
            expert_stack = torch.empty((len(kept_indices), 0), device=device)

        # --- The Aggregator (Modular Softmax) ---
        # Create the background column with the constant bg_logit
        bg_column = torch.full((len(kept_indices), 1), self.bg_logit, device=device)
        
        # Combine Expert Logits + Background Logit
        # Shape: [Filtered_Batch_Size, Num_Experts + 1]
        combined_logits = torch.cat([expert_stack, bg_column], dim=1)
        
        # Apply Softmax to get mutually exclusive probabilities
        filtered_probs = F.softmax(combined_logits, dim=1)

        # --- Reconstruct Output ---
        # Place the computed probabilities back into their original batch positions
        # so that Row 5 in the output strictly matches Row 5 from your input CSV.
        final_probs[kept_indices] = filtered_probs

        return {
            "probabilities": final_probs,
            "kept_mask": keep_mask,
            "classes": self.class_names
        }