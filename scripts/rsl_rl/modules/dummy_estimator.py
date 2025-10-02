"""Dummy estimator that returns zeros for environments that don't use privileged states."""

import torch
import torch.nn as nn
from typing import Dict, Any


class DummyEstimator(nn.Module):
    """A dummy estimator that returns zeros instead of estimating privileged states."""
    
    def __init__(self, **kwargs):
        """Initialize the dummy estimator.
        
        Args:
            **kwargs: Ignored arguments for compatibility
        """
        super().__init__()
        # Create a dummy parameter so the module is not empty
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return zeros with the same batch size as input.
        
        Args:
            x: Input tensor with shape (batch_size, input_dim)
            
        Returns:
            Zero tensor with shape (batch_size, 0) - no privileged states for rough terrain
        """
        batch_size = x.shape[0]
        device = x.device
        # Return empty tensor since rough terrain has no privileged states
        # Use dummy_param to ensure gradients are tracked
        return torch.zeros(batch_size, 0, device=device) + self.dummy_param * 0
    
    def to(self, device):
        """Move module to device."""
        return super().to(device)
