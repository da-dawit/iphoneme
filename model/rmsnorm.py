"""
RMSNorm: Root Mean Square Layer Normalization
Replaces LayerNorm for non-stationary iEEG signals
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    More stable than LayerNorm for non-stationary biological signals
    Used in LLaMA, Falcon, T5, and modern BCI decoders
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] or [B, D] or [T, D]
        Returns:
            Normalized tensor with same shape
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed