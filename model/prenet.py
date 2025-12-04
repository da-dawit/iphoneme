"""
Temporal Prenet: Channel-mixer with dilated convolutions and GRU
Removes iEEG channel noise and aligns temporal structure
"""
import torch
import torch.nn as nn
from .rmsnorm import RMSNorm


class TemporalPrenet(nn.Module):
    """
    Temporal channel-mixer prenet for iEEG
    
    Pipeline:
    1. Conv1D (kernel=5, dilation=1) - fast temporal patterns
    2. Conv1D (kernel=5, dilation=2) - slow temporal patterns  
    3. Bidirectional GRU - temporal alignment
    4. Linear projection to d_model
    
    Purpose:
    - Remove electrode noise
    - Align electrodes → phoneme transitions
    - Extract multi-scale temporal structure
    
    Expected gain: +3-5% accuracy
    """
    def __init__(self, input_dim: int = 512, d_model: int = 384, gru_hidden: int = 256):
        super().__init__()
        
        # Dilated temporal convolutions
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=5,
            dilation=1,
            padding=2  # same padding
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=5,
            dilation=2,
            padding=4  # same padding for dilation=2
        )
        
        self.norm1 = RMSNorm(input_dim)
        self.norm2 = RMSNorm(input_dim)
        
        # Bidirectional GRU for temporal alignment
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Project to model dimension
        self.proj = nn.Linear(gru_hidden * 2, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] - raw iEEG features
        Returns:
            [B, T, d_model] - cleaned temporal features
        """
        # x: [B, T, 512]
        
        # Conv expects [B, C, T]
        x_conv = x.transpose(1, 2)  # [B, 512, T]
        
        # First dilated conv (fast patterns)
        h1 = self.conv1(x_conv)  # [B, 512, T]
        h1 = h1.transpose(1, 2)  # [B, T, 512]
        h1 = self.norm1(h1)
        h1 = self.activation(h1)
        
        # Second dilated conv (slow patterns)
        h2 = self.conv2(h1.transpose(1, 2))  # [B, 512, T]
        h2 = h2.transpose(1, 2)  # [B, T, 512]
        h2 = self.norm2(h2)
        h2 = self.activation(h2)
        
        # Bidirectional GRU for temporal alignment
        gru_out, _ = self.gru(h2)  # [B, T, 512] (256*2)
        
        # Project to d_model
        out = self.proj(gru_out)  # [B, T, 384]
        
        return out