"""
Conformer-XL: 12-layer Conformer with 8× subsampling
Optimized for iEEG → phoneme decoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rmsnorm import RMSNorm
from .prenet import TemporalPrenet


class ConvModule(nn.Module):
    """Conformer Convolution Module with depthwise separable convolutions"""
    def __init__(self, d_model: int, kernel_size: int = 15, expansion: int = 2, dropout: float = 0.15):
        super().__init__()
        
        # Pointwise expansion
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * expansion, kernel_size=1)
        
        # GLU activation
        self.glu = nn.GLU(dim=1)
        
        # Depthwise conv with GroupNorm (only place we keep GroupNorm)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model  # depthwise
        )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=d_model)
        self.activation = nn.SiLU()
        
        # Pointwise compression
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        # x: [B, T, D] → [B, D, T]
        x = x.transpose(1, 2)
        
        # Pointwise expansion
        x = self.pointwise_conv1(x)  # [B, 2D, T]
        
        # GLU
        x = self.glu(x)  # [B, D, T]
        
        # Depthwise conv
        x = self.depthwise_conv(x)  # [B, D, T]
        x = self.norm(x)
        x = self.activation(x)
        
        # Pointwise compression
        x = self.pointwise_conv2(x)  # [B, D, T]
        x = self.dropout(x)
        
        # Back to [B, T, D]
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.15):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        B, T, D = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)  # [B, T, 3D]
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, d]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, T, d]
        out = out.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Conformer Feed-Forward Module with Swish activation"""
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.15):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerBlock(nn.Module):
    """
    Single Conformer block with proper ordering:
    FFN → MHSA → ConvModule → FFN
    All with RMSNorm and residual connections
    """
    def __init__(self, d_model: int, num_heads: int, ff_expansion: int = 4, 
                 conv_kernel: int = 15, dropout: float = 0.15):
        super().__init__()
        
        # First FFN (half-step residual)
        self.norm_ff1 = RMSNorm(d_model)
        self.ff1 = FeedForward(d_model, ff_expansion, dropout)
        
        # Multi-head self-attention
        self.norm_mhsa = RMSNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Convolution module
        self.norm_conv = RMSNorm(d_model)
        self.conv = ConvModule(d_model, conv_kernel, expansion=2, dropout=dropout)
        
        # Second FFN (half-step residual)
        self.norm_ff2 = RMSNorm(d_model)
        self.ff2 = FeedForward(d_model, ff_expansion, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        # First FFN (half-step)
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        
        # MHSA
        x = x + self.mhsa(self.norm_mhsa(x))
        
        # Conv module
        x = x + self.conv(self.norm_conv(x))
        
        # Second FFN (half-step)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        
        return x


class Subsampling8x(nn.Module):
    """
    8× strided subsampling using three Conv1D layers
    Critical for CTC stability with iEEG
    """
    def __init__(self, d_model: int):
        super().__init__()
        
        # Three stride-2 convs: 2 × 2 × 2 = 8×
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T/8, D]
        """
        # [B, T, D] → [B, D, T]
        x = x.transpose(1, 2)
        
        # Three stride-2 convolutions
        x = self.activation(self.conv1(x))  # T/2
        x = self.activation(self.conv2(x))  # T/4
        x = self.activation(self.conv3(x))  # T/8
        
        # [B, D, T/8] → [B, T/8, D]
        x = x.transpose(1, 2)
        return x


class ConformerXL(nn.Module):
    """
    Conformer-XL for iEEG → Phoneme decoding
    
    Architecture:
    1. Temporal Prenet (Conv + GRU)
    2. 8× Subsampling
    3. 12 Conformer blocks
    4. CTC output layer
    
    Specs:
    - d_model: 384
    - num_layers: 12
    - num_heads: 6
    - ff_expansion: 4
    - conv_kernel: 15
    
    Expected performance: 90-95% accuracy with WFST decoding
    """
    def __init__(self, config: dict):
        super().__init__()
        
        input_dim = config['model']['input_dim']
        d_model = config['model']['d_model']
        num_layers = config['model']['num_layers']
        num_heads = config['model']['num_heads']
        ff_expansion = config['model']['ff_expansion']
        conv_kernel = config['model']['conv_kernel']
        dropout = config['model']['dropout']
        num_classes = config['model']['num_classes']
        
        # Temporal prenet
        self.prenet = TemporalPrenet(input_dim, d_model)
        
        # 8× subsampling
        self.subsampling = Subsampling8x(d_model)
        
        # 12 Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_expansion, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(d_model)
        
        # CTC output projection
        self.ctc_proj = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """
        Args:
            x: [B, T, input_dim] - raw iEEG features
            lengths: [B] - sequence lengths (optional)
        Returns:
            logits: [B, T', num_classes] where T' = T/8
            output_lengths: [B] - output sequence lengths
        """
        # Prenet: [B, T, 512] → [B, T, 384]
        x = self.prenet(x)
        
        # 8× subsampling: [B, T, 384] → [B, T/8, 384]
        x = self.subsampling(x)
        
        # Update lengths
        if lengths is not None:
            output_lengths = lengths // 8
        else:
            output_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        
        # 12 Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # CTC projection: [B, T/8, 384] → [B, T/8, 42]
        logits = self.ctc_proj(x)
        
        return logits, output_lengths