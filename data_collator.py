"""
Data collator with intelligent batching and SpecAugment
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple
import random


class SpecAugment:
    """
    SpecAugment for iEEG data
    Applies time and frequency masking for regularization
    """
    
    def __init__(self, time_mask_count: int = 2, time_mask_size: int = 80,
                 freq_mask_count: int = 2, freq_mask_size: int = 20):
        self.time_mask_count = time_mask_count
        self.time_mask_size = time_mask_size
        self.freq_mask_count = freq_mask_count
        self.freq_mask_size = freq_mask_size
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input
        
        Args:
            x: [T, D] - single sequence
        
        Returns:
            [T, D] - augmented sequence
        """
        T, D = x.shape
        x = x.clone()
        
        # Time masking
        for _ in range(self.time_mask_count):
            t = random.randint(0, min(self.time_mask_size, T - 1))
            t0 = random.randint(0, T - t)
            x[t0:t0 + t] = 0
        
        # Frequency (channel) masking
        for _ in range(self.freq_mask_count):
            f = random.randint(0, min(self.freq_mask_size, D - 1))
            f0 = random.randint(0, D - f)
            x[:, f0:f0 + f] = 0
        
        return x


class iEEGCollator:
    """
    Collate function for iEEG dataset
    
    Operations:
    1. Sort by length (longest first) for efficient packing
    2. Pad sequences to max length in batch
    3. Apply SpecAugment (during training)
    4. Return batch tensors
    """
    
    def __init__(self, specaugment_config: dict = None, training: bool = True):
        self.training = training
        
        if specaugment_config and training:
            self.specaugment = SpecAugment(
                time_mask_count=specaugment_config.get('time_mask_count', 2),
                time_mask_size=specaugment_config.get('time_mask_size', 80),
                freq_mask_count=specaugment_config.get('freq_mask_count', 2),
                freq_mask_size=specaugment_config.get('freq_mask_size', 20)
            )
        else:
            self.specaugment = None
    
    def __call__(self, batch: List[Tuple]) -> dict:
        """
        Collate batch of samples
        
        Args:
            batch: List of (x, y, x_len, y_len) tuples
        
        Returns:
            dict with:
                - neural: [B, T_max, 512]
                - phonemes: [B, L_max]
                - neural_lengths: [B]
                - phoneme_lengths: [B]
        """
        # Sort by neural length (descending) for efficiency
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        
        # Unpack
        neural_seqs = [item[0] for item in batch]
        phoneme_seqs = [item[1] for item in batch]
        neural_lengths = torch.LongTensor([item[2] for item in batch])
        phoneme_lengths = torch.LongTensor([item[3] for item in batch])
        
        # Apply SpecAugment if training
        if self.specaugment is not None:
            neural_seqs = [self.specaugment(x) for x in neural_seqs]
        
        # Pad neural sequences
        max_neural_len = max(x.size(0) for x in neural_seqs)
        neural_padded = []
        for x in neural_seqs:
            pad_len = max_neural_len - x.size(0)
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))  # Pad time dimension
            neural_padded.append(x)
        neural_tensor = torch.stack(neural_padded)  # [B, T_max, 512]
        
        # Pad phoneme sequences
        max_phoneme_len = max(y.size(0) for y in phoneme_seqs)
        phoneme_padded = []
        for y in phoneme_seqs:
            pad_len = max_phoneme_len - y.size(0)
            if pad_len > 0:
                y = F.pad(y, (0, pad_len), value=0)  # Pad with blank token
            phoneme_padded.append(y)
        phoneme_tensor = torch.stack(phoneme_padded)  # [B, L_max]
        
        return {
            'neural': neural_tensor,
            'phonemes': phoneme_tensor,
            'neural_lengths': neural_lengths,
            'phoneme_lengths': phoneme_lengths
        }


def get_collator(config: dict, training: bool = True):
    """
    Factory function to create collator
    
    Args:
        config: Full config dict
        training: Whether for training (enables SpecAugment)
    
    Returns:
        iEEGCollator instance
    """
    specaugment_config = config.get('specaugment', None) if training else None
    return iEEGCollator(specaugment_config, training)