"""
Metrics for phoneme error rate (PER) calculation
"""
import torch
import numpy as np
from typing import List


def edit_distance(ref: List[int], hyp: List[int]) -> int:
    """
    Compute Levenshtein edit distance between two sequences
    
    Args:
        ref: Reference sequence
        hyp: Hypothesis sequence
    
    Returns:
        Edit distance
    """
    m, n = len(ref), len(hyp)
    
    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )
    
    return dp[m][n]


def phoneme_error_rate(references: List[List[int]], hypotheses: List[List[int]]) -> float:
    """
    Calculate Phoneme Error Rate (PER)
    
    PER = (substitutions + deletions + insertions) / total_reference_phonemes
    
    Args:
        references: List of reference phoneme sequences
        hypotheses: List of hypothesis phoneme sequences
    
    Returns:
        PER as percentage (0-100)
    """
    total_distance = 0
    total_length = 0
    
    for ref, hyp in zip(references, hypotheses):
        # Remove blanks and padding
        ref = [p for p in ref if p > 0]
        hyp = [p for p in hyp if p > 0]
        
        total_distance += edit_distance(ref, hyp)
        total_length += len(ref)
    
    if total_length == 0:
        return 0.0
    
    return (total_distance / total_length) * 100.0


def greedy_decode(logits: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    """
    Greedy CTC decoding
    
    Args:
        logits: [B, T, C] - model output logits
        blank_idx: Index of blank token
    
    Returns:
        List of decoded sequences (one per batch item)
    """
    # Get argmax predictions
    predictions = torch.argmax(logits, dim=-1)  # [B, T]
    
    decoded = []
    for pred in predictions:
        # Remove consecutive duplicates
        collapsed = []
        prev = None
        for p in pred.tolist():
            if p != prev:
                collapsed.append(p)
                prev = p
        
        # Remove blanks
        decoded_seq = [p for p in collapsed if p != blank_idx]
        decoded.append(decoded_seq)
    
    return decoded


class Metrics:
    """
    Metrics tracker for training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_ctc_loss = 0.0
        self.total_ce_loss = 0.0
        self.count = 0
    
    def update(self, loss_dict: dict):
        """Update metrics with batch results"""
        self.total_loss += loss_dict.get('loss', 0.0)
        self.total_ctc_loss += loss_dict.get('ctc_loss', 0.0)
        self.total_ce_loss += loss_dict.get('ce_loss', 0.0)
        self.count += 1
    
    def get_average(self) -> dict:
        """Get average metrics"""
        if self.count == 0:
            return {'loss': 0.0, 'ctc_loss': 0.0, 'ce_loss': 0.0}
        
        return {
            'loss': self.total_loss / self.count,
            'ctc_loss': self.total_ctc_loss / self.count,
            'ce_loss': self.total_ce_loss / self.count
        }