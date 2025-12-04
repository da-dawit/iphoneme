"""
Loss functions for iEEG phoneme decoding
"""
from .ctc_softce_loss import CTCSoftCELoss

__all__ = ['CTCSoftCELoss']