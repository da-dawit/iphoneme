"""
Exponential Moving Average (EMA) for model parameters
Provides stable "teacher model" for improved decoding
"""
import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters
    
    Maintains a smoothed version of the model that typically decodes better
    than the training model, especially for noisy biological signals.
    
    Usage:
        ema = EMA(model, decay=0.9995)
        
        # During training:
        loss.backward()
        optimizer.step()
        ema.update(model)
        
        # For evaluation:
        ema.apply_shadow()
        evaluate(model)
        ema.restore()
    
    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9995)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        self.register()
    
    def register(self):
        """Register all model parameters for EMA tracking"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module = None):
        """
        Update EMA parameters
        
        Args:
            model: Model to update from (uses self.model if None)
        """
        if model is not None:
            self.model = model
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not registered in EMA"
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """
        Apply EMA parameters to model for evaluation
        Backs up original parameters first
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not registered in EMA"
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model parameters after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup, f"No backup for parameter {name}"
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state for checkpointing"""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']