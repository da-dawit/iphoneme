"""
Learning rate scheduler with warmup and cosine decay
"""
import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of base LR (default: 0.01)
    """
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr_ratio: float = 0.01, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step"""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = float(step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]


def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Factory function to create warmup + cosine scheduler
    
    Args:
        optimizer: Optimizer to wrap
        warmup_steps: Number of linear warmup steps
        total_steps: Total training steps
    
    Returns:
        WarmupCosineScheduler instance
    """
    return WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.01
    )