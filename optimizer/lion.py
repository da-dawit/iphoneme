"""
Lion Optimizer (Google 2023)
Faster convergence and cleaner gradients for attention models with noisy biological input
"""
import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion optimizer: Evolved sign momentum
    
    Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al. 2023)
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-4)
        betas: coefficients for momentum (default: (0.9, 0.99))
        weight_decay: weight decay coefficient (default: 0.0)
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Weight decay
                if group['weight_decay'] > 0.0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update (Lion's unique sign-based update)
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Update exponential moving average
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss