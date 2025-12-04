"""
COMPLETE INTEGRATED TRAINING SYSTEM
Integrates Ultimate Architecture with full training pipeline

Usage:
    python train_ultimate_integrated.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# Import our ultimate architecture
from ultimate_architecture import (
    UltimateConformer,
    UltimateAugmentation,
    ULTIMATE_CONFIG
)
from dataset import iEEGPhonemeDataset


class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=0.9998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class LionOptimizer(optim.Optimizer):
    """Lion optimizer - more efficient than Adam"""
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # EMA update
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper that applies augmentation"""
    
    def __init__(self, base_dataset, augmentation, training=True):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.training = training
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        neural, phonemes, neural_len, phoneme_len = self.base_dataset[idx]
        
        if self.training:
            # Apply augmentation
            num_copies = self.augmentation.get_num_copies() if hasattr(self.augmentation, 'get_num_copies') else np.random.randint(6, 17)
            
            augmented = []
            for _ in range(num_copies):
                aug = self.augmentation.augment(neural)
                augmented.append(aug)
            
            augmented = torch.stack(augmented)
            phonemes = phonemes.unsqueeze(0).expand(num_copies, -1)
            neural_len = neural_len.unsqueeze(0).expand(num_copies) if isinstance(neural_len, torch.Tensor) else torch.tensor([neural_len] * num_copies)
            phoneme_len = phoneme_len.unsqueeze(0).expand(num_copies) if isinstance(phoneme_len, torch.Tensor) else torch.tensor([phoneme_len] * num_copies)
            
            return augmented, phonemes, neural_len, phoneme_len
        else:
            return neural.unsqueeze(0), phonemes.unsqueeze(0), neural_len.unsqueeze(0) if isinstance(neural_len, torch.Tensor) else torch.tensor([neural_len]), phoneme_len.unsqueeze(0) if isinstance(phoneme_len, torch.Tensor) else torch.tensor([phoneme_len])


def collate_fn(batch):
    """Custom collate function for variable-length sequences with augmentation"""
    
    # Flatten augmented copies
    all_neural = []
    all_phonemes = []
    all_neural_len = []
    all_phoneme_len = []
    
    for neural, phonemes, neural_len, phoneme_len in batch:
        # neural: [num_copies, T, 512]
        for i in range(neural.shape[0]):
            all_neural.append(neural[i])
            all_phonemes.append(phonemes[i])
            all_neural_len.append(neural_len[i] if isinstance(neural_len, torch.Tensor) else neural_len)
            all_phoneme_len.append(phoneme_len[i] if isinstance(phoneme_len, torch.Tensor) else phoneme_len)
    
    # Pad sequences
    max_neural_len = max(n.shape[0] for n in all_neural)
    max_phoneme_len = max(p.shape[0] for p in all_phonemes)
    
    neural_padded = torch.zeros(len(all_neural), max_neural_len, 512)
    phonemes_padded = torch.zeros(len(all_phonemes), max_phoneme_len, dtype=torch.long)
    
    for i, (n, p) in enumerate(zip(all_neural, all_phonemes)):
        neural_padded[i, :n.shape[0]] = n
        phonemes_padded[i, :p.shape[0]] = p
    
    neural_lengths = torch.tensor(all_neural_len)
    phoneme_lengths = torch.tensor(all_phoneme_len)
    
    return {
        'neural': neural_padded,
        'phonemes': phonemes_padded,
        'neural_lengths': neural_lengths,
        'phoneme_lengths': phoneme_lengths
    }


class UltimateTrainer:
    """Complete training pipeline for Ultimate architecture"""
    
    def __init__(self, config_path, device='cuda'):
        self.device = device
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Create model
        print("Creating Ultimate model...")
        self.model = UltimateConformer(**self.config['model']).to(device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params / 1e6:.1f}M")
        
        # Create datasets
        print("Loading datasets...")
        train_sessions = [s for s, p in zip(self.config['data']['sessions'], self.config['data']['dataset_probability_val']) if p == 1]
        val_sessions = [s for s, p in zip(self.config['data']['sessions'], self.config['data']['dataset_probability_val']) if p == 0]
        
        base_train_dataset = iEEGPhonemeDataset(
            dataset_root=self.config['data']['dataset_root'],
            sessions=train_sessions,
            split='train'
        )
        
        base_val_dataset = iEEGPhonemeDataset(
            dataset_root=self.config['data']['dataset_root'],
            sessions=val_sessions,
            split='train'
        )
        
        # Wrap with augmentation
        augmentation = UltimateAugmentation(**self.config['augmentation'])
        self.train_dataset = AugmentedDataset(base_train_dataset, augmentation, training=True)
        self.val_dataset = AugmentedDataset(base_val_dataset, augmentation, training=False)
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer
        self.optimizer = LionOptimizer(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            betas=self.config['optimizer']['betas'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # LR scheduler with warmup
        self.scheduler = self.create_scheduler()
        
        # EMA
        self.ema = EMA(self.model, decay=self.config['training']['ema_decay'])
        
        # Loss
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config['training']['mixed_precision'] else None
    
    def create_scheduler(self):
        """Create LR scheduler with linear warmup"""
        warmup_steps = self.config['training']['warmup_steps']
        total_steps = len(self.train_loader) * self.config['training']['num_epochs']
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_ctc = 0
        total_boundary = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch in pbar:
            neural = batch['neural'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            neural_lengths = batch['neural_lengths'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            
            # Forward
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits, output_lengths, boundary_logits = self.model(
                        neural, neural_lengths, return_boundary=True
                    )
                    
                    # CTC loss
                    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
                    ctc_loss = self.ctc_loss(log_probs, phonemes, output_lengths, phoneme_lengths)
                    
                    # Boundary loss (auxiliary)
                    boundary_loss = 0
                    if boundary_logits is not None and self.config['loss'].get('boundary_weight', 0) > 0:
                        # Create boundary targets (simplified - 1 at phoneme changes)
                        boundary_targets = torch.zeros_like(phonemes, dtype=torch.long)
                        for i in range(len(phonemes)):
                            for j in range(1, phoneme_lengths[i]):
                                if phonemes[i, j] != phonemes[i, j-1]:
                                    boundary_targets[i, j] = 1
                        
                        boundary_loss = F.cross_entropy(
                            boundary_logits.reshape(-1, 2),
                            boundary_targets.reshape(-1)
                        ) * self.config['loss']['boundary_weight']
                    
                    loss = ctc_loss + boundary_loss
                
                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, output_lengths, boundary_logits = self.model(
                    neural, neural_lengths, return_boundary=True
                )
                
                log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
                ctc_loss = self.ctc_loss(log_probs, phonemes, output_lengths, phoneme_lengths)
                
                boundary_loss = 0
                if boundary_logits is not None and self.config['loss'].get('boundary_weight', 0) > 0:
                    boundary_targets = torch.zeros_like(phonemes, dtype=torch.long)
                    for i in range(len(phonemes)):
                        for j in range(1, phoneme_lengths[i]):
                            if phonemes[i, j] != phonemes[i, j-1]:
                                boundary_targets[i, j] = 1
                    
                    boundary_loss = F.cross_entropy(
                        boundary_logits.reshape(-1, 2),
                        boundary_targets.reshape(-1)
                    ) * self.config['loss']['boundary_weight']
                
                loss = ctc_loss + boundary_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                self.optimizer.step()
            
            # EMA update
            self.ema.update()
            
            # LR scheduler
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            total_ctc += ctc_loss.item()
            if isinstance(boundary_loss, torch.Tensor):
                total_boundary += boundary_loss.item()
            
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ctc': f'{ctc_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'ctc_loss': total_ctc / len(self.train_loader),
            'boundary_loss': total_boundary / len(self.train_loader),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_shadow': self.ema.shadow,
            'config': self.config
        }, path)
        
        print(f"✓ Saved checkpoint: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.ema.shadow = checkpoint['ema_shadow']
        self.epoch = checkpoint['epoch']
        
        print(f"✓ Loaded checkpoint: {path}")


def main():
    """Main training function"""
    
    # Save config
    os.makedirs('outputs', exist_ok=True)
    config_path = 'outputs/ultimate_config.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(ULTIMATE_CONFIG, f)
    
    print("="*80)
    print("ULTIMATE ARCHITECTURE - INTEGRATED TRAINING")
    print("="*80)
    print("\nTarget: 97-98% accuracy")
    print("\nInnovations:")
    print("  1. NEJM augmentation (SD=1.2)")
    print("  2. Delta + delta-delta features")
    print("  3. Multi-scale temporal pyramid")
    print("  4. Region-aware attention")
    print("  5. FiLM adaptation")
    print("  6. Phoneme boundary detection")
    print("  7. Electrode dropout")
    print("="*80 + "\n")
    
    # Initialize W&B
    wandb.init(
        project='ieeg-ultimate-97pct',
        name='ultimate_integrated_v1',
        config=ULTIMATE_CONFIG,
        tags=['ultimate', '97pct-target', 'all-innovations']
    )
    
    # Create trainer
    trainer = UltimateTrainer(config_path, device='cuda')
    
    # Training loop
    num_epochs = ULTIMATE_CONFIG['training']['num_epochs']
    
    for epoch in range(num_epochs):
        trainer.epoch = epoch
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = trainer.train_epoch()
        
        # Log to W&B
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_metrics['loss'],
            'train/ctc_loss': train_metrics['ctc_loss'],
            'train/boundary_loss': train_metrics['boundary_loss'],
            'train/lr': train_metrics['lr']
        })
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(f'outputs/ultimate_model_epoch_{epoch+1}.pt')
    
    # Save final model
    trainer.save_checkpoint('outputs/ultimate_model_final.pt')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Decode with 6-gram LM")
    print("  2. Ensemble with v4 models")
    print("  3. WFST beam search")
    print("\nExpected: 97-98% final accuracy")
    print("="*80 + "\n")
    
    wandb.finish()


if __name__ == '__main__':
    main()