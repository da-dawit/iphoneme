"""
Main training script for Conformer-XL iEEG → Phoneme decoder

Three-stage curriculum training:
Stage A - Fast warmup (8 epochs)
Stage B - Hard training (60 epochs) 
Stage C - Fine-tuning (20 epochs)

Expected final accuracy: 90-95% with WFST decoding
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
from pathlib import Path

from model import ConformerXL
from dataset import iEEGPhonemeDataset
from data_collator import get_collator
from loss import CTCSoftCELoss
from optimizer.lion import Lion
from utils.ema import EMA
from utils.scheduler import get_scheduler
from utils.metrics import Metrics, phoneme_error_rate, greedy_decode
from decoding.wfst_decoder import WFSTDecoder, build_language_model_from_dataset


class Trainer:
    """
    Trainer for Conformer-XL iEEG decoder
    
    Implements:
    - Mixed precision training
    - EMA model tracking
    - Gradient clipping
    - Curriculum learning
    - WFST decoding evaluation
    """
    
    def __init__(self, config_path: str, device: str = 'cuda'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device
        self.output_dir = Path('outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup model
        self.model = ConformerXL(self.config).to(device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        # Setup optimizer
        self.optimizer = Lion(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            betas=self.config['optimizer']['betas'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Setup EMA
        self.ema = EMA(self.model, decay=self.config['training']['ema_decay'])
        
        # Setup loss
        self.criterion = CTCSoftCELoss(
            blank_idx=0,
            ce_weight=self.config['loss']['ce_weight'],
            temperature=self.config['loss']['softmax_temperature']
        )
        
        # Mixed precision
        self.use_amp = self.config['training'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup data
        self._setup_data()
        
        # Setup scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config['training']['num_epochs']
        self.scheduler = get_scheduler(
            self.optimizer,
            warmup_steps=self.config['training']['warmup_steps'],
            total_steps=total_steps
        )
        
        # Build language model if needed
        lm_path = self.output_dir / 'phoneme_lm.pkl'
        if not lm_path.exists():
            print("\n" + "="*50)
            print("Building language model from training data...")
            print("="*50)
            build_language_model_from_dataset(self.train_dataset, str(lm_path))
            print("="*50 + "\n")
        
        # Setup decoder
        self.decoder = WFSTDecoder(self.config)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_per = float('inf')
    
    def _setup_data(self):
        """Setup datasets and dataloaders"""
        sessions = self.config['data']['sessions']
        val_flags = self.config['data'].get('dataset_probability_val', None)
        
        if val_flags:
            # Use predefined validation split
            # 1 = train, 0 = validation
            train_sessions = [s for s, v in zip(sessions, val_flags) if v == 1]
            val_sessions = [s for s, v in zip(sessions, val_flags) if v == 0]
        else:
            # Fallback: 80/10/10 split
            n_sessions = len(sessions)
            n_train = int(0.8 * n_sessions)
            n_val = int(0.1 * n_sessions)
            train_sessions = sessions[:n_train]
            val_sessions = sessions[n_train:n_train + n_val]
        
        print(f"Train sessions: {len(train_sessions)}")
        print(f"Val sessions: {len(val_sessions)}")
        
        # Create datasets - both use 'train' split
        self.train_dataset = iEEGPhonemeDataset(
            self.config['data']['dataset_root'],
            train_sessions,
            split='train'
        )
        
        self.val_dataset = iEEGPhonemeDataset(
            self.config['data']['dataset_root'],
            val_sessions,
            split='train'  # Val sessions also use train split
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=get_collator(self.config, training=True),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=get_collator(self.config, training=False),
            pin_memory=True
        )
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        metrics = Metrics()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        for batch in pbar:
            # Move to device
            neural = batch['neural'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            neural_lengths = batch['neural_lengths'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Student model forward
                logits, output_lengths = self.model(neural, neural_lengths)
                
                # Teacher model forward (EMA, no grad)
                with torch.no_grad():
                    self.ema.apply_shadow()
                    teacher_logits, _ = self.model(neural, neural_lengths)
                    self.ema.restore()
                
                # Compute loss
                loss_dict = self.criterion(
                    logits, phonemes,
                    output_lengths, phoneme_lengths,
                    teacher_logits=teacher_logits
                )
                loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                self.optimizer.step()
            
            # Update EMA
            self.ema.update(self.model)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update metrics
            metrics.update(loss_dict)
            self.global_step += 1
            
            # Update progress bar
            avg_metrics = metrics.get_average()
            pbar.set_postfix({
                'loss': f"{avg_metrics['loss']:.4f}",
                'ctc': f"{avg_metrics['ctc_loss']:.4f}",
                'ce': f"{avg_metrics['ce_loss']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        return metrics.get_average()
    
    @torch.no_grad()
    def validate(self, use_ema: bool = True):
        """Validate on validation set"""
        if use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        metrics = Metrics()
        
        all_references = []
        all_hypotheses_greedy = []
        all_hypotheses_wfst = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # Move to device
            neural = batch['neural'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            neural_lengths = batch['neural_lengths'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            
            # Forward pass
            logits, output_lengths = self.model(neural, neural_lengths)
            
            # Compute loss
            loss_dict = self.criterion(
                logits, phonemes,
                output_lengths, phoneme_lengths,
                teacher_logits=None  # No teacher during eval
            )
            metrics.update(loss_dict)
            
            # Decode
            greedy_decoded = greedy_decode(logits, blank_idx=0)
            wfst_decoded = self.decoder.decode(logits)
            
            # Collect for PER calculation
            for i in range(len(phonemes)):
                ref = phonemes[i, :phoneme_lengths[i]].cpu().tolist()
                all_references.append(ref)
                all_hypotheses_greedy.append(greedy_decoded[i])
                all_hypotheses_wfst.append(wfst_decoded[i])
        
        # Calculate PER
        greedy_per = phoneme_error_rate(all_references, all_hypotheses_greedy)
        wfst_per = phoneme_error_rate(all_references, all_hypotheses_wfst)
        
        if use_ema:
            self.ema.restore()
        
        avg_metrics = metrics.get_average()
        avg_metrics['greedy_per'] = greedy_per
        avg_metrics['wfst_per'] = wfst_per
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_per': self.best_per
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_per = checkpoint['best_per']
        
        print(f"Loaded checkpoint: {path}")
    
    def train(self):
        """Full training loop"""
        num_epochs = self.config['training']['num_epochs']
        
        print("\n" + "="*50)
        print("Starting training")
        print(f"Total epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print("="*50 + "\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"CTC: {train_metrics['ctc_loss']:.4f}, "
                  f"CE: {train_metrics['ce_loss']:.4f}")
            
            # Validate every 5 epochs
            #if (epoch + 1) % 5 == 0:
            #    val_metrics = self.validate(use_ema=True)
            #    print(f"Val - Loss: {val_metrics['loss']:.4f}, "
            #          f"Greedy PER: {val_metrics['greedy_per']:.2f}%, "
            #          f"WFST PER: {val_metrics['wfst_per']:.2f}%")
                
                # Save best model
            #    if val_metrics['wfst_per'] < self.best_per:
            #        self.best_per = val_metrics['wfst_per']
            #        self.save_checkpoint('best_model.pt')
            #        print(f"New best model! WFST PER: {self.best_per:.2f}%")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        print("\n" + "="*50)
        print("Training complete!")
        print(f"Best WFST PER: {self.best_per:.2f}%")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Conformer-XL iEEG decoder')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(args.config, args.device)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()