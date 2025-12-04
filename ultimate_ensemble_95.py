"""
V4: Enable LM validation ONLY after epoch 40
Before epoch 40: pure greedy only (LM would hurt early training)
After epoch 40: greedy + LM validation (model is good enough for LM to help)
"""
import optuna
import wandb
import yaml
import torch
import numpy as np
from pathlib import Path
from train import Trainer


def train_single_model(trial, model_id, base_config):
    """Train a single model with LM enabled after epoch 40"""
    
    config = base_config.copy()
    
    # ARCHITECTURE - optimized for 90%+ baseline
    config['model']['d_model'] = trial.suggest_categorical(f'm{model_id}_d_model', [576, 672])
    config['model']['num_layers'] = trial.suggest_int(f'm{model_id}_num_layers', 16, 18)
    config['model']['num_heads'] = 12
    config['model']['dropout'] = trial.suggest_float(f'm{model_id}_dropout', 0.12, 0.18)
    config['model']['ff_expansion'] = 6
    
    # TRAINING - stable
    config['training']['lr'] = trial.suggest_float(f'm{model_id}_lr', 2e-5, 5e-5, log=True)
    config['training']['batch_size'] = 16
    config['training']['num_epochs'] = 120
    config['training']['max_grad_norm'] = 0.8
    config['training']['warmup_steps'] = 5000
    
    # REGULARIZATION
    config['loss']['ce_weight'] = 0.0
    config['specaugment']['time_mask_size'] = 120
    config['specaugment']['freq_mask_size'] = 25
    
    # Random seed for diversity
    seed = [42, 123, 456, 789, 999][model_id]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Save config
    config_path = f'outputs/v4_model_{model_id}_config.yaml'
    Path('outputs').mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Initialize W&B
    wandb.init(
        project='ieeg-decoder-95pct-v4',
        name=f'v4_model_{model_id}_trial_{trial.number}',
        config=config,
        reinit=True,
        tags=['lm-after-epoch-40']
    )
    
    try:
        # Train model
        trainer = Trainer(config_path, device='cuda')
        
        # Load LM decoder (will only use after epoch 40)
        from greedy_lm_decoder import GreedyLMDecoder
        lm_decoder = GreedyLMDecoder(
            lm_path='outputs/phoneme_lm.pkl',
            lm_weight=0.5  # Lower weight to not overpower acoustic model
        )
        
        for epoch in range(120):
            trainer.epoch = epoch
            train_metrics = trainer.train_epoch()
            
            # Log training metrics
            wandb.log({
                'epoch': epoch + 1,
                f'model_{model_id}/train_loss': train_metrics['loss'],
                f'model_{model_id}/train_ctc': train_metrics['ctc_loss'],
                f'model_{model_id}/lr': train_metrics.get('lr', 0)
            })
            
            # Validate every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Model {model_id} - Epoch {epoch + 1}: Validation")
                print(f"{'='*60}")
                
                trainer.model.eval()
                all_refs = []
                all_hyps_greedy = []
                all_hyps_lm = []
                
                with torch.no_grad():
                    for batch in trainer.val_loader:
                        neural = batch['neural'].to('cuda')
                        phonemes = batch['phonemes']
                        neural_lengths = batch['neural_lengths'].to('cuda')
                        phoneme_lengths = batch['phoneme_lengths']
                        
                        logits, _ = trainer.model(neural, neural_lengths)
                        
                        # ALWAYS do pure greedy
                        decoded_greedy = lm_decoder.greedy_decode(logits)
                        all_hyps_greedy.extend(decoded_greedy)
                        
                        # ONLY do LM if epoch > 40
                        if epoch + 1 > 40:
                            decoded_lm = lm_decoder.greedy_with_lm_rescore(logits, top_k=5)
                            all_hyps_lm.extend(decoded_lm)
                        
                        # References
                        for i in range(len(phonemes)):
                            all_refs.append(phonemes[i, :phoneme_lengths[i]].tolist())
                
                from utils.metrics import phoneme_error_rate
                
                greedy_per = phoneme_error_rate(all_refs, all_hyps_greedy)
                greedy_acc = 100 - greedy_per
                
                log_dict = {
                    'epoch': epoch + 1,
                    f'model_{model_id}/greedy_acc': greedy_acc,
                }
                
                print(f"\n  Pure Greedy: {greedy_acc:.2f}%")
                
                # Only compute LM if epoch > 40
                if epoch + 1 > 40:
                    lm_per = phoneme_error_rate(all_refs, all_hyps_lm)
                    lm_acc = 100 - lm_per
                    
                    log_dict[f'model_{model_id}/lm_acc'] = lm_acc
                    log_dict[f'model_{model_id}/lm_boost'] = lm_acc - greedy_acc
                    
                    print(f"  + Fixed LM:  {lm_acc:.2f}% (+{lm_acc-greedy_acc:.2f}%)")
                else:
                    print(f"  (LM validation disabled until epoch 40)")
                
                wandb.log(log_dict)
                print(f"{'='*60}\n")
        
        # FINAL EVALUATION with EMA
        print(f"\n{'='*60}")
        print(f"Model {model_id}: FINAL EVALUATION")
        print(f"{'='*60}")
        
        trainer.model.eval()
        all_refs = []
        all_hyps_greedy = []
        all_hyps_lm = []
        
        with torch.no_grad():
            trainer.ema.apply_shadow()
            for batch in trainer.val_loader:
                neural = batch['neural'].to('cuda')
                phonemes = batch['phonemes']
                neural_lengths = batch['neural_lengths'].to('cuda')
                phoneme_lengths = batch['phoneme_lengths']
                
                logits, _ = trainer.model(neural, neural_lengths)
                
                decoded_greedy = lm_decoder.greedy_decode(logits)
                all_hyps_greedy.extend(decoded_greedy)
                
                decoded_lm = lm_decoder.greedy_with_lm_rescore(logits, top_k=5)
                all_hyps_lm.extend(decoded_lm)
                
                for i in range(len(phonemes)):
                    all_refs.append(phonemes[i, :phoneme_lengths[i]].tolist())
            
            trainer.ema.restore()
        
        final_greedy_per = phoneme_error_rate(all_refs, all_hyps_greedy)
        final_lm_per = phoneme_error_rate(all_refs, all_hyps_lm)
        
        final_greedy_acc = 100 - final_greedy_per
        final_lm_acc = 100 - final_lm_per
        
        # Save model
        trainer.save_checkpoint(f'v4_model_{model_id}_final.pt')
        
        wandb.log({
            f'model_{model_id}/final_greedy_acc': final_greedy_acc,
            f'model_{model_id}/final_lm_acc': final_lm_acc,
            f'model_{model_id}/final_lm_boost': final_lm_acc - final_greedy_acc,
        })
        
        print(f"\nModel {model_id} FINAL:")
        print(f"  Greedy: {final_greedy_acc:.2f}%")
        print(f"  + LM:   {final_lm_acc:.2f}%")
        print(f"{'='*60}\n")
        
        wandb.finish()
        
        return final_lm_acc
        
    except Exception as e:
        print(f"Model {model_id} failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        raise optuna.exceptions.TrialPruned()


def objective(trial):
    """Train 5 diverse models"""
    
    base_config = {
        'model': {
            'input_dim': 512,
            'subsampling': 8,
            'conv_kernel': 31,
            'num_classes': 42
        },
        'training': {
            'mixed_precision': True,
            'ema_decay': 0.9997
        },
        'loss': {
            'type': 'ctc_softce',
            'softmax_temperature': 1.5
        },
        'optimizer': {
            'type': 'lion',
            'betas': [0.9, 0.99],
            'weight_decay': 0.01
        },
        'specaugment': {
            'time_mask_count': 3,
            'freq_mask_count': 3,
        },
        'decoding': {
            'type': 'wfst',
            'beam_size': 64,
            'lm_scale': 0.8,
            'word_penalty': 0.2
        },
        'data': {
            'dataset_root': '/workspace/data/hdf5_data_final',
            'sessions': [
                't15.2023.08.11', 't15.2023.08.13', 't15.2023.08.18',
                't15.2023.08.20', 't15.2023.08.25', 't15.2023.08.27',
                't15.2023.09.01', 't15.2023.09.03', 't15.2023.09.24',
                't15.2023.09.29', 't15.2023.10.01', 't15.2023.10.06',
                't15.2023.10.08', 't15.2023.10.13', 't15.2023.10.15',
                't15.2023.10.20', 't15.2023.10.22', 't15.2023.11.03',
                't15.2023.11.04', 't15.2023.11.17', 't15.2023.11.19',
                't15.2023.11.26', 't15.2023.12.03', 't15.2023.12.08',
                't15.2023.12.10', 't15.2023.12.17', 't15.2023.12.29',
                't15.2024.02.25', 't15.2024.03.03', 't15.2024.03.08',
                't15.2024.03.15', 't15.2024.03.17', 't15.2024.04.25',
                't15.2024.04.28', 't15.2024.05.10', 't15.2024.06.14',
                't15.2024.07.19', 't15.2024.07.21', 't15.2024.07.28',
                't15.2025.01.10', 't15.2025.01.12', 't15.2025.03.14',
                't15.2025.03.16', 't15.2025.03.30', 't15.2025.04.13'
            ],
            'dataset_probability_val': [
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]
        }
    }
    
    # Train 5 models
    accuracies = []
    for model_id in range(5):
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL {model_id + 1}/5")
        print(f"{'='*80}\n")
        
        acc = train_single_model(trial, model_id, base_config)
        accuracies.append(acc)
        
        print(f"\n✓ Model {model_id + 1} Final: {acc:.2f}%\n")
    
    # Return average
    avg_accuracy = np.mean(accuracies)
    
    print(f"\n{'='*80}")
    print(f"TRIAL COMPLETE")
    print(f"{'='*80}")
    print(f"\nIndividual: {[f'{a:.2f}%' for a in accuracies]}")
    print(f"Average:    {avg_accuracy:.2f}%")
    print(f"Expected ensemble: {avg_accuracy + 1.5:.1f}%")
    print(f"Expected WFST:     {avg_accuracy + 4:.1f}%")
    print(f"{'='*80}\n")
    
    return avg_accuracy


def run_training(n_trials=1):
    """Run Optuna optimization"""
    
    study = optuna.create_study(
        study_name='ultimate-95-v4',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print("="*80)
    print("V4: LM ENABLED AFTER EPOCH 40")
    print("="*80)
    print(f"\nW&B Project: ieeg-decoder-95pct-v4")
    print(f"Strategy:")
    print(f"  Epoch 1-40:  Pure greedy validation (LM disabled)")
    print(f"  Epoch 41+:   Greedy + LM validation (LM enabled)")
    print(f"\nTarget: 88-90% individual, 95%+ final")
    print("="*80 + "\n")
    
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nBest: {study.best_value:.2f}%")
    print(f"Expected final: {study.best_value + 4:.1f}%")
    print("\n5 Best Models:")
    for i in range(5):
        print(f"  v4_model_{i}_final.pt")
    print("="*80 + "\n")
    
    return study


if __name__ == '__main__':
    study = run_training(n_trials=1)