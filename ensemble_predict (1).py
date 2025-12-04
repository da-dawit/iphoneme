"""
Ensemble Prediction: Combine 5 models for 92-93% accuracy
"""
import torch
import yaml
import numpy as np
from pathlib import Path
from train import Trainer
from tqdm import tqdm


def load_ensemble_models(model_ids=[0, 1, 2, 3, 4]):
    """Load 5 trained models"""
    models = []
    configs = []
    
    for model_id in model_ids:
        config_path = f'outputs/ensemble_model_{model_id}_config.yaml'
        checkpoint_path = f'outputs/ensemble_model_{model_id}_final.pt'
        
        print(f"Loading model {model_id}...")
        trainer = Trainer(config_path, device='cuda')
        trainer.load_checkpoint(f'ensemble_model_{model_id}_final.pt')
        trainer.ema.apply_shadow()  # Use EMA weights
        trainer.model.eval()
        
        models.append(trainer.model)
        configs.append(trainer.config)
    
    print(f"✓ Loaded {len(models)} models")
    return models, configs


def ensemble_predict_greedy(models, val_loader):
    """
    Ensemble prediction with greedy decoding
    Strategy: Average logits from all models
    """
    all_refs = []
    all_hyps = []
    
    print("\nRunning ensemble greedy prediction...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            neural = batch['neural'].to('cuda')
            phonemes = batch['phonemes']
            neural_lengths = batch['neural_lengths'].to('cuda')
            phoneme_lengths = batch['phoneme_lengths']
            
            # Collect logits from all models
            all_logits = []
            for model in models:
                logits, _ = model(neural, neural_lengths)
                all_logits.append(logits)
            
            # Average logits
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
            
            # Greedy decode
            predictions = torch.argmax(ensemble_logits, dim=-1)
            
            # CTC collapse
            for pred in predictions:
                collapsed = []
                prev = None
                for p in pred.tolist():
                    if p != prev:
                        collapsed.append(p)
                        prev = p
                decoded_seq = [p for p in collapsed if p != 0]  # Remove blanks
                all_hyps.append(decoded_seq)
            
            # Collect references
            for i in range(len(phonemes)):
                all_refs.append(phonemes[i, :phoneme_lengths[i]].tolist())
    
    return all_refs, all_hyps


def ensemble_predict_wfst(models, val_loader, decoder):
    """
    Ensemble prediction with WFST decoding
    Strategy: Average logits, then WFST decode
    """
    all_refs = []
    all_hyps = []
    
    print("\nRunning ensemble WFST prediction (this will take time)...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            neural = batch['neural'].to('cuda')
            phonemes = batch['phonemes']
            neural_lengths = batch['neural_lengths'].to('cuda')
            phoneme_lengths = batch['phoneme_lengths']
            
            # Collect logits from all models
            all_logits = []
            for model in models:
                logits, _ = model(neural, neural_lengths)
                all_logits.append(logits)
            
            # Average logits
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
            
            # WFST decode
            decoded = decoder.decode(ensemble_logits)
            all_hyps.extend(decoded)
            
            # Collect references
            for i in range(len(phonemes)):
                all_refs.append(phonemes[i, :phoneme_lengths[i]].tolist())
    
    return all_refs, all_hyps


def main():
    """
    Main ensemble evaluation script
    """
    print("="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)
    
    # Load models
    models, configs = load_ensemble_models()
    
    # Setup validation loader (use first model's trainer for this)
    print("\nSetting up validation data...")
    trainer = Trainer(f'outputs/ensemble_model_0_config.yaml', device='cuda')
    val_loader = trainer.val_loader
    
    # 1. Greedy ensemble evaluation
    print("\n" + "="*60)
    print("GREEDY ENSEMBLE EVALUATION")
    print("="*60)
    
    refs_greedy, hyps_greedy = ensemble_predict_greedy(models, val_loader)
    
    from utils.metrics import phoneme_error_rate
    greedy_per = phoneme_error_rate(refs_greedy, hyps_greedy)
    greedy_acc = 100 - greedy_per
    
    print(f"\n{'='*60}")
    print(f"GREEDY ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Greedy PER: {greedy_per:.2f}%")
    print(f"Greedy Accuracy: {greedy_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # 2. WFST ensemble evaluation
    use_wfst = input("\nRun WFST ensemble? (will take 1-2 hours) [y/N]: ").lower() == 'y'
    
    if use_wfst:
        print("\n" + "="*60)
        print("WFST ENSEMBLE EVALUATION")
        print("="*60)
        
        # Update decoder config for larger beam
        wfst_config = configs[0].copy()
        wfst_config['decoding']['beam_size'] = 64  # Larger beam for final eval
        
        # Save temp config
        with open('outputs/wfst_eval_config.yaml', 'w') as f:
            yaml.dump(wfst_config, f)
        
        # Create decoder with larger beam
        wfst_trainer = Trainer('outputs/wfst_eval_config.yaml', device='cuda')
        decoder = wfst_trainer.decoder
        
        refs_wfst, hyps_wfst = ensemble_predict_wfst(models, val_loader, decoder)
        
        wfst_per = phoneme_error_rate(refs_wfst, hyps_wfst)
        wfst_acc = 100 - wfst_per
        
        print(f"\n{'='*60}")
        print(f"WFST ENSEMBLE RESULTS")
        print(f"{'='*60}")
        print(f"WFST PER: {wfst_per:.2f}%")
        print(f"WFST Accuracy: {wfst_acc:.2f}%")
        print(f"{'='*60}\n")
        
        print(f"\nImprovement over single model:")
        print(f"  Single model WFST: 89.5%")
        print(f"  Ensemble WFST: {wfst_acc:.2f}%")
        print(f"  Gain: +{wfst_acc - 89.5:.2f}%")
    
    # Save results
    results = {
        'greedy_accuracy': greedy_acc,
        'greedy_per': greedy_per,
    }
    
    if use_wfst:
        results['wfst_accuracy'] = wfst_acc
        results['wfst_per'] = wfst_per
    
    import json
    with open('outputs/ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to outputs/ensemble_results.json")


if __name__ == '__main__':
    main()