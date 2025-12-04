"""
WFST Optimization to Reach 93%+

Current: 91.62% (beam=64, lm_weight=0.8)
Target: 93%+

Strategies:
1. Larger beam size (64 → 128)
2. Optimize LM weight (0.8 → grid search)
3. Add word insertion penalty
4. Ensemble predictions from top-5 models
"""
import torch
import yaml
import pickle
import numpy as np
from pathlib import Path
from train import Trainer
from tqdm import tqdm
from eval_wfst_beam import WFSTBeamDecoder


def fine_grained_grid_search():
    """
    Fine-grained grid search around best params
    Focus on reaching 93%+
    """
    print("="*80)
    print("FINE-GRAINED WFST OPTIMIZATION FOR 93%+")
    print("="*80)
    print(f"\nCurrent best: 91.62% (beam=64, lm_weight=0.8)")
    print(f"Target: 93%+\n")
    print("="*80 + "\n")
    
    # Load model
    trainer = Trainer('outputs/v4_model_1_config.yaml', device='cuda')
    trainer.load_checkpoint('v4_model_1_final.pt')
    trainer.model.eval()
    
    # Fine-grained parameter grid
    beam_sizes = [64, 96, 128]
    lm_weights = [0.6, 0.7, 0.8, 0.9, 1.0]
    length_penalties = [0.9, 1.0, 1.1]
    
    best_acc = 91.62
    best_params = {'beam': 64, 'lm_weight': 0.8, 'length_penalty': 1.0}
    
    results = []
    total_tests = len(beam_sizes) * len(lm_weights) * len(length_penalties)
    test_num = 0
    
    for beam in beam_sizes:
        for lm_w in lm_weights:
            for len_p in length_penalties:
                test_num += 1
                print(f"\n[{test_num}/{total_tests}] Testing: beam={beam}, lm_w={lm_w:.1f}, len_p={len_p:.1f}")
                
                try:
                    # Create decoder
                    decoder = WFSTBeamDecoder(
                        lm_path='outputs/t15_phoneme_lm.pkl',
                        beam_size=beam,
                        lm_weight=lm_w,
                        length_penalty=len_p
                    )
                    
                    # Evaluate
                    all_refs = []
                    all_hyps = []
                    
                    with torch.no_grad():
                        trainer.ema.apply_shadow()
                        
                        for batch in tqdm(trainer.val_loader, desc="Decoding", leave=False):
                            neural = batch['neural'].to('cuda')
                            phonemes = batch['phonemes']
                            neural_lengths = batch['neural_lengths'].to('cuda')
                            phoneme_lengths = batch['phoneme_lengths']
                            
                            logits, _ = trainer.model(neural, neural_lengths)
                            decoded = decoder.decode(logits)
                            all_hyps.extend(decoded)
                            
                            for i in range(len(phonemes)):
                                all_refs.append(phonemes[i, :phoneme_lengths[i]].tolist())
                        
                        trainer.ema.restore()
                    
                    from utils.metrics import phoneme_error_rate
                    per = phoneme_error_rate(all_refs, all_hyps)
                    acc = 100 - per
                    
                    results.append({
                        'beam': beam,
                        'lm_weight': lm_w,
                        'length_penalty': len_p,
                        'accuracy': acc
                    })
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {'beam': beam, 'lm_weight': lm_w, 'length_penalty': len_p}
                        print(f"✓ NEW BEST: {acc:.2f}%")
                    else:
                        print(f"  Result: {acc:.2f}%")
                    
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    continue
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest accuracy: {best_acc:.2f}%")
    print(f"Best parameters:")
    print(f"  Beam size: {best_params['beam']}")
    print(f"  LM weight: {best_params['lm_weight']}")
    print(f"  Length penalty: {best_params['length_penalty']}")
    
    print(f"\nTop 10 results:")
    for i, r in enumerate(sorted(results, key=lambda x: x['accuracy'], reverse=True)[:10], 1):
        print(f"  {i}. {r['accuracy']:.2f}% - beam={r['beam']}, lm={r['lm_weight']:.1f}, len={r['length_penalty']:.1f}")
    
    print(f"{'='*80}\n")
    
    # Save results
    with open('wfst_optimization_results.pkl', 'wb') as f:
        pickle.dump({'results': results, 'best': best_params, 'best_acc': best_acc}, f)
    
    return best_params, best_acc


def ensemble_top5_models():
    """
    Ensemble predictions from all 5 trained models
    Expected boost: +0.5-1.0%
    """
    print("="*80)
    print("ENSEMBLE WFST EVALUATION (5 MODELS)")
    print("="*80 + "\n")
    
    # Load all 5 models
    models = []
    for i in range(5):
        config_path = f'outputs/v4_model_{i}_config.yaml'
        model_path = f'v4_model_{i}_final.pt'
        
        if not Path(model_path).exists():
            print(f"⚠️  Model {i} not found, skipping")
            continue
        
        print(f"Loading model {i}...")
        trainer = Trainer(config_path, device='cuda')
        trainer.load_checkpoint(model_path)
        trainer.model.eval()
        models.append(trainer)
    
    print(f"\n✓ Loaded {len(models)} models")
    
    # Use best WFST params from grid search
    decoder = WFSTBeamDecoder(
        lm_path='outputs/t15_phoneme_lm.pkl',
        beam_size=128,
        lm_weight=0.9,
        length_penalty=1.0
    )
    
    # Ensemble decode
    all_refs = []
    all_hyps = []
    
    print("\nRunning ensemble WFST decoding...")
    
    with torch.no_grad():
        # Use first model's dataloader
        for batch in tqdm(models[0].val_loader, desc="Ensemble decoding"):
            neural = batch['neural'].to('cuda')
            phonemes = batch['phonemes']
            neural_lengths = batch['neural_lengths'].to('cuda')
            phoneme_lengths = batch['phoneme_lengths']
            
            # Get logits from all models
            all_logits = []
            for trainer in models:
                trainer.ema.apply_shadow()
                logits, _ = trainer.model(neural, neural_lengths)
                all_logits.append(logits)
                trainer.ema.restore()
            
            # Average logits (probability averaging)
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
            
            # Decode ensemble
            decoded = decoder.decode(ensemble_logits)
            all_hyps.extend(decoded)
            
            # References
            for i in range(len(phonemes)):
                all_refs.append(phonemes[i, :phoneme_lengths[i]].tolist())
    
    from utils.metrics import phoneme_error_rate
    per = phoneme_error_rate(all_refs, all_hyps)
    acc = 100 - per
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE RESULTS")
    print(f"{'='*80}")
    print(f"Ensemble Accuracy: {acc:.2f}%")
    print(f"PER: {per:.2f}%")
    print(f"{'='*80}\n")
    
    return acc


def optimize_for_93_plus():
    """
    Complete optimization pipeline to reach 93%+
    """
    print("="*80)
    print("COMPLETE OPTIMIZATION PIPELINE FOR 93%+")
    print("="*80 + "\n")
    
    print("Stage 1: Fine-grained grid search on Model 1...")
    best_params, best_single = fine_grained_grid_search()
    
    print(f"\nStage 1 complete: Best single model = {best_single:.2f}%\n")
    
    if best_single >= 93.0:
        print("✓ Already reached 93%+ with single model!")
        return best_single
    
    print("Stage 2: Ensemble all 5 models...")
    ensemble_acc = ensemble_top5_models()
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best single model: {best_single:.2f}%")
    print(f"Ensemble (5 models): {ensemble_acc:.2f}%")
    
    if ensemble_acc >= 93.0:
        print(f"\n✓ TARGET REACHED: {ensemble_acc:.2f}% ≥ 93%")
    else:
        print(f"\n⚠️  Close but not quite: {ensemble_acc:.2f}% < 93%")
        print(f"   Gap: {93.0 - ensemble_acc:.2f}%")
        print(f"\nSuggestions:")
        print(f"  1. Train longer (120 → 150 epochs)")
        print(f"  2. Use even larger models (d_model 672 → 768)")
        print(f"  3. Collect more training data")
    
    print(f"{'='*80}\n")
    
    return ensemble_acc


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'grid':
            # Just grid search
            fine_grained_grid_search()
        elif sys.argv[1] == 'ensemble':
            # Just ensemble
            ensemble_top5_models()
        else:
            # Full pipeline
            optimize_for_93_plus()
    else:
        # Default: full pipeline
        optimize_for_93_plus()