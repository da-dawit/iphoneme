"""
WFST Beam Search Evaluation for Best Model
Optimized for maximum accuracy with proper beam search + 6-gram LM

Expected boost: +3-5% over greedy (89.5% → 93-94%)
"""
import torch
import yaml
import pickle
import numpy as np
from pathlib import Path
from train import Trainer
from tqdm import tqdm


class WFSTBeamDecoder:
    """
    WFST-style beam search decoder with n-gram LM
    Much better than greedy+LM because it explores multiple hypotheses
    """
    
    def __init__(self, lm_path, beam_size=64, lm_weight=1.0, length_penalty=1.0):
        """
        Args:
            lm_path: Path to n-gram LM
            beam_size: Number of hypotheses to track
            lm_weight: Weight for LM score (higher = more LM influence)
            length_penalty: Penalty for sequence length
        """
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty
        
        # Load LM
        print(f"Loading LM from {lm_path}")
        with open(lm_path, 'rb') as f:
            lm_data = pickle.load(f)
        
        if isinstance(lm_data, dict):
            self.lm = type('obj', (object,), {
                'order': lm_data.get('order', 6),
                'vocab': lm_data.get('vocab', set()),
                'ngram_counts': lm_data.get('ngram_counts', {})
            })()
        else:
            self.lm = lm_data
        
        print(f"✓ Loaded {self.lm.order}-gram LM")
        print(f"  Beam size: {beam_size}")
        print(f"  LM weight: {lm_weight}")
        print(f"  Length penalty: {length_penalty}")
    
    def decode(self, logits):
        """
        Beam search decode with LM
        
        Args:
            logits: [batch, time, vocab_size]
        Returns:
            List of decoded sequences
        """
        batch_size = logits.shape[0]
        decoded = []
        
        for b in range(batch_size):
            seq = self._beam_search(logits[b])
            decoded.append(seq)
        
        return decoded
    
    def _beam_search(self, logits):
        """
        Beam search for single sequence
        
        Args:
            logits: [time, vocab_size]
        Returns:
            Best decoded sequence
        """
        import math
        
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        T, V = log_probs.shape
        
        # Initialize beam
        # Each hypothesis: (sequence, last_phoneme, am_score, lm_score)
        beam = [([], None, 0.0, 0.0)]
        
        for t in range(T):
            candidates = []
            
            for seq, last_phoneme, am_score, lm_score in beam:
                # For each phoneme in vocabulary
                for phoneme in range(V):
                    # CTC: if same as last, just update score
                    if phoneme == last_phoneme:
                        new_am_score = am_score + log_probs[t, phoneme]
                        candidates.append((seq, last_phoneme, new_am_score, lm_score))
                        continue
                    
                    # Blank: don't add to sequence
                    if phoneme == 0:
                        new_am_score = am_score + log_probs[t, phoneme]
                        candidates.append((seq, phoneme, new_am_score, lm_score))
                        continue
                    
                    # Real phoneme: add to sequence and score with LM
                    new_seq = seq + [phoneme]
                    new_am_score = am_score + log_probs[t, phoneme]
                    new_lm_score = lm_score + self._get_lm_score(new_seq)
                    
                    candidates.append((new_seq, phoneme, new_am_score, new_lm_score))
            
            # Prune to beam size based on combined score
            scored_candidates = []
            for seq, last_p, am, lm in candidates:
                # Combined score with length penalty
                length = len(seq) if len(seq) > 0 else 1
                combined = am + self.lm_weight * lm
                combined = combined / (length ** self.length_penalty)
                scored_candidates.append((combined, seq, last_p, am, lm))
            
            # Keep top beam_size
            scored_candidates.sort(reverse=True)
            beam = [(seq, last_p, am, lm) for _, seq, last_p, am, lm in scored_candidates[:self.beam_size]]
        
        # Return best hypothesis
        if len(beam) > 0:
            return beam[0][0]
        return []
    
    def _get_lm_score(self, sequence):
        """Get LM score with backoff"""
        if len(sequence) == 0:
            return 0.0
        
        import math
        
        phoneme = sequence[-1]
        vocab_size = len(self.lm.vocab) if len(self.lm.vocab) > 0 else 40
        
        # Try longest context first, back off if not found
        max_context_size = min(len(sequence) - 1, self.lm.order - 1)
        
        for context_size in range(max_context_size, -1, -1):
            if context_size == 0:
                context = tuple()
            else:
                context = tuple(sequence[-context_size-1:-1])
            
            ngram = context + (phoneme,)
            ngram_level = len(ngram)
            
            ngram_count = self.lm.ngram_counts.get(ngram_level, {}).get(ngram, 0)
            
            if ngram_count > 0:
                if len(context) == 0:
                    total = sum(self.lm.ngram_counts.get(1, {}).values())
                    context_count = total if total > 0 else 1
                else:
                    context_count = self.lm.ngram_counts.get(len(context), {}).get(context, 0)
                
                if context_count > 0:
                    prob = ngram_count / context_count
                    return math.log(prob + 1e-10)
        
        return math.log(1.0 / vocab_size)


def evaluate_wfst(
    model_path='v4_model_1_final.pt',
    config_path='outputs/v4_model_1_config.yaml',
    lm_path='outputs/phoneme_lm.pkl',
    beam_size=64,
    lm_weight=1.0,
    length_penalty=1.0
):
    """
    Evaluate model with WFST beam search
    
    Args:
        model_path: Path to trained model
        config_path: Path to model config
        lm_path: Path to n-gram LM
        beam_size: Beam size (32-128, higher = better but slower)
        lm_weight: LM weight (0.5-2.0, higher = more LM influence)
        length_penalty: Length penalty (0.5-1.5, higher = prefer longer sequences)
    """
    print("="*80)
    print("WFST BEAM SEARCH EVALUATION")
    print("="*80)
    print(f"\nModel: {model_path}")
    print(f"Beam size: {beam_size}")
    print(f"LM weight: {lm_weight}")
    print(f"Length penalty: {length_penalty}")
    print("="*80 + "\n")
    
    # Load model
    trainer = Trainer(config_path, device='cuda')
    trainer.load_checkpoint(model_path)
    trainer.model.eval()
    
    # Create WFST decoder
    decoder = WFSTBeamDecoder(
        lm_path=lm_path,
        beam_size=beam_size,
        lm_weight=lm_weight,
        length_penalty=length_penalty
    )
    
    # Evaluate
    all_refs = []
    all_hyps = []
    
    print("\nRunning WFST beam search evaluation...")
    print("This will take ~30-60 minutes depending on beam size...\n")
    
    with torch.no_grad():
        trainer.ema.apply_shadow()
        
        for batch in tqdm(trainer.val_loader, desc="Decoding"):
            neural = batch['neural'].to('cuda')
            phonemes = batch['phonemes']
            neural_lengths = batch['neural_lengths'].to('cuda')
            phoneme_lengths = batch['phoneme_lengths']
            
            # Get logits
            logits, _ = trainer.model(neural, neural_lengths)
            
            # Beam search decode
            decoded = decoder.decode(logits)
            all_hyps.extend(decoded)
            
            # References
            for i in range(len(phonemes)):
                all_refs.append(phonemes[i, :phoneme_lengths[i]].tolist())
        
        trainer.ema.restore()
    
    # Calculate accuracy
    from utils.metrics import phoneme_error_rate
    
    per = phoneme_error_rate(all_refs, all_hyps)
    acc = 100 - per
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"WFST Accuracy: {acc:.2f}%")
    print(f"PER: {per:.2f}%")
    print(f"{'='*80}\n")
    
    return acc


def grid_search_wfst_params(model_path='v4_model_1_final.pt'):
    """
    Grid search over WFST parameters to find best settings
    """
    print("="*80)
    print("WFST PARAMETER GRID SEARCH")
    print("="*80 + "\n")
    
    # Parameter grid
    beam_sizes = [32, 64, 96]
    lm_weights = [0.7, 1.0, 1.3, 1.6]
    length_penalties = [0.8, 1.0, 1.2]
    
    best_acc = 0
    best_params = {}
    
    results = []
    
    for beam in beam_sizes:
        for lm_w in lm_weights:
            for len_p in length_penalties:
                print(f"\nTesting: beam={beam}, lm_weight={lm_w:.1f}, len_penalty={len_p:.1f}")
                
                try:
                    acc = evaluate_wfst(
                        model_path=model_path,
                        beam_size=beam,
                        lm_weight=lm_w,
                        length_penalty=len_p
                    )
                    
                    results.append({
                        'beam': beam,
                        'lm_weight': lm_w,
                        'length_penalty': len_p,
                        'accuracy': acc
                    })
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {
                            'beam': beam,
                            'lm_weight': lm_w,
                            'length_penalty': len_p
                        }
                    
                    print(f"✓ Accuracy: {acc:.2f}%")
                    
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    continue
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest accuracy: {best_acc:.2f}%")
    print(f"Best parameters:")
    print(f"  Beam size: {best_params['beam']}")
    print(f"  LM weight: {best_params['lm_weight']}")
    print(f"  Length penalty: {best_params['length_penalty']}")
    
    print(f"\nAll results:")
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"  {r['accuracy']:.2f}% - beam={r['beam']}, lm_w={r['lm_weight']:.1f}, len_p={r['length_penalty']:.1f}")
    
    print(f"{'='*80}\n")
    
    return best_params, best_acc


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'grid':
        # Grid search mode
        grid_search_wfst_params(model_path='v4_model_1_final.pt')
    else:
        # Single evaluation with optimized params
        # Based on best model: model 1 (89.54% greedy)
        evaluate_wfst(
            model_path='v4_model_1_final.pt',
            config_path='outputs/v4_model_1_config.yaml',
            lm_path='outputs/phoneme_lm.pkl',
            beam_size=96,           # Large beam for best results
            lm_weight=1.3,          # Higher LM influence
            length_penalty=1.0      # Balanced length
        )