"""
Greedy Decoder with N-gram Language Model Rescoring - FIXED VERSION
Fast alternative to WFST that still uses the LM
KEY FIX: Proper backoff for unseen n-grams instead of useless Laplace smoothing
"""
import torch
import pickle
from pathlib import Path


class GreedyLMDecoder:
    """
    Greedy CTC decoder with n-gram LM rescoring
    Much faster than WFST but still benefits from LM
    """
    
    def __init__(self, lm_path='outputs/phoneme_lm.pkl', lm_weight=0.5):
        """
        Args:
            lm_path: Path to n-gram language model
            lm_weight: How much to weight LM vs acoustic model (0-1)
        """
        self.lm_weight = lm_weight
        
        # Load LM
        if Path(lm_path).exists():
            print(f"Loading LM from {lm_path}")
            with open(lm_path, 'rb') as f:
                lm_data = pickle.load(f)
            
            # Handle both dict and object formats
            if isinstance(lm_data, dict):
                # LM saved as dict - reconstruct object
                from decoding.wfst_decoder import NgramLanguageModel
                self.lm = NgramLanguageModel(order=lm_data.get('order', 6))
                self.lm.vocab = lm_data.get('vocab', set())
                self.lm.ngram_counts = lm_data.get('ngram_counts', {})
                print(f"✓ Loaded {self.lm.order}-gram LM from dict")
            else:
                # LM saved as object
                self.lm = lm_data
                print(f"✓ Loaded {self.lm.order}-gram LM")
        else:
            print(f"⚠️  LM not found at {lm_path}")
            self.lm = None
    
    def greedy_decode(self, logits):
        """
        Standard greedy CTC decode (no LM)
        
        Args:
            logits: [batch, time, num_classes]
        Returns:
            List of decoded sequences
        """
        predictions = torch.argmax(logits, dim=-1)  # [batch, time]
        
        decoded = []
        for pred in predictions:
            # CTC collapse
            collapsed = []
            prev = None
            for p in pred.tolist():
                if p != prev:
                    collapsed.append(p)
                    prev = p
            
            # Remove blanks (index 0)
            seq = [p for p in collapsed if p != 0]
            decoded.append(seq)
        
        return decoded
    
    def greedy_with_lm_rescore(self, logits, top_k=3):
        """
        Greedy decode with LM rescoring
        For each timestep, consider top-k phonemes and pick best with LM
        
        Args:
            logits: [batch, time, num_classes]
            top_k: Number of alternatives to consider per frame
        Returns:
            List of decoded sequences
        """
        if self.lm is None:
            print("⚠️  No LM loaded, using pure greedy")
            return self.greedy_decode(logits)
        
        batch_size = logits.shape[0]
        decoded = []
        
        for b in range(batch_size):
            seq = self._decode_with_lm_simple(logits[b], top_k)
            decoded.append(seq)
        
        return decoded
    
    def _decode_with_lm_simple(self, logits, top_k):
        """
        Simple greedy with LM: at each step, pick from top-k using LM
        
        Args:
            logits: [time, num_classes]
            top_k: Consider top-k alternatives
        Returns:
            Decoded sequence
        """
        import numpy as np
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        T, C = probs.shape
        
        sequence = []
        last_phoneme = None
        
        for t in range(T):
            # Get top-k phonemes at this timestep
            top_k_indices = probs[t].argsort()[-top_k:][::-1]
            
            best_phoneme = None
            best_score = -float('inf')
            
            for phoneme_idx in top_k_indices:
                phoneme = int(phoneme_idx)
                am_score = np.log(probs[t, phoneme] + 1e-10)
                
                # CTC: skip if same as last (would collapse anyway)
                if phoneme == last_phoneme:
                    score = am_score
                    if score > best_score:
                        best_score = score
                        best_phoneme = phoneme
                    continue
                
                # Blank: don't add to sequence
                if phoneme == 0:
                    score = am_score
                    if score > best_score:
                        best_score = score
                        best_phoneme = phoneme
                    continue
                
                # Real phoneme: get LM score
                temp_seq = sequence + [phoneme]
                lm_score = self._get_lm_score_fast(temp_seq)
                
                # Combined score
                score = am_score + self.lm_weight * lm_score
                
                if score > best_score:
                    best_score = score
                    best_phoneme = phoneme
            
            # Add best phoneme
            if best_phoneme is not None and best_phoneme != 0 and best_phoneme != last_phoneme:
                sequence.append(best_phoneme)
            
            last_phoneme = best_phoneme
        
        return sequence
    
    def _get_lm_score_fast(self, sequence):
        """
        Fast LM scoring with PROPER BACKOFF for unseen n-grams
        
        KEY FIX: Instead of giving all unseen n-grams the same prob (1/40),
        we back off to shorter context until we find seen data.
        
        Example: If (w1, w2, w3, w4) unseen, try (w2, w3, w4), then (w3, w4), etc.
        """
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
            
            # Check if this n-gram exists
            ngram_count = self.lm.ngram_counts.get(ngram_level, {}).get(ngram, 0)
            
            if ngram_count > 0:
                # Found it! Calculate probability
                if len(context) == 0:
                    # Unigram probability
                    total = sum(self.lm.ngram_counts.get(1, {}).values())
                    context_count = total if total > 0 else 1
                else:
                    context_count = self.lm.ngram_counts.get(len(context), {}).get(context, 0)
                
                if context_count > 0:
                    # Use actual probability, not Laplace (Laplace only for smoothing)
                    prob = ngram_count / context_count
                    return math.log(prob + 1e-10)
        
        # Ultimate fallback: uniform probability
        return math.log(1.0 / vocab_size)


def main():
    """Test greedy+LM decoder"""
    from train import Trainer
    
    print("="*60)
    print("GREEDY + 6-GRAM LM EVALUATION (FIXED VERSION)")
    print("="*60)
    
    # Load model
    trainer = Trainer('config.yaml', 'cuda')
    trainer.load_checkpoint('final_model.pt')
    trainer.model.eval()
    
    # Create greedy+LM decoder
    decoder = GreedyLMDecoder(
        lm_path='outputs/phoneme_lm.pkl',
        lm_weight=0.7  # Tune this: higher = more LM influence
    )
    
    # Validate with greedy+LM
    all_refs = []
    all_hyps_greedy = []
    all_hyps_lm = []
    
    print("\nRunning validation...")
    with torch.no_grad():
        trainer.ema.apply_shadow()
        for batch in trainer.val_loader:
            neural = batch['neural'].to('cuda')
            phonemes = batch['phonemes']
            neural_lengths = batch['neural_lengths'].to('cuda')
            phoneme_lengths = batch['phoneme_lengths']
            
            logits, _ = trainer.model(neural, neural_lengths)
            
            # Pure greedy
            decoded_greedy = decoder.greedy_decode(logits)
            all_hyps_greedy.extend(decoded_greedy)
            
            # Greedy + LM rescore
            decoded_lm = decoder.greedy_with_lm_rescore(logits, top_k=5)
            all_hyps_lm.extend(decoded_lm)
            
            # References
            for i in range(len(phonemes)):
                all_refs.append(phonemes[i, :phoneme_lengths[i]].tolist())
        
        trainer.ema.restore()
    
    # Calculate PER
    from utils.metrics import phoneme_error_rate
    
    greedy_per = phoneme_error_rate(all_refs, all_hyps_greedy)
    lm_per = phoneme_error_rate(all_refs, all_hyps_lm)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Pure Greedy: {100 - greedy_per:.2f}% accuracy")
    print(f"Greedy + 6-gram LM (FIXED): {100 - lm_per:.2f}% accuracy")
    print(f"Improvement: +{greedy_per - lm_per:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()