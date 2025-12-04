"""
Build 8-gram LM and decode with phoneme sequence output
"""
import pickle
from collections import defaultdict
from dataset import iEEGPhonemeDataset
import torch
from train import Trainer
from eval_wfst_beam import WFSTBeamDecoder
from tqdm import tqdm


def build_8gram_lm():
    """Build 8-gram language model from T15 training data"""
    
    print("="*80)
    print("BUILDING 8-GRAM LANGUAGE MODEL")
    print("="*80 + "\n")
    
    sessions = [
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
    ]
    
    print("Loading T15 training data...")
    dataset = iEEGPhonemeDataset(
        dataset_root='/workspace/data/hdf5_data_final',
        sessions=sessions,
        split='train'
    )
    
    print(f"Extracting phoneme sequences from {len(dataset)} samples...")
    
    sequences = []
    for i in range(len(dataset)):
        _, phonemes, _, _ = dataset[i]
        seq = phonemes.tolist()
        if len(seq) > 0:
            sequences.append(seq)
    
    print(f"Extracted {len(sequences)} sequences")
    print(f"Building 8-gram LM...\n")
    
    # Build n-gram counts
    ngram_counts = {}
    vocab = set()
    
    for n in range(1, 9):  # 1-gram to 8-gram
        ngram_counts[n] = defaultdict(int)
    
    for seq in sequences:
        for phoneme in seq:
            vocab.add(phoneme)
        
        # Add start/end markers
        padded = [40] + seq + [40]  # 40 as boundary marker
        
        # Count n-grams
        for n in range(1, 9):
            for i in range(len(padded) - n + 1):
                ngram = tuple(padded[i:i+n])
                ngram_counts[n][ngram] += 1
    
    # Save LM
    lm_data = {
        'order': 8,
        'vocab': vocab,
        'ngram_counts': dict(ngram_counts)
    }
    
    with open('outputs/t15_8gram_lm.pkl', 'wb') as f:
        pickle.dump(lm_data, f)
    
    print(f"✓ Built 8-gram LM from T15 data")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Sequences: {len(sequences)}")
    for n in range(1, 9):
        print(f"  {n}-grams: {len(ngram_counts[n]):,}")
    print(f"\n✓ Saved to outputs/t15_8gram_lm.pkl")
    print("="*80 + "\n")


def decode_with_phoneme_output(
    model_path='v4_model_1_final.pt',
    config_path='outputs/v4_model_1_config.yaml',
    lm_path='outputs/t15_8gram_lm.pkl',
    output_file='phoneme_predictions.txt',
    beam_size=128,
    lm_weight=1.0
):
    """
    Decode validation set and output phoneme sequences
    """
    
    print("="*80)
    print("DECODING WITH 8-GRAM LM + PHONEME OUTPUT")
    print("="*80)
    print(f"\nModel: {model_path}")
    print(f"LM: {lm_path}")
    print(f"Beam: {beam_size}, LM weight: {lm_weight}")
    print(f"Output: {output_file}\n")
    print("="*80 + "\n")
    
    # Phoneme mapping (ARPAbet symbols)
    PHONEME_MAP = {
        0: '<blank>',
        1: 'AA', 2: 'AE', 3: 'AH', 4: 'AO', 5: 'AW',
        6: 'AY', 7: 'B', 8: 'CH', 9: 'D', 10: 'DH',
        11: 'EH', 12: 'ER', 13: 'EY', 14: 'F', 15: 'G',
        16: 'HH', 17: 'IH', 18: 'IY', 19: 'JH', 20: 'K',
        21: 'L', 22: 'M', 23: 'N', 24: 'NG', 25: 'OW',
        26: 'OY', 27: 'P', 28: 'R', 29: 'S', 30: 'SH',
        31: 'T', 32: 'TH', 33: 'UH', 34: 'UW', 35: 'V',
        36: 'W', 37: 'Y', 38: 'Z', 39: 'ZH', 40: 'SIL', 41: 'SP'
    }
    
    # Load model
    print("Loading model...")
    trainer = Trainer(config_path, device='cuda')
    trainer.load_checkpoint(model_path)
    trainer.model.eval()
    
    # Create decoder
    print("Loading 8-gram LM...")
    decoder = WFSTBeamDecoder(
        lm_path=lm_path,
        beam_size=beam_size,
        lm_weight=lm_weight,
        length_penalty=0.9
    )
    
    # Decode
    print("\nDecoding validation set...\n")
    
    all_refs = []
    all_hyps = []
    all_refs_text = []
    all_hyps_text = []
    
    with torch.no_grad():
        trainer.ema.apply_shadow()
        
        for batch_idx, batch in enumerate(tqdm(trainer.val_loader, desc="Decoding")):
            neural = batch['neural'].to('cuda')
            phonemes = batch['phonemes']
            neural_lengths = batch['neural_lengths'].to('cuda')
            phoneme_lengths = batch['phoneme_lengths']
            
            # Get logits
            logits, _ = trainer.model(neural, neural_lengths)
            
            # Decode with 8-gram WFST
            decoded = decoder.decode(logits)
            all_hyps.extend(decoded)
            
            # References
            for i in range(len(phonemes)):
                ref_seq = phonemes[i, :phoneme_lengths[i]].tolist()
                all_refs.append(ref_seq)
                
                # Convert to text
                ref_text = ' '.join([PHONEME_MAP.get(p, f'UNK{p}') for p in ref_seq])
                all_refs_text.append(ref_text)
            
            # Hypotheses to text
            for hyp_seq in decoded:
                hyp_text = ' '.join([PHONEME_MAP.get(p, f'UNK{p}') for p in hyp_seq])
                all_hyps_text.append(hyp_text)
        
        trainer.ema.restore()
    
    # Calculate accuracy
    from utils.metrics import phoneme_error_rate
    per = phoneme_error_rate(all_refs, all_hyps)
    acc = 100 - per
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"8-gram WFST Accuracy: {acc:.2f}%")
    print(f"PER: {per:.2f}%")
    print(f"Total samples: {len(all_refs)}")
    print(f"{'='*80}\n")
    
    # Save predictions to file
    print(f"Writing predictions to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"8-GRAM WFST DECODING RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"PER: {per:.2f}%\n")
        f.write(f"Beam size: {beam_size}\n")
        f.write(f"LM weight: {lm_weight}\n")
        f.write("="*80 + "\n\n")
        
        for i in range(len(all_refs)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  REF: {all_refs_text[i]}\n")
            f.write(f"  HYP: {all_hyps_text[i]}\n")
            
            # Calculate sample-level accuracy
            ref_seq = all_refs[i]
            hyp_seq = all_hyps[i]
            
            if ref_seq == hyp_seq:
                f.write(f"  ✓ CORRECT\n")
            else:
                sample_per = phoneme_error_rate([ref_seq], [hyp_seq])
                f.write(f"  ✗ PER: {sample_per:.1f}%\n")
            
            f.write("\n")
    
    print(f"✓ Saved predictions to {output_file}")
    print(f"✓ Also saved to /mnt/user-data/outputs/{output_file}\n")
    
    # Copy to outputs for download
    import shutil
    shutil.copy(output_file, f'/mnt/user-data/outputs/{output_file}')
    
    return acc


if __name__ == '__main__':
    import sys
    
    # Step 1: Build 8-gram LM
    print("Step 1: Building 8-gram LM...\n")
    build_8gram_lm()
    
    # Step 2: Decode with phoneme output
    print("\nStep 2: Decoding with 8-gram LM...\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'ensemble':
        print("TODO: Implement ensemble with 8-gram")
    else:
        # Single model decode
        acc = decode_with_phoneme_output(
            model_path='v4_model_1_final.pt',
            config_path='outputs/v4_model_1_config.yaml',
            lm_path='outputs/t15_8gram_lm.pkl',
            output_file='phoneme_predictions_8gram.txt',
            beam_size=128,
            lm_weight=1.0
        )
        
        print(f"\n{'='*80}")
        print(f"COMPLETE!")
        print(f"{'='*80}")
        print(f"\nFinal accuracy with 8-gram LM: {acc:.2f}%")
        print(f"Predictions saved to: phoneme_predictions_8gram.txt")
        print(f"{'='*80}\n")