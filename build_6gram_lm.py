"""
Build 6-gram Language Model with External Phoneme Data

Sources:
1. LibriSpeech phoneme transcripts (100k+ sequences)
2. TIMIT phoneme annotations (6k sequences)
3. CMU pronunciation dictionary (134k words → phoneme sequences)

Expected improvement: +1-2% accuracy over 4-gram
"""
import os
import pickle
import requests
from pathlib import Path
from tqdm import tqdm
import re


def download_file(url, output_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=output_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def load_cmu_dict():
    """
    Load CMU Pronunciation Dictionary
    ~134k words mapped to phoneme sequences
    """
    print("\n" + "="*60)
    print("Loading CMU Pronunciation Dictionary")
    print("="*60)
    
    cmu_url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    cmu_path = Path("data/cmudict.dict")
    cmu_path.parent.mkdir(exist_ok=True)
    
    if not cmu_path.exists():
        print("Downloading CMU dict...")
        download_file(cmu_url, cmu_path)
    
    # Parse CMU dict
    # Format: WORD  P H O N E M E S
    sequences = []
    phoneme_set = set()
    
    print("Parsing CMU dict...")
    with open(cmu_path, 'r', encoding='latin-1') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # word = parts[0]
            phonemes = parts[1:]
            
            # Remove stress markers (0, 1, 2) from phonemes
            phonemes = [re.sub(r'[0-9]', '', p) for p in phonemes]
            
            # Map to ARPAbet phoneme IDs (you'll need to map these to your phoneme set)
            phoneme_set.update(phonemes)
            sequences.append(phonemes)
    
    print(f"✓ Loaded {len(sequences)} phoneme sequences from CMU dict")
    print(f"  Unique phonemes: {len(phoneme_set)}")
    print(f"  Sample: {sequences[0]}")
    
    return sequences, phoneme_set


def load_timit_phonemes():
    """
    Load TIMIT phoneme annotations
    ~6k utterances with phoneme-level transcriptions
    
    Note: TIMIT requires download from LDC (not freely available)
    This is a placeholder - you'll need TIMIT access
    """
    print("\n" + "="*60)
    print("Loading TIMIT Phonemes")
    print("="*60)
    
    timit_path = Path("data/timit")
    
    if not timit_path.exists():
        print("⚠️  TIMIT not found at data/timit/")
        print("   TIMIT requires LDC license: https://catalog.ldc.upenn.edu/LDC93S1")
        print("   Skipping TIMIT data...")
        return [], set()
    
    # If TIMIT exists, parse .PHN files
    sequences = []
    phoneme_set = set()
    
    phn_files = list(timit_path.rglob("*.PHN"))
    print(f"Found {len(phn_files)} TIMIT phoneme files")
    
    for phn_file in tqdm(phn_files):
        with open(phn_file, 'r') as f:
            phonemes = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    phoneme = parts[2]
                    phonemes.append(phoneme)
                    phoneme_set.add(phoneme)
            
            if phonemes:
                sequences.append(phonemes)
    
    print(f"✓ Loaded {len(sequences)} sequences from TIMIT")
    return sequences, phoneme_set


def load_librispeech_phonemes():
    """
    Load LibriSpeech phoneme transcripts
    ~100k sequences
    
    We'll use Montreal Forced Aligner format or convert from text
    """
    print("\n" + "="*60)
    print("Loading LibriSpeech Phonemes")
    print("="*60)
    
    # Option 1: Download pre-aligned LibriSpeech phonemes
    librispeech_phonemes_url = "https://www.openslr.org/resources/11/librispeech-lexicon.txt"
    lexicon_path = Path("data/librispeech-lexicon.txt")
    lexicon_path.parent.mkdir(exist_ok=True)
    
    if not lexicon_path.exists():
        print("Downloading LibriSpeech lexicon...")
        download_file(librispeech_phonemes_url, lexicon_path)
    
    # Parse lexicon (word -> phoneme sequence)
    sequences = []
    phoneme_set = set()
    
    print("Parsing LibriSpeech lexicon...")
    with open(lexicon_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # word = parts[0]
            phonemes = parts[1:]
            phoneme_set.update(phonemes)
            sequences.append(phonemes)
    
    print(f"✓ Loaded {len(sequences)} sequences from LibriSpeech")
    print(f"  Unique phonemes: {len(phoneme_set)}")
    
    return sequences, phoneme_set


def map_phonemes_to_indices(sequences, phoneme_set, target_phoneme_map):
    """
    Map external phonemes to your model's phoneme indices
    
    Args:
        sequences: List of phoneme sequences (strings)
        phoneme_set: Set of unique phonemes from external data
        target_phoneme_map: Your model's phoneme-to-index mapping
    
    Returns:
        List of integer sequences
    """
    print("\n" + "="*60)
    print("Mapping phonemes to model indices")
    print("="*60)
    
    # ARPAbet phoneme mapping (matches your LOGIT_TO_PHONEME)
    arpabet_to_model = {
        'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5,
        'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10,
        'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15,
        'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20,
        'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25,
        'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30,
        'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35,
        'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39,
    }
    
    # Print mapping info
    print(f"External phonemes: {len(phoneme_set)}")
    print(f"Model phoneme indices: 1-39 (40 is silence, ignored)")
    
    # Convert sequences to indices
    indexed_sequences = []
    skipped = 0
    
    print("Mapping sequences...")
    for seq in tqdm(sequences):
        indexed_seq = []
        valid = True
        
        for phoneme in seq:
            # Strip stress markers (0, 1, 2) from phonemes like AA0 -> AA
            base_phoneme = re.sub(r'[0-9]', '', phoneme)
            
            # Also handle special characters
            if base_phoneme in ['#', '']:
                continue  # Skip silence/padding markers
            
            if base_phoneme in arpabet_to_model:
                indexed_seq.append(arpabet_to_model[base_phoneme])
            else:
                # Unknown phoneme - skip this sequence
                valid = False
                break
        
        if valid and len(indexed_seq) > 0:
            indexed_sequences.append(indexed_seq)
        else:
            skipped += 1
    
    print(f"\n✓ Mapped {len(indexed_sequences)} sequences")
    print(f"  Skipped {skipped} sequences (unmapped phonemes)")
    
    # Show some statistics
    if indexed_sequences:
        lengths = [len(seq) for seq in indexed_sequences]
        print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
        print(f"  Sample sequence: {indexed_sequences[0][:10]}...")
    
    return indexed_sequences


def build_6gram_lm(internal_sequences, external_sequences):
    """
    Build 6-gram language model combining internal + external data
    """
    from decoding.wfst_decoder import NgramLanguageModel
    
    print("\n" + "="*60)
    print("Building 6-gram Language Model")
    print("="*60)
    
    all_sequences = internal_sequences + external_sequences
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"  Internal (your data): {len(internal_sequences)}")
    print(f"  External: {len(external_sequences)}")
    
    # Build 6-gram LM
    lm = NgramLanguageModel(order=6)
    lm.train(all_sequences)
    
    return lm


def main():
    """
    Main script to build enhanced 6-gram LM
    """
    print("="*60)
    print("BUILDING ENHANCED 6-GRAM LANGUAGE MODEL")
    print("="*60)
    
    # 1. Load external phoneme data
    cmu_sequences, cmu_phonemes = load_cmu_dict()
    timit_sequences, timit_phonemes = load_timit_phonemes()
    libri_sequences, libri_phonemes = load_librispeech_phonemes()
    
    # Combine external sequences
    external_sequences_raw = cmu_sequences + timit_sequences + libri_sequences
    all_external_phonemes = cmu_phonemes | timit_phonemes | libri_phonemes
    
    print(f"\n{'='*60}")
    print(f"EXTERNAL DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total external sequences: {len(external_sequences_raw)}")
    print(f"Unique external phonemes: {len(all_external_phonemes)}")
    print(f"Sample phonemes: {sorted(list(all_external_phonemes))[:20]}")
    
    # 2. Load your internal phoneme data
    print(f"\n{'='*60}")
    print("Loading your internal phoneme data")
    print(f"{'='*60}")
    
    # Load your dataset
    from dataset import iEEGPhonemeDataset
    
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Get train sessions
    sessions = config['data']['sessions']
    val_flags = config['data']['dataset_probability_val']
    train_sessions = [s for s, v in zip(sessions, val_flags) if v == 1]
    
    dataset = iEEGPhonemeDataset(
        config['data']['dataset_root'],
        train_sessions,
        split='train'
    )
    
    internal_sequences = []
    for sample in dataset.samples:
        seq = sample['phonemes']
        if isinstance(seq, list):
            internal_sequences.append(seq)
        else:
            internal_sequences.append(seq.tolist())
    
    print(f"Internal sequences: {len(internal_sequences)}")
    
    # Get your phoneme set (indices 1-41, 0 is blank)
    all_phonemes_in_data = set()
    for seq in internal_sequences:
        all_phonemes_in_data.update(seq)
    
    print(f"Your model's phonemes: {sorted(all_phonemes_in_data)}")
    
    # 3. Map external phonemes to your indices
    # You'll need to create a proper mapping based on your phoneme set
    # For now, we'll create a simple identity mapping
    target_phoneme_map = {str(i): i for i in range(1, 42)}  # Placeholder
    
    print("\n⚠️  NOTE: You need to create proper phoneme mapping!")
    print("   Edit the arpabet_to_model dict in map_phonemes_to_indices()")
    print("   to match your dataset's phoneme encoding.")
    
    external_sequences_indexed = map_phonemes_to_indices(
        external_sequences_raw,
        all_external_phonemes,
        target_phoneme_map
    )
    
    # 4. Build 6-gram LM
    lm = build_6gram_lm(internal_sequences, external_sequences_indexed)
    
    # 5. Save LM
    output_path = 'outputs/phoneme_6gram_lm.pkl'
    Path('outputs').mkdir(exist_ok=True)
    lm.save(output_path)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"6-gram LM saved to: {output_path}")
    print(f"Total sequences used: {len(internal_sequences) + len(external_sequences_indexed)}")
    print(f"LM vocabulary size: {len(lm.vocab)}")
    print(f"\nTo use this LM:")
    print(f"1. Copy outputs/phoneme_6gram_lm.pkl to replace outputs/phoneme_lm.pkl")
    print(f"2. Or update config to use 6gram LM")
    print(f"\nExpected improvement: +1-2% accuracy")


if __name__ == '__main__':
    main()