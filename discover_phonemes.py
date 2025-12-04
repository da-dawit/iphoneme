"""
Phoneme Mapping Helper

This file helps you map external phoneme sets (ARPAbet, IPA, etc.)
to your model's phoneme indices.

You need to customize this based on your actual phoneme encoding.
"""

# Step 1: Define YOUR phoneme set
# Look at your data to see what phonemes you have (indices 1-41)
# Common phoneme sets for English speech:

# ARPAbet phonemes (CMU, LibriSpeech use this)
ARPABET_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY',  # Vowels
    'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH',  # Consonants
    'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S',
    'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
]

# TIMIT phonemes (61 phonemes, more detailed)
TIMIT_PHONEMES = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
    'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',
    'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy',
    'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy',
    'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw',
    'ux', 'v', 'w', 'y', 'z', 'zh'
]


def discover_your_phonemes(dataset):
    """
    Discover what phonemes are in your dataset
    """
    print("Discovering phonemes in your dataset...")
    
    all_phonemes = set()
    phoneme_counts = {}
    
    for sample in dataset.samples:
        seq = sample['phonemes']
        if not isinstance(seq, list):
            seq = seq.tolist()
        
        for p in seq:
            if p != 0:  # Skip blank
                all_phonemes.add(p)
                phoneme_counts[p] = phoneme_counts.get(p, 0) + 1
    
    print(f"\nYour dataset has {len(all_phonemes)} unique phonemes:")
    print(f"Indices: {sorted(all_phonemes)}")
    print(f"\nTop 10 most common:")
    for phoneme, count in sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Phoneme {phoneme}: {count} occurrences")
    
    return sorted(all_phonemes), phoneme_counts


def create_mapping_template(your_phonemes):
    """
    Create a template for manual phoneme mapping
    """
    print("\n" + "="*60)
    print("PHONEME MAPPING TEMPLATE")
    print("="*60)
    print("\nEdit this mapping to match your dataset:\n")
    
    print("PHONEME_MAPPING = {")
    for i, phoneme_idx in enumerate(your_phonemes):
        # Try to guess based on common patterns
        if phoneme_idx <= len(ARPABET_PHONEMES):
            guess = ARPABET_PHONEMES[phoneme_idx - 1]
        else:
            guess = f"UNKNOWN_{phoneme_idx}"
        
        print(f"    {phoneme_idx}: '{guess}',  # YOUR_PHONEME_HERE")
    print("}")
    
    print("\n" + "="*60)
    print("REVERSE MAPPING (for external data)")
    print("="*60)
    print("\nARPABET_TO_MODEL = {")
    for i, phoneme_idx in enumerate(your_phonemes):
        if phoneme_idx <= len(ARPABET_PHONEMES):
            arpabet = ARPABET_PHONEMES[phoneme_idx - 1]
            print(f"    '{arpabet}': {phoneme_idx},")
    print("}")


def main():
    """
    Run this to discover your phoneme mapping
    """
    print("="*60)
    print("PHONEME MAPPING DISCOVERY")
    print("="*60)
    
    # Load your dataset
    from dataset import iEEGPhonemeDataset
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    sessions = config['data']['sessions']
    val_flags = config['data']['dataset_probability_val']
    train_sessions = [s for s, v in zip(sessions, val_flags) if v == 1]
    
    dataset = iEEGPhonemeDataset(
        config['data']['dataset_root'],
        train_sessions[:5],  # Just first 5 sessions for speed
        split='train'
    )
    
    your_phonemes, counts = discover_your_phonemes(dataset)
    
    # Create mapping template
    create_mapping_template(your_phonemes)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review the mapping template above")
    print("2. Update build_6gram_lm.py with correct mapping")
    print("3. Run: python build_6gram_lm.py")
    print("\nNOTE: You may need to check your data documentation")
    print("      to see what phoneme encoding is used.")


if __name__ == '__main__':
    main()