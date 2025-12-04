"""
Build 6-gram LM from actual T15 training data
This will match your phoneme distribution!
"""
import pickle
from collections import defaultdict
from dataset import iEEGPhonemeDataset

print("Loading T15 training data...")

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
print(f"Building 6-gram LM...")

# Build n-gram counts
ngram_counts = {}
vocab = set()

for n in range(1, 7):  # 1-gram to 6-gram
    ngram_counts[n] = defaultdict(int)

for seq in sequences:
    for phoneme in seq:
        vocab.add(phoneme)
    
    # Add start/end markers
    padded = [40] + seq + [40]  # 40 is silence/boundary
    
    # Count n-grams
    for n in range(1, 7):
        for i in range(len(padded) - n + 1):
            ngram = tuple(padded[i:i+n])
            ngram_counts[n][ngram] += 1

# Save LM
lm_data = {
    'order': 6,
    'vocab': vocab,
    'ngram_counts': dict(ngram_counts)
}

with open('outputs/t15_phoneme_lm.pkl', 'wb') as f:
    pickle.dump(lm_data, f)

print(f"\n✓ Built 6-gram LM from T15 data")
print(f"  Vocab size: {len(vocab)}")
print(f"  Sequences: {len(sequences)}")
for n in range(1, 7):
    print(f"  {n}-grams: {len(ngram_counts[n])}")
print(f"\n✓ Saved to outputs/t15_phoneme_lm.pkl")
