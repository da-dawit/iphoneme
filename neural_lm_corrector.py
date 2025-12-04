"""
Neural Language Model Corrector with Bidirectional GRU

This is a 2-stage system:
1. Acoustic model predicts phoneme sequence (may have errors)
2. Bidirectional GRU looks at full context and corrects mistakes

This is MUCH more powerful than n-gram LM because:
- Sees both left AND right context (bidirectional)
- Can learn complex patterns (neural)
- Can correct systematic acoustic model errors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm


class PhonemeSequenceCorrector(nn.Module):
    """
    Bidirectional GRU that corrects phoneme sequences
    
    Input: Noisy phoneme sequence from acoustic model
    Output: Corrected phoneme probabilities at each position
    """
    
    def __init__(
        self,
        vocab_size=41,  # 39 phonemes + blank + padding
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, phoneme_seq, lengths):
        """
        Args:
            phoneme_seq: [batch, max_len] - noisy phoneme indices
            lengths: [batch] - actual sequence lengths
        Returns:
            logits: [batch, max_len, vocab_size] - corrected predictions
        """
        # Embed
        embedded = self.embedding(phoneme_seq)  # [batch, len, embed_dim]
        embedded = self.dropout(embedded)
        
        # Pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Bidirectional GRU
        packed_output, _ = self.gru(packed)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # [batch, len, hidden_dim*2]
        
        output = self.dropout(output)
        
        # Project to vocab
        logits = self.output_proj(output)  # [batch, len, vocab_size]
        
        return logits


class CorrectionDataset(Dataset):
    """
    Dataset of (noisy_sequence, clean_sequence) pairs
    
    We generate this by:
    1. Taking real phoneme sequences
    2. Corrupting them with realistic errors
    3. Training model to recover original
    """
    
    def __init__(self, phoneme_sequences, corruption_rate=0.15):
        """
        Args:
            phoneme_sequences: List of clean phoneme sequences
            corruption_rate: Probability of corrupting each phoneme
        """
        self.sequences = phoneme_sequences
        self.corruption_rate = corruption_rate
        self.vocab_size = 41  # Including blank and padding
        
    def __len__(self):
        return len(self.sequences)
    
    def corrupt_sequence(self, seq):
        """
        Corrupt sequence with realistic errors:
        - Substitution (most common)
        - Deletion
        - Insertion
        """
        corrupted = []
        for phoneme in seq:
            if np.random.random() < self.corruption_rate:
                error_type = np.random.choice(['substitute', 'delete', 'insert'], p=[0.7, 0.2, 0.1])
                
                if error_type == 'substitute':
                    # Replace with random phoneme
                    corrupted.append(np.random.randint(1, 40))
                elif error_type == 'delete':
                    # Skip this phoneme
                    continue
                else:  # insert
                    # Add random phoneme before
                    corrupted.append(np.random.randint(1, 40))
                    corrupted.append(phoneme)
            else:
                corrupted.append(phoneme)
        
        return corrupted if len(corrupted) > 0 else [1]  # At least one phoneme
    
    def __getitem__(self, idx):
        clean_seq = self.sequences[idx]
        
        # Create corrupted version
        noisy_seq = self.corrupt_sequence(clean_seq)
        
        return {
            'noisy': torch.tensor(noisy_seq, dtype=torch.long),
            'clean': torch.tensor(clean_seq, dtype=torch.long),
            'noisy_length': len(noisy_seq),
            'clean_length': len(clean_seq)
        }


def collate_fn(batch):
    """Collate batch with padding"""
    noisy_seqs = [item['noisy'] for item in batch]
    clean_seqs = [item['clean'] for item in batch]
    noisy_lengths = torch.tensor([item['noisy_length'] for item in batch])
    clean_lengths = torch.tensor([item['clean_length'] for item in batch])
    
    # Pad sequences
    noisy_padded = nn.utils.rnn.pad_sequence(noisy_seqs, batch_first=True, padding_value=0)
    clean_padded = nn.utils.rnn.pad_sequence(clean_seqs, batch_first=True, padding_value=0)
    
    return {
        'noisy': noisy_padded,
        'clean': clean_padded,
        'noisy_lengths': noisy_lengths,
        'clean_lengths': clean_lengths
    }


def train_corrector(
    phoneme_sequences,
    save_path='outputs/neural_corrector.pt',
    epochs=20,
    batch_size=64,
    lr=1e-3,
    device='cuda'
):
    """
    Train the bidirectional GRU corrector
    
    Args:
        phoneme_sequences: List of clean phoneme sequences for training
        save_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    print("="*60)
    print("TRAINING NEURAL PHONEME CORRECTOR")
    print("="*60)
    
    # Create dataset
    dataset = CorrectionDataset(phoneme_sequences, corruption_rate=0.15)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print(f"Train sequences: {train_size}")
    print(f"Val sequences: {val_size}")
    print()
    
    # Create model
    model = PhonemeSequenceCorrector(
        vocab_size=41,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            noisy_lengths = batch['noisy_lengths']
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(noisy, noisy_lengths)
            
            # Loss: predict clean sequence given noisy input
            # We align by truncating to min length
            min_len = min(logits.shape[1], clean.shape[1])
            logits = logits[:, :min_len]
            targets = clean[:, :min_len]
            
            # Cross entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=0  # Ignore padding
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            mask = targets != 0
            train_correct += ((predictions == targets) & mask).sum().item()
            train_total += mask.sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.1f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch['noisy'].to(device)
                clean = batch['clean'].to(device)
                noisy_lengths = batch['noisy_lengths']
                
                logits = model(noisy, noisy_lengths)
                
                min_len = min(logits.shape[1], clean.shape[1])
                logits = logits[:, :min_len]
                targets = clean[:, :min_len]
                
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=0
                )
                
                val_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                mask = targets != 0
                val_correct += ((predictions == targets) & mask).sum().item()
                val_total += mask.sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")
        print()
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch
            }, save_path)
            print(f"✓ Saved best model to {save_path}")
    
    print("\nTraining complete!")
    return model


class NeuralLMDecoder:
    """
    Decoder that uses neural corrector instead of n-gram LM
    
    Two-stage process:
    1. Acoustic model predicts sequence (greedy or beam)
    2. Neural corrector refines using bidirectional context
    """
    
    def __init__(self, corrector_path='outputs/neural_corrector.pt', device='cuda'):
        """
        Args:
            corrector_path: Path to trained corrector model
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Load corrector
        print(f"Loading neural corrector from {corrector_path}")
        checkpoint = torch.load(corrector_path, map_location=device)
        
        self.corrector = PhonemeSequenceCorrector(
            vocab_size=41,
            embed_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.0  # No dropout at inference
        ).to(device)
        
        self.corrector.load_state_dict(checkpoint['model_state_dict'])
        self.corrector.eval()
        
        print(f"✓ Loaded corrector (val_acc: {checkpoint['val_acc']:.1f}%)")
    
    def decode_with_correction(self, logits, greedy_first=True):
        """
        Two-stage decoding:
        1. Get initial prediction (greedy)
        2. Refine with neural corrector
        
        Args:
            logits: [batch, time, vocab_size] from acoustic model
            greedy_first: If True, use greedy for stage 1
        Returns:
            corrected_sequences: List of corrected phoneme sequences
        """
        batch_size = logits.shape[0]
        
        # Stage 1: Initial prediction (greedy CTC)
        if greedy_first:
            initial_seqs = self._greedy_decode(logits)
        else:
            # Could also use beam search here
            initial_seqs = self._greedy_decode(logits)
        
        # Stage 2: Neural correction
        corrected_seqs = []
        
        with torch.no_grad():
            for seq in initial_seqs:
                if len(seq) == 0:
                    corrected_seqs.append([])
                    continue
                
                # Convert to tensor
                seq_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
                seq_length = torch.tensor([len(seq)])
                
                # Get corrector predictions
                corrector_logits = self.corrector(seq_tensor, seq_length)
                
                # Take argmax
                corrected = corrector_logits[0].argmax(dim=-1).cpu().tolist()
                
                # Remove padding
                corrected = [p for p in corrected if p != 0]
                
                corrected_seqs.append(corrected)
        
        return corrected_seqs
    
    def _greedy_decode(self, logits):
        """Standard greedy CTC decode"""
        predictions = torch.argmax(logits, dim=-1)
        
        decoded = []
        for pred in predictions:
            collapsed = []
            prev = None
            for p in pred.tolist():
                if p != prev:
                    collapsed.append(p)
                    prev = p
            
            seq = [p for p in collapsed if p != 0]
            decoded.append(seq)
        
        return decoded


def main():
    """
    Train neural corrector from actual training data
    """
    import os
    
    # Check if sequences already extracted
    if not os.path.exists('outputs/train_phoneme_sequences.pkl'):
        print("="*60)
        print("ERROR: Training sequences not found!")
        print("="*60)
        print("\nYou need to extract phoneme sequences first:")
        print("\nRun this command:")
        print("-"*60)
        print("""
python -c "
import sys
sys.path.append('.')
from dataset import iEEGPhonemeDataset
import pickle
import os

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

print('Loading training data from HDF5...')
dataset = iEEGPhonemeDataset(
    dataset_root='/workspace/data/hdf5_data_final',
    sessions=sessions,
    split='train'
)

train_phonemes = []
for i in range(len(dataset)):
    _, phonemes, _, _ = dataset[i]
    seq = phonemes.tolist()
    if len(seq) > 0:
        train_phonemes.append(seq)

print(f'Extracted {len(train_phonemes)} sequences')

os.makedirs('outputs', exist_ok=True)
with open('outputs/train_phoneme_sequences.pkl', 'wb') as f:
    pickle.dump(train_phonemes, f)

print('✓ Saved to outputs/train_phoneme_sequences.pkl')
"
""")
        print("-"*60)
        return
    
    # Load phoneme sequences
    print("Loading phoneme sequences for training...")
    with open('outputs/train_phoneme_sequences.pkl', 'rb') as f:
        sequences = pickle.load(f)
    
    print(f"Found {len(sequences)} sequences for training")
    
    if len(sequences) < 1000:
        print("⚠️  Not enough sequences! Need more training data.")
        return
    
    # Train corrector
    train_corrector(
        sequences,
        save_path='outputs/neural_corrector.pt',
        epochs=20,
        batch_size=64,
        lr=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n✓ Training complete!")
    print("\nNow you can use NeuralLMDecoder in your validation:")
    print("  decoder = NeuralLMDecoder('outputs/neural_corrector.pt')")
    print("  corrected = decoder.decode_with_correction(logits)")


if __name__ == '__main__':
    main()