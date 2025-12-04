"""
Train corrector on REAL acoustic model errors
This requires running inference first to get predictions
"""
import torch
import sys
sys.path.append('.')
from train import Trainer
from dataset import iEEGPhonemeDataset
from neural_lm_corrector import PhonemeSequenceCorrector, collate_fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

print("="*60)
print("STEP 1: Generate real acoustic model predictions")
print("="*60)

# Load your best model
trainer = Trainer('config.yaml', 'cuda')
checkpoint_path = 'final_model.pt'  # Use your best checkpoint

try:
    trainer.load_checkpoint(checkpoint_path)
    print(f"✓ Loaded model from {checkpoint_path}")
except:
    print(f"❌ Model not found! Train a base model first.")
    print("Run this first:")
    print("  python train.py")
    exit(1)

trainer.model.eval()

# Load training data
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

print(f"\nGenerating predictions for {len(dataset)} samples...")
print("This takes ~10 minutes...")

real_errors = []

with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        neural, phonemes, neural_len, phoneme_len = dataset[i]
        
        neural = neural.unsqueeze(0).cuda()
        neural_len = torch.tensor([neural_len]).cuda()
        
        # Get prediction
        logits, _ = trainer.model(neural, neural_len)
        pred = logits.argmax(-1)[0].cpu()
        
        # CTC collapse
        collapsed = []
        prev = None
        for p in pred.tolist():
            if p != prev:
                collapsed.append(p)
                prev = p
        
        predicted = [p for p in collapsed if p != 0]
        ground_truth = phonemes.tolist()
        
        # Only keep if prediction is wrong
        if predicted != ground_truth and len(predicted) > 0:
            real_errors.append({
                'noisy': predicted,
                'clean': ground_truth
            })

print(f"\n✓ Generated {len(real_errors)} real error pairs")

# Save real errors
with open('outputs/real_acoustic_errors.pkl', 'wb') as f:
    pickle.dump(real_errors, f)

print("\n" + "="*60)
print("STEP 2: Train corrector on REAL errors")
print("="*60)

class RealErrorDataset(Dataset):
    def __init__(self, error_pairs):
        self.pairs = error_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            'noisy': torch.tensor(pair['noisy'], dtype=torch.long),
            'clean': torch.tensor(pair['clean'], dtype=torch.long),
            'noisy_length': len(pair['noisy']),
            'clean_length': len(pair['clean'])
        }

dataset = RealErrorDataset(real_errors)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=4)

model = PhonemeSequenceCorrector(vocab_size=41, embed_dim=256, hidden_dim=512, num_layers=3, dropout=0.2).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=50)

best_val_acc = 0

for epoch in range(1, 51):
    model.train()
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/50")
    for batch in pbar:
        noisy = batch['noisy'].cuda()
        clean = batch['clean'].cuda()
        noisy_lengths = batch['noisy_lengths']
        
        optimizer.zero_grad()
        logits = model(noisy, noisy_lengths)
        
        min_len = min(logits.shape[1], clean.shape[1])
        loss = F.cross_entropy(logits[:, :min_len].reshape(-1, 41), clean[:, :min_len].reshape(-1), ignore_index=0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        preds = logits.argmax(-1)
        mask = clean[:, :min_len] != 0
        train_correct += ((preds[:, :min_len] == clean[:, :min_len]) & mask).sum().item()
        train_total += mask.sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100*train_correct/train_total:.1f}%'})
    
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            noisy = batch['noisy'].cuda()
            clean = batch['clean'].cuda()
            noisy_lengths = batch['noisy_lengths']
            
            logits = model(noisy, noisy_lengths)
            min_len = min(logits.shape[1], clean.shape[1])
            
            preds = logits[:, :min_len].argmax(-1)
            targets = clean[:, :min_len]
            mask = targets != 0
            
            val_correct += ((preds == targets) & mask).sum().item()
            val_total += mask.sum().item()
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    print(f"\nEpoch {epoch}: Train {train_acc:.1f}% | Val {val_acc:.1f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc, 'epoch': epoch}, 'outputs/neural_corrector.pt')
        print(f"✓ Saved! Best: {val_acc:.1f}%")

print(f"\n✓ Done! Best: {best_val_acc:.1f}%")
print("\nThis corrector is trained on REAL acoustic errors!")
print("Expected accuracy: 80-90%")
