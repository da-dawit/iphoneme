import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from neural_lm_corrector import PhonemeSequenceCorrector, collate_fn

class BetterDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.confusions = {
            1: [2, 3], 2: [1, 3], 3: [1, 2], 4: [5, 6],
            16: [17, 18], 17: [16, 18], 18: [17, 19], 19: [18, 20]
        }
    
    def __len__(self):
        return len(self.sequences)
    
    def corrupt(self, seq):
        import random
        corrupted = []
        for i, p in enumerate(seq):
            if random.random() < 0.20:
                choice = random.choice(['sub', 'del', 'rep'])
                if choice == 'sub' and p in self.confusions:
                    corrupted.append(random.choice(self.confusions[p]))
                elif choice == 'del':
                    continue
                elif choice == 'rep' and corrupted:
                    corrupted.append(corrupted[-1])
                else:
                    corrupted.append(p)
            else:
                corrupted.append(p)
        return corrupted if corrupted else [1]
    
    def __getitem__(self, idx):
        clean = self.sequences[idx]
        noisy = self.corrupt(clean)
        return {
            'noisy': torch.tensor(noisy, dtype=torch.long),
            'clean': torch.tensor(clean, dtype=torch.long),
            'noisy_length': len(noisy),
            'clean_length': len(clean)
        }

with open('outputs/train_phoneme_sequences.pkl', 'rb') as f:
    sequences = pickle.load(f)

print(f"Training on {len(sequences)} sequences")

dataset = BetterDataset(sequences)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=4)

model = PhonemeSequenceCorrector(vocab_size=41, embed_dim=256, hidden_dim=512, num_layers=3, dropout=0.2).cuda()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
