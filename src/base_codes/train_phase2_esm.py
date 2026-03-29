import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 512
EPOCHS = 10
LR = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmbeddingDataset(Dataset):
    """Dataset that loads precomputed ESM-2 embeddings"""
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.embeddings = data['embeddings']  # Shape: (N, 1280)
        self.labels = data['labels']          # Shape: (N,)
        print(f"Loaded {len(self.embeddings):,} samples from {npz_file}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class BindingNet(nn.Module):
    """Neural network using ESM-2 embeddings (1280-dim input)"""
    def __init__(self, input_size=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # No sigmoid - using BCEWithLogitsLoss
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / len(loader)

print("="*60)
print("Phase 2: Training with ESM-2 Embeddings (DisProt Only)")
print("="*60)

# Load precomputed embeddings
train_data = EmbeddingDataset('disprot_train_embeddings.npz')
val_data = EmbeddingDataset('disprot_val_embeddings.npz')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2)

# Create model
model = BindingNet(input_size=1280).to(DEVICE)

# Loss with class weighting (DisProt has ~24% positives, so lower weight than ScanNet)
pos_weight = torch.tensor([2.5]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer with weight decay for regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

print(f"\nDataset sizes:")
print(f"  Train: {len(train_data):,} residues")
print(f"  Val: {len(val_data):,} residues")
print(f"\nTraining settings:")
print(f"  Device: {DEVICE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LR}")
print(f"  Positive weight: {pos_weight.item()}")
print(f"  Weight decay: 0.01")

best_loss = float('inf')
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    
    print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'protein_phase2_esm_model.pt')
        print("    Saved best model")

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Best val loss: {best_loss:.4f}")
print(f"Model saved: protein_phase2_esm_model.pt")
print(f"{'='*60}")