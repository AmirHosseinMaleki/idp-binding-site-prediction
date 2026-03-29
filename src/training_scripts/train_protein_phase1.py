import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WINDOW_SIZE = 31
BATCH_SIZE = 512
EPOCHS = 10
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

class BindingDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.samples = []
        self.labels = []
        half_w = WINDOW_SIZE // 2
        
        for _, row in df.iterrows():
            seq = row['sequence']
            ann = row['annotation']
            padded = 'X' * half_w + seq + 'X' * half_w
            
            for i in range(len(seq)):
                window = padded[i:i+WINDOW_SIZE]
                encoded = np.zeros((WINDOW_SIZE, len(AA_VOCAB)), dtype=np.float32)
                for j, aa in enumerate(window):
                    idx = AA_TO_IDX.get(aa, AA_TO_IDX['X'])
                    encoded[j, idx] = 1.0
                
                self.samples.append(encoded.flatten())
                self.labels.append(int(ann[i]))
        
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

class BindingNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
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
print("Phase 1: Training on ScanNet (Structured Protein-Protein)")
print("="*60)

train_data = BindingDataset('/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/scannet_train_clustered.csv')
val_data = BindingDataset('/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/scannet_val_clustered.csv')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2)

input_size = WINDOW_SIZE * len(AA_VOCAB)
model = BindingNet(input_size).to(DEVICE)

# Adjusted weight for ~18% positive (instead of 1.5%)
# Ratio = (100-18)/18 = 4.5
pos_weight = torch.tensor([3.0]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

best_loss = float('inf')
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    
    print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), '/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/protein_phase1_model.pt')
        print("  Saved best model")

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Best val loss: {best_loss:.4f}")