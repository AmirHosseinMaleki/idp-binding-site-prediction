import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WINDOW_SIZE = 31
BATCH_SIZE = 512
EPOCHS = 5
LR = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

class BindingDataset(Dataset):
    def __init__(self, tsv_file):
        df = pd.read_csv(tsv_file, sep='\t')
        self.samples = []
        self.labels = []
        half_w = WINDOW_SIZE // 2
        
        for _, row in df.iterrows():
            seq = row['sequence']
            labels_str = row['labels']
            labels_list = [int(x) for x in labels_str.split(',')]
            
            padded = 'X' * half_w + seq + 'X' * half_w
            
            for i in range(len(seq)):
                window = padded[i:i+WINDOW_SIZE]
                encoded = np.zeros((WINDOW_SIZE, len(AA_VOCAB)), dtype=np.float32)
                for j, aa in enumerate(window):
                    idx = AA_TO_IDX.get(aa, AA_TO_IDX['X'])
                    encoded[j, idx] = 1.0
                
                self.samples.append(encoded.flatten())
                self.labels.append(labels_list[i])
        
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
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
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

train_data = BindingDataset('ion_binding_train.tsv')
val_data = BindingDataset('ion_binding_val.tsv')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2)

input_size = WINDOW_SIZE * len(AA_VOCAB)
model = BindingNet(input_size).to(DEVICE)

pos_weight = torch.tensor([2.0]).to(DEVICE)
criterion = nn.BCELoss(reduction='none')

def weighted_loss(out, y):
    loss = criterion(out, y)
    weights = torch.where(y == 1, pos_weight, 0.5)
    return (loss * weights).mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_loss = float('inf')
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, weighted_loss, optimizer)
    val_loss = validate(model, val_loader, weighted_loss)
    
    print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'phase2_model.pt')

print(f"\nBest val loss: {best_loss:.4f}")