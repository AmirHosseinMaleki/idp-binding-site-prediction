import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import time

BATCH_SIZE = 512
EPOCHS = 10
LR = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmbeddingDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class CombinedDataset(Dataset):
    def __init__(self, npz_files):
        all_embeddings = []
        all_labels = []
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            all_embeddings.append(data['embeddings'])
            all_labels.append(data['labels'])
        
        self.embeddings = np.concatenate(all_embeddings, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ========== ARCHITECTURE 1: MLP (Baseline) ==========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

# ========== ARCHITECTURE 2: 1D CNN ==========
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Treat 1280-dim embedding as a 1D signal
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, 1280)
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),  # -> (batch, 64, 640)
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),  # -> (batch, 128, 320)
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),  # -> (batch, 256, 160)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 160, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x: (batch, 1280)
        x = x.unsqueeze(1)  # (batch, 1, 1280)
        x = self.conv_layers(x)
        return self.fc(x).squeeze()

# ========== ARCHITECTURE 3: LSTM ==========
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Split 1280 into sequence of 20 steps with 64 features each
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # Bidirectional for better context
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x: (batch, 1280) -> reshape to (batch, 20, 64)
        x = x.view(x.size(0), 20, 64)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Concatenate final hidden states from both directions
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_final = torch.cat([h_forward, h_backward], dim=1)
        
        return self.fc(h_final).squeeze()

# ========== ARCHITECTURE 4: GRU ==========
class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), 20, 64)
        gru_out, h_n = self.gru(x)
        
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_final = torch.cat([h_forward, h_backward], dim=1)
        
        return self.fc(h_final).squeeze()

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

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            probs = torch.sigmoid(out)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.7, 0.05):
        preds_binary = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    preds_binary = (all_preds >= best_thresh).astype(int)
    
    return {
        'AUC': roc_auc_score(all_labels, all_preds),
        'F1': f1_score(all_labels, preds_binary),
        'Accuracy': accuracy_score(all_labels, preds_binary),
        'Threshold': best_thresh
    }

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader):
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    best_val_auc = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader)
        
        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
            best_epoch = epoch + 1
            # Save best model
            torch.save(model.state_dict(), f'{model_name.lower().replace(" ", "_")}_model.pt')
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: "
                  f"Loss={train_loss:.4f}, Val AUC={val_metrics['AUC']:.4f}")
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(f'{model_name.lower().replace(" ", "_")}_model.pt'))
    test_metrics = evaluate(model, test_loader)
    
    print(f"\n  Best Validation (Epoch {best_epoch}): AUC={best_val_auc:.4f}")
    print(f"  Test Results:")
    print(f"    AUC: {test_metrics['AUC']:.4f}")
    print(f"    F1: {test_metrics['F1']:.4f}")
    print(f"    Accuracy: {test_metrics['Accuracy']:.4f}")
    print(f"    Threshold: {test_metrics['Threshold']:.2f}")
    print(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f} min)")
    
    return {
        'name': model_name,
        'val_auc': best_val_auc,
        'test_auc': test_metrics['AUC'],
        'test_f1': test_metrics['F1'],
        'test_acc': test_metrics['Accuracy'],
        'time': training_time
    }

# ========== MAIN EXECUTION ==========
print("="*70)
print("ARCHITECTURE COMPARISON: Protein-Protein Binding")
print("="*70)

print("\nLoading data...")
train_data = CombinedDataset([
    'scannet_train_embeddings.npz',
    'disprot_train_embeddings.npz'
])
val_data = EmbeddingDataset('disprot_val_embeddings.npz')
test_data = EmbeddingDataset('disprot_test_embeddings.npz')

print(f"  Train: {len(train_data):,} samples")
print(f"  Val: {len(val_data):,} samples")
print(f"  Test: {len(test_data):,} samples")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2)

# Test all architectures
architectures = [
    (MLP(), "MLP (Baseline)"),
    (CNN1D(), "1D CNN"),
    (BiLSTM(), "Bi-LSTM"),
    (BiGRU(), "Bi-GRU")
]

results = []

for model, name in architectures:
    result = train_and_evaluate(model, name, train_loader, val_loader, test_loader)
    results.append(result)

# ========== FINAL SUMMARY ==========
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"{'Architecture':<20} {'Val AUC':<12} {'Test AUC':<12} {'Test F1':<12} {'Time (min)':<12}")
print("-"*70)

for r in sorted(results, key=lambda x: x['test_auc'], reverse=True):
    print(f"{r['name']:<20} {r['val_auc']:<12.4f} {r['test_auc']:<12.4f} "
          f"{r['test_f1']:<12.4f} {r['time']/60:<12.1f}")

print("="*70)

# Interpretation
best = max(results, key=lambda x: x['test_auc'])
print(f"\nBest Architecture: {best['name']} (Test AUC: {best['test_auc']:.4f})")

mlp_result = next(r for r in results if r['name'] == 'MLP (Baseline)')
print(f"\nComparison to Baseline (MLP):")
for r in results:
    if r['name'] != 'MLP (Baseline)':
        diff = r['test_auc'] - mlp_result['test_auc']
        if abs(diff) < 0.01:
            comparison = "≈ similar"
        elif diff > 0:
            comparison = f"↑ {diff:.4f} better"
        else:
            comparison = f"↓ {abs(diff):.4f} worse"
        print(f"  {r['name']:<20} {comparison}")

print("\n" + "="*70)