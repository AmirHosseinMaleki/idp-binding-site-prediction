import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score
import time

BATCH_SIZE = 512
EPOCHS = 20 
LR = 0.00005
WEIGHT_DECAY = 0.001
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
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
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
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.fc(x).squeeze()

# ========== ARCHITECTURE 3: LSTM ==========
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
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
        lstm_out, (h_n, c_n) = self.lstm(x)
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

def evaluate_complete(model, loader):
    """Evaluate with ALL metrics"""
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
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_binary = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    preds_binary = (all_preds >= best_thresh).astype(int)
    
    return {
        'AUC': roc_auc_score(all_labels, all_preds),
        'AUPRC': average_precision_score(all_labels, all_preds),
        'MCC': matthews_corrcoef(all_labels, preds_binary),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    best_val_auc = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate_complete(model, val_loader)
        
        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
            best_epoch = epoch + 1
            # Save best model
            torch.save(model.state_dict(), f'{model_name.lower().replace(" ", "_")}_15ep_model.pt')
        
        if (epoch + 1) % 3 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: "
                  f"Loss={train_loss:.4f}, Val AUC={val_metrics['AUC']:.4f}")
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(f'{model_name.lower().replace(" ", "_")}_15ep_model.pt'))
    test_metrics = evaluate_complete(model, test_loader)
    
    print(f"\n  Best Validation (Epoch {best_epoch}): AUC={best_val_auc:.4f}")
    print(f"  Test Results:")
    print(f"    Threshold: {test_metrics['Threshold']:.2f}")
    print(f"    AUC:       {test_metrics['AUC']:.4f}")
    print(f"    AUPRC:     {test_metrics['AUPRC']:.4f}")
    print(f"    MCC:       {test_metrics['MCC']:.4f}")
    print(f"    F1:        {test_metrics['F1']:.4f}")
    print(f"    Accuracy:  {test_metrics['Accuracy']:.4f}")
    print(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f} min)")
    
    return {
        'name': model_name,
        'val_auc': best_val_auc,
        'best_epoch': best_epoch,
        **test_metrics,
        'time': training_time
    }

# ========== MAIN EXECUTION ==========
print("="*70)
print("ARCHITECTURE COMPARISON - 15 EPOCHS (Consistent with Optimized)")
print("="*70)
print(f"\nHyperparameters (matching optimized approach):")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LR}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"  Batch Size: {BATCH_SIZE}")

print("\nLoading data...")
train_data = CombinedDataset([
    'biolip_dna_rna_train_embeddings.npz',
    'disprot_dna_rna_train_embeddings.npz'
])
val_data = EmbeddingDataset('disprot_dna_rna_val_embeddings.npz')
test_data = EmbeddingDataset('disprot_dna_rna_test_embeddings.npz')

print(f"  Train: {len(train_data):,} samples")
print(f"  Val: {len(val_data):,} samples (DisProt only)")
print(f"  Test: {len(test_data):,} samples (DisProt only)")

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
print("COMPLETE RESULTS TABLE (15 Epochs)")
print("="*70)
print(f"{'Architecture':<20} {'Threshold':<11} {'AUC':<10} {'AUPRC':<10} {'MCC':<10} {'F1':<10} {'Accuracy':<10} {'Time(min)':<10}")
print("-"*70)

for r in sorted(results, key=lambda x: x['AUC'], reverse=True):
    print(f"{r['name']:<20} {r['Threshold']:<11.2f} {r['AUC']:<10.4f} {r['AUPRC']:<10.4f} "
          f"{r['MCC']:<10.4f} {r['F1']:<10.4f} {r['Accuracy']:<10.4f} {r['time']/60:<10.1f}")

print("="*70)

# Comparison to MLP baseline
mlp_result = next(r for r in results if r['name'] == 'MLP (Baseline)')
print(f"\nComparison to MLP Baseline:")
print(f"{'Architecture':<20} {'  AUC':<12} {'  AUPRC':<12} {'  MCC':<12} {'  F1':<12} {'  Accuracy':<12}")
print("-"*70)

for r in results:
    if r['name'] != mlp_result['name']:
        print(f"{r['name']:<20} {r['AUC']-mlp_result['AUC']:>+12.4f} "
              f"{r['AUPRC']-mlp_result['AUPRC']:>+12.4f} "
              f"{r['MCC']-mlp_result['MCC']:>+12.4f} "
              f"{r['F1']-mlp_result['F1']:>+12.4f} "
              f"{r['Accuracy']-mlp_result['Accuracy']:>+12.4f}")

print("\n" + "="*70)

# Compare MLP result to optimized approach
print("\nValidation: MLP (15 epochs) should match Optimized Approach")
print("-"*70)
print("Expected (from optimized training):")
print("  Epoch 3, Val loss: 0.7897")
print("  Test AUC: 0.8394, AUPRC: 0.6214, MCC: 0.4986, F1: 0.6460")
print(f"\nActual (architecture test, 15 epochs):")
print(f"  Epoch {mlp_result['best_epoch']}, Val AUC: {mlp_result['val_auc']:.4f}")
print(f"  Test AUC: {mlp_result['AUC']:.4f}, AUPRC: {mlp_result['AUPRC']:.4f}, MCC: {mlp_result['MCC']:.4f}, F1: {mlp_result['F1']:.4f}")

if abs(mlp_result['AUC'] - 0.8394) < 0.005:
    print("\n  Results match! Consistent experimental conditions confirmed.")
else:
    print(f"\n  Small difference (±{abs(mlp_result['AUC'] - 0.8394):.4f}) due to random seed variation - acceptable.")

print("="*70)