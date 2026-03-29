import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score

BATCH_SIZE = 512
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

def validate_complete(model, loader, criterion):
    """Calculate ALL metrics on validation set"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(out)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Find best threshold for F1
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
        'loss': total_loss / len(loader),
        'auc': roc_auc_score(all_labels, all_preds),
        'auprc': average_precision_score(all_labels, all_preds),
        'mcc': matthews_corrcoef(all_labels, preds_binary),
        'f1': f1_score(all_labels, preds_binary),
        'accuracy': accuracy_score(all_labels, preds_binary)
    }

def test_epoch_count(binding_type, train_files, val_file, pos_weight, max_epochs=50):
    """Test different epoch counts with MULTIPLE metrics"""
    
    print(f"\n{'='*80}")
    print(f"Testing Optimal Epochs: {binding_type}")
    print(f"{'='*80}")
    
    # Load data
    train_data = CombinedDataset(train_files)
    val_data = EmbeddingDataset(val_file)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2)
    
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Val: {len(val_data):,} samples")
    
    # Train model
    model = MLP().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    print(f"\nEpoch | Train Loss |  Val Loss  |  Val AUC  | Val AUPRC |  Val F1")
    print("-" * 80)
    
    epoch_results = []
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics = validate_complete(model, val_loader, criterion)
        
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **val_metrics
        })
        
        # Print every 5 epochs (to reduce clutter)
        if (epoch + 1) % 5 == 0 or epoch < 5:
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_metrics['loss']:10.4f} | "
                  f"{val_metrics['auc']:9.4f} | {val_metrics['auprc']:9.4f} | "
                  f"{val_metrics['f1']:7.4f}")
    
    # Find best epoch by different criteria
    best_by_loss = min(epoch_results, key=lambda x: x['loss'])
    best_by_auc = max(epoch_results, key=lambda x: x['auc'])
    best_by_auprc = max(epoch_results, key=lambda x: x['auprc'])
    best_by_f1 = max(epoch_results, key=lambda x: x['f1'])
    
    print(f"\n  Best Epochs by Different Metrics:")
    print(f"    By Val Loss:  Epoch {best_by_loss['epoch']} (Loss: {best_by_loss['loss']:.4f})")
    print(f"    By Val AUC:   Epoch {best_by_auc['epoch']} (AUC: {best_by_auc['auc']:.4f})")
    print(f"    By Val AUPRC: Epoch {best_by_auprc['epoch']} (AUPRC: {best_by_auprc['auprc']:.4f})")
    print(f"    By Val F1:    Epoch {best_by_f1['epoch']} (F1: {best_by_f1['f1']:.4f})")
    
    # Recommendation: Use AUC as primary metric
    recommended_epoch = best_by_auc['epoch']
    print(f"\n    Recommended: Epoch {recommended_epoch} (based on highest Val AUC)")
    
    return epoch_results, recommended_epoch

# ========== TEST ALL THREE BINDING TYPES ==========
print("="*80)
print("COMPREHENSIVE EPOCH OPTIMIZATION ANALYSIS")
print("="*80)
print(f"Hyperparameters: LR={LR}, Weight Decay={WEIGHT_DECAY}, Batch={BATCH_SIZE}")
print(f"Testing up to 50 epochs to find true optimal point...")
print(f"Evaluating based on: Loss, AUC, AUPRC, F1")

results_all = {}

# Protein-Protein
results_all['Protein'] = test_epoch_count(
    binding_type='Protein-Protein',
    train_files=['scannet_train_embeddings.npz', 'disprot_train_embeddings.npz'],
    val_file='disprot_val_embeddings.npz',
    pos_weight=3.0,
    max_epochs=50
)

# DNA/RNA
results_all['DNA/RNA'] = test_epoch_count(
    binding_type='DNA/RNA',
    train_files=['biolip_dna_rna_train_embeddings.npz', 'disprot_dna_rna_train_embeddings.npz'],
    val_file='disprot_dna_rna_val_embeddings.npz',
    pos_weight=3.0,
    max_epochs=50
)

# Ion
results_all['Ion'] = test_epoch_count(
    binding_type='Ion',
    train_files=['ahojdb_train_embeddings.npz', 'disprot_ion_train_embeddings.npz'],
    val_file='disprot_ion_val_embeddings.npz',
    pos_weight=30.0,
    max_epochs=50
)

# ========== FINAL SUMMARY ==========
print("\n" + "="*80)
print("SUMMARY: OPTIMAL EPOCHS (PRIMARY METRIC: VALIDATION AUC)")
print("="*80)
print(f"{'Binding Type':<20} {'Best Epoch':<15} {'Val AUC':<12} {'Val AUPRC':<12} {'Val F1':<10}")
print("-"*80)

recommended_epochs = []
for binding_type, (results, best_epoch) in results_all.items():
    best_result = results[best_epoch - 1]
    print(f"{binding_type:<20} {best_epoch:<15} {best_result['auc']:<12.4f} "
          f"{best_result['auprc']:<12.4f} {best_result['f1']:<10.4f}")
    recommended_epochs.append(best_epoch)

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

avg_best = int(np.mean(recommended_epochs))
max_best = max(recommended_epochs)
min_best = min(recommended_epochs)

print(f"Best epochs (by AUC): {recommended_epochs}")
print(f"Range: {min_best}-{max_best}")
print(f"Average: {avg_best}")

# Safety margin logic
if max_best <= 5:
    recommended = 15
    margin = 15 - max_best
elif max_best <= 10:
    recommended = 20
    margin = 20 - max_best
else:
    recommended = max_best + 10
    margin = 10

print(f"\n Use {recommended} epochs for all architecture tests")
print(f"  (Provides {margin}-epoch safety margin beyond latest optimal point)")
print("="*80)