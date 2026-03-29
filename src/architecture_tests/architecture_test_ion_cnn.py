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
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            probs = torch.sigmoid(out)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
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

print("="*70)
print("Ion Binding - 1D CNN Architecture (20 Epochs)")
print("="*70)

print("\nLoading data...")
train_data = CombinedDataset(['ahojdb_train_embeddings.npz', 'disprot_ion_train_embeddings.npz'])
val_data = EmbeddingDataset('disprot_ion_val_embeddings.npz')
test_data = EmbeddingDataset('disprot_ion_test_embeddings.npz')

print(f"  Train: {len(train_data):,} samples")
print(f"  Val: {len(val_data):,} samples")
print(f"  Test: {len(test_data):,} samples")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2)

model = CNN1D().to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]).to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(f"\nTraining 1D CNN for {EPOCHS} epochs...")
best_val_auc = 0
best_epoch = 0
start_time = time.time()

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_metrics = evaluate_complete(model, val_loader)
    
    if val_metrics['AUC'] > best_val_auc:
        best_val_auc = val_metrics['AUC']
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'ion_cnn_model.pt')
    
    if (epoch + 1) % 3 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}: Loss={train_loss:.4f}, Val AUC={val_metrics['AUC']:.4f}")

training_time = time.time() - start_time

model.load_state_dict(torch.load('ion_cnn_model.pt'))
test_metrics = evaluate_complete(model, test_loader)

print(f"\n{'='*70}")
print(f"1D CNN Results:")
print(f"  Best Epoch: {best_epoch}")
print(f"  Val AUC: {best_val_auc:.4f}")
print(f"  Test Threshold: {test_metrics['Threshold']:.2f}")
print(f"  Test AUC: {test_metrics['AUC']:.4f}")
print(f"  Test AUPRC: {test_metrics['AUPRC']:.4f}")
print(f"  Test MCC: {test_metrics['MCC']:.4f}")
print(f"  Test F1: {test_metrics['F1']:.4f}")
print(f"  Test Accuracy: {test_metrics['Accuracy']:.4f}")
print(f"  Training Time: {training_time/60:.1f} min")
print(f"{'='*70}")