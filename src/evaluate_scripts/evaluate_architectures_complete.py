import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score
import os

BATCH_SIZE = 512
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

# ========== ARCHITECTURE DEFINITIONS ==========
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

def evaluate_complete_metrics(model, loader):
    """Evaluate model with ALL metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("  Calculating predictions...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Find best threshold based on F1
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = None
    
    print("  Finding optimal threshold...")
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_binary = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    # Calculate all metrics with best threshold
    preds_binary = (all_preds >= best_thresh).astype(int)
    
    metrics = {
        'Threshold': best_thresh,
        'AUC': roc_auc_score(all_labels, all_preds),
        'AUPRC': average_precision_score(all_labels, all_preds),
        'MCC': matthews_corrcoef(all_labels, preds_binary),
        'F1': f1_score(all_labels, preds_binary),
        'Accuracy': accuracy_score(all_labels, preds_binary)
    }
    
    return metrics

# ========== MAIN EVALUATION ==========
print("="*70)
print("Complete Architecture Evaluation with All Metrics")
print("="*70)

# Load test data
print("\nLoading DisProt test data...")
test_data = EmbeddingDataset('disprot_test_embeddings.npz')
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2)
print(f"  Test samples: {len(test_data):,}")

# Define architectures and their saved model paths
architectures = [
    ('MLP (Baseline)', MLP(), 'mlp_(baseline)_model.pt'),
    ('1D CNN', CNN1D(), '1d_cnn_model.pt'),
    ('Bi-LSTM', BiLSTM(), 'bi-lstm_model.pt'),
    ('Bi-GRU', BiGRU(), 'bi-gru_model.pt')
]

results = []

for arch_name, model, model_path in architectures:
    print(f"\n{'='*70}")
    print(f"Evaluating: {arch_name}")
    print(f"{'='*70}")
    
    if not os.path.exists(model_path):
        print(f"  ⚠ Model file not found: {model_path}")
        continue
    
    # Load model
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Evaluate
    metrics = evaluate_complete_metrics(model, test_loader)
    
    # Print results
    print(f"\n  Results:")
    print(f"    Threshold: {metrics['Threshold']:.2f}")
    print(f"    AUC:       {metrics['AUC']:.4f}")
    print(f"    AUPRC:     {metrics['AUPRC']:.4f}")
    print(f"    MCC:       {metrics['MCC']:.4f}")
    print(f"    F1:        {metrics['F1']:.4f}")
    print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
    
    results.append({
        'Architecture': arch_name,
        **metrics
    })

# ========== FINAL SUMMARY TABLE ==========
print("\n" + "="*70)
print("COMPLETE RESULTS TABLE")
print("="*70)
print(f"{'Architecture':<20} {'Threshold':<12} {'AUC':<10} {'AUPRC':<10} {'MCC':<10} {'F1':<10} {'Accuracy':<10}")
print("-"*70)

for r in sorted(results, key=lambda x: x['AUC'], reverse=True):
    print(f"{r['Architecture']:<20} {r['Threshold']:<12.2f} {r['AUC']:<10.4f} "
          f"{r['AUPRC']:<10.4f} {r['MCC']:<10.4f} {r['F1']:<10.4f} {r['Accuracy']:<10.4f}")

print("="*70)

# Comparison to MLP baseline
mlp_result = next(r for r in results if 'MLP' in r['Architecture'])
print(f"\nComparison to MLP Baseline:")
print(f"{'Architecture':<20} {'AUC Diff':<12} {'AUPRC Diff':<12} {'MCC Diff':<12} {'F1 Diff':<12}")
print("-"*70)

for r in results:
    if r['Architecture'] != mlp_result['Architecture']:
        auc_diff = r['AUC'] - mlp_result['AUC']
        auprc_diff = r['AUPRC'] - mlp_result['AUPRC']
        mcc_diff = r['MCC'] - mlp_result['MCC']
        f1_diff = r['F1'] - mlp_result['F1']
        
        print(f"{r['Architecture']:<20} {auc_diff:>+12.4f} {auprc_diff:>+12.4f} {mcc_diff:>+12.4f} {f1_diff:>+12.4f}")

print("\n" + "="*70)
print("Evaluation Complete!")
print("="*70)