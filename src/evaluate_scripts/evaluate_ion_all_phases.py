import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score

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

class BindingNet(nn.Module):
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
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

def evaluate_with_best_threshold(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = None
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_binary = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
            best_metrics = {
                'Threshold': thresh,
                'AUC': roc_auc_score(all_labels, all_preds),
                'AUPRC': average_precision_score(all_labels, all_preds),
                'MCC': matthews_corrcoef(all_labels, preds_binary),
                'F1': f1,
                'Accuracy': accuracy_score(all_labels, preds_binary)
            }
    
    return best_metrics

print("="*60)
print("Ion Binding Site Prediction - Evaluation")
print("="*60)

# Load test data
print("\nLoading test data...")
ahojdb_test = EmbeddingDataset('ahojdb_test_embeddings.npz')
disprot_test = EmbeddingDataset('disprot_ion_test_embeddings.npz')

ahojdb_loader = DataLoader(ahojdb_test, batch_size=BATCH_SIZE, num_workers=2)
disprot_loader = DataLoader(disprot_test, batch_size=BATCH_SIZE, num_workers=2)

print(f"  AHoJ-DB test: {len(ahojdb_test):,} residues")
print(f"  DisProt test: {len(disprot_test):,} residues")

# Evaluate all 3 phases
models = [
    ('Phase 1 (AHoJ-DB only)', 'ion_phase1_model.pt'),
    ('Phase 2 (DisProt only)', 'ion_phase2_model.pt'),
    ('Phase 3 (Hybrid)', 'ion_phase3_model.pt')
]

results_summary = []

for phase_name, model_file in models:
    print("\n" + "="*60)
    print(f"Evaluating {phase_name}")
    print("="*60)
    
    # Load model
    model = BindingNet(input_size=1280).to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    
    # Test on AHoJ-DB
    print(f"\nTesting on AHoJ-DB (Structured):")
    ahojdb_metrics = evaluate_with_best_threshold(model, ahojdb_loader)
    print(f"  Threshold: {ahojdb_metrics['Threshold']:.2f}")
    print(f"  AUC: {ahojdb_metrics['AUC']:.4f}")
    print(f"  AUPRC: {ahojdb_metrics['AUPRC']:.4f}")
    print(f"  MCC: {ahojdb_metrics['MCC']:.4f}")
    print(f"  F1: {ahojdb_metrics['F1']:.4f}")
    print(f"  Accuracy: {ahojdb_metrics['Accuracy']:.4f}")
    
    # Test on DisProt
    print(f"\nTesting on DisProt (IDPs):")
    disprot_metrics = evaluate_with_best_threshold(model, disprot_loader)
    print(f"  Threshold: {disprot_metrics['Threshold']:.2f}")
    print(f"  AUC: {disprot_metrics['AUC']:.4f}")
    print(f"  AUPRC: {disprot_metrics['AUPRC']:.4f}")
    print(f"  MCC: {disprot_metrics['MCC']:.4f}")
    print(f"  F1: {disprot_metrics['F1']:.4f}")
    print(f"  Accuracy: {disprot_metrics['Accuracy']:.4f}")
    
    results_summary.append({
        'Phase': phase_name,
        'AHoJ_AUC': ahojdb_metrics['AUC'],
        'DisProt_AUC': disprot_metrics['AUC']
    })

# Summary comparison
print("\n" + "="*60)
print("SUMMARY: AUC Comparison")
print("="*60)
print(f"{'Phase':<25} {'AHoJ-DB AUC':<15} {'DisProt AUC':<15}")
print("-"*60)
for r in results_summary:
    print(f"{r['Phase']:<25} {r['AHoJ_AUC']:<15.4f} {r['DisProt_AUC']:<15.4f}")