import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score, recall_score
# from src.utils.config import load_config, get_embedding_path, get_model_path

BATCH_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cfg = load_config()

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
                'Recall': recall_score(all_labels, preds_binary),
                'Accuracy': accuracy_score(all_labels, preds_binary)
            }
    
    return best_metrics

print("="*60)
print("Protein-Protein Binding Site Prediction - Evaluation")
print("="*60)

# Load test data
print("\nLoading test data...")
scannet_test = EmbeddingDataset('scannet_test_embeddings.npz')
disprot_test = EmbeddingDataset('disprot_test_embeddings.npz')
# scannet_test = EmbeddingDataset(get_embedding_path(cfg, "scannet_test"))
# disprot_test = EmbeddingDataset(get_embedding_path(cfg, "disprot_protein_test"))

scannet_loader = DataLoader(scannet_test, batch_size=BATCH_SIZE, num_workers=2)
disprot_loader = DataLoader(disprot_test, batch_size=BATCH_SIZE, num_workers=2)

print(f"  ScanNet test: {len(scannet_test):,} residues")
print(f"  DisProt test: {len(disprot_test):,} residues")

# Evaluate all 3 phases
models = [
    ('Phase 1 (ScanNet only)', 'protein_phase1_esm_model.pt'),
    ('Phase 2 (DisProt only)', 'protein_phase2_esm_model.pt'),
    ('Phase 3 (Hybrid)', 'protein_phase3_esm_model.pt')
]
# models = [
#     ('Phase 1 (ScanNet only)', get_model_path(cfg, "protein_phase1")),
#     ('Phase 2 (DisProt only)', get_model_path(cfg, "protein_phase2")),
#     ('Phase 3 (Hybrid)', get_model_path(cfg, "protein_phase3"))
# ]

results_summary = []

for phase_name, model_file in models:
    print("\n" + "="*60)
    print(f"Evaluating {phase_name}")
    print("="*60)
    
    # Load model
    model = BindingNet(input_size=1280).to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    
    # Test on ScanNet
    print(f"\nTesting on ScanNet (Structured):")
    scannet_metrics = evaluate_with_best_threshold(model, scannet_loader)
    print(f"  Threshold: {scannet_metrics['Threshold']:.2f}")
    print(f"  AUC: {scannet_metrics['AUC']:.4f}")
    print(f"  AUPRC: {scannet_metrics['AUPRC']:.4f}")
    print(f"  MCC: {scannet_metrics['MCC']:.4f}")
    print(f"  F1: {scannet_metrics['F1']:.4f}")
    print(f"  Recall: {scannet_metrics['Recall']:.4f}")
    print(f"  Accuracy: {scannet_metrics['Accuracy']:.4f}")

    # Test on DisProt
    print(f"\nTesting on DisProt (IDPs):")
    disprot_metrics = evaluate_with_best_threshold(model, disprot_loader)
    print(f"  Threshold: {disprot_metrics['Threshold']:.2f}")
    print(f"  AUC: {disprot_metrics['AUC']:.4f}")
    print(f"  AUPRC: {disprot_metrics['AUPRC']:.4f}")
    print(f"  MCC: {disprot_metrics['MCC']:.4f}")
    print(f"  F1: {disprot_metrics['F1']:.4f}")
    print(f"  Recall: {disprot_metrics['Recall']:.4f}")
    print(f"  Accuracy: {disprot_metrics['Accuracy']:.4f}")
    
    results_summary.append({
        'Phase': phase_name,
        'ScanNet_AUC': scannet_metrics['AUC'],
        'DisProt_AUC': disprot_metrics['AUC'],
        'ScanNet_AUPRC': scannet_metrics['AUPRC'],
        'DisProt_AUPRC': disprot_metrics['AUPRC'],
        'ScanNet_MCC': scannet_metrics['MCC'],
        'DisProt_MCC': disprot_metrics['MCC']
    })

# Summary comparison
print("\n" + "="*60)
print("SUMMARY: AUC Comparison")
print("="*60)
print(f"{'Phase':<25} {'ScanNet AUC':<15} {'DisProt AUC':<15} {'ScanNet AUPRC':<15} {'DisProt AUPRC':<15} {'ScanNet MCC':<15} {'DisProt MCC':<15}")
print("-"*60)
for r in results_summary:
    print(f"{r['Phase']:<25} {r['ScanNet_AUC']:<15.4f} {r['DisProt_AUC']:<15.4f} {r['ScanNet_AUPRC']:<15.4f} {r['DisProt_AUPRC']:<15.4f} {r['ScanNet_MCC']:<15.4f} {r['DisProt_MCC']:<15.4f}")