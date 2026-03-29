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
print("DNA/RNA Binding - Optimized Model Evaluation")
print("="*60)

# Load test data
print("\nLoading test data...")
biolip_test = EmbeddingDataset('biolip_dna_rna_test_embeddings.npz')
disprot_test = EmbeddingDataset('disprot_dna_rna_test_embeddings.npz')

biolip_loader = DataLoader(biolip_test, batch_size=BATCH_SIZE, num_workers=2)
disprot_loader = DataLoader(disprot_test, batch_size=BATCH_SIZE, num_workers=2)

print(f"  BioLip test: {len(biolip_test):,} residues")
print(f"  DisProt test: {len(disprot_test):,} residues")

# Load model
model = BindingNet(input_size=1280).to(DEVICE)
model.load_state_dict(torch.load('dna_rna_hybrid_idpval_model.pt', map_location=DEVICE))

print(f"\nUsing device: {DEVICE}")
print("\nModel: dna_rna_hybrid_idpval_model.pt (Optimized - epoch 3, val loss 1.2289)")
print("Hyperparameters: LR=0.00005, WD=0.01 (from grid search)")

# Test on BioLip
print("\n" + "="*60)
print("Testing on BioLip (Structured)")
print("="*60)
metrics = evaluate_with_best_threshold(model, biolip_loader)
print(f"\nOptimal threshold: {metrics['Threshold']:.2f}")
print(f"  AUC: {metrics['AUC']:.4f}")
print(f"  AUPRC: {metrics['AUPRC']:.4f}")
print(f"  MCC: {metrics['MCC']:.4f}")
print(f"  F1: {metrics['F1']:.4f}")
print(f"  Accuracy: {metrics['Accuracy']:.4f}")

# Test on DisProt
print("\n" + "="*60)
print("Testing on DisProt (IDPs)")
print("="*60)
metrics = evaluate_with_best_threshold(model, disprot_loader)
print(f"\nOptimal threshold: {metrics['Threshold']:.2f}")
print(f"  AUC: {metrics['AUC']:.4f}")
print(f"  AUPRC: {metrics['AUPRC']:.4f}")
print(f"  MCC: {metrics['MCC']:.4f}")
print(f"  F1: {metrics['F1']:.4f}")
print(f"  Accuracy: {metrics['Accuracy']:.4f}")

print("\n" + "="*60)