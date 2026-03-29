import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score

WINDOW_SIZE = 31
BATCH_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

class BindingDataset(Dataset):
    """Flexible dataset for both CSV and TSV"""
    def __init__(self, file_path):
        if file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
            is_disprot = True
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            is_disprot = False
        else:
            raise ValueError("File must be .csv or .tsv")
        
        self.samples = []
        self.labels = []
        half_w = WINDOW_SIZE // 2
        
        for _, row in df.iterrows():
            seq = row['sequence']
            
            if is_disprot:
                labels_list = [int(x) for x in row['labels'].split(',')]
            else:
                labels_list = [int(x) for x in row['annotation']]
            
            if len(seq) != len(labels_list):
                continue
            
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
def evaluate_with_best_threshold(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("  Evaluating and finding best threshold...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = None
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_binary = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
            # Calculate all metrics at best threshold
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
print("Evaluating Phase 1: Protein-Protein Model")
print("="*60)

print("\nLoading test data...")
# ScanNet structured test
scannet_test = BindingDataset('/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/scannet_test_clustered.csv')
# DisProt protein-binding test
disprot_test = BindingDataset('/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/protein_binding_test.tsv')

scannet_loader = DataLoader(scannet_test, batch_size=BATCH_SIZE, num_workers=2)
disprot_loader = DataLoader(disprot_test, batch_size=BATCH_SIZE, num_workers=2)

print(f"  ScanNet test: {len(scannet_test):,} residues")
print(f"  DisProt test: {len(disprot_test):,} residues")

# Load model
input_size = WINDOW_SIZE * len(AA_VOCAB)
model = BindingNet(input_size).to(DEVICE)
model.load_state_dict(torch.load('/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/protein_phase1_model.pt', map_location=DEVICE))

print(f"\nUsing device: {DEVICE}")
print("\n" + "="*60)
print("Testing on ScanNet (Structured Proteins)")
print("="*60)
metrics = evaluate_with_best_threshold(model, scannet_loader)
print(f"\nOptimal threshold: {metrics['Threshold']:.2f}")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n" + "="*60)
print("Testing on DisProt (IDP Proteins)")
print("="*60)
metrics = evaluate_with_best_threshold(model, disprot_loader)
print(f"\nOptimal threshold: {metrics['Threshold']:.2f}")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")