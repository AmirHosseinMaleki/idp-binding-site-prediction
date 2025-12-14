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

class StructuredDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.samples = []
        self.labels = []
        half_w = WINDOW_SIZE // 2
        
        for _, row in df.iterrows():
            seq = row['sequence']
            ann = row['annotation']
            padded = 'X' * half_w + seq + 'X' * half_w
            
            for i in range(len(seq)):
                window = padded[i:i+WINDOW_SIZE]
                encoded = np.zeros((WINDOW_SIZE, len(AA_VOCAB)), dtype=np.float32)
                for j, aa in enumerate(window):
                    idx = AA_TO_IDX.get(aa, AA_TO_IDX['X'])
                    encoded[j, idx] = 1.0
                
                self.samples.append(encoded.flatten())
                self.labels.append(int(ann[i]))
        
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

class DisprotDataset(Dataset):
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

def evaluate_model(model, loader, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            all_preds.extend(out.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    preds_binary = (all_preds >= threshold).astype(int)
    
    auc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary)
    acc = accuracy_score(all_labels, preds_binary)
    
    return {
        'AUC': auc,
        'AUPRC': auprc,
        'MCC': mcc,
        'F1': f1,
        'Accuracy': acc
    }

input_size = WINDOW_SIZE * len(AA_VOCAB)

struct_test = StructuredDataset('test_data.csv')
disprot_test = DisprotDataset('ion_binding_test.tsv')

struct_loader = DataLoader(struct_test, batch_size=BATCH_SIZE, num_workers=4)
disprot_loader = DataLoader(disprot_test, batch_size=BATCH_SIZE, num_workers=4)

print("\n" + "="*60)
print("PHASE 1: Structured Data Model")
print("="*60)
model1 = BindingNet(input_size).to(DEVICE)
model1.load_state_dict(torch.load('phase1_model.pt', map_location=DEVICE))

print("\nTesting on Structured test data:")
metrics = evaluate_model(model1, struct_loader, threshold=0.5)
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nTesting on DisProt test data:")
metrics = evaluate_model(model1, disprot_loader, threshold=0.5)
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n" + "="*60)
print("PHASE 2: DisProt Only Model")
print("="*60)
model2 = BindingNet(input_size).to(DEVICE)
model2.load_state_dict(torch.load('phase2_model.pt', map_location=DEVICE))

print("\nTesting on Structured test data:")
metrics = evaluate_model(model2, struct_loader, threshold=0.5)
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nTesting on DisProt test data:")
metrics = evaluate_model(model2, disprot_loader, threshold=0.5)
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n" + "="*60)
print("PHASE 3: Mixed Data Model")
print("="*60)
model3 = BindingNet(input_size).to(DEVICE)
model3.load_state_dict(torch.load('phase3_model.pt', map_location=DEVICE))

print("\nTesting on Structured test data:")
metrics = evaluate_model(model3, struct_loader, threshold=0.5)
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nTesting on DisProt test data:")
metrics = evaluate_model(model3, disprot_loader, threshold=0.5)
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n" + "="*60)