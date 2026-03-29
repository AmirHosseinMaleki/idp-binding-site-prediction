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

# ========== DATASET WITH TASK LABELS ==========
class MultiTaskDataset(Dataset):
    def __init__(self, protein_files, dna_rna_files, ion_files):
        """
        Combine all three binding types into one dataset
        Each sample has: (embedding, label, task_id)
        task_id: 0=protein, 1=dna_rna, 2=ion
        """
        all_embeddings = []
        all_labels = []
        all_task_ids = []
        
        # Load protein data (task_id = 0)
        for npz_file in protein_files:
            data = np.load(npz_file)
            n_samples = len(data['embeddings'])
            all_embeddings.append(data['embeddings'])
            all_labels.append(data['labels'])
            all_task_ids.append(np.zeros(n_samples, dtype=np.int64))  # task 0
        
        # Load DNA/RNA data (task_id = 1)
        for npz_file in dna_rna_files:
            data = np.load(npz_file)
            n_samples = len(data['embeddings'])
            all_embeddings.append(data['embeddings'])
            all_labels.append(data['labels'])
            all_task_ids.append(np.ones(n_samples, dtype=np.int64))  # task 1
        
        # Load Ion data (task_id = 2)
        for npz_file in ion_files:
            data = np.load(npz_file)
            n_samples = len(data['embeddings'])
            all_embeddings.append(data['embeddings'])
            all_labels.append(data['labels'])
            all_task_ids.append(np.full(n_samples, 2, dtype=np.int64))  # task 2
        
        self.embeddings = np.concatenate(all_embeddings, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.task_ids = np.concatenate(all_task_ids, axis=0)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.task_ids[idx]

# ========== MULTI-TASK MODEL ==========
class MultiTaskBindingNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared encoder (learns common features across all binding types)
        self.shared_encoder = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads (specialized for each binding type)
        self.protein_head = nn.Linear(128, 1)
        self.dna_rna_head = nn.Linear(128, 1)
        self.ion_head = nn.Linear(128, 1)
    
    def forward(self, x, task_ids):
        """
        x: embeddings (batch_size, 1280)
        task_ids: which task each sample belongs to (batch_size,)
        """
        # Shared encoding
        shared_features = self.shared_encoder(x)
        
        # Route through appropriate head based on task
        outputs = torch.zeros(x.size(0), device=x.device)
        
        # Process each task
        protein_mask = (task_ids == 0)
        dna_rna_mask = (task_ids == 1)
        ion_mask = (task_ids == 2)
        
        if protein_mask.any():
            outputs[protein_mask] = self.protein_head(shared_features[protein_mask]).squeeze()
        if dna_rna_mask.any():
            outputs[dna_rna_mask] = self.dna_rna_head(shared_features[dna_rna_mask]).squeeze()
        if ion_mask.any():
            outputs[ion_mask] = self.ion_head(shared_features[ion_mask]).squeeze()
        
        return outputs

# ========== TRAINING FUNCTION ==========
def train_epoch(model, loader, criterion_dict, optimizer):
    model.train()
    total_loss = 0
    
    for x, y, task_ids in loader:
        x, y, task_ids = x.to(DEVICE), y.to(DEVICE), task_ids.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(x, task_ids)
        
        # Use task-specific loss weights
        loss = torch.tensor(0.0, device=DEVICE)
        for task_id in [0, 1, 2]:
            mask = (task_ids == task_id)
            if mask.any():
                task_loss = criterion_dict[task_id](outputs[mask], y[mask])
                loss = loss + task_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

# ========== EVALUATION FUNCTION ==========
def evaluate_by_task(model, loader, task_id):
    """Evaluate model on specific task"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, task_ids in loader:
            # Filter for specific task
            mask = (task_ids == task_id).numpy()
            if not mask.any():
                continue
            
            x_task = x[mask].to(DEVICE)
            y_task = y[mask]
            task_ids_tensor = torch.full((x_task.size(0),), task_id, dtype=torch.long, device=DEVICE)
            
            outputs = model(x_task, task_ids_tensor)
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y_task.numpy())
    
    if len(all_preds) == 0:
        return None
    
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

# ========== MAIN TRAINING ==========
print("="*80)
print("MULTI-TASK UNIFIED MODEL - All Three Binding Types")
print("="*80)

print("\nLoading datasets...")

# Training data (all three types combined)
train_data = MultiTaskDataset(
    protein_files=['scannet_train_embeddings.npz', 'disprot_train_embeddings.npz'],
    dna_rna_files=['biolip_dna_rna_train_embeddings.npz', 'disprot_dna_rna_train_embeddings.npz'],
    ion_files=['ahojdb_train_embeddings.npz', 'disprot_ion_train_embeddings.npz']
)

# Validation data (DisProt only for all three)
val_data = MultiTaskDataset(
    protein_files=['disprot_val_embeddings.npz'],
    dna_rna_files=['disprot_dna_rna_val_embeddings.npz'],
    ion_files=['disprot_ion_val_embeddings.npz']
)

# Test data (DisProt only)
test_data = MultiTaskDataset(
    protein_files=['disprot_test_embeddings.npz'],
    dna_rna_files=['disprot_dna_rna_test_embeddings.npz'],
    ion_files=['disprot_ion_test_embeddings.npz']
)

print(f"  Train: {len(train_data):,} samples (all three tasks)")
print(f"  Val: {len(val_data):,} samples (DisProt only)")
print(f"  Test: {len(test_data):,} samples (DisProt only)")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2)

# Task-specific loss functions
criterion_dict = {
    0: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE)),   # Protein
    1: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE)),   # DNA/RNA
    2: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]).to(DEVICE))   # Ion
}

model = MultiTaskBindingNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(f"\nTraining unified model for {EPOCHS} epochs...")
print(f"Architecture: Shared encoder + 3 task-specific heads")

best_avg_auc = 0
best_epoch = 0
start_time = time.time()

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion_dict, optimizer)
    
    # Evaluate on each task separately
    protein_metrics = evaluate_by_task(model, val_loader, task_id=0)
    dna_rna_metrics = evaluate_by_task(model, val_loader, task_id=1)
    ion_metrics = evaluate_by_task(model, val_loader, task_id=2)
    
    # Average AUC across tasks
    avg_auc = (protein_metrics['AUC'] + dna_rna_metrics['AUC'] + ion_metrics['AUC']) / 3
    
    if avg_auc > best_avg_auc:
        best_avg_auc = avg_auc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'multitask_unified_model.pt')
    
    if (epoch + 1) % 3 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}: Loss={train_loss:.4f}, "
              f"Val AUC: Protein={protein_metrics['AUC']:.4f}, "
              f"DNA/RNA={dna_rna_metrics['AUC']:.4f}, Ion={ion_metrics['AUC']:.4f}")

training_time = time.time() - start_time

# Load best model and evaluate on test set
model.load_state_dict(torch.load('multitask_unified_model.pt'))

print(f"\n{'='*80}")
print(f"Best Model (Epoch {best_epoch})")
print(f"{'='*80}")

test_results = {}
for task_id, task_name in [(0, 'Protein'), (1, 'DNA/RNA'), (2, 'Ion')]:
    metrics = evaluate_by_task(model, test_loader, task_id)
    test_results[task_name] = metrics
    
    print(f"\n{task_name} Binding:")
    print(f"  Threshold: {metrics['Threshold']:.2f}")
    print(f"  AUC:       {metrics['AUC']:.4f}")
    print(f"  AUPRC:     {metrics['AUPRC']:.4f}")
    print(f"  MCC:       {metrics['MCC']:.4f}")
    print(f"  F1:        {metrics['F1']:.4f}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")

print(f"\nTraining Time: {training_time/60:.1f} min")

# ========== COMPARISON TO INDIVIDUAL MODELS ==========
print(f"\n{'='*80}")
print("COMPARISON: Multi-Task vs Individual Models")
print(f"{'='*80}")

individual_results = {
    'Protein': {'AUC': 0.8438, 'AUPRC': 0.6267, 'MCC': 0.5060, 'F1': 0.6507},
    'DNA/RNA': {'AUC': 0.7071, 'AUPRC': 0.5548, 'MCC': 0.3304, 'F1': 0.5803},
    'Ion': {'AUC': 0.8571, 'AUPRC': 0.6352, 'MCC': 0.5034, 'F1': 0.6256}
}

print(f"{'Task':<12} {'Individual AUC':<16} {'Multi-Task AUC':<16} {'Difference':<12}")
print("-"*80)
for task in ['Protein', 'DNA/RNA', 'Ion']:
    ind_auc = individual_results[task]['AUC']
    mt_auc = test_results[task]['AUC']
    diff = mt_auc - ind_auc
    print(f"{task:<12} {ind_auc:<16.4f} {mt_auc:<16.4f} {diff:>+12.4f}")

print(f"{'='*80}")