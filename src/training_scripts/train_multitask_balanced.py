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
        all_embeddings = []
        all_labels = []
        all_task_ids = []
        
        # Load protein data (task_id = 0)
        for npz_file in protein_files:
            data = np.load(npz_file)
            n_samples = len(data['embeddings'])
            all_embeddings.append(data['embeddings'])
            all_labels.append(data['labels'])
            all_task_ids.append(np.zeros(n_samples, dtype=np.int64))
        
        # Load DNA/RNA data (task_id = 1)
        for npz_file in dna_rna_files:
            data = np.load(npz_file)
            n_samples = len(data['embeddings'])
            all_embeddings.append(data['embeddings'])
            all_labels.append(data['labels'])
            all_task_ids.append(np.ones(n_samples, dtype=np.int64))
        
        # Load Ion data (task_id = 2)
        for npz_file in ion_files:
            data = np.load(npz_file)
            n_samples = len(data['embeddings'])
            all_embeddings.append(data['embeddings'])
            all_labels.append(data['labels'])
            all_task_ids.append(np.full(n_samples, 2, dtype=np.int64))
        
        self.embeddings = np.concatenate(all_embeddings, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.task_ids = np.concatenate(all_task_ids, axis=0)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.task_ids[idx]
    
    def get_task_indices(self):
        """Return indices for each task"""
        protein_idx = np.where(self.task_ids == 0)[0]
        dna_rna_idx = np.where(self.task_ids == 1)[0]
        ion_idx = np.where(self.task_ids == 2)[0]
        return protein_idx, dna_rna_idx, ion_idx

# ========== BALANCED BATCH SAMPLER ==========
class BalancedBatchSampler:
    """
    Creates batches with equal samples from each task
    Avoids WeightedRandomSampler's 2^24 limit
    """
    def __init__(self, protein_idx, dna_rna_idx, ion_idx, batch_size, num_batches):
        self.protein_idx = protein_idx
        self.dna_rna_idx = dna_rna_idx
        self.ion_idx = ion_idx
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        # Each task gets 1/3 of batch size
        self.samples_per_task = batch_size // 3
    
    def __iter__(self):
        for _ in range(self.num_batches):
            # Sample equal amounts from each task
            protein_sample = np.random.choice(self.protein_idx, self.samples_per_task, replace=True)
            dna_rna_sample = np.random.choice(self.dna_rna_idx, self.samples_per_task, replace=True)
            ion_sample = np.random.choice(self.ion_idx, self.samples_per_task, replace=True)
            
            # Combine and shuffle
            batch_indices = np.concatenate([protein_sample, dna_rna_sample, ion_sample])
            np.random.shuffle(batch_indices)
            
            yield batch_indices.tolist()
    
    def __len__(self):
        return self.num_batches

# ========== MULTI-TASK MODEL ==========
class MultiTaskBindingNet(nn.Module):
    def __init__(self):
        super().__init__()
        
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
        
        self.protein_head = nn.Linear(128, 1)
        self.dna_rna_head = nn.Linear(128, 1)
        self.ion_head = nn.Linear(128, 1)
    
    def forward(self, x, task_ids):
        shared_features = self.shared_encoder(x)
        outputs = torch.zeros(x.size(0), device=x.device)
        
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

def train_epoch(model, loader, criterion_dict, optimizer):
    model.train()
    total_loss = 0
    
    for x, y, task_ids in loader:
        x, y, task_ids = x.to(DEVICE), y.to(DEVICE), task_ids.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(x, task_ids)
        
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

def evaluate_by_task(model, loader, task_id):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, task_ids in loader:
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
print("MULTI-TASK MODEL WITH BALANCED SAMPLING")
print("="*80)

print("\nLoading datasets...")

# Training data
train_data = MultiTaskDataset(
    protein_files=['scannet_train_embeddings.npz', 'disprot_train_embeddings.npz'],
    dna_rna_files=['biolip_dna_rna_train_embeddings.npz', 'disprot_dna_rna_train_embeddings.npz'],
    ion_files=['ahojdb_train_embeddings.npz', 'disprot_ion_train_embeddings.npz']
)

val_data = MultiTaskDataset(
    protein_files=['disprot_val_embeddings.npz'],
    dna_rna_files=['disprot_dna_rna_val_embeddings.npz'],
    ion_files=['disprot_ion_val_embeddings.npz']
)

test_data = MultiTaskDataset(
    protein_files=['disprot_test_embeddings.npz'],
    dna_rna_files=['disprot_dna_rna_test_embeddings.npz'],
    ion_files=['disprot_ion_test_embeddings.npz']
)

print(f"\nTotal samples:")
print(f"  Train: {len(train_data):,}")
print(f"  Val: {len(val_data):,}")
print(f"  Test: {len(test_data):,}")

# Get task indices
protein_idx, dna_rna_idx, ion_idx = train_data.get_task_indices()

print(f"\nDataset composition:")
print(f"  Protein: {len(protein_idx):,} samples ({100*len(protein_idx)/len(train_data):.1f}%)")
print(f"  DNA/RNA: {len(dna_rna_idx):,} samples ({100*len(dna_rna_idx)/len(train_data):.1f}%)")
print(f"  Ion: {len(ion_idx):,} samples ({100*len(ion_idx)/len(train_data):.1f}%)")

# Create balanced batch sampler
# Each epoch: sample equal amounts from each task
num_batches_per_epoch = max(len(protein_idx), len(dna_rna_idx), len(ion_idx)) // (BATCH_SIZE // 3)

print(f"\nBalanced sampling strategy:")
print(f"  Batches per epoch: {num_batches_per_epoch:,}")
print(f"  Samples per batch: {BATCH_SIZE} ({BATCH_SIZE//3} from each task)")
print(f"  Each task sampled equally per epoch!")

train_sampler = BalancedBatchSampler(
    protein_idx, dna_rna_idx, ion_idx, 
    batch_size=BATCH_SIZE, 
    num_batches=num_batches_per_epoch
)

train_loader = DataLoader(train_data, batch_sampler=train_sampler, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2)

# Task-specific loss functions
criterion_dict = {
    0: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE)),
    1: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE)),
    2: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]).to(DEVICE))
}

model = MultiTaskBindingNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(f"\nTraining balanced multi-task model for {EPOCHS} epochs...")
print(f"Architecture: Shared encoder + 3 task-specific heads")

best_avg_auc = 0
best_epoch = 0
start_time = time.time()

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion_dict, optimizer)
    
    protein_metrics = evaluate_by_task(model, val_loader, task_id=0)
    dna_rna_metrics = evaluate_by_task(model, val_loader, task_id=1)
    ion_metrics = evaluate_by_task(model, val_loader, task_id=2)
    
    avg_auc = (protein_metrics['AUC'] + dna_rna_metrics['AUC'] + ion_metrics['AUC']) / 3
    
    if avg_auc > best_avg_auc:
        best_avg_auc = avg_auc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'multitask_balanced_model.pt')
    
    if (epoch + 1) % 3 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}: Loss={train_loss:.4f}, "
              f"Val AUC: Protein={protein_metrics['AUC']:.4f}, "
              f"DNA/RNA={dna_rna_metrics['AUC']:.4f}, Ion={ion_metrics['AUC']:.4f}, "
              f"Avg={avg_auc:.4f}")

training_time = time.time() - start_time

# Load best model
model.load_state_dict(torch.load('multitask_balanced_model.pt'))

print(f"\n{'='*80}")
print(f"Best Model (Epoch {best_epoch}) - BALANCED SAMPLING")
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

# Comparison
print(f"\n{'='*80}")
print("COMPARISON: Balanced vs Unbalanced vs Individual Models")
print(f"{'='*80}")

individual_results = {
    'Protein': {'AUC': 0.8438},
    'DNA/RNA': {'AUC': 0.7071},
    'Ion': {'AUC': 0.8571}
}

unbalanced_results = {
    'Protein': {'AUC': 0.8390},
    'DNA/RNA': {'AUC': 0.7109},
    'Ion': {'AUC': 0.8445}
}

print(f"{'Task':<12} {'Individual':<12} {'Unbalanced':<12} {'Balanced':<12} {'Δ Ind':<10} {'Δ Unbal':<10}")
print("-"*80)
for task in ['Protein', 'DNA/RNA', 'Ion']:
    ind_auc = individual_results[task]['AUC']
    unbal_auc = unbalanced_results[task]['AUC']
    bal_auc = test_results[task]['AUC']
    diff_ind = bal_auc - ind_auc
    diff_unbal = bal_auc - unbal_auc
    print(f"{task:<12} {ind_auc:<12.4f} {unbal_auc:<12.4f} {bal_auc:<12.4f} {diff_ind:>+10.4f} {diff_unbal:>+10.4f}")

print(f"{'='*80}")