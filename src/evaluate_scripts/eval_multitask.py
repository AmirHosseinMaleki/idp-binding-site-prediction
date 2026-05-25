import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

DEVICE = torch.device('cpu')
BATCH_SIZE = 512

class EmbeddingDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx): return self.embeddings[idx], self.labels[idx]

class MultiTaskBindingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.3)
        )
        self.protein_head = nn.Linear(128, 1)
        self.dna_rna_head = nn.Linear(128, 1)
        self.ion_head     = nn.Linear(128, 1)

    def forward(self, x, task_id):
        feat = self.shared_encoder(x)
        if task_id == 0: return self.protein_head(feat).squeeze()
        if task_id == 1: return self.dna_rna_head(feat).squeeze()
        return self.ion_head(feat).squeeze()

def get_auc(model, npz_file, task_id):
    dataset = EmbeddingDataset(npz_file)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE), task_id)
            preds.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(y.numpy())
    return roc_auc_score(np.array(labels), np.array(preds))

tasks = [
    (0, 'Protein',  'disprot_test_embeddings.npz'),
    (1, 'DNA/RNA',  'disprot_dna_rna_test_embeddings.npz'),
    (2, 'Ion',      'disprot_ion_test_embeddings.npz'),
]

for model_name, path in [
    ('Unified',  'multitask_unified_model.pt'),
    ('Balanced', 'multitask_balanced_model.pt'),
]:
    model = MultiTaskBindingNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f"\n{model_name}:")
    for task_id, task_name, npz in tasks:
        auc = get_auc(model, npz, task_id)
        print(f"  {task_name}: AUC = {auc:.4f}")