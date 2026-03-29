import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameter grid
GRID = {
    'lr': [0.0001, 0.00005, 0.0002],
    'dropout': [(0.3, 0.3, 0.2), (0.5, 0.5, 0.3), (0.6, 0.6, 0.4)],
    'weight_decay': [0.001, 0.01, 0.05],
    'batch_size': [256, 512],
    'epochs': [10]
}

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

class BindingNet(nn.Module):
    def __init__(self, dropout1, dropout2, dropout3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

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

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_config(config, train_data, val_data, config_id):
    """Train a single configuration"""
    
    print(f"\n{'='*60}")
    print(f"Configuration {config_id}")
    print(f"{'='*60}")
    print(f"  LR: {config['lr']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Weight Decay: {config['weight_decay']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], 
                           num_workers=2)
    
    # Create model
    dropout1, dropout2, dropout3 = config['dropout']
    model = BindingNet(dropout1, dropout2, dropout3).to(DEVICE)
    
    # Loss and optimizer
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config['lr'], 
                                  weight_decay=config['weight_decay'])
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # Save best model
            torch.save(model.state_dict(), f'protein_grid_config_{config_id}_model.pt')
        
        if (epoch + 1) % 2 == 0:  # Print every 2 epochs
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss:.4f} | Val {val_loss:.4f}")
    
    print(f"\n  Best: Epoch {best_epoch}, Val Loss {best_val_loss:.4f}")
    
    return {
        'config_id': config_id,
        'config': config,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

# Load data once
print("="*60)
print("Protein-Protein Binding - Grid Search")
print("="*60)

print("\nLoading training data:")
train_data = CombinedDataset([
    'scannet_train_embeddings.npz',
    'disprot_train_embeddings.npz'
])

print("\nLoading validation data (DisProt only):")
val_data = EmbeddingDataset('disprot_val_embeddings.npz')

# Generate all configurations
configs = []
config_id = 1

for lr in GRID['lr']:
    for dropout in GRID['dropout']:
        for wd in GRID['weight_decay']:
            for bs in GRID['batch_size']:
                for ep in GRID['epochs']:
                    configs.append({
                        'lr': lr,
                        'dropout': dropout,
                        'weight_decay': wd,
                        'batch_size': bs,
                        'epochs': ep
                    })

print(f"\nTotal configurations to test: {len(configs)}")
print(f"Estimated time: ~{len(configs) * 15} minutes")

# Train all configurations
results = []

for i, config in enumerate(configs, 1):
    result = train_config(config, train_data, val_data, config_id=i)
    results.append(result)

# Sort by best validation loss
results_sorted = sorted(results, key=lambda x: x['best_val_loss'])

print("\n" + "="*60)
print("GRID SEARCH RESULTS - TOP 5 CONFIGURATIONS")
print("="*60)

for i, result in enumerate(results_sorted[:5], 1):
    print(f"\nRank {i}:")
    print(f"  Config ID: {result['config_id']}")
    print(f"  Val Loss: {result['best_val_loss']:.4f} (Epoch {result['best_epoch']})")
    print(f"  LR: {result['config']['lr']}")
    print(f"  Dropout: {result['config']['dropout']}")
    print(f"  Weight Decay: {result['config']['weight_decay']}")
    print(f"  Batch Size: {result['config']['batch_size']}")

results_json = {
    'timestamp': datetime.now().isoformat(),
    'device': str(DEVICE),
    'total_configs': len(configs),
    'results': results_sorted
}

with open('protein_grid_search_results.json', 'w') as f:
    json.dump(results_json, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"Results saved to: protein_grid_search_results.json")
print(f"Best model: protein_grid_config_{results_sorted[0]['config_id']}_model.pt")
print(f"{'='*60}")