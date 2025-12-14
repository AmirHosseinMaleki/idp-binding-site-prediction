#!/usr/bin/env python3
"""
Neural Network Architecture for Ion Binding Site Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from typing import Tuple, Dict, List

class IonBindingDataset(Dataset):
    """Dataset for ion binding site prediction"""
    
    def __init__(self, csv_file: str, vocab_file: str, window_size: int = 15):
        """
        Args:
            csv_file: Path to train/val/test CSV
            vocab_file: Path to amino acid vocabulary JSON
            window_size: Context window for each residue (default 15 as in CryptoBench)
        """
        self.data = pd.read_csv(csv_file)
        
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
            self.aa_to_idx = vocab_data['aa_to_idx']
            self.vocab_size = len(vocab_data['amino_acids'])
        
        self.window_size = window_size
        self.pad_idx = self.aa_to_idx.get('X', 20)  # Use X for padding
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        annotation = row['annotation']
        
        # Convert sequence to indices
        seq_indices = [self.aa_to_idx.get(aa, self.pad_idx) for aa in sequence]
        
        # Convert annotation string to list of ints
        labels = [int(x) for x in annotation]
        
        # Create windows for each position
        padded_seq = [self.pad_idx] * self.window_size + seq_indices + [self.pad_idx] * self.window_size
        
        windows = []
        for i in range(len(seq_indices)):
            window = padded_seq[i:i + 2*self.window_size + 1]
            windows.append(window)
        
        return {
            'windows': torch.tensor(windows, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'length': len(sequence),
            'protein_id': f"{row['pdb_id']}_{row['chain_id']}"
        }


class IonBindingNet(nn.Module):
    """
    Neural Network for Ion Binding Site Prediction
    - 3-5 layers
    - 256-2048 layer width
    - Dropout 0.3-0.7
    - AdamW optimizer
    """
    
    def __init__(self, config: Dict):
        super(IonBindingNet, self).__init__()
        
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.window_size = config['window_size']
        self.hidden_dims = config['hidden_dims']  # List of hidden layer dimensions
        self.dropout_rate = config['dropout_rate']
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=20)
        
        input_dim = (self.window_size * 2 + 1) * self.embedding_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, return_logits=False):
        # x shape: (batch_size, seq_len, window_size*2+1)
        batch_size, seq_len, window_len = x.shape
        
        # Reshape for embedding: (batch_size * seq_len, window_len)
        x = x.view(-1, window_len)
        
        # Embed amino acids
        embedded = self.embedding(x)  # (batch_size * seq_len, window_len, embedding_dim)
        
        # Flatten the window embeddings
        embedded = embedded.view(-1, window_len * self.embedding_dim)
        
        # Pass through network
        logits = self.network(embedded)
        
        # Reshape back to (batch_size, seq_len)
        logits = logits.view(batch_size, seq_len)
        
        if return_logits:
            return logits
        else:
            # Apply sigmoid for binary classification
            return torch.sigmoid(logits)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        BCE_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)


def get_model_configs():
    """
    Generate different model configurations
    Returns list of configurations to try
    """
    configs = []
    
    # Config 1: Small model (3 layers, 256-512 width)
    configs.append({
        'name': 'small_3layer',
        'vocab_size': 21,
        'embedding_dim': 20,
        'window_size': 15,
        'hidden_dims': [256, 512, 256],
        'dropout_rate': 0.3,
        'learning_rate': 1e-3,
        'loss_type': 'bce'  # binary cross-entropy
    })
    
    # Config 2: Medium model (4 layers, 512-1024 width)
    configs.append({
        'name': 'medium_4layer',
        'vocab_size': 21,
        'embedding_dim': 20,
        'window_size': 15,
        'hidden_dims': [512, 1024, 1024, 512],
        'dropout_rate': 0.5,
        'learning_rate': 5e-4,
        'loss_type': 'focal'  # focal loss
    })
    
    # Config 3: Large model (5 layers, 1024-2048 width)
    configs.append({
        'name': 'large_5layer',
        'vocab_size': 21,
        'embedding_dim': 20,
        'window_size': 15,
        'hidden_dims': [1024, 2048, 2048, 1024, 512],
        'dropout_rate': 0.5,
        'learning_rate': 3e-4,
        'loss_type': 'bce'
    })
    
    # Config 4: Deep narrow (5 layers, 256-512 width, high dropout)
    configs.append({
        'name': 'deep_narrow',
        'vocab_size': 21,
        'embedding_dim': 20,
        'window_size': 15,
        'hidden_dims': [256, 512, 512, 512, 256],
        'dropout_rate': 0.7,
        'learning_rate': 1e-3,
        'loss_type': 'focal'
    })
    
    # Config 5: Wide shallow (3 layers, 2048 width)
    configs.append({
        'name': 'wide_shallow',
        'vocab_size': 21,
        'embedding_dim': 20,
        'window_size': 15,
        'hidden_dims': [2048, 2048, 1024],
        'dropout_rate': 0.4,
        'learning_rate': 2e-4,
        'loss_type': 'bce'
    })
    
    return configs


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    max_len = max([item['windows'].shape[0] for item in batch])
    
    padded_windows = []
    padded_labels = []
    lengths = []
    protein_ids = []
    
    for item in batch:
        seq_len = item['windows'].shape[0]
        
        # Pad sequences to max length in batch
        if seq_len < max_len:
            pad_len = max_len - seq_len
            padded_window = torch.cat([
                item['windows'],
                torch.zeros(pad_len, item['windows'].shape[1], dtype=torch.long)
            ])
            padded_label = torch.cat([
                item['labels'],
                torch.zeros(pad_len, dtype=torch.float32)
            ])
        else:
            padded_window = item['windows']
            padded_label = item['labels']
        
        padded_windows.append(padded_window)
        padded_labels.append(padded_label)
        lengths.append(seq_len)
        protein_ids.append(item['protein_id'])
    
    return {
        'windows': torch.stack(padded_windows),
        'labels': torch.stack(padded_labels),
        'lengths': torch.tensor(lengths),
        'protein_ids': protein_ids
    }


def test_model():
    """Test model creation and forward pass"""
    
    with open('dataset_stats.json', 'r') as f:
        stats = json.load(f)
    
    print("Class weights from dataset:")
    print(f"  Positive weight: {stats['weight_positive']:.3f}")
    print(f"  Negative weight: {stats['weight_negative']:.3f}")
    
    configs = get_model_configs()
    
    print(f"\nTesting {len(configs)} model configurations:")
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Layers: {len(config['hidden_dims'])}")
        print(f"  Hidden dims: {config['hidden_dims']}")
        print(f"  Dropout: {config['dropout_rate']}")
        print(f"  Loss type: {config['loss_type']}")
        
        model = IonBindingNet(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 4
        seq_len = 100
        window_size = config['window_size']
        
        dummy_input = torch.randint(0, 21, (batch_size, seq_len, window_size*2+1))
        output = model(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        assert output.shape == (batch_size, seq_len), "Output shape mismatch!"
        print("Forward pass successful")


if __name__ == "__main__":
    test_model()