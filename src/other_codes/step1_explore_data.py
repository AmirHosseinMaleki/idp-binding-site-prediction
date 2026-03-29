#!/usr/bin/env python3
"""
Explore ion-binding dataset and check for imbalance
"""

import pandas as pd
import numpy as np

df = pd.read_csv('final_complete_ion_dataset.csv')

print("=" * 60)
print("ION-BINDING DATASET EXPLORATION")
print("=" * 60)

# Basic statistics
print(f"\nDataset size:")
print(f"  Total proteins: {len(df)}")
print(f"  Total residues: {df['length'].sum()}")
print(f"  Total binding sites: {df['binding_sites'].sum()}")

# Sequence length statistics
print(f"\nSequence length statistics:")
print(f"  Mean: {df['length'].mean():.1f}")
print(f"  Median: {df['length'].median():.1f}")
print(f"  Min: {df['length'].min()}")
print(f"  Max: {df['length'].max()}")

# Binding site statistics
print(f"\nBinding sites per protein:")
print(f"  Mean: {df['binding_sites'].mean():.1f}")
print(f"  Median: {df['binding_sites'].median():.1f}")
print(f"  Min: {df['binding_sites'].min()}")
print(f"  Max: {df['binding_sites'].max()}")

total_residues = df['length'].sum()
total_binding = df['binding_sites'].sum()
total_non_binding = total_residues - total_binding

positive_percentage = (total_binding / total_residues) * 100
negative_percentage = (total_non_binding / total_residues) * 100

print(f"\nClass Imbalance (Vit's concern):")
print(f"  Binding residues (positive): {total_binding} ({positive_percentage:.2f}%)")
print(f"  Non-binding residues (negative): {total_non_binding} ({negative_percentage:.2f}%)")
print(f"  Imbalance ratio: 1:{total_non_binding/total_binding:.1f}")

weight_positive = total_residues / (2 * total_binding)
weight_negative = total_residues / (2 * total_non_binding)

print(f"\nClass weights for training:")
print(f"  Weight for positive class: {weight_positive:.3f}")
print(f"  Weight for negative class: {weight_negative:.3f}")

print(f"\nIon type distribution:")
ion_counts = df['ligand'].value_counts()
for ion, count in ion_counts.head(10).items():
    print(f"  {ion}: {count}")

stats = {
    'total_proteins': len(df),
    'total_residues': int(total_residues),
    'total_binding': int(total_binding),
    'positive_percentage': float(positive_percentage),
    'negative_percentage': float(negative_percentage),
    'weight_positive': float(weight_positive),
    'weight_negative': float(weight_negative),
    'mean_length': float(df['length'].mean()),
    'max_length': int(df['length'].max())
}

import json
with open('dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)