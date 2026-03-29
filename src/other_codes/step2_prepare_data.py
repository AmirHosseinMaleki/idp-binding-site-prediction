#!/usr/bin/env python3
"""
Prepare data for neural network training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

df = pd.read_csv('final_complete_ion_dataset.csv')
print(f"Loaded {len(df)} proteins")

# 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"\nDataset split:")
print(f"  Train: {len(train_df)} proteins ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Validation: {len(val_df)} proteins ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test: {len(test_df)} proteins ({len(test_df)/len(df)*100:.1f}%)")

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}

print(f"\nAmino acid vocabulary size: {len(amino_acids)}")

train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

with open('aa_vocabulary.json', 'w') as f:
    json.dump({'amino_acids': amino_acids, 'aa_to_idx': aa_to_idx}, f, indent=2)

for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    total_res = split_df['length'].sum()
    total_bind = split_df['binding_sites'].sum()
    pos_pct = (total_bind / total_res) * 100
    print(f"\n{name} set balance:")
    print(f"  Binding sites: {pos_pct:.2f}%")