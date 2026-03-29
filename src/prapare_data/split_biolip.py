"""
Step 3: Create train/val/test splits for BioLiP DNA and RNA data
70% train, 15% val, 15% test
"""

import pandas as pd
import numpy as np

print("="*60)
print("Creating Train/Val/Test Splits for BioLiP Data")
print("="*60)

def create_splits(input_file, output_prefix):
    """Create 70/15/15 train/val/test splits"""
    print(f"\nProcessing: {input_file}")
    
    df = pd.read_csv(input_file)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split sizes
    n = len(df)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    # Split
    train = df[:train_size]
    val = df[train_size:train_size + val_size]
    test = df[train_size + val_size:]
    
    # Save
    train.to_csv(f'{output_prefix}_train.csv', index=False)
    val.to_csv(f'{output_prefix}_val.csv', index=False)
    test.to_csv(f'{output_prefix}_test.csv', index=False)
    
    # Statistics
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train):,} proteins ({len(train)/n*100:.1f}%)")
    print(f"    Residues: {train['length'].sum():,}")
    print(f"    Binding: {train['binding_sites'].sum():,} ({100*train['binding_sites'].sum()/train['length'].sum():.2f}%)")
    
    print(f"  Val: {len(val):,} proteins ({len(val)/n*100:.1f}%)")
    print(f"    Residues: {val['length'].sum():,}")
    print(f"    Binding: {val['binding_sites'].sum():,} ({100*val['binding_sites'].sum()/val['length'].sum():.2f}%)")
    
    print(f"  Test: {len(test):,} proteins ({len(test)/n*100:.1f}%)")
    print(f"    Residues: {test['length'].sum():,}")
    print(f"    Binding: {test['binding_sites'].sum():,} ({100*test['binding_sites'].sum()/test['length'].sum():.2f}%)")
    
    print(f"\nSaved:")
    print(f"  {output_prefix}_train.csv")
    print(f"  {output_prefix}_val.csv")
    print(f"  {output_prefix}_test.csv")

# Process DNA
print("\n" + "="*60)
print("DNA Binding Data")
print("="*60)
create_splits('biolip_dna_all.csv', 'biolip_dna')

# Process RNA
print("\n" + "="*60)
print("RNA Binding Data")
print("="*60)
create_splits('biolip_rna_all.csv', 'biolip_rna')

print("\n" + "="*60)
print("Splits Complete!")
print("="*60)
print("\nYou now have:")
print("  DNA: biolip_dna_train.csv, biolip_dna_val.csv, biolip_dna_test.csv")
print("  RNA: biolip_rna_train.csv, biolip_rna_val.csv, biolip_rna_test.csv")