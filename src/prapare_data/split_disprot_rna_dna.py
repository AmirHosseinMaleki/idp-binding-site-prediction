"""
Split DisProt DNA and RNA binding data into train/val/test (70/15/15)
"""

import pandas as pd
import numpy as np

print("="*60)
print("Splitting DisProt DNA and RNA Binding Data")
print("="*60)

def split_disprot_data(input_file, output_prefix):
    """Split DisProt data into 70/15/15 train/val/test"""
    print(f"\nProcessing: {input_file}")
    
    df = pd.read_csv(input_file, sep='\t')
    
    print(f"  Total proteins: {len(df):,}")
    
    # Check data format
    if 'labels' in df.columns:
        # Calculate statistics
        total_residues = 0
        binding_residues = 0
        for _, row in df.iterrows():
            labels = [int(x) for x in row['labels'].split(',')]
            total_residues += len(labels)
            binding_residues += sum(labels)
        
        print(f"  Total residues: {total_residues:,}")
        print(f"  Binding residues: {binding_residues:,}")
        print(f"  Binding percentage: {100*binding_residues/total_residues:.2f}%")
    
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
    train.to_csv(f'{output_prefix}_train.tsv', sep='\t', index=False)
    val.to_csv(f'{output_prefix}_val.tsv', sep='\t', index=False)
    test.to_csv(f'{output_prefix}_test.tsv', sep='\t', index=False)
    
    # Statistics per split
    def get_split_stats(split_df, split_name):
        if 'labels' not in split_df.columns:
            return
        
        total_res = 0
        binding_res = 0
        for _, row in split_df.iterrows():
            labels = [int(x) for x in row['labels'].split(',')]
            total_res += len(labels)
            binding_res += sum(labels)
        
        print(f"\n  {split_name}: {len(split_df):,} proteins ({len(split_df)/n*100:.1f}%)")
        print(f"    Residues: {total_res:,}")
        print(f"    Binding: {binding_res:,} ({100*binding_res/total_res:.2f}%)")
    
    get_split_stats(train, "Train")
    get_split_stats(val, "Val")
    get_split_stats(test, "Test")
    
    print(f"\n  Saved:")
    print(f"    {output_prefix}_train.tsv")
    print(f"    {output_prefix}_val.tsv")
    print(f"    {output_prefix}_test.tsv")

# Process DNA
print("\n" + "="*60)
print("DisProt DNA Binding Data")
print("="*60)
split_disprot_data('/home/malekia/idp-binding-site-prediction/data/training/dna_binding_training_data.tsv', 'dna_binding')

# Process RNA
print("\n" + "="*60)
print("DisProt RNA Binding Data")
print("="*60)
split_disprot_data('/home/malekia/idp-binding-site-prediction/data/training/rna_binding_training_data.tsv', 'rna_binding')

print("\n" + "="*60)
print("Splits Complete!")
print("="*60)
print("\nYou now have:")
print("  DNA: dna_binding_train.tsv, dna_binding_val.tsv, dna_binding_test.tsv")
print("  RNA: rna_binding_train.tsv, rna_binding_val.tsv, rna_binding_test.tsv")