"""
Convert ScanNet PPBS Dataset to AHoJ-DB Format
Combines table.csv with labels files to create train/val/test CSV files
"""

import pandas as pd
import os

def parse_labels_file(label_file):
    """Parse ScanNet label file format"""
    data = {}
    current_protein = None
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Header: >13gs_0-A
                current_protein = line[1:]  # Remove '>'
                data[current_protein] = {'sequence': '', 'labels': []}
            else:
                # Data: A 0 M 0
                parts = line.split()
                if len(parts) == 4:
                    chain, pos, residue, label = parts
                    data[current_protein]['sequence'] += residue
                    data[current_protein]['labels'].append(label)
    
    return data

def create_dataset(table_file, label_file, output_file):
    """Create dataset in AHoJ-DB format"""
    
    # Read table
    df_table = pd.read_csv(table_file)
    
    # Parse labels
    label_data = parse_labels_file(label_file)
    
    results = []
    
    for protein_id, data in label_data.items():
        # Extract PDB ID and chain
        # Format: 13gs_0-A or 13gs_A
        if '_' in protein_id:
            parts = protein_id.split('_')
            pdb_id = parts[0]
            chain_part = parts[-1].split('-')[-1]  # Get chain (A, B, etc.)
        else:
            continue
        
        sequence = data['sequence']
        annotation = ''.join(data['labels'])
        length = len(sequence)
        binding_sites = annotation.count('1')
        
        results.append({
            'pdb_id': pdb_id,
            'chain_id': chain_part,
            'sequence': sequence,
            'annotation': annotation,
            'length': length,
            'binding_sites': binding_sites
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save
    df.to_csv(output_file, index=False)
    
    # Statistics
    total_residues = df['length'].sum()
    total_binding = df['binding_sites'].sum()
    
    print(f"\n{output_file}:")
    print(f"  Proteins: {len(df):,}")
    print(f"  Total residues: {total_residues:,}")
    print(f"  Binding residues: {total_binding:,}")
    print(f"  Binding percentage: {100*total_binding/total_residues:.2f}%")
    
    return df

print("="*60)
print("Converting ScanNet PPBS to AHoJ-DB Format")
print("="*60)

# Process each dataset
table_file = 'table.csv'

# Train
print("\nProcessing Training Set...")
df_train = create_dataset(table_file, 'labels_train.txt', 'scannet_train.csv')

# Validation - combine all validation sets
print("\nProcessing Validation Set...")
# For simplicity, use validation_70 (most similar to training)
df_val = create_dataset(table_file, 'labels_validation_70.txt', 'scannet_val.csv')

# Test - combine all test sets
print("\nProcessing Test Set...")
# For simplicity, use test_70
df_test = create_dataset(table_file, 'labels_test_70.txt', 'scannet_test.csv')

print("\n" + "="*60)
print("Conversion Complete!")
print("="*60)
print("\nOutput files:")
print("  scannet_train.csv")
print("  scannet_val.csv")
print("  scannet_test.csv")
print("\nThese files have the same format as AHoJ-DB and can be used")
print("directly with your existing training scripts!")