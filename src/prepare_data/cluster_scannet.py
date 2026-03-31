"""
Cluster ScanNet PPBS and Filter Against CAID3
Same approach as used for AHoJ-DB ion-binding data
"""

import pandas as pd
import subprocess
import os
from pathlib import Path

print("="*60)
print("Clustering and Filtering ScanNet PPBS")
print("="*60)

# Step 1: Combine all data for clustering
print("\nStep 1: Combining datasets...")
df_train = pd.read_csv('scannet_train.csv')
df_val = pd.read_csv('scannet_val.csv')
df_test = pd.read_csv('scannet_test.csv')

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
print(f"Total proteins: {len(df_all):,}")

# Step 2: Create FASTA file for MMseqs2
print("\nStep 2: Creating FASTA file...")
fasta_file = 'scannet_all.fasta'
with open(fasta_file, 'w') as f:
    for idx, row in df_all.iterrows():
        header = f"{row['pdb_id']}_{row['chain_id']}"
        f.write(f">{header}\n{row['sequence']}\n")

print(f"Wrote {len(df_all):,} sequences to {fasta_file}")

# Step 3: Run MMseqs2 clustering at 10% identity
print("\nStep 3: Running MMseqs2 clustering at 10% identity...")
print("(This may take 10-30 minutes)")

# MMseqs2 commands
db_file = 'scannet_db'
cluster_file = 'scannet_cluster'
result_file = 'scannet_cluster_result'

commands = [
    f"mmseqs createdb {fasta_file} {db_file}",
    f"mmseqs cluster {db_file} {cluster_file} tmp --min-seq-id 0.1 -c 0.8 --cov-mode 0",
    f"mmseqs createtsv {db_file} {db_file} {cluster_file} {result_file}.tsv"
]

for cmd in commands:
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        print("  Make sure MMseqs2 is installed: conda install -c bioconda mmseqs2")
        exit(1)

print("  Clustering complete!")

# Step 4: Parse clusters
print("\nStep 4: Parsing clusters...")
clusters = {}
with open(f'{result_file}.tsv', 'r') as f:
    for line in f:
        rep, member = line.strip().split('\t')
        if rep not in clusters:
            clusters[rep] = []
        clusters[rep].append(member)

print(f"Found {len(clusters):,} clusters")

# Step 5: Select one representative per cluster
print("\nStep 5: Selecting cluster representatives...")
representatives = set(clusters.keys())

df_clustered = df_all[df_all.apply(lambda row: f"{row['pdb_id']}_{row['chain_id']}" in representatives, axis=1)]
print(f"After clustering: {len(df_clustered):,} proteins")

# Step 6: Filter against CAID3 (if file exists)
caid3_files = [
    'binding.fasta',       # CAID3 with ordered regions as negative
    'binding-idr.fasta'    # CAID3 with ordered regions excluded
]

caid3_proteins = set()
for caid3_file in caid3_files:
    if os.path.exists(caid3_file):
        print(f"\nStep 6: Filtering against CAID3: {caid3_file}...")
        with open(caid3_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # Parse header: >PDB_ID_CHAIN or similar
                    protein_id = line[1:].strip().split()[0]
                    # Handle different formats: A0A001_A, A0A001-A, etc.
                    protein_id = protein_id.replace('-', '_')
                    caid3_proteins.add(protein_id)
        print(f"  Found {len(caid3_proteins)} unique CAID3 proteins")

if caid3_proteins:
    initial_count = len(df_clustered)
    
    # Check both pdb_chain and just pdb_id
    df_clustered = df_clustered[~df_clustered.apply(
        lambda row: (
            f"{row['pdb_id']}_{row['chain_id']}" in caid3_proteins or
            f"{row['pdb_id']}-{row['chain_id']}" in caid3_proteins or
            row['pdb_id'] in caid3_proteins
        ), axis=1
    )]
    
    removed = initial_count - len(df_clustered)
    print(f"\nRemoved {removed} proteins overlapping with CAID3")
    print(f"Remaining: {len(df_clustered):,} proteins")
else:
    print(f"\nStep 6: CAID3 files not found, skipping filtering...")
    print("CAID3 files should be:")
    print("  - binding.fasta")
    print("  - binding-idr.fasta")
    print("If you have them, place them in the current directory")

# Step 7: Re-split into train/val/test (70/15/15)
print("\nStep 7: Creating new train/val/test splits...")
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df_clustered, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

# Save
train_df.to_csv('scannet_train_clustered.csv', index=False)
val_df.to_csv('scannet_val_clustered.csv', index=False)
test_df.to_csv('scannet_test_clustered.csv', index=False)

# Statistics
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    total_res = df['length'].sum()
    binding_res = df['binding_sites'].sum()
    print(f"\n{name}:")
    print(f"  Proteins: {len(df):,}")
    print(f"  Residues: {total_res:,}")
    print(f"  Binding: {binding_res:,} ({100*binding_res/total_res:.2f}%)")

print("\n" + "="*60)
print("Clustering and Filtering Complete!")
print("="*60)
print("\nOutput files:")
print("  scannet_train_clustered.csv")
print("  scannet_val_clustered.csv")
print("  scannet_test_clustered.csv")
print("\nUse these for training to avoid data leakage!")