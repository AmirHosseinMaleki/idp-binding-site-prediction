"""
Cluster BioLiP DNA and RNA data and Filter Against CAID3
Same approach as used for ScanNet PPBS
"""

import pandas as pd
import subprocess
import os

print("="*60)
print("Clustering and Filtering BioLiP DNA and RNA Data")
print("="*60)

def cluster_and_filter(input_file, output_prefix, name):
    """Cluster dataset and filter against CAID3"""
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")
    
    # Step 1: Load data
    print(f"\nStep 1: Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Total proteins: {len(df):,}")
    
    # Step 2: Create FASTA file
    print("\nStep 2: Creating FASTA file...")
    fasta_file = f'{output_prefix}_all.fasta'
    with open(fasta_file, 'w') as f:
        for idx, row in df.iterrows():
            header = f"{row['pdb_id']}_{row['chain_id']}"
            f.write(f">{header}\n{row['sequence']}\n")
    
    print(f"Wrote {len(df):,} sequences to {fasta_file}")
    
    # Step 3: Run MMseqs2 clustering at 10% identity
    print("\nStep 3: Running MMseqs2 clustering at 10% identity...")
    print("(This may take 5-15 minutes)")
    
    db_file = f'{output_prefix}_db'
    cluster_file = f'{output_prefix}_cluster'
    result_file = f'{output_prefix}_cluster_result'
    
    commands = [
        f"mmseqs createdb {fasta_file} {db_file}",
        f"mmseqs cluster {db_file} {cluster_file} tmp_{output_prefix} --min-seq-id 0.1 -c 0.8 --cov-mode 0",
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
    
    # Step 5: Select representatives
    print("\nStep 5: Selecting cluster representatives...")
    representatives = set(clusters.keys())
    
    df_clustered = df[df.apply(lambda row: f"{row['pdb_id']}_{row['chain_id']}" in representatives, axis=1)]
    print(f"After clustering: {len(df_clustered):,} proteins")
    
    # Step 6: Filter against CAID3 (if files exist)
    caid3_files = [
        'binding.fasta',
        'binding-idr.fasta'
    ]
    
    caid3_proteins = set()
    for caid3_file in caid3_files:
        if os.path.exists(caid3_file):
            print(f"\nStep 6: Filtering against CAID3: {caid3_file}...")
            with open(caid3_file, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        protein_id = line[1:].strip().split()[0]
                        protein_id = protein_id.replace('-', '_')
                        caid3_proteins.add(protein_id)
            print(f"  Found {len(caid3_proteins)} unique CAID3 proteins")
    
    if caid3_proteins:
        initial_count = len(df_clustered)
        
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
        print(f"\nStep 6: CAID3 files not found, skipping filtering")
    
    # Step 7: Split into train/val/test (70/15/15)
    print("\nStep 7: Creating train/val/test splits...")
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(df_clustered, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    # Save
    train_df.to_csv(f'{output_prefix}_train.csv', index=False)
    val_df.to_csv(f'{output_prefix}_val.csv', index=False)
    test_df.to_csv(f'{output_prefix}_test.csv', index=False)
    
    # Statistics
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        total_res = split_df['length'].sum()
        binding_res = split_df['binding_sites'].sum()
        print(f"\n{split_name}:")
        print(f"  Proteins: {len(split_df):,}")
        print(f"  Residues: {total_res:,}")
        print(f"  Binding: {binding_res:,} ({100*binding_res/total_res:.2f}%)")
    
    print(f"\nSaved:")
    print(f"  {output_prefix}_train.csv")
    print(f"  {output_prefix}_val.csv")
    print(f"  {output_prefix}_test.csv")

# Process DNA
cluster_and_filter('biolip_dna_all.csv', 'biolip_dna_clustered', 'DNA Binding')

# Process RNA
cluster_and_filter('biolip_rna_all.csv', 'biolip_rna_clustered', 'RNA Binding')

print("\n" + "="*60)
print("Clustering and Filtering Complete!")
print("="*60)
print("\nFinal output files:")
print("  DNA: biolip_dna_clustered_train.csv, biolip_dna_clustered_val.csv, biolip_dna_clustered_test.csv")
print("  RNA: biolip_rna_clustered_train.csv, biolip_rna_clustered_val.csv, biolip_rna_clustered_test.csv")
print("\nUse these clustered files for training to avoid data leakage!")