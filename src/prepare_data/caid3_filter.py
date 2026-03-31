#!/usr/bin/env python3
"""
Remove training sequences similar to CAID3 test set
"""

import subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict

def create_disprot_combined_fasta():
    """
    Create a single FASTA file from all DisProt sequences
    """
    print("Creating combined DisProt FASTA...")
    
    all_sequences = {}
    
    categories = ['protein_binding', 'dna_binding', 'rna_binding', 'ion_binding']
    
    for category in categories:
        folder = Path(f'data/uniprot_sequences/{category}')
        if not folder.exists():
            print(f"  Warning: {folder} not found")
            continue
            
        # Read all FASTA files in this category
        for fasta_file in folder.glob('*.fasta'):
            with open(fasta_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    protein_id = fasta_file.stem
                    header = lines[0].strip()
                    sequence = ''.join(lines[1:]).replace('\n', '')
                    
                    if protein_id not in all_sequences:
                        all_sequences[protein_id] = (header, sequence)
    
    with open('disprot_all_sequences.fasta', 'w') as f:
        for protein_id, (header, sequence) in all_sequences.items():
            if not header.startswith('>'):
                header = f'>{protein_id}'
            f.write(f'{header}\n{sequence}\n')
    
    return 'disprot_all_sequences.fasta'

def run_clustering_with_caid3(training_fasta, caid3_fasta, output_prefix):
    """
    Cluster training sequences with CAID3 to find similar ones
    """
    
    combined = f'{output_prefix}_combined.fasta'
    with open(combined, 'w') as outf:
        # Add training sequences
        with open(training_fasta, 'r') as inf:
            outf.write(inf.read())
        # Add CAID3 sequences
        with open(caid3_fasta, 'r') as inf:
            outf.write(inf.read())
    
    n_training = sum(1 for line in open(training_fasta) if line.startswith('>'))
    n_caid3 = sum(1 for line in open(caid3_fasta) if line.startswith('>'))
    print(f"  Training sequences: {n_training}")
    print(f"  CAID3 sequences: {n_caid3}")
    
    output_dir = Path(f'{output_prefix}_clustering')
    output_dir.mkdir(exist_ok=True)
    
    cmd = [
        'mmseqs', 'easy-cluster',
        combined,
        str(output_dir / 'clusters'),
        str(output_dir / 'tmp'),
        '--min-seq-id', '0.1',
        '--threads', '4'
    ]
    
    print(f"  Running MMseqs2...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Clustering complete!")
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return None
    
    return str(output_dir / 'clusters_cluster.tsv')

def find_sequences_to_remove(cluster_file, caid3_fasta):
    """
    Find training sequences that cluster with CAID3
    """
    caid3_ids = set()
    with open(caid3_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Extract ID from header
                seq_id = line[1:].strip().split()[0]
                # Handle different header formats
                if '|' in seq_id:
                    seq_id = seq_id.split('|')[1]
                caid3_ids.add(seq_id)
    
    
    clusters = defaultdict(set)
    with open(cluster_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep = parts[0].split()[0]  # Just the ID part
                member = parts[1].split()[0]
                clusters[rep].add(member)
    
    sequences_to_remove = set()
    contaminated_clusters = 0
    
    for rep, members in clusters.items():
        all_members = members | {rep}
        
        if any(caid_id in member or member in caid_id for member in all_members for caid_id in caid3_ids):
            contaminated_clusters += 1
            for member in all_members:
                if not any(caid_id in member or member in caid_id for caid_id in caid3_ids):
                    sequences_to_remove.add(member)
    
    
    return sequences_to_remove

def filter_datasets(sequences_to_remove):
    """
    Remove similar sequences from all datasets
    """
    
    categories = ['protein_binding', 'dna_binding', 'rna_binding', 'ion_binding']
    
    for category in categories:
        tsv_file = f'data/training/{category}_training_data.tsv'
        if not Path(tsv_file).exists():
            continue
            
        df = pd.read_csv(tsv_file, sep='\t')
        initial_size = len(df)
        
        df_filtered = df[~df['protein_id'].isin(sequences_to_remove)]
        
        output_file = f'data/training/{category}_filtered_caid3.tsv'
        df_filtered.to_csv(output_file, sep='\t', index=False)
        
        print(f"  {category}: {initial_size} → {len(df_filtered)} proteins ({len(df_filtered)/initial_size*100:.1f}% retained)")
    
    if Path('train_data_clustered.csv').exists():
        df = pd.read_csv('train_data_clustered.csv')
        initial_size = len(df)
        
        df['protein_id'] = df['pdb_id'] + '_' + df['chain_id']
        
        df_filtered = df[~df['protein_id'].isin(sequences_to_remove)]
        df_filtered = df_filtered.drop('protein_id', axis=1)
        
        df_filtered.to_csv('train_data_filtered_caid3.csv', index=False)
        print(f"  Ion binding (AHoJ-DB): {initial_size} → {len(df_filtered)} proteins ({len(df_filtered)/initial_size*100:.1f}% retained)")

def main():
    print("="*60)
    print("CAID3 FILTERING PIPELINE")
    print("="*60)
    
    caid3_files = ['binding.fasta', 'binding-idr.fasta']
    caid3_file = None
    
    for f in caid3_files:
        if Path(f).exists():
            caid3_file = f
            print(f"Found CAID3 test set: {f}")
            break
    
    if not caid3_file:
        print("ERROR: No CAID3 test file found!")
        print("Need either binding.fasta or binding-idr.fasta")
        return
    
    all_sequences_to_remove = set()
    
    disprot_fasta = create_disprot_combined_fasta()
    
    cluster_file = run_clustering_with_caid3(
        disprot_fasta,
        caid3_file,
        'disprot_caid3'
    )
    
    if cluster_file:
        to_remove = find_sequences_to_remove(cluster_file, caid3_file)
        all_sequences_to_remove.update(to_remove)
    
    if Path('ion_sequences.fasta').exists():
        cluster_file = run_clustering_with_caid3(
            'ion_sequences.fasta',
            caid3_file,
            'ion_caid3'
        )
        
        if cluster_file:
            to_remove = find_sequences_to_remove(cluster_file, caid3_file)
            all_sequences_to_remove.update(to_remove)
    
    print(f"\nTotal sequences to remove: {len(all_sequences_to_remove)}")
    
    with open('caid3_removed_sequences.txt', 'w') as f:
        for seq_id in sorted(all_sequences_to_remove):
            f.write(f"{seq_id}\n")
    
    filter_datasets(all_sequences_to_remove)

if __name__ == "__main__":
    main()