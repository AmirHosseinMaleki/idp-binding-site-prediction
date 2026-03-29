#!/usr/bin/env python3
"""
Cluster sequences with MMseqs2 to prevent information leakage
"""

import pandas as pd
import subprocess
import os
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

def create_fasta_file(csv_path='final_complete_ion_dataset.csv', 
                      output_path='ion_sequences.fasta'):
    """
    Create FASTA file from the ion binding dataset
    """
    print("Creating FASTA file from dataset...")
    
    df = pd.read_csv(csv_path)
    
    # Remove duplicates based on sequence
    df_unique = df.drop_duplicates(subset=['sequence'])
    
    with open(output_path, 'w') as f:
        for idx, row in df_unique.iterrows():
            # Create unique ID from PDB and chain
            protein_id = f"{row['pdb_id']}_{row['chain_id']}"
            sequence = row['sequence']
            
            # Write FASTA format
            f.write(f">{protein_id}\n")
            f.write(f"{sequence}\n")
    
    print(f"Created FASTA file with {len(df_unique)} unique sequences")
    print(f"Saved to: {output_path}")
    
    return len(df_unique)


def run_mmseqs_clustering(fasta_file='ion_sequences.fasta',
                         min_seq_identity=0.1,
                         threads=4):
    """
    Run MMseqs2 clustering to group similar sequences
    min_seq_identity=0.1 means sequences with >10% identity will be clustered together
    """
    print("\n" + "="*60)
    print("Running MMseqs2 Clustering")
    print("="*60)
    print(f"Min sequence identity: {min_seq_identity}")
    print(f"Threads: {threads}")
    
    output_dir = Path("mmseqs_output")
    output_dir.mkdir(exist_ok=True)
    
    cluster_result = output_dir / "clusterRes"
    tmp_dir = output_dir / "tmp"
    cluster_tsv = output_dir / "clusters.tsv"
    
    tmp_dir.mkdir(exist_ok=True)
    
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta_file),
        str(cluster_result),
        str(tmp_dir),
        "--min-seq-id", str(min_seq_identity),
        "--threads", str(threads)
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("MMseqs2 clustering completed successfully!")
        
        cluster_file = Path(str(cluster_result) + "_cluster.tsv")
        
        if cluster_file.exists():
            subprocess.run(["cp", str(cluster_file), str(cluster_tsv)])
            print(f"Cluster results saved to: {cluster_tsv}")
            return cluster_tsv
        else:
            print(f"Warning: Cluster file not found at {cluster_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running MMseqs2: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        
        # Check if MMseqs2 is installed
        print("\nChecking if MMseqs2 is installed...")
        try:
            subprocess.run(["mmseqs", "--version"], capture_output=True, check=True)
            print("MMseqs2 is installed")
        except:
            print("ERROR: MMseqs2 is not installed!")
            print("Install it with: conda install -c bioconda mmseqs2")
            print("Or download from: https://github.com/soedinglab/MMseqs2")
        
        return None


def parse_clusters(cluster_file='mmseqs_output/clusters.tsv'):
    """
    Parse MMseqs2 cluster output into a dictionary
    """
    print("\n" + "="*60)
    print("Parsing Cluster Results")
    print("="*60)
    
    clusters = defaultdict(list)
    
    with open(cluster_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                representative = parts[0]
                member = parts[1]
                clusters[representative].append(member)
    
    print(f"Total clusters: {len(clusters)}")
    
    cluster_sizes = [len(members) for members in clusters.values()]
    
    print(f"\nCluster size statistics:")
    print(f"  Min size: {min(cluster_sizes)}")
    print(f"  Max size: {max(cluster_sizes)}")
    print(f"  Mean size: {np.mean(cluster_sizes):.2f}")
    print(f"  Median size: {np.median(cluster_sizes):.2f}")
    
    singletons = sum(1 for size in cluster_sizes if size == 1)
    print(f"  Singleton clusters: {singletons} ({singletons/len(clusters)*100:.1f}%)")
    
    return dict(clusters)


def create_cluster_aware_splits(csv_path='final_complete_ion_dataset.csv',
                               cluster_file='mmseqs_output/clusters.tsv',
                               train_ratio=0.7,
                               val_ratio=0.15,
                               test_ratio=0.15):
    """
    Create train/val/test splits based on clusters to prevent information leakage
    """
    print("\n" + "="*60)
    print("Creating Cluster-Aware Data Splits")
    print("="*60)
    
    # Load original dataset
    df = pd.read_csv(csv_path)
    
    # Add protein ID column
    df['protein_id'] = df['pdb_id'] + '_' + df['chain_id']
    
    # Parse clusters
    clusters = parse_clusters(cluster_file)
    
    # Create mapping from protein to cluster representative
    protein_to_cluster = {}
    for rep, members in clusters.items():
        for member in members:
            protein_to_cluster[member] = rep
    
    # Add cluster information to dataframe
    df['cluster'] = df['protein_id'].map(protein_to_cluster)
    
    # Check for proteins without cluster assignment
    missing_clusters = df['cluster'].isna().sum()
    if missing_clusters > 0:
        print(f"Warning: {missing_clusters} proteins without cluster assignment")
        # Assign them to their own cluster
        df.loc[df['cluster'].isna(), 'cluster'] = df.loc[df['cluster'].isna(), 'protein_id']
    
    unique_clusters = df['cluster'].unique()
    print(f"\nTotal unique clusters in dataset: {len(unique_clusters)}")
    
    train_clusters, temp_clusters = train_test_split(
        unique_clusters, 
        test_size=(val_ratio + test_ratio),
        random_state=42
    )
    
    val_clusters, test_clusters = train_test_split(
        temp_clusters,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=42
    )
    
    train_df = df[df['cluster'].isin(train_clusters)]
    val_df = df[df['cluster'].isin(val_clusters)]
    test_df = df[df['cluster'].isin(test_clusters)]
    
    print(f"\nCluster-based split:")
    print(f"  Train clusters: {len(train_clusters)}")
    print(f"  Val clusters: {len(val_clusters)}")
    print(f"  Test clusters: {len(test_clusters)}")
    
    print(f"\nProtein distribution:")
    print(f"  Train: {len(train_df)} proteins ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} proteins ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} proteins ({len(test_df)/len(df)*100:.1f}%)")
    
    # Check class balance in each split
    print(f"\nClass balance (% positive):")
    print(f"  Overall: {df['binding_sites'].sum() / df['length'].sum() * 100:.2f}%")
    print(f"  Train: {train_df['binding_sites'].sum() / train_df['length'].sum() * 100:.2f}%")
    print(f"  Val: {val_df['binding_sites'].sum() / val_df['length'].sum() * 100:.2f}%")
    print(f"  Test: {test_df['binding_sites'].sum() / test_df['length'].sum() * 100:.2f}%")
    
    train_df = train_df.drop(columns=['cluster', 'protein_id'])
    val_df = val_df.drop(columns=['cluster', 'protein_id'])
    test_df = test_df.drop(columns=['cluster', 'protein_id'])
    
    train_df.to_csv('train_data_clustered.csv', index=False)
    val_df.to_csv('val_data_clustered.csv', index=False)
    test_df.to_csv('test_data_clustered.csv', index=False)
    
    print(f"\nSaved cluster-aware splits:")
    print(f"  train_data_clustered.csv")
    print(f"  val_data_clustered.csv")
    print(f"  test_data_clustered.csv")
    
    cluster_info = {
        'train_clusters': list(train_clusters),
        'val_clusters': list(val_clusters),
        'test_clusters': list(test_clusters),
        'total_clusters': len(unique_clusters),
        'cluster_mapping': {k: list(v) for k, v in clusters.items()}
    }
    
    with open('cluster_splits.json', 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    print(f"  cluster_splits.json (cluster assignments)")
    
    return train_df, val_df, test_df


def verify_no_leakage(cluster_file='cluster_splits.json'):
    """
    Verify that there's no sequence overlap between splits
    """
    print("\n" + "="*60)
    print("Verifying No Information Leakage")
    print("="*60)
    
    with open(cluster_file, 'r') as f:
        cluster_info = json.load(f)
    
    train_clusters = set(cluster_info['train_clusters'])
    val_clusters = set(cluster_info['val_clusters'])
    test_clusters = set(cluster_info['test_clusters'])
    
    # Check for overlaps
    train_val_overlap = train_clusters.intersection(val_clusters)
    train_test_overlap = train_clusters.intersection(test_clusters)
    val_test_overlap = val_clusters.intersection(test_clusters)
    
    print(f"Cluster overlaps:")
    print(f"  Train-Val overlap: {len(train_val_overlap)} clusters")
    print(f"  Train-Test overlap: {len(train_test_overlap)} clusters")
    print(f"  Val-Test overlap: {len(val_test_overlap)} clusters")
    
    if any([train_val_overlap, train_test_overlap, val_test_overlap]):
        print("\n WARNING: Found cluster overlaps! Information leakage detected!")
        return False
    else:
        print("\n No cluster overlaps found. No information leakage!")
        return True


def main():
    """
    Complete pipeline for cluster-aware data splitting
    """
    print("="*60)
    print("CLUSTER-AWARE DATA SPLITTING PIPELINE")
    print("="*60)
    
    num_sequences = create_fasta_file()
    
    cluster_file = run_mmseqs_clustering(
        fasta_file='ion_sequences.fasta',
        min_seq_identity=0.1,
        threads=4
    )
    
    if cluster_file is None:
        print("\nERROR: MMseqs2 clustering failed!")
        print("Please install MMseqs2 and try again:")
        return
    
    train_df, val_df, test_df = create_cluster_aware_splits(
        csv_path='final_complete_ion_dataset.csv',
        cluster_file=cluster_file
    )
    
    no_leakage = verify_no_leakage()
    
    if no_leakage:
        print("\n" + "="*60)
        print("Cluster-aware splits created")
        print("="*60)
    else:
        print("\nERROR: Information leakage detected!")
        print("Please check the splitting logic")


if __name__ == "__main__":
    main()