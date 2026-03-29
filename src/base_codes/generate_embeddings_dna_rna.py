import pandas as pd
import torch
import esm
import numpy as np
import os

# Load ESM-2 model
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)
print(f"Model loaded on {DEVICE}")

MAX_SEQ_LENGTH = 2000

def generate_embeddings_for_csv(csv_file, output_file):
    """Generate ESM-2 embeddings for CSV files (BioLip format)"""
    
    print(f"\nProcessing: {csv_file}")
    df = pd.read_csv(csv_file)
    
    all_embeddings = []
    all_labels = []
    skipped_count = 0
    
    total = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{total} sequences")
        
        seq = row['sequence']
        ann = row['annotation']
        
        if len(seq) > MAX_SEQ_LENGTH:
            print(f"    Skipping sequence {idx} (length {len(seq)} > {MAX_SEQ_LENGTH})")
            skipped_count += 1
            continue
        
        data = [("protein", seq)]
        
        try:
            with torch.no_grad():
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(DEVICE)
                results = model(batch_tokens, repr_layers=[33])
                
                embeddings = results["representations"][33][0, 1:-1, :].cpu().numpy()
            
            for i, emb in enumerate(embeddings):
                all_embeddings.append(emb)
                all_labels.append(int(ann[i]))
            
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    OOM on sequence {idx} (length {len(seq)}), skipping...")
                skipped_count += 1
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.float32)
    
    np.savez_compressed(output_file, 
                       embeddings=embeddings_array, 
                       labels=labels_array)
    
    print(f"  Saved {len(embeddings_array):,} residue embeddings to {output_file}")
    print(f"  Embedding shape: {embeddings_array.shape}")
    if skipped_count > 0:
        print(f"    Skipped {skipped_count} sequences")

def generate_embeddings_for_tsv(tsv_file, output_file):
    """Generate ESM-2 embeddings for TSV files (DisProt format)"""
    
    print(f"\nProcessing: {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    
    all_embeddings = []
    all_labels = []
    skipped_count = 0
    
    total = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{total} sequences")
        
        seq = row['sequence']
        labels_list = [int(x) for x in row['labels'].split(',')]
        
        if len(seq) != len(labels_list):
            continue
        
        if len(seq) > MAX_SEQ_LENGTH:
            print(f"    Skipping sequence {idx} (length {len(seq)} > {MAX_SEQ_LENGTH})")
            skipped_count += 1
            continue
        
        data = [("protein", seq)]
        
        try:
            with torch.no_grad():
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(DEVICE)
                results = model(batch_tokens, repr_layers=[33])
                
                embeddings = results["representations"][33][0, 1:-1, :].cpu().numpy()
            
            for i, emb in enumerate(embeddings):
                all_embeddings.append(emb)
                all_labels.append(labels_list[i])
            
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    OOM on sequence {idx} (length {len(seq)}), skipping...")
                skipped_count += 1
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.float32)
    
    np.savez_compressed(output_file, 
                       embeddings=embeddings_array, 
                       labels=labels_array)
    
    print(f"  Saved {len(embeddings_array):,} residue embeddings to {output_file}")
    print(f"  Embedding shape: {embeddings_array.shape}")
    if skipped_count > 0:
        print(f"    Skipped {skipped_count} sequences")

def combine_embeddings(file1, file2, output_file):
    """Combine two embedding files (e.g., DNA + RNA)"""
    
    print(f"\nCombining {file1} and {file2}...")
    
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    combined_embeddings = np.concatenate([data1['embeddings'], data2['embeddings']], axis=0)
    combined_labels = np.concatenate([data1['labels'], data2['labels']], axis=0)
    
    np.savez_compressed(output_file,
                       embeddings=combined_embeddings,
                       labels=combined_labels)
    
    print(f"  Combined: {len(combined_embeddings):,} residues saved to {output_file}")

# Generate embeddings for all datasets
print("="*60)
print("Generating ESM-2 Embeddings for DNA/RNA Datasets")
print("="*60)

BASE_PATH = '/home/malekia/idp-binding-site-prediction/data/biolip/'

# BioLip DNA datasets
print("\n" + "="*60)
print("BioLip DNA Datasets")
print("="*60)
generate_embeddings_for_csv(
    BASE_PATH + 'biolip_dna_clustered_train.csv',
    'biolip_dna_train_embeddings.npz'
)
generate_embeddings_for_csv(
    BASE_PATH + 'biolip_dna_clustered_val.csv',
    'biolip_dna_val_embeddings.npz'
)
generate_embeddings_for_csv(
    BASE_PATH + 'biolip_dna_clustered_test.csv',
    'biolip_dna_test_embeddings.npz'
)

# BioLip RNA datasets
print("\n" + "="*60)
print("BioLip RNA Datasets")
print("="*60)
generate_embeddings_for_csv(
    BASE_PATH + 'biolip_rna_clustered_train.csv',
    'biolip_rna_train_embeddings.npz'
)
generate_embeddings_for_csv(
    BASE_PATH + 'biolip_rna_clustered_val.csv',
    'biolip_rna_val_embeddings.npz'
)
generate_embeddings_for_csv(
    BASE_PATH + 'biolip_rna_clustered_test.csv',
    'biolip_rna_test_embeddings.npz'
)

# DisProt DNA datasets
print("\n" + "="*60)
print("DisProt DNA Datasets")
print("="*60)
generate_embeddings_for_tsv(
    BASE_PATH + 'dna_binding_train.tsv',
    'disprot_dna_train_embeddings.npz'
)
generate_embeddings_for_tsv(
    BASE_PATH + 'dna_binding_val.tsv',
    'disprot_dna_val_embeddings.npz'
)
generate_embeddings_for_tsv(
    BASE_PATH + 'dna_binding_test.tsv',
    'disprot_dna_test_embeddings.npz'
)

# DisProt RNA datasets
print("\n" + "="*60)
print("DisProt RNA Datasets")
print("="*60)
generate_embeddings_for_tsv(
    BASE_PATH + 'rna_binding_train.tsv',
    'disprot_rna_train_embeddings.npz'
)
generate_embeddings_for_tsv(
    BASE_PATH + 'rna_binding_val.tsv',
    'disprot_rna_val_embeddings.npz'
)
generate_embeddings_for_tsv(
    BASE_PATH + 'rna_binding_test.tsv',
    'disprot_rna_test_embeddings.npz'
)

# Combine DNA + RNA for each split
print("\n" + "="*60)
print("Combining DNA and RNA Embeddings")
print("="*60)

# BioLip combined (DNA + RNA)
combine_embeddings('biolip_dna_train_embeddings.npz', 'biolip_rna_train_embeddings.npz', 'biolip_dna_rna_train_embeddings.npz')
combine_embeddings('biolip_dna_val_embeddings.npz', 'biolip_rna_val_embeddings.npz', 'biolip_dna_rna_val_embeddings.npz')
combine_embeddings('biolip_dna_test_embeddings.npz', 'biolip_rna_test_embeddings.npz', 'biolip_dna_rna_test_embeddings.npz')

# DisProt combined (DNA + RNA)
combine_embeddings('disprot_dna_train_embeddings.npz', 'disprot_rna_train_embeddings.npz', 'disprot_dna_rna_train_embeddings.npz')
combine_embeddings('disprot_dna_val_embeddings.npz', 'disprot_rna_val_embeddings.npz', 'disprot_dna_rna_val_embeddings.npz')
combine_embeddings('disprot_dna_test_embeddings.npz', 'disprot_rna_test_embeddings.npz', 'disprot_dna_rna_test_embeddings.npz')

print("\n" + "="*60)
print("All DNA/RNA embeddings generated successfully!")
print("="*60)