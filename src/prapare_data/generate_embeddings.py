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

# ADD THIS: Maximum sequence length to prevent OOM
MAX_SEQ_LENGTH = 2000  # Skip sequences longer than this

def generate_embeddings_for_csv(csv_file, output_file):
    """Generate ESM-2 embeddings for all sequences in a CSV file"""
    
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
        
        # ← SKIP VERY LONG SEQUENCES
        if len(seq) > MAX_SEQ_LENGTH:
            print(f"  Skipping sequence {idx} (length {len(seq)} > {MAX_SEQ_LENGTH})")
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
            
            # ← CLEAR GPU CACHE AFTER EACH SEQUENCE
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM on sequence {idx} (length {len(seq)}), skipping...")
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
    
    print(f"Saved {len(embeddings_array):,} residue embeddings to {output_file}")
    print(f"  Embedding shape: {embeddings_array.shape}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} sequences (too long or OOM)")

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
        
        # SKIP VERY LONG SEQUENCES
        if len(seq) > MAX_SEQ_LENGTH:
            print(f"Skipping sequence {idx} (length {len(seq)} > {MAX_SEQ_LENGTH})")
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
            
            # CLEAR GPU CACHE AFTER EACH SEQUENCE
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM on sequence {idx} (length {len(seq)}), skipping...")
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
        print(f"Skipped {skipped_count} sequences (too long or OOM)")

# Generate embeddings for all datasets
print("="*60)
print("Generating ESM-2 Embeddings for All Datasets")
print("="*60)

# ScanNet datasets (already done, but keeping for completeness)
generate_embeddings_for_csv(
    '/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/scannet_train_clustered.csv',
    'scannet_train_embeddings.npz'
)

generate_embeddings_for_csv(
    '/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/scannet_val_clustered.csv',
    'scannet_val_embeddings.npz'
)

generate_embeddings_for_csv(
    '/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/scannet_test_clustered.csv',
    'scannet_test_embeddings.npz'
)

# DisProt datasets
generate_embeddings_for_tsv(
    '/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/protein_binding_train.tsv',
    'disprot_train_embeddings.npz'
)

generate_embeddings_for_tsv(
    '/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/protein_binding_val.tsv',
    'disprot_val_embeddings.npz'
)

generate_embeddings_for_tsv(
    '/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/protein_binding_test.tsv',
    'disprot_test_embeddings.npz'
)

print("\n" + "="*60)
print("All embeddings generated successfully!")
print("="*60)