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
    """Generate ESM-2 embeddings for CSV files (AHoJ-DB format)"""
    
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

# Generate embeddings for all datasets
print("="*60)
print("Generating ESM-2 Embeddings for Ion Binding Datasets")
print("="*60)

BASE_PATH = '/home/malekia/idp-binding-site-prediction/data/'

# AHoJ-DB datasets
print("\n" + "="*60)
print("AHoJ-DB (Structured Ion Binding)")
print("="*60)
generate_embeddings_for_csv(
    BASE_PATH + 'train_data.csv',
    'ahojdb_train_embeddings.npz'
)
generate_embeddings_for_csv(
    BASE_PATH + 'val_data.csv',
    'ahojdb_val_embeddings.npz'
)
generate_embeddings_for_csv(
    BASE_PATH + 'test_data.csv',
    'ahojdb_test_embeddings.npz'
)

# DisProt ion binding datasets
print("\n" + "="*60)
print("DisProt (IDP Ion Binding)")
print("="*60)
generate_embeddings_for_tsv(
    BASE_PATH + 'ion_binding_train.tsv',
    'disprot_ion_train_embeddings.npz'
)
generate_embeddings_for_tsv(
    BASE_PATH + 'ion_binding_val.tsv',
    'disprot_ion_val_embeddings.npz'
)
generate_embeddings_for_tsv(
    BASE_PATH + 'ion_binding_test.tsv',
    'disprot_ion_test_embeddings.npz'
)

print("\n" + "="*60)
print("All ion binding embeddings generated successfully!")
print("="*60)