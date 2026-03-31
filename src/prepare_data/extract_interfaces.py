"""
Step 2: Extract Interface Residues (PARALLEL VERSION)
Uses multiple CPU cores to speed up processing
"""

import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

INTERFACE_CUTOFF = 6.0
N_WORKERS = cpu_count()  # Use all available CPU cores

# 3-letter to 1-letter amino acid code
AA_DICT = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def process_single_pdb(pdb_path):
    """Process one PDB file - returns list of results or None"""
    if not os.path.exists(pdb_path):
        return None
    
    try:
        pdb_file_obj = pdb.PDBFile.read(pdb_path)
        structure = pdb_file_obj.get_structure(model=1)
    except:
        return None
    
    structure = structure[struc.filter_amino_acids(structure)]
    
    if len(structure) == 0:
        return None
    
    chain_ids = np.unique(structure.chain_id)
    
    if len(chain_ids) < 2:
        return None
    
    basename = os.path.basename(pdb_path)
    pdb_id = basename.split('_')[0]
    
    results = []
    
    for chain_id in chain_ids:
        chain_mask = structure.chain_id == chain_id
        chain_structure = structure[chain_mask]
        
        if len(chain_structure) == 0:
            continue
        
        residue_ids, residue_names = struc.get_residues(chain_structure)
        
        sequence = ''.join([AA_DICT.get(res, 'X') for res in residue_names])
        
        if len(sequence) == 0:
            continue
        
        other_mask = structure.chain_id != chain_id
        other_structure = structure[other_mask]
        
        if len(other_structure) == 0:
            continue
        
        labels = []
        for res_id in residue_ids:
            res_mask = chain_structure.res_id == res_id
            res_atoms = chain_structure[res_mask]
            
            if len(res_atoms) == 0:
                labels.append(0)
                continue
            
            min_distance = float('inf')
            for atom in res_atoms:
                distances = np.linalg.norm(other_structure.coord - atom.coord, axis=1)
                min_dist = np.min(distances)
                if min_dist < min_distance:
                    min_distance = min_dist
            
            labels.append(1 if min_distance <= INTERFACE_CUTOFF else 0)
        
        results.append({
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'sequence': sequence,
            'annotation': ''.join(str(x) for x in labels),
            'length': len(sequence),
            'binding_sites': sum(labels)
        })
    
    return results if results else None

print("="*60)
print("Extracting Interface Residues from PPIRef (PARALLEL)")
print("="*60)

print(f"\nUsing {N_WORKERS} CPU cores")

print("\nReading PPI paths...")
with open('ppiref_paths.txt', 'r') as f:
    ppi_paths = [line.strip() for line in f]

print(f"Found {len(ppi_paths):,} PPIs to process")

print("\nProcessing PDB files in parallel...")

all_data = []
failed = 0

with Pool(N_WORKERS) as pool:
    for result in tqdm(pool.imap_unordered(process_single_pdb, ppi_paths), total=len(ppi_paths)):
        if result is None:
            failed += 1
        else:
            all_data.extend(result)

print(f"\nProcessed: {len(ppi_paths):,} PDB files")
print(f"Failed: {failed:,}")
print(f"Extracted: {len(all_data):,} protein chains")

df = pd.DataFrame(all_data)
df.to_csv('ppiref_all_data.csv', index=False)

print(f"\nSaved to: ppiref_all_data.csv")

total_residues = df['length'].sum()
total_binding = df['binding_sites'].sum()
print(f"\nStatistics:")
print(f"  Total proteins: {len(df):,}")
print(f"  Total residues: {total_residues:,}")
print(f"  Binding residues: {total_binding:,}")
print(f"  Binding percentage: {100*total_binding/total_residues:.2f}%")

print("\n" + "="*60)
print("Extraction complete!")
print("="*60)