"""
Step 2: Process BioLiP to Extract DNA and RNA Binding Sites
Reads the annotation file and FASTA files to create labeled datasets
"""

import pandas as pd
import biotite.sequence.io.fasta as fasta
from collections import defaultdict

print("="*60)
print("Processing BioLiP for DNA/RNA Binding Sites")
print("="*60)

# Read receptor sequences using Biotite
print("\nReading receptor sequences...")
receptor_seqs = {}
fasta_file = fasta.FastaFile.read('protein_nr.fasta')
for header, sequence_string in fasta_file.items():
    # FASTA header format: >207lA (PDB_ID + CHAIN, no separator)
    receptor_seqs[header] = sequence_string

print(f"Loaded {len(receptor_seqs):,} receptor sequences")

# Read annotation file to get binding sites
print("\nReading BioLiP annotation file...")

# Column names from README
columns = [
    'pdb_id', 'chain', 'resolution', 'bs_code', 'ligand_id',
    'ligand_chain', 'ligand_serial', 'bs_residues_pdb', 'bs_residues_renum',
    'cat_site_pdb', 'cat_site_renum', 'ec_number', 'go_terms',
    'affinity_manual', 'affinity_moad', 'affinity_pdbind', 'affinity_bindingdb',
    'uniprot_id', 'pubmed_id', 'ligand_auth_seq', 'sequence'
]

df = pd.read_csv('BioLiP_nr.txt', sep='\t', header=None, names=columns)
print(f"Total entries: {len(df):,}")

def process_binding_data(ligand_type, df_all, name):
    """Process binding site data for DNA or RNA"""
    print(f"\n{'-'*60}")
    print(f"Processing {name} Binding Data")
    print(f"{'-'*60}")
    
    results = []
    binding_sites_dict = defaultdict(set)
    
    # Determine ligand IDs based on type
    if ligand_type == 'DNA':
        # DNA ligands appear as "dna" in the ligand_id column
        target_ligands = {'dna'}
    else:  # RNA
        # RNA ligands appear as "rna" in the ligand_id column
        target_ligands = {'rna'}
    
    # First pass: collect all binding sites for each protein
    for _, row in df_all.iterrows():
        pdb_id = str(row['pdb_id']).lower()
        chain = str(row['chain'])
        ligand_id = str(row['ligand_id']).lower()
        
        # Skip if not our target ligand type
        if ligand_id not in target_ligands:
            continue
        
        # Create protein key matching FASTA format: pdb_id + chain (e.g., "10mhA")
        protein_key = f"{pdb_id}{chain}"
        
        bs_residues = row['bs_residues_renum']
        
        if pd.isna(bs_residues):
            continue
        
        # Parse binding site residues (renumbered from 1)
        # Format: "N1 L2 A3 V15 H18..."
        try:
            for residue in str(bs_residues).split():
                # Extract position number (e.g., "N1" -> 1)
                # Find first digit
                for i, char in enumerate(residue):
                    if char.isdigit():
                        pos = int(residue[i:])
                        binding_sites_dict[protein_key].add(pos)
                        break
        except Exception as e:
            continue
    
    print(f"Found binding sites for {len(binding_sites_dict):,} proteins")
    
    # Second pass: create annotations
    for protein_key, bs_positions in binding_sites_dict.items():
        if protein_key not in receptor_seqs:
            continue
        
        sequence = receptor_seqs[protein_key]
        
        # Create binary annotation
        annotation = ''
        for i in range(1, len(sequence) + 1):
            annotation += '1' if i in bs_positions else '0'
        
        if len(annotation) != len(sequence):
            continue
        
        binding_sites = annotation.count('1')
        
        if binding_sites == 0:
            continue
        
        # Extract PDB ID and chain from protein_key
        # protein_key format: "10mhA" -> pdb_id="10mh", chain="A"
        pdb_id = protein_key[:-1]
        chain_id = protein_key[-1]
        
        results.append({
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'sequence': sequence,
            'annotation': annotation,
            'length': len(sequence),
            'binding_sites': binding_sites
        })
    
    df_result = pd.DataFrame(results)
    
    # Statistics
    total_residues = df_result['length'].sum()
    total_binding = df_result['binding_sites'].sum()
    
    print(f"\nResults:")
    print(f"  Proteins: {len(df_result):,}")
    print(f"  Total residues: {total_residues:,}")
    print(f"  Binding residues: {total_binding:,}")
    print(f"  Binding percentage: {100*total_binding/total_residues:.2f}%")
    
    return df_result

# Process DNA
df_dna = process_binding_data('DNA', df, "DNA")
df_dna.to_csv('biolip_dna_all.csv', index=False)
print(f"\nSaved: biolip_dna_all.csv")

# Process RNA
df_rna = process_binding_data('RNA', df, "RNA")
df_rna.to_csv('biolip_rna_all.csv', index=False)
print(f"Saved: biolip_rna_all.csv")

print("\n" + "="*60)
print("Processing Complete!")
print("="*60)
print("\nNext steps:")
print("1. Apply CAID3 filtering (if needed)")
print("2. Create train/val/test splits (70/15/15)")