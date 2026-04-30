#!/usr/bin/env python3

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import get_structure
from biotite.structure import get_residues
import biotite.structure

# Amino acid mapping
mapping = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 
    'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 
    'Tyr': 'Y', 'Val': 'V'
}

def three_to_one(three_letter):
    return mapping.get(three_letter[0].upper() + three_letter[1:].lower(), 'X')

def process_single_protein(entry_data):
    """Process a single protein - to run in parallel"""
    dir_name, pdb_id, chain_id, ligand, binding_residues = entry_data
    
    try:
        # Create worker-specific CIF directory
        worker_id = mp.current_process().name
        cif_dir = f"cif_files_{worker_id}"
        os.makedirs(cif_dir, exist_ok=True)
        
        # Download structure
        cif_path = rcsb.fetch(pdb_id, "cif", target_path=cif_dir)
        cif_file = pdbx.CIFFile.read(cif_path)
        atoms = get_structure(cif_file, model=1)
        
        # Filter for this chain
        chain_atoms = atoms[
            (atoms.chain_id == chain_id) & 
            (biotite.structure.filter_peptide_backbone(atoms))
        ]
        
        if len(chain_atoms) == 0:
            return None
        
        # Extract sequence
        residues = get_residues(chain_atoms)
        residue_indices, residue_names = residues
        sequence = ''.join([three_to_one(res) for res in residue_names])
        
        # Create binding annotation
        binding_indices = []
        for res in binding_residues:
            try:
                res_chain, res_num = res.split('_')
                if res_chain == chain_id:
                    binding_indices.append(int(res_num))
            except:
                continue
        
        annotation = [1 if idx in binding_indices else 0 for idx in residue_indices]
        
        result = {
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'ligand': ligand,
            'sequence': sequence,
            'annotation': ''.join(map(str, annotation)),
            'length': len(sequence),
            'binding_sites': sum(annotation)
        }
        
        return result
        
    except Exception as e:
        return None

def load_existing_results():
    """Load results from previous runs"""
    dataset = []
    processed_proteins = set()
    
    # Look for existing result files
    for filename in os.listdir('.'):
        if filename.startswith('progress_') and filename.endswith('.csv'):
            print(f"Found existing progress file: {filename}")
            
            with open(filename, 'r') as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split(',', 6)
                    if len(parts) >= 7:
                        pdb_id, chain_id, ligand, sequence, annotation, length, binding_sites = parts
                        
                        dataset.append({
                            'pdb_id': pdb_id,
                            'chain_id': chain_id,
                            'ligand': ligand,
                            'sequence': sequence,
                            'annotation': annotation,
                            'length': int(length),
                            'binding_sites': int(binding_sites)
                        })
                        
                        # Track processed proteins
                        processed_proteins.add(f"{pdb_id}_{chain_id}")
    
    return dataset, processed_proteins

def main():
    all_entries = []
    with open('all_binding_sites.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(',', 4)
            if len(parts) >= 5:
                dir_name, pdb_id, chain_id, ligand, binding_residues = parts
                all_entries.append((dir_name, pdb_id, chain_id, ligand, binding_residues.split()))

    print(f"Total entries to process: {len(all_entries)}")
    
    existing_dataset, processed_proteins = load_existing_results()
    
    if existing_dataset:
        print(f"Resuming: Found {len(existing_dataset)} already processed proteins")
        
        # Filter out already processed entries
        remaining_entries = []
        for entry in all_entries:
            dir_name, pdb_id, chain_id, ligand, binding_residues = entry
            protein_key = f"{pdb_id}_{chain_id}"
            if protein_key not in processed_proteins:
                remaining_entries.append(entry)
        
        entries = remaining_entries
        dataset = existing_dataset
        print(f"Remaining entries to process: {len(entries)}")
    else:
        print("Starting fresh")
        entries = all_entries
        dataset = []
    
    if not entries:
        print("All entries already processed!")
        return
    
    num_cores = min(mp.cpu_count(), 8)
    print(f"Using {num_cores} CPU cores")
    
    os.makedirs("cif_files", exist_ok=True)
    
    batch_size = 100
    
    for batch_start in range(0, len(entries), batch_size):
        batch_end = min(batch_start + batch_size, len(entries))
        batch = entries[batch_start:batch_end]
        
        print(f"Processing batch: entries {batch_start+1}-{batch_end} (total processed so far: {len(dataset)})")
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(process_single_protein, entry): entry for entry in batch}
            
            batch_results = []
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    batch_results.append(result)
            
            dataset.extend(batch_results)
            print(f"  Batch completed: {len(batch_results)}/{len(batch)} successful")
        
        progress_file = f'progress_{len(dataset)}_proteins.csv'
        with open(progress_file, 'w') as f:
            f.write('pdb_id,chain_id,ligand,sequence,annotation,length,binding_sites\n')
            for entry in dataset:
                f.write(f"{entry['pdb_id']},{entry['chain_id']},{entry['ligand']},{entry['sequence']},{entry['annotation']},{entry['length']},{entry['binding_sites']}\n")
        
        print(f"  Progress saved to: {progress_file}")
    
    print(f"Successfully processed: {len(dataset)} proteins")
    
    if dataset:
        with open('final_complete_ion_dataset.csv', 'w') as f:
            f.write('pdb_id,chain_id,ligand,sequence,annotation,length,binding_sites\n')
            for entry in dataset:
                f.write(f"{entry['pdb_id']},{entry['chain_id']},{entry['ligand']},{entry['sequence']},{entry['annotation']},{entry['length']},{entry['binding_sites']}\n")
        
        total_residues = sum(entry['length'] for entry in dataset)
        total_binding_sites = sum(entry['binding_sites'] for entry in dataset)
        
        print(f"Final dataset saved to: final_complete_ion_dataset.csv")
        print(f"Total proteins: {len(dataset)}")
        print(f"Total residues: {total_residues}")
        print(f"Total binding sites: {total_binding_sites}")

if __name__ == "__main__":
    main()