#!/usr/bin/env python3

import os

with open('all_ion_directories.txt', 'r') as f:
    ion_dirs = [line.strip() for line in f]

print(f"Processing ALL {len(ion_dirs)} ion directories:")

results = []

for i, dir_name in enumerate(ion_dirs):
    if (i + 1) % 100 == 0:  # Progress update every 100
        print(f"Progress: {i+1}/{len(ion_dirs)}")
    
    parts = dir_name.split('-')
    pdb_id = parts[0]
    chain_id = parts[1] 
    ligand = parts[2]
    
    found = False
    subdirs = os.listdir('data')
    
    for subdir in subdirs:
        dir_path = f'data/{subdir}/{dir_name}'
        pocket_file = f'{dir_path}/pocket_residues.csv'
        
        if os.path.exists(pocket_file):
            try:
                with open(pocket_file, 'r') as f:
                    header = f.readline().strip()
                    content = f.readline().strip()
                
                if content:
                    parts = content.split(',', 2)
                    if len(parts) >= 3:
                        binding_residues = parts[2].split()
                        
                        results.append({
                            'dir_name': dir_name,
                            'pdb_id': pdb_id,
                            'chain_id': chain_id,
                            'ligand': ligand,
                            'binding_residues': binding_residues,
                            'num_sites': len(binding_residues)
                        })
                        
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break

print(f"\nSuccessfully processed: {len(results)} directories")

with open('all_binding_sites.txt', 'w') as f:
    for result in results:
        f.write(f"{result['dir_name']},{result['pdb_id']},{result['chain_id']},{result['ligand']},{' '.join(result['binding_residues'])}\n")

print(f"Saved binding site data to all_binding_sites.txt")