"""
Download UniProt sequences organized by binding category
"""

import pandas as pd
import requests
import os
from pathlib import Path

def download_uniprot_sequence(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text.strip() if response.text.strip() else None
    except requests.exceptions.RequestException:
        return None

def main():
    categories = {
        'protein_binding': 'data/protein_binding_filtered.tsv',
        'dna_binding': 'data/dna_binding_filtered.tsv',
        'rna_binding': 'data/rna_binding_filtered.tsv',
        'ion_binding': 'data/ion_binding_filtered.tsv'
    }
    
    base_dir = Path("data/uniprot_sequences")
    category_proteins = {}
    all_unique_proteins = set()
    
    for category, filename in categories.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename, sep='\t')
            unique_ids = set(df['acc'].unique())
            category_proteins[category] = unique_ids
            all_unique_proteins.update(unique_ids)
            
            # Create category directory
            (base_dir / category).mkdir(parents=True, exist_ok=True)
            print(f"{category}: {len(unique_ids)} unique proteins")
        else:
            category_proteins[category] = set()
    
    print(f"Total unique proteins: {len(all_unique_proteins)}")
    
    # Download sequences by category
    total_successful = 0
    total_failed = 0
    category_results = {}
    
    for category, protein_ids in category_proteins.items():
        if not protein_ids:
            continue
            
        category_dir = base_dir / category
        successful = {}
        failed = []
        
        for i, protein_id in enumerate(sorted(protein_ids), 1):
            output_file = category_dir / f"{protein_id}.fasta"
            
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        content = f.read()
                        seq_lines = content.strip().split('\n')[1:]
                        successful[protein_id] = len(''.join(seq_lines))
                    continue
                except:
                    pass
            
            sequence = download_uniprot_sequence(protein_id)
            if sequence:
                with open(output_file, 'w') as f:
                    f.write(sequence)
                seq_lines = sequence.split('\n')[1:]
                successful[protein_id] = len(''.join(seq_lines))
            else:
                failed.append(protein_id)
        
        category_results[category] = {
            'successful': len(successful),
            'failed': len(failed),
            'total': len(protein_ids)
        }
        
        total_successful += len(successful)
        total_failed += len(failed)
    
    # Create protein-category mapping
    overlap_mapping = {}
    for protein_id in all_unique_proteins:
        categories_for_protein = [cat for cat, proteins in category_proteins.items() if protein_id in proteins]
        overlap_mapping[protein_id] = categories_for_protein
    
    mapping_data = [{
        'protein_id': protein_id,
        'categories': ','.join(cats),
        'category_count': len(cats)
    } for protein_id, cats in overlap_mapping.items()]
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_file = base_dir / "protein_category_mapping.tsv"
    mapping_df.to_csv(mapping_file, sep='\t', index=False)
    
    print(f"\nTotal successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Multi-category proteins: {len(mapping_df[mapping_df['category_count'] > 1])}")

if __name__ == "__main__":
    main()