#!/usr/bin/env python3
"""
Search directly for all binding terms and organize by category
Terms are from Table S1
(https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-023-01803-y/MediaObjects/12915_2023_1803_MOESM1_ESM.pdf)
(excluding Lipid binding and Flexible linker)
"""

import pandas as pd
import requests
import time
import os
from pathlib import Path

def search_disprot_term(term_name, release="2024_12"):
    go_term = term_name.lower()
    url = f"https://disprot.org/api/search?go_name={go_term.replace(' ', '%20')}&format=tsv&release={release}&page=0&sort_field=disprot_id&sort_value=asc"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        if response.text.strip():
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), sep='\t')
            return df
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

def main():
    terms = {
        'protein_binding': [
            "protein binding", "P53 binding", "TBP-class protein binding", 
            "Beta-catenin binding", "Calmodulin binding", "Transcription coactivator binding",
            "Growth factor binding", "Histone binding", "SH3 domain binding",
            "Ubiquitin protein ligase binding", "Platelet-derived growth factor receptor binding",
            "Protein kinase binding", "MHC class I protein binding", "MDM2/MDM4 family protein binding",
            "14-3-3 protein binding", "RNA polymerase binding", "Importin-alpha family protein binding"
        ],
        
        'dna_binding': [
            "DNA binding, bending", "DNA binding", "Single-stranded DNA binding"
        ],
        
        'rna_binding': [
            "RNA binding", "Single-stranded RNA binding", "G-quadruplex RNA binding",
            "Regulatory region RNA binding", "RNA stem-loop binding", "rRNA binding",
            "mRNA binding", "tRNA binding"
        ],
        
        'ion_binding': [
            "Ion binding", "Iron ion binding", "Zinc ion binding", "Copper ion binding",
            "Potassium ion binding", "Metal ion binding", "Calcium ion binding"
        ]
    }
    
    base_dir = Path("data")
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    
    all_results = []
    found_terms = []
    missing_terms = []
    
    total_terms = sum(len(term_list) for term_list in terms.values())
    current_term = 0
    
    for category, term_list in terms.items():
        print(f"\nSearching {category} terms:")
        
        for term in term_list:
            current_term += 1
            print(f"  {current_term}/{total_terms}: {term}")
            
            df = search_disprot_term(term)
            
            if not df.empty:
                df['go_name'] = term 
                df['category'] = category
                all_results.append(df)
                found_terms.append(term)
                print(f"    Found: {len(df)} entries")
            else:
                missing_terms.append(term)
                print(f"    No data found")
            
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        raw_file = raw_dir / "all_terms_raw.tsv"
        combined_df.to_csv(raw_file, sep='\t', index=False)
        
        print(f"\nSummary:")
        print(f"Terms found: {len(found_terms)}/{total_terms}")
        print(f"Total entries: {len(combined_df)}")
        print(f"Unique proteins: {combined_df['acc'].nunique()}")
        
        print(f"\nResults by category:")
        category_summary = combined_df.groupby('category').agg({
            'go_name': 'nunique',
            'acc': 'nunique'
        }).rename(columns={'go_name': 'terms_found', 'acc': 'unique_proteins'})
        
        for category, row in category_summary.iterrows():
            print(f"  {category}: {row['terms_found']} terms, {row['unique_proteins']} proteins")
        
        filtered_dir = base_dir / "filtered"
        
        for category in terms.keys():
            category_data = combined_df[combined_df['category'] == category]
            if not category_data.empty:
                category_dir = filtered_dir
                category_dir.mkdir(exist_ok=True)
                
                category_file = category_dir / f"{category}_filtered.tsv"
                category_data.to_csv(category_file, sep='\t', index=False)
                print(f"  Saved {category}: {len(category_data)} entries")
        
        if missing_terms:
            print(f"\nTerms with no data:")
            for term in missing_terms:
                print(f"  - {term}")
                
    else:
        print(f"No data found for any terms")

if __name__ == "__main__":
    main()