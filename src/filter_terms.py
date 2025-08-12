"""
Filter the collected data for only the terms from Table S1
(https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-023-01803-y/MediaObjects/12915_2023_1803_MOESM1_ESM.pdf)
(excluding Lipid binding and Flexible linker)
"""

import pandas as pd

table_s1_terms = {
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

def case_insensitive_filter(df, terms_list):
    terms_lower = [term.lower() for term in terms_list]
    
    mask = df['term_name'].str.lower().isin(terms_lower)
    
    return df[mask]

files_to_process = [
    ('protein_binding', 'data/protein_binding_raw.tsv'),
    ('dna_binding', 'data/dna_binding_raw.tsv'),
    ('rna_binding', 'data/rna_binding_raw.tsv'),
    ('ion_binding', 'data/ion_binding_raw.tsv')
]

for category, filename in files_to_process:
    print(f"\n{'='*50}")
    print(f"Category {category}")
    
    df = pd.read_csv(filename, sep='\t')
    print(f"Original entries: {len(df)}")
    
    relevant_terms = table_s1_terms[category]
    filtered_df = case_insensitive_filter(df, relevant_terms)
    
    print(f"Filtered entries: {len(filtered_df)}")
    print(f"Unique proteins: {filtered_df['acc'].nunique()}")
    
    if len(filtered_df) > 0:
        print(f"\nFound terms:")
        found_terms = filtered_df['term_name'].value_counts()
        for term, count in found_terms.items():
            print(f"  - {term}: {count} entries")
        
        output_file = f"data/{category}_filtered.tsv"
        filtered_df.to_csv(output_file, sep='\t', index=False)
        print(f"Saved to: {output_file}")
        
        sample_accs = filtered_df['acc'].unique()[:5]
        print(f"Sample protein IDs: {sample_accs}")
        
    else:
        print("No matching terms found")