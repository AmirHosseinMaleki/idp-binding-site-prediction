import pandas as pd
from src.utils.config import load_config, get_dataset_path

cfg = load_config()
filtered_dir = cfg["datasets"]["disprot"]["filtered_dir"]

# Check Protein binding terms  
# rna_df = pd.read_csv('data/filtered/protein_binding_filtered.tsv', sep='\t')
rna_df = pd.read_csv(f"{filtered_dir}/protein_binding_filtered.tsv", sep='\t')

# same for dna, rna, ion
print("Protein binding terms:")
print(rna_df['term_name'].value_counts())

print("\n" + "="*50 + "\n")

# Check DNA binding terms
# dna_df = pd.read_csv('data/filtered/dna_binding_filtered.tsv', sep='\t')
dna_df = pd.read_csv(f"{filtered_dir}/dna_binding_filtered.tsv", sep='\t')
print("DNA binding terms:")
print(dna_df['term_name'].value_counts())

print("\n" + "="*50 + "\n")

# Check RNA binding terms  
# rna_df = pd.read_csv('data/filtered/rna_binding_filtered.tsv', sep='\t')
rna_df = pd.read_csv(f"{filtered_dir}/rna_binding_filtered.tsv", sep='\t')
print("RNA binding terms:")
print(rna_df['term_name'].value_counts())

print("\n" + "="*50 + "\n")

# Check Ion binding terms
# ion_df = pd.read_csv('data/filtered/ion_binding_filtered.tsv', sep='\t')
ion_df = pd.read_csv(f"{filtered_dir}/ion_binding_filtered.tsv", sep='\t')
print("Ion binding terms:")
print(ion_df['term_name'].value_counts())