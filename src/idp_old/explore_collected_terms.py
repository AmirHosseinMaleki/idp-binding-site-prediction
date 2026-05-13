import pandas as pd

# Check Protein binding terms  
rna_df = pd.read_csv('data/filtered/protein_binding_filtered.tsv', sep='\t')
print("Protein binding terms:")
print(rna_df['term_name'].value_counts())

print("\n" + "="*50 + "\n")

# Check DNA binding terms
dna_df = pd.read_csv('data/filtered/dna_binding_filtered.tsv', sep='\t')
print("DNA binding terms:")
print(dna_df['term_name'].value_counts())

print("\n" + "="*50 + "\n")

# Check RNA binding terms  
rna_df = pd.read_csv('data/filtered/rna_binding_filtered.tsv', sep='\t')
print("RNA binding terms:")
print(rna_df['term_name'].value_counts())

print("\n" + "="*50 + "\n")

# Check Ion binding terms
ion_df = pd.read_csv('data/filtered/ion_binding_filtered.tsv', sep='\t')
print("Ion binding terms:")
print(ion_df['term_name'].value_counts())