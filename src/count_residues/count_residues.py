import numpy as np

# Define all data files
files = {
    'Protein-Protein': {
        'train': ['scannet_train_embeddings.npz', 'disprot_train_embeddings.npz'],
        'val': ['disprot_val_embeddings.npz'],
        'test': ['disprot_test_embeddings.npz']
    },
    'DNA/RNA': {
        'train': ['biolip_dna_rna_train_embeddings.npz', 'disprot_dna_rna_train_embeddings.npz'],
        'val': ['disprot_dna_rna_val_embeddings.npz'],
        'test': ['disprot_dna_rna_test_embeddings.npz']
    },
    'Ion': {
        'train': ['ahojdb_train_embeddings.npz', 'disprot_ion_train_embeddings.npz'],
        'val': ['disprot_ion_val_embeddings.npz'],
        'test': ['disprot_ion_test_embeddings.npz']
    }
}

def count_residues_in_file(npz_file):
    """Count total residues and binding sites in a .npz file"""
    try:
        data = np.load(npz_file)
        total_residues = len(data['labels'])
        binding_sites = int(np.sum(data['labels']))
        non_binding = total_residues - binding_sites
        return total_residues, binding_sites, non_binding
    except FileNotFoundError:
        print(f"  ⚠ File not found: {npz_file}")
        return 0, 0, 0

print("="*90)
print("RESIDUE COUNTS BY BINDING TYPE AND SPLIT")
print("="*90)

# Store results for summary table
all_results = {}

for binding_type, splits in files.items():
    print(f"\n{binding_type} Binding")
    print("-"*90)
    
    all_results[binding_type] = {}
    
    for split_name, file_list in splits.items():
        total_res = 0
        total_binding = 0
        total_non_binding = 0
        
        print(f"\n  {split_name.upper()}:")
        for file in file_list:
            res, binding, non_binding = count_residues_in_file(file)
            total_res += res
            total_binding += binding
            total_non_binding += non_binding
            
            if res > 0:
                print(f"    {file:<50} {res:>12,} residues ({binding:>10,} binding, {non_binding:>10,} non-binding)")
        
        # Summary for this split
        if total_res > 0:
            binding_pct = 100 * total_binding / total_res
            print(f"    {'TOTAL ' + split_name.upper():<50} {total_res:>12,} residues ({binding:>10,} binding, {non_binding:>10,} non-binding, {binding_pct:.1f}%)")
        
        all_results[binding_type][split_name] = {
            'total': total_res,
            'binding': total_binding,
            'non_binding': total_non_binding
        }

# Summary table
print("\n" + "="*90)
print("SUMMARY TABLE")
print("="*90)
print(f"{'Binding Type':<20} {'Split':<10} {'Total Residues':<20} {'Binding Sites':<20} {'Non-Binding':<20}")
print("-"*90)

for binding_type, splits in all_results.items():
    for split_name, counts in splits.items():
        print(f"{binding_type:<20} {split_name:<10} {counts['total']:>19,} {counts['binding']:>19,} {counts['non_binding']:>19,}")

print("="*90)

# Grand totals
print("\nGRAND TOTALS:")
print("-"*90)

for binding_type, splits in all_results.items():
    total = sum(s['total'] for s in splits.values())
    binding = sum(s['binding'] for s in splits.values())
    non_binding = sum(s['non_binding'] for s in splits.values())
    binding_pct = 100 * binding / total if total > 0 else 0
    print(f"{binding_type:<20} {total:>19,} total ({binding:>10,} binding, {non_binding:>10,} non-binding, {binding_pct:.1f}% positive)")

print("="*90)