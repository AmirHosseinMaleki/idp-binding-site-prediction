import numpy as np
import os
from src.utils.config import load_config, get_embedding_path

cfg = load_config()
# Define all possible data files
# files = {
#     'Protein-Protein': {
#         'Structured (ScanNet)': {
#             'train': 'scannet_train_embeddings.npz',
#             'val': 'scannet_val_embeddings.npz',
#             'test': 'scannet_test_embeddings.npz'
#         },
#         'IDP (DisProt)': {
#             'train': 'disprot_train_embeddings.npz',
#             'val': 'disprot_val_embeddings.npz',
#             'test': 'disprot_test_embeddings.npz'
#         }
#     },
#     'DNA/RNA': {
#         'Structured (BioLiP)': {
#             'train': 'biolip_dna_rna_train_embeddings.npz',
#             'val': 'biolip_dna_rna_val_embeddings.npz',
#             'test': 'biolip_dna_rna_test_embeddings.npz'
#         },
#         'IDP (DisProt)': {
#             'train': 'disprot_dna_rna_train_embeddings.npz',
#             'val': 'disprot_dna_rna_val_embeddings.npz',
#             'test': 'disprot_dna_rna_test_embeddings.npz'
#         }
#     },
#     'Ion': {
#         'Structured (AHoJ-DB)': {
#             'train': 'ahojdb_train_embeddings.npz',
#             'val': 'ahojdb_val_embeddings.npz',
#             'test': 'ahojdb_test_embeddings.npz'
#         },
#         'IDP (DisProt)': {
#             'train': 'disprot_ion_train_embeddings.npz',
#             'val': 'disprot_ion_val_embeddings.npz',
#             'test': 'disprot_ion_test_embeddings.npz'
#         }
#     }
# }

files = {
    'Protein-Protein': {
        'Structured (ScanNet)': {
            'train': get_embedding_path(cfg, "scannet_train"),
            'val':   get_embedding_path(cfg, "scannet_val"),
            'test':  get_embedding_path(cfg, "scannet_test"),
        },
        'IDP (DisProt)': {
            'train': get_embedding_path(cfg, "disprot_protein_train"),
            'val':   get_embedding_path(cfg, "disprot_protein_val"),
            'test':  get_embedding_path(cfg, "disprot_protein_test"),
        }
    },
    'DNA/RNA': {
        'Structured (BioLiP)': {
            'train': get_embedding_path(cfg, "biolip_dna_rna_train"),
            'val':   get_embedding_path(cfg, "biolip_dna_rna_val"),
            'test':  get_embedding_path(cfg, "biolip_dna_rna_test"),
        },
        'IDP (DisProt)': {
            'train': get_embedding_path(cfg, "disprot_dna_rna_train"),
            'val':   get_embedding_path(cfg, "disprot_dna_rna_val"),
            'test':  get_embedding_path(cfg, "disprot_dna_rna_test"),
        }
    },
    'Ion': {
        'Structured (AHoJ-DB)': {
            'train': get_embedding_path(cfg, "ahojdb_train"),
            'val':   get_embedding_path(cfg, "ahojdb_val"),
            'test':  get_embedding_path(cfg, "ahojdb_test"),
        },
        'IDP (DisProt)': {
            'train': get_embedding_path(cfg, "disprot_ion_train"),
            'val':   get_embedding_path(cfg, "disprot_ion_val"),
            'test':  get_embedding_path(cfg, "disprot_ion_test"),
        }
    }
}

def count_residues_in_file(npz_file):
    """Count total residues and binding sites in a .npz file"""
    if not os.path.exists(npz_file):
        return None, None, None
    
    try:
        data = np.load(npz_file)
        total_residues = len(data['labels'])
        binding_sites = int(np.sum(data['labels']))
        non_binding = total_residues - binding_sites
        return total_residues, binding_sites, non_binding
    except Exception as e:
        print(f"  ⚠ Error reading {npz_file}: {e}")
        return None, None, None

print("="*100)
print("COMPLETE RESIDUE COUNTS - ALL DATASETS")
print("="*100)

# Store results for summary
all_results = {}

for binding_type, sources in files.items():
    print(f"\n{'='*100}")
    print(f"{binding_type} Binding")
    print('='*100)
    
    all_results[binding_type] = {}
    
    for source_name, splits in sources.items():
        print(f"\n{source_name}:")
        print("-"*100)
        
        all_results[binding_type][source_name] = {}
        
        for split_name, file in splits.items():
            res, binding, non_binding = count_residues_in_file(file)
            
            if res is None:
                print(f"  {split_name.upper():<10} {file:<55} ✗ NOT FOUND")
                all_results[binding_type][source_name][split_name] = {
                    'total': 0, 'binding': 0, 'non_binding': 0, 'exists': False
                }
            else:
                binding_pct = 100 * binding / res if res > 0 else 0
                print(f"  {split_name.upper():<10} {file:<55} ✓ {res:>12,} residues "
                      f"({binding:>10,} binding, {non_binding:>10,} non-binding, {binding_pct:.1f}%)")
                all_results[binding_type][source_name][split_name] = {
                    'total': res, 'binding': binding, 'non_binding': non_binding, 'exists': True
                }

# Summary tables
print("\n" + "="*100)
print("SUMMARY: COMBINED TRAIN/VAL/TEST BY BINDING TYPE")
print("="*100)

combined_summary = {}

for binding_type, sources in all_results.items():
    print(f"\n{binding_type}:")
    print("-"*100)
    print(f"{'Split':<10} {'Total Residues':<20} {'Binding Sites':<20} {'Non-Binding':<20} {'% Positive':<15}")
    print("-"*100)
    
    combined_summary[binding_type] = {}
    
    for split in ['train', 'val', 'test']:
        total_res = sum(source[split]['total'] for source in sources.values())
        total_binding = sum(source[split]['binding'] for source in sources.values())
        total_non_binding = sum(source[split]['non_binding'] for source in sources.values())
        
        if total_res > 0:
            pct = 100 * total_binding / total_res
            print(f"{split.upper():<10} {total_res:>19,} {total_binding:>19,} {total_non_binding:>19,} {pct:>14.1f}%")
            combined_summary[binding_type][split] = {
                'total': total_res, 'binding': total_binding, 'non_binding': total_non_binding
            }
        else:
            print(f"{split.upper():<10} {'N/A - no files':>19}")
            combined_summary[binding_type][split] = {
                'total': 0, 'binding': 0, 'non_binding': 0
            }

# Grand summary
print("\n" + "="*100)
print("GRAND TOTALS")
print("="*100)
print(f"{'Binding Type':<20} {'Total Residues':<20} {'Binding Sites':<20} {'Non-Binding':<20} {'% Positive':<15}")
print("-"*100)

for binding_type, splits in combined_summary.items():
    total = sum(s['total'] for s in splits.values())
    binding = sum(s['binding'] for s in splits.values())
    non_binding = sum(s['non_binding'] for s in splits.values())
    pct = 100 * binding / total if total > 0 else 0
    print(f"{binding_type:<20} {total:>19,} {binding:>19,} {non_binding:>19,} {pct:>14.1f}%")

# Dataset availability summary
print("\n" + "="*100)
print("DATASET AVAILABILITY MATRIX")
print("="*100)
print(f"{'Source':<40} {'Train':<10} {'Val':<10} {'Test':<10}")
print("-"*100)

for binding_type, sources in all_results.items():
    print(f"\n{binding_type}:")
    for source_name, splits in sources.items():
        train_status = "✓" if splits['train']['exists'] else "✗"
        val_status = "✓" if splits['val']['exists'] else "✗"
        test_status = "✓" if splits['test']['exists'] else "✗"
        print(f"  {source_name:<38} {train_status:<10} {val_status:<10} {test_status:<10}")

print("="*100)