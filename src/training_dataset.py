"""
Create complete training dataset for all proteins
"""

import pandas as pd
from pathlib import Path
import numpy as np

def load_protein_sequence(protein_id, sequences_dir="data/uniprot_sequences"):
    """Load protein sequence from FASTA file"""
    for category in ['protein_binding', 'dna_binding', 'rna_binding', 'ion_binding']:
        fasta_file = Path(sequences_dir) / category / f"{protein_id}.fasta"
        if fasta_file.exists():
            with open(fasta_file, 'r') as f:
                lines = f.read().strip().split('\n')
                header = lines[0]
                sequence = ''.join(lines[1:])
                return sequence, header
    return None, None

def load_binding_annotations_by_category(protein_id, filtered_dir="data/filtered"):
    """Load binding annotations grouped by category"""
    annotations_by_category = {
        'protein_binding': [],
        'dna_binding': [],
        'rna_binding': [],
        'ion_binding': []
    }
    
    for category in ['protein_binding', 'dna_binding', 'rna_binding', 'ion_binding']:
        file_path = Path(filtered_dir) / f"{category}_filtered.tsv"
        if file_path.exists():
            df = pd.read_csv(file_path, sep='\t')
            protein_data = df[df['acc'] == protein_id]
            
            for _, row in protein_data.iterrows():
                annotations_by_category[category].append({
                    'go_name': row['go_name'],
                    'start': int(row['start']),
                    'end': int(row['end']),
                    'length': int(row['end']) - int(row['start']) + 1
                })
    
    return annotations_by_category

def create_binary_labels(sequence_length, annotations):
    """Create binary labels for specific binding type"""
    labels = np.zeros(sequence_length, dtype=int)
    
    for ann in annotations:
        start = ann['start'] - 1 
        end = ann['end']         
        labels[start:end] = 1
    
    return labels

def get_all_protein_ids(filtered_dir="data/filtered"):
    """Get all unique protein IDs from filtered datasets"""
    all_protein_ids = set()
    
    for category in ['protein_binding', 'dna_binding', 'rna_binding', 'ion_binding']:
        file_path = Path(filtered_dir) / f"{category}_filtered.tsv"
        if file_path.exists():
            df = pd.read_csv(file_path, sep='\t')
            protein_ids = set(df['acc'].unique())
            all_protein_ids.update(protein_ids)
    
    return sorted(list(all_protein_ids))

def process_protein_for_training(protein_id):
    """Process one protein and create training examples for each binding category"""
    sequence, header = load_protein_sequence(protein_id)
    if not sequence:
        return []
    
    annotations_by_category = load_binding_annotations_by_category(protein_id)
    
    training_examples = []
    
    for category, annotations in annotations_by_category.items():
        if annotations:
            labels = create_binary_labels(len(sequence), annotations)
            binding_positions = np.sum(labels)
            coverage = (binding_positions / len(sequence)) * 100
            
            unique_terms = set(ann['go_name'] for ann in annotations)
            
            training_examples.append({
                'protein_id': protein_id,
                'sequence': sequence,
                'header': header,
                'length': len(sequence),
                'binding_category': category,
                'annotations_count': len(annotations),
                'binding_positions': binding_positions,
                'coverage_percent': coverage,
                'labels': labels,
                'go_terms': list(unique_terms)
            })
    
    return training_examples

def save_training_datasets(all_examples, output_dir="data/training"):
    """Save training datasets by category as simple TSV files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    examples_by_category = {}
    for example in all_examples:
        category = example['binding_category']
        if category not in examples_by_category:
            examples_by_category[category] = []
        examples_by_category[category].append(example)
    
    for category, examples in examples_by_category.items():
        tsv_data = []
        for example in examples:
            labels_str = ','.join(map(str, example['labels']))
            
            tsv_data.append({
                'protein_id': example['protein_id'],
                'sequence': example['sequence'],
                'labels': labels_str,
                'binding_category': example['binding_category'],
                'length': example['length'],
                'binding_positions': example['binding_positions'],
                'coverage_percent': round(example['coverage_percent'], 2),
                'annotations_count': example['annotations_count'],
                'go_terms': '; '.join(example['go_terms']),
                'header': example['header']
            })
        
        df = pd.DataFrame(tsv_data)
        tsv_file = output_dir / f"{category}_training_data.tsv"
        df.to_csv(tsv_file, sep='\t', index=False)

def main():
    all_protein_ids = get_all_protein_ids()
    print(f"Processing {len(all_protein_ids)} proteins")
    
    all_training_examples = []
    failed_proteins = []
    
    for protein_id in all_protein_ids:
        try:
            examples = process_protein_for_training(protein_id)
            all_training_examples.extend(examples)
        except Exception as e:
            failed_proteins.append(protein_id)
    
    print(f"Total training examples: {len(all_training_examples)}")
    
    if failed_proteins:
        print(f"Failed proteins: {len(failed_proteins)}")
    
    examples_by_category = {}
    for example in all_training_examples:
        category = example['binding_category']
        if category not in examples_by_category:
            examples_by_category[category] = []
        examples_by_category[category].append(example)
    
    for category, examples in examples_by_category.items():
        print(f"{category}: {len(examples)} examples")
    
    save_training_datasets(all_training_examples)

if __name__ == "__main__":
    main()