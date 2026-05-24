# Data Preparation Guide

This document describes the full data preparation pipeline for all three binding types: ion binding (AHoJ-DB), DNA/RNA binding (BioLiP + DisProt), and protein-protein binding (ScanNet + DisProt).

All scripts assume you are working from the repository root. Training scripts are designed to run on a SLURM cluster; adjust paths in the scripts if running locally.

---

## Table of Contents
- [Overview of Data Sources](#overview-of-data-sources)
- [Ion Binding (AHoJ-DB + DisProt)](#ion-binding-ahoj-db--disprot)
- [DNA/RNA Binding (BioLiP + DisProt)](#dnarna-binding-biolip--disprot)
- [Protein-Protein Binding (ScanNet + DisProt)](#protein-protein-binding-scannet--disprot)
- [Dataset Statistics](#dataset-statistics)

---

## Overview of Data Sources

| Source | Binding Type | Role | How to obtain |
|--------|-------------|------|----------|
| [AHoJ-DB](https://apoholo.cz/db/archive) | Ion | Structured training data | Archive download |
| [BioLiP](https://aideepmed.com/BioLiP/download.html) | DNA/RNA | Structured training data | Files: `protein_nr.fasta`, `BioLiP_nr.txt` |
| [ScanNet PPBS](https://github.com/jertubiana/ScanNet/tree/main/datasets) | Protein-Protein | Structured training data | GitHub datasets folder |
| [DisProt](https://disprot.org/download) | All types | IDP training and test data | TSV export |
| [UniProt REST API](https://rest.uniprot.org/) | All types | Sequence retrieval for DisProt entries | Programmatic |
| [CAID3](https://caid.idpcentral.org/) | All types | Benchmark test set, contamination filter | Challenge download |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | All types | Sequence clustering tool | `conda install -c bioconda mmseqs2` |

All datasets must be downloaded manually before running any preparation scripts. Place downloaded files in `data/` at the repository root.

---

## Ion Binding (AHoJ-DB + DisProt)

### Step 1: Download AHoJ-DB

Download the full archive from [https://apoholo.cz/db/archive](https://apoholo.cz/db/archive). Extract into `data/ahojdb/`. You will need:
- `ligand.tsv` — ligand metadata with SMILES strings
- Per-entry subdirectories containing `pocket_residues.csv` files

### Step 2: Identify Ion Ligands

Filter the ligand list to keep only monoatomic charged ions using RDKit:

```bash
python src/staticp_old/1_filter_ions.py
```

**Input:** `data/ahojdb/ligand.tsv`  
**Output:** `ions.txt` — list of ion codes (e.g., ZN, MG, CA)  
**Logic:** Keeps ligands with formal charge ≠ 0 and atom count < 2.

### Step 3: Find Ion-Containing Entries

Scan all AHoJ-DB subdirectories to identify entries containing the ions from Step 2:

```bash
python src/staticp_old/2_find_ion_directories.py
```

**Input:** `ions.txt`, `data/ahojdb/` directory structure  
**Output:** `all_ion_directories.txt`

### Step 4: Extract Binding Site Residues

Parse `pocket_residues.csv` from each ion entry to get the binding residue list:

```bash
python src/staticp_old/3_extract_binding_sites.py
```

**Input:** `all_ion_directories.txt`  
**Output:** `all_binding_sites.txt` — per-entry: PDB ID, chain, ligand, binding residue list

### Step 5: Download PDB Structures and Generate Dataset

For each entry, download the CIF structure from RCSB, extract the protein sequence, and create binary annotations (1 = binding residue, 0 = non-binding):

```bash
python src/staticp_old/4_generate_dataset.py
```

**Input:** `all_binding_sites.txt`  
**Output:** `final_complete_ion_dataset.csv`  
**Columns:** `pdb_id`, `chain_id`, `ligand`, `sequence`, `annotation`, `length`, `binding_sites`  
**Note:** This script downloads CIF files in parallel using all available CPU cores. It saves progress checkpoints every 100 proteins.

### Step 6: Cluster Sequences to Prevent Data Leakage

Create a FASTA file from the dataset and run MMseqs2 clustering at 10% sequence identity:

```bash
python src/other_codes/step1b_cluster_sequences.py
```

**Input:** `final_complete_ion_dataset.csv`  
**Output:**
- `ion_sequences.fasta`
- `mmseqs_output/clusters.tsv`
- `train_data_clustered.csv`, `val_data_clustered.csv`, `test_data_clustered.csv` (70/15/15 split at cluster level)
- `cluster_splits.json`

Internally runs:
```bash
mmseqs easy-cluster ion_sequences.fasta mmseqs_results/clusterRes mmseqs_results/tmp --min-seq-id 0.1
```

### Step 7: Filter Against CAID3 Benchmark

Remove training sequences that are similar to the CAID3 test set:

```bash
python src/prepare_data/caid3_filter.py
```

**Input:** `train_data_clustered.csv`, CAID3 files (`binding.fasta` and/or `binding-idr.fasta`)  
**Output:** `train_data_filtered_caid3.csv`

Place CAID3 FASTA files in the working directory before running.

### Step 8: Prepare DisProt Ion Binding Data

The DisProt ion binding data is collected by searching DisProt for ion-related GO terms (Ion binding, Metal ion binding, Zinc ion binding, etc.) using `src/idp_old/term_search.py`, downloading sequences via the UniProt REST API using `src/idp_old/download_sequences.py`, and assembling per-residue binary labels using `src/idp_old/training_dataset.py`.

The resulting TSV file (`ion_binding_training_data.tsv`) is then split manually into train/val/test at the training script level. For DisProt ion, the splits are applied directly when loading data in the training scripts.

**DisProt ion file format (TSV):**

| Column | Description |
|--------|-------------|
| `sequence` | Full amino acid sequence string |
| `labels` | Comma-separated binary labels, one per residue (e.g., `0,0,1,1,0,0`) |

Example row:
```
sequence                labels
MSEQNNTEMTFQIQR...     0,0,0,1,1,0,0,1,0,...
```

The label `1` indicates a residue that participates in ion binding according to DisProt annotations. Unlike AHoJ-DB where binding is inferred from 3D structure, DisProt annotations are curated from literature and cover disordered regions only.

### Step 9: Generate ESM-2 Embeddings for Ion Binding

```bash
sbatch src/base_codes/generate_embeddings_ion.sh
```

Or run directly (requires GPU):
```bash
python src/base_codes/generate_embeddings_ion.py
```

**Input files read:**
- `data/train_data.csv` → `ahojdb_train_embeddings.npz`
- `data/val_data.csv` → `ahojdb_val_embeddings.npz`
- `data/test_data.csv` → `ahojdb_test_embeddings.npz`
- `data/ion_binding_train.tsv` → `disprot_ion_train_embeddings.npz`
- `data/ion_binding_val.tsv` → `disprot_ion_val_embeddings.npz`
- `data/ion_binding_test.tsv` → `disprot_ion_test_embeddings.npz`

**ESM-2 model used:** `esm2_t33_650M_UR50D` (650M parameter model, layer 33 representations)  
**Embedding dimension:** 1280 per residue  
**Output format:** `.npz` files with keys `embeddings` (shape: N×1280) and `labels` (shape: N)  
**Sequences longer than 2000 residues are skipped** to prevent GPU out-of-memory errors.

---

## DNA/RNA Binding (BioLiP + DisProt)

### Step 1: Download BioLiP

Download from [https://aideepmed.com/BioLiP/download.html](https://aideepmed.com/BioLiP/download.html). You need:
- `protein_nr.fasta` — non-redundant receptor sequences
- `BioLiP_nr.txt` — annotation file (tab-separated)

Place both files in `data/biolip/`.

### Step 2: Extract DNA and RNA Binding Sites

Parse the BioLiP annotation file to find entries where the ligand is `dna` or `rna`, then create per-residue binary annotations:

```bash
python src/prepare_data/process_biolip.py
```

**Input:** `data/biolip/protein_nr.fasta`, `data/biolip/BioLiP_nr.txt`  
**Output:** `biolip_dna_all.csv`, `biolip_rna_all.csv`  
**Logic:** For each protein-chain pair, collects all binding site residue positions from the renumbered (`bs_residues_renum`) column, then creates a binary string annotation aligned to the sequence.

**BioLiP output CSV format:**

| Column | Description |
|--------|-------------|
| `pdb_id` | 4-character PDB identifier |
| `chain_id` | Chain letter |
| `sequence` | Full amino acid sequence |
| `annotation` | Binary string, one character per residue (e.g., `00110001`) |
| `length` | Sequence length |
| `binding_sites` | Count of binding residues |

### Step 3: Create Train/Val/Test Splits for BioLiP

```bash
python src/prepare_data/split_biolip.py
```

**Input:** `biolip_dna_all.csv`, `biolip_rna_all.csv`  
**Output:** `biolip_dna_train.csv`, `biolip_dna_val.csv`, `biolip_dna_test.csv`, and RNA equivalents  
**Split:** 70/15/15 random shuffle with seed 42

### Step 4: Cluster BioLiP and Filter Against CAID3

```bash
python src/prepare_data/cluster_biolip.py
```

This script runs the full MMseqs2 clustering pipeline on both DNA and RNA datasets:
1. Creates FASTA file from all sequences
2. Runs `mmseqs cluster` at 10% sequence identity, 80% coverage
3. Selects one representative per cluster
4. Filters out sequences overlapping with CAID3 test set
5. Re-splits into 70/15/15

**Input:** `biolip_dna_all.csv`, `biolip_rna_all.csv`, CAID3 files (optional)  
**Output:** `biolip_dna_clustered_train.csv`, `biolip_dna_clustered_val.csv`, `biolip_dna_clustered_test.csv`, and RNA equivalents

### Step 5: Prepare DisProt DNA and RNA Binding Data

Download DisProt annotations and search for DNA and RNA binding GO terms (DNA binding, RNA binding, Single-stranded DNA binding, G-quadruplex RNA binding, etc.) using `src/idp_old/term_search.py`, download sequences using `src/idp_old/download_sequences.py`, and assemble per-residue labels using `src/idp_old/training_dataset.py`. This produces:
- `data/training/dna_binding_training_data.tsv`
- `data/training/rna_binding_training_data.tsv`

Then split:

```bash
python src/prepare_data/split_disprot_rna_dna.py
```

**Input:** `data/training/dna_binding_training_data.tsv`, `data/training/rna_binding_training_data.tsv`  
**Output:** `dna_binding_train.tsv`, `dna_binding_val.tsv`, `dna_binding_test.tsv`, and RNA equivalents  
**Split:** 70/15/15 random shuffle with seed 42

**DisProt DNA/RNA file format (TSV):** Same format as DisProt ion — `sequence` and comma-separated `labels` columns.

### Step 6: Generate ESM-2 Embeddings for DNA/RNA

```bash
sbatch src/base_codes/generate_embeddings_dna_rna.sh
```

Or run directly:
```bash
python src/base_codes/generate_embeddings_dna_rna.py
```

This script handles both BioLiP (CSV format, using `annotation` column) and DisProt (TSV format, using comma-separated `labels` column), and combines DNA and RNA embeddings at the end.

**Output files produced:**
- `biolip_dna_train_embeddings.npz` + `biolip_rna_train_embeddings.npz` → `biolip_dna_rna_train_embeddings.npz`
- `biolip_dna_val_embeddings.npz` + `biolip_rna_val_embeddings.npz` → `biolip_dna_rna_val_embeddings.npz`
- `biolip_dna_test_embeddings.npz` + `biolip_rna_test_embeddings.npz` → `biolip_dna_rna_test_embeddings.npz`
- `disprot_dna_train_embeddings.npz` + `disprot_rna_train_embeddings.npz` → `disprot_dna_rna_train_embeddings.npz`
- Same for val and test splits

DNA and RNA are combined into a single binary classification task (binding vs non-binding to nucleic acids). This is justified because both involve nucleotide-binding interfaces and the combined dataset improves coverage.

---

## Protein-Protein Binding (ScanNet + DisProt)

### Step 1: Download ScanNet PPBS Dataset

Download from [https://github.com/jertubiana/ScanNet/tree/main/datasets](https://github.com/jertubiana/ScanNet/tree/main/datasets). You need the PPBS (Protein-Protein Binding Sites) dataset files:
- `table.csv`
- `labels_train.txt`
- `labels_validation_70.txt`
- `labels_test_70.txt`

Place in `data/ScanNet/datasets/PPBS/`.

### Step 2: Convert ScanNet Format

ScanNet uses a custom label format where each line contains chain, position, residue, and label. Convert to the same CSV format used by AHoJ-DB:

```bash
python src/prepare_data/convert_scannet.py
```

**Input:** `table.csv`, `labels_train.txt`, `labels_validation_70.txt`, `labels_test_70.txt`  
**Output:** `scannet_train.csv`, `scannet_val.csv`, `scannet_test.csv`  
**Format:** `pdb_id`, `chain_id`, `sequence`, `annotation` (binary string), `length`, `binding_sites`

**ScanNet label file format (input):**
```
>13gs_0-A
A 0 M 0
A 1 T 0
A 2 I 1
...
```
Each line is: chain, position, residue, label. The script reads these and assembles them into a sequence string and annotation string.

**Note:** The script uses validation set `_70` (sequences with up to 70% identity to training), which is the most standard split used in the ScanNet paper.

### Step 3: Cluster ScanNet and Filter Against CAID3

```bash
python src/prepare_data/cluster_scannet.py
```

This script:
1. Combines all three splits into one pool for clustering
2. Creates FASTA and runs MMseqs2 at 10% identity, 80% coverage
3. Selects one representative per cluster
4. Filters out sequences overlapping with CAID3 test set
5. Re-splits into 70/15/15

**Output:** `scannet_train_clustered.csv`, `scannet_val_clustered.csv`, `scannet_test_clustered.csv`

### Step 4: Prepare DisProt Protein Binding Data

**4a. Search DisProt for protein binding terms:**

```bash
python src/idp_old/term_search.py
```

Searches the DisProt API for all protein-binding GO terms (protein binding, calmodulin binding, SH3 domain binding, MDM2/MDM4 binding, 14-3-3 protein binding, etc.). The full list of terms is from Table S1 of the CAID2 paper.

**Output:** `data/filtered/protein_binding_filtered.tsv`

**4b. Download UniProt sequences:**

```bash
python src/idp_old/download_sequences.py
```

For each unique UniProt accession in the filtered dataset, fetches the FASTA sequence from `https://rest.uniprot.org/uniprotkb/{accession}.fasta`. Sequences are saved per-category.

**Output:** `data/uniprot_sequences/protein_binding/*.fasta`

**4c. Assemble per-residue training labels:**

```bash
python src/idp_old/training_dataset.py
```

Combines DisProt positional start/end annotations with downloaded sequences to create binary per-residue label strings. A residue is labelled 1 if it falls within any annotated binding region for that protein.

**Output:** `data/training/protein_binding_training_data.tsv`

**4d. Split into train/val/test:**

```bash
python src/prepare_data/split_disprot.py
```

**Output:** `protein_binding_train.tsv`, `protein_binding_val.tsv`, `protein_binding_test.tsv`  
**Split:** 70/15/15 random shuffle with seed 42

### Step 5: Generate ESM-2 Embeddings for Protein Binding

```bash
sbatch src/base_codes/run-sbatch.sh
```

Or run directly:
```bash
python src/prepare_data/generate_embeddings.py
```

**Input files read:**
- `data/ScanNet/datasets/PPBS/scannet_train_clustered.csv` → `scannet_train_embeddings.npz`
- `data/ScanNet/datasets/PPBS/scannet_val_clustered.csv` → `scannet_val_embeddings.npz`
- `data/ScanNet/datasets/PPBS/scannet_test_clustered.csv` → `scannet_test_embeddings.npz`
- `data/ScanNet/datasets/PPBS/protein_binding_train.tsv` → `disprot_train_embeddings.npz`
- `data/ScanNet/datasets/PPBS/protein_binding_val.tsv` → `disprot_val_embeddings.npz`
- `data/ScanNet/datasets/PPBS/protein_binding_test.tsv` → `disprot_test_embeddings.npz`

**Output files:**
- `scannet_train_embeddings.npz`, `scannet_val_embeddings.npz`, `scannet_test_embeddings.npz`
- `disprot_train_embeddings.npz`, `disprot_val_embeddings.npz`, `disprot_test_embeddings.npz`

---

## Dataset Statistics

After running all preparation steps, the final datasets have the following characteristics:

| Dataset | Source | Train Residues | Val Residues | Test Residues | Positive % |
|---------|--------|---------------|-------------|--------------|------------|
| Ion (AHoJ-DB) | Structured | ~41,992,000 | ~8,998,000 | 8,998,264 | 1.5% |
| Ion (DisProt) | IDP | ~19,189 | ~4,112 | 4,112 | 31.3% |
| DNA/RNA (BioLiP) | Structured | ~734,146 | ~157,317 | 157,317 | ~10% |
| DNA/RNA (DisProt) | IDP | ~46,289 | ~9,919 | 9,919 | ~24% |
| Protein (ScanNet) | Structured | ~1,036,882 | ~222,189 | 222,189 | ~18% |
| Protein (DisProt) | IDP | ~265,879 | ~56,974 | 56,974 | 23.9% |

**Class imbalance notes:**
- Ion binding from AHoJ-DB has the most extreme imbalance (1.5% positive), requiring `pos_weight=30.0` in the loss function
- DisProt datasets are far more balanced (~20-30% positive) as they specifically annotate binding regions in disordered proteins
- DNA/RNA and protein binding use `pos_weight=3.0`

**Embedding storage:** Each `.npz` file stores float32 arrays. The full set of embedding files requires approximately 50GB of disk space.

**Train/val/test split strategy:** All splits are done at the sequence cluster level (not protein level) to ensure no similar sequences appear across splits. MMseqs2 clusters at 10% sequence identity threshold. The CAID3 benchmark sequences are additionally removed from training data to allow unbiased evaluation on the official benchmark.