# Code Architecture

This document describes the structure of the repository, the responsibility of each module, and the key classes and functions within each script.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Module Descriptions](#module-descriptions)
  - [src/prepare_data/](#srcprepare_data)
  - [src/staticp_old/](#srcstaticp_old)
  - [src/idp_old/](#srcidp_old)
  - [src/other_codes/](#srcother_codes)
  - [src/base_codes/](#srcbase_codes)
  - [src/training_scripts/](#srctraining_scripts)
  - [src/architecture_tests/](#srcarchitecture_tests)
  - [src/parameter_testing/](#srcparameter_testing)
  - [src/optimal_epoch_testing/](#srcoptimal_epoch_testing)
  - [src/evaluate_scripts/](#srcevaluate_scripts)
  - [src/without_embedding/](#srcwithout_embedding)
- [Core Classes](#core-classes)
- [Data Flow](#data-flow)

---

## Repository Structure

```
idp-binding-site-prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── docs/
│   ├── data_preparation.md
│   ├── code_architecture.md        ← this file
│   └── results_summary.md
│
├── demo/                           ← demo input and expected output
│
├── predict.py                      ← standalone prediction script
│
└── src/
    ├── prepare_data/               ← data preprocessing pipeline
    ├── staticp_old/                ← AHoJ-DB ion extraction scripts
    ├── idp_old/                    ← DisProt data collection scripts
    ├── other_codes/                ← ion dataset preparation utilities
    ├── base_codes/                 ← phase 1-3 training and embedding scripts
    ├── training_scripts/           ← hybrid and multi-task training
    ├── architecture_tests/         ← MLP vs CNN vs LSTM vs GRU comparison
    ├── parameter_testing/          ← hyperparameter grid search
    ├── optimal_epoch_testing/      ← epoch count optimisation
    ├── evaluate_scripts/           ← model evaluation and comparison
    └── without_embedding/          ← sequence-only baseline (no ESM-2)
```

---

## Module Descriptions

### `src/prepare_data/`

Converts raw downloaded datasets into training-ready format. This is the main data pipeline for all three binding types.

| Script | Responsibility |
|--------|---------------|
| `generate_embeddings.py` | ESM-2 embedding generation for ScanNet + DisProt protein binding |
| `process_biolip.py` | Parses `BioLiP_nr.txt` and `protein_nr.fasta` to extract DNA/RNA binding site annotations |
| `split_biolip.py` | Creates 70/15/15 random train/val/test splits for BioLiP DNA and RNA |
| `cluster_biolip.py` | MMseqs2 clustering + CAID3 filtering + re-split for BioLiP |
| `cluster_scannet.py` | MMseqs2 clustering + CAID3 filtering + re-split for ScanNet PPBS |
| `convert_scannet.py` | Converts ScanNet label file format to the standard CSV format used throughout |
| `extract_interfaces.py` | Extracts interface residues from PPIRef PDB files using biotite (parallel) |
| `split_disprot.py` | 70/15/15 split for DisProt protein binding TSV |
| `split_disprot_rna_dna.py` | 70/15/15 split for DisProt DNA and RNA binding TSVs |
| `caid3_filter.py` | Removes training sequences similar to CAID3 benchmark using MMseqs2 |

---

### `src/staticp_old/`

Scripts for extracting ion binding site data from the AHoJ-DB archive. These run sequentially to produce `final_complete_ion_dataset.csv`.

| Script | Responsibility |
|--------|---------------|
| `1_filter_ions.py` | Filters `ligand.tsv` using RDKit to identify monoatomic charged ions |
| `2_find_ion_directories.py` | Scans AHoJ-DB directory structure to find entries containing target ions |
| `3_extract_binding_sites.py` | Reads `pocket_residues.csv` per entry to get binding residue lists |
| `4_generate_dataset.py` | Downloads CIF structures from RCSB, extracts sequences, generates binary annotations. Runs in parallel across CPU cores. |

---

### `src/idp_old/`

Scripts for collecting DisProt binding annotations and downloading UniProt sequences. Used for all three binding types (protein, DNA, RNA, ion).

| Script | Responsibility |
|--------|---------------|
| `term_search.py` | Queries DisProt API for binding-related GO terms and saves results as TSV |
| `download_sequences.py` | Downloads FASTA sequences from UniProt REST API for each accession in DisProt |
| `training_dataset.py` | Assembles per-residue binary labels from DisProt positional annotations + sequences |
| `explore_collected_terms.py` | Utility to inspect term distribution in collected DisProt data |

---

### `src/other_codes/`

Utilities for the AHoJ-DB ion dataset after initial extraction.

| Script | Responsibility |
|--------|---------------|
| `step1_explore_data.py` | Computes class imbalance statistics and dataset summary for `final_complete_ion_dataset.csv` |
| `step1b_cluster_sequences.py` | Runs MMseqs2 clustering on ion sequences and creates cluster-aware 70/15/15 splits |
| `step2_prepare_data.py` | Legacy random split script (pre-clustering) |
| `step3_neural_network.py` | Early prototype neural network with window-based encoding (pre-ESM-2) |

---

### `src/base_codes/`

Core training scripts implementing the three-phase training strategy for all binding types. Each binding type (ion, DNA/RNA, protein) has Phase 1, 2, and 3 scripts plus embedding generation scripts. Each `.py` script has a corresponding `.sh` SLURM submission script.

**Embedding generation:**

| Script | Responsibility |
|--------|---------------|
| `generate_embeddings_ion.py` | ESM-2 embeddings for AHoJ-DB (CSV format) and DisProt ion (TSV format) |
| `generate_embeddings_dna_rna.py` | ESM-2 embeddings for BioLiP DNA/RNA (CSV) and DisProt DNA/RNA (TSV), then combines DNA+RNA |

**Phase training (ion binding):**

| Script | Training Data | Validation Data |
|--------|--------------|----------------|
| `train_ion_phase1.py` | AHoJ-DB only | AHoJ-DB val |
| `train_ion_phase2.py` | DisProt ion only | DisProt ion val |
| `train_ion_phase3.py` | AHoJ-DB + DisProt ion | AHoJ-DB + DisProt val |

**Phase training (DNA/RNA binding):**

| Script | Training Data | Validation Data |
|--------|--------------|----------------|
| `train_dna_rna_phase1.py` | BioLiP only | BioLiP val |
| `train_dna_rna_phase2.py` | DisProt DNA/RNA only | DisProt DNA/RNA val |
| `train_dna_rna_phase3.py` | BioLiP + DisProt | BioLiP + DisProt val |

**Phase training (protein-protein binding):**

| Script | Training Data | Validation Data |
|--------|--------------|----------------|
| `train_phase1_esm.py` | ScanNet only | ScanNet val |
| `train_phase2_esm.py` | DisProt protein only | DisProt protein val |
| `train_phase3_esm.py` | ScanNet + DisProt | ScanNet + DisProt val |

**Evaluation (protein-protein):**

| Script | Responsibility |
|--------|---------------|
| `evaluate_phase1_esm.py` | Evaluates Phase 1 model on ScanNet and DisProt test sets |
| `evaluate_phase2_esm.py` | Evaluates Phase 2 model on ScanNet and DisProt test sets |
| `evaluate_phase3_esm.py` | Evaluates Phase 3 model on ScanNet and DisProt test sets |

---

### `src/training_scripts/`

Advanced training approaches beyond the basic three-phase strategy.

| Script | Responsibility |
|--------|---------------|
| `train_protein_hybrid_idp_val.py` | Hybrid training (ScanNet + DisProt) with **DisProt-only** validation — the key innovation ensuring the saved model performs best on IDPs |
| `train_protein_hybrid_idp_val_optimized.py` | Same but with optimized hyperparameters from grid search (LR=0.00005, WD=0.001) |
| `train_ion_hybrid_idp_val.py` | Hybrid ion training with DisProt-only validation |
| `train_ion_hybrid_idp_val_optimized.py` | Optimized hyperparameter version |
| `train_dna_rna_hybrid_idp_val.py` | Hybrid DNA/RNA training with DisProt-only validation |
| `train_dna_rna_hybrid_idp_val_optimized.py` | Optimized hyperparameter version |
| `train_multitask_unified.py` | Single model with shared encoder + 3 task-specific heads for all binding types simultaneously |
| `train_multitask_balanced.py` | Multi-task model with custom batch sampler enforcing equal representation per task per batch |
| `train_finetune.py` | Two-stage fine-tuning: pre-train on structured data, then fine-tune on DisProt |

---

### `src/architecture_tests/`

Benchmarks four neural architectures against each other for each binding type. All architectures are trained under identical conditions for fair comparison.

| Script | Responsibility |
|--------|---------------|
| `architecture_test_protein.py` | Compares MLP, 1D CNN, Bi-LSTM, Bi-GRU for protein-protein binding |
| `architecture_test_dna_rna.py` | Same comparison for DNA/RNA binding |
| `architecture_test_ion.py` | Same comparison for ion binding |
| `architecture_test_ion_mlp.py` | Standalone MLP test for ion binding (extended epochs) |
| `architecture_test_ion_cnn.py` | Standalone 1D CNN test for ion binding |
| `architecture_test_ion_lstm.py` | Standalone Bi-LSTM test for ion binding |
| `architecture_test_ion_gru.py` | Standalone Bi-GRU test for ion binding |
| `architecture_test copy.py` | Early prototype comparison script (protein binding, fewer metrics) |

Each script also has a corresponding `.sh` SLURM submission file.

---

### `src/parameter_testing/`

Hyperparameter optimisation via grid search.

| Script | Responsibility |
|--------|---------------|
| `grid_search_protein.py` | Grid search over LR, dropout, weight decay, batch size for protein-protein binding. Tests 54 configurations, saves results to JSON. |

**Grid search space:**

| Parameter | Values |
|-----------|--------|
| Learning rate | 0.0001, 0.00005, 0.0002 |
| Dropout | (0.3/0.3/0.2), (0.5/0.5/0.3), (0.6/0.6/0.4) |
| Weight decay | 0.001, 0.01, 0.05 |
| Batch size | 256, 512 |

---

### `src/optimal_epoch_testing/`

Determines the optimal number of training epochs before the model begins to overfit.

| Script | Responsibility |
|--------|---------------|
| `find_optimal_epochs.py` | Trains MLP for up to 50 epochs on all three binding types, reports best epoch by AUC, AUPRC, F1, and loss. Recommends epoch count with safety margin. |

---

### `src/evaluate_scripts/`

Comprehensive evaluation and comparison scripts. These load saved `.pt` model files and evaluate on test sets.

| Script | Responsibility |
|--------|---------------|
| `evaluate_ion_all_phases.py` | Evaluates Phase 1, 2, 3 ion models on both AHoJ-DB and DisProt test sets |
| `evaluate_ion_comparison.py` | Compares Phase 3 vs hybrid IDP-val ion model |
| `evaluate_ion_optimized.py` | Evaluates the final optimized ion model |
| `evaluate_dna_rna_all_phases.py` | Evaluates Phase 1, 2, 3 DNA/RNA models on BioLiP and DisProt test sets |
| `evaluate_dna_rna_comparison.py` | Compares Phase 3 vs hybrid IDP-val DNA/RNA model |
| `evaluate_dna_rna_optimized.py` | Evaluates the final optimized DNA/RNA model |
| `evaluate_protein_comparison.py` | Compares Phase 3 vs hybrid IDP-val protein model on ScanNet and DisProt |
| `evaluate_protein_optimized.py` | Evaluates the final optimized protein model |
| `evaluate_protein_phase1.py` | Detailed Phase 1 protein evaluation with threshold search |
| `evaluate_architectures_complete.py` | Loads all four architecture models and evaluates with full metrics |
| `evaluate_all.py` | Legacy evaluation script (pre-ESM-2, window-based models) |
| `evaluate.py` | Early prototype evaluation script |

---

### `src/without_embedding/`

Baseline approach using one-hot encoded sliding windows instead of ESM-2 embeddings. Kept for comparison to demonstrate the benefit of protein language model features.

| Script | Responsibility |
|--------|---------------|
| `train_phase1.py` | Phase 1 training with 31-residue one-hot encoded windows (structured data) |
| `train_phase2.py` | Phase 2 training with window encoding (DisProt data) |
| `train_phase3.py` | Phase 3 hybrid training with window encoding |

**Window encoding:** Each residue is represented as a one-hot vector over 21 amino acids (20 standard + X for unknown/padding), concatenated across a 31-residue window (15 residues each side of the target). Input dimensionality: 31 × 21 = 651.

---

## Core Classes

The following classes appear throughout the codebase and are shared across multiple modules.

### `EmbeddingDataset`
**Location:** Used in `base_codes/`, `training_scripts/`, `architecture_tests/`, `evaluate_scripts/`

```python
class EmbeddingDataset(Dataset):
    def __init__(self, npz_file):
        # Loads precomputed ESM-2 embeddings from .npz file
        # Keys: 'embeddings' (N, 1280) float32, 'labels' (N,) float32
```

Loads a single `.npz` embedding file. Used for DisProt-only validation and test sets.

### `CombinedDataset`
**Location:** Used in `base_codes/`, `training_scripts/`, `architecture_tests/`

```python
class CombinedDataset(Dataset):
    def __init__(self, npz_files: list):
        # Concatenates multiple .npz files into one dataset
        # Used to combine structured + IDP data for hybrid training
```

Concatenates multiple embedding files into a single dataset. Used whenever combining structured and IDP data (Phase 3 training).

### `BindingNet` (MLP)
**Location:** Used in all training and evaluation scripts

```python
class BindingNet(nn.Module):
    # Input:  (batch_size, 1280) ESM-2 embedding
    # Output: (batch_size,) logit score per residue
    net = Sequential(
        Linear(1280, 512), ReLU, Dropout(0.5),
        Linear(512, 256),  ReLU, Dropout(0.5),
        Linear(256, 128),  ReLU, Dropout(0.3),
        Linear(128, 1)
    )
```

The primary model used in all final experiments. Takes a single residue ESM-2 embedding and outputs a binding probability logit.

### `MultiTaskBindingNet`
**Location:** `training_scripts/train_multitask_unified.py`, `train_multitask_balanced.py`

```python
class MultiTaskBindingNet(nn.Module):
    # Shared encoder: 1280 → 512 → 256 → 128
    # Task heads: protein_head, dna_rna_head, ion_head (each 128 → 1)
    # forward(x, task_ids) routes each sample to its task-specific head
```

### `CNN1D`, `BiLSTM`, `BiGRU`
**Location:** `architecture_tests/`

Alternative architectures tested against MLP baseline:
- **CNN1D:** Treats the 1280-dim embedding as a 1D signal, applies three Conv1d layers with MaxPool, then FC layers. Input: (batch, 1, 1280) → Output: (batch, 1)
- **BiLSTM:** Reshapes 1280-dim embedding to (20, 64) sequence, passes through 2-layer bidirectional LSTM, concatenates final hidden states. Input: (batch, 1280) → Output: (batch, 1)
- **BiGRU:** Same as BiLSTM but uses GRU cells. Faster training, similar performance.

### `BalancedBatchSampler`
**Location:** `training_scripts/train_multitask_balanced.py`

```python
class BalancedBatchSampler:
    # Yields batches with equal samples from each task (protein, DNA/RNA, ion)
    # Avoids WeightedRandomSampler's 2^24 index limit for large datasets
    # samples_per_task = batch_size // 3
```

---

## Data Flow

The complete pipeline from raw data to trained model:

```
Raw Databases
    │
    ├── AHoJ-DB archive ──────────────────────────────────────────────────────┐
    │   src/staticp_old/1_filter_ions.py                                       │
    │   src/staticp_old/2_find_ion_directories.py                              │
    │   src/staticp_old/3_extract_binding_sites.py                             │
    │   src/staticp_old/4_generate_dataset.py                                  │
    │   → final_complete_ion_dataset.csv                                        │
    │   src/other_codes/step1b_cluster_sequences.py (MMseqs2 + CAID3 filter)   │
    │   → train_data_clustered.csv / val / test                                 │
    │                                                                           │
    ├── BioLiP (protein_nr.fasta + BioLiP_nr.txt) ────────────────────────────┤
    │   src/prepare_data/process_biolip.py                                      │
    │   → biolip_dna_all.csv, biolip_rna_all.csv                               │
    │   src/prepare_data/cluster_biolip.py (MMseqs2 + CAID3 filter)            │
    │   → biolip_dna_clustered_train/val/test.csv (+ RNA)                       │
    │                                                                           │
    ├── ScanNet PPBS (table.csv + labels_*.txt) ───────────────────────────────┤
    │   src/prepare_data/convert_scannet.py                                     │
    │   → scannet_train/val/test.csv                                            │
    │   src/prepare_data/cluster_scannet.py (MMseqs2 + CAID3 filter)           │
    │   → scannet_train/val/test_clustered.csv                                  │
    │                                                                           │
    └── DisProt (GO term search + UniProt sequences) ─────────────────────────┤
        src/idp_old/term_search.py                                              │
        src/idp_old/download_sequences.py                                       │
        src/idp_old/training_dataset.py                                         │
        → protein/dna/rna/ion_binding_training_data.tsv                        │
        src/prepare_data/split_disprot*.py                                      │
        → *_train/val/test.tsv                                                  │
                                                                                │
                            ▼                                                   │
              ESM-2 Embedding Generation ◄──────────────────────────────────────┘
              src/base_codes/generate_embeddings*.py
              src/prepare_data/generate_embeddings.py
                            │
                            ▼
              *.npz files (embeddings + labels per residue)
              [e.g. scannet_train_embeddings.npz,
                    disprot_train_embeddings.npz, ...]
                            │
                            ▼
              Model Training
              ├── Phase 1: structured only   → src/base_codes/train_*_phase1.py
              ├── Phase 2: DisProt only      → src/base_codes/train_*_phase2.py
              ├── Phase 3: hybrid mixed val  → src/base_codes/train_*_phase3.py
              └── Hybrid IDP val (final)     → src/training_scripts/train_*_hybrid_idp_val*.py
                            │
                            ▼
              Saved Models (.pt files in data/)
                            │
                            ▼
              Evaluation
              src/evaluate_scripts/evaluate_*_all_phases.py
              src/evaluate_scripts/evaluate_*_comparison.py
              src/evaluate_scripts/evaluate_*_optimized.py
```