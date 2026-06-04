# IDP Binding Site Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Binding site prediction on a disordered protein sequence](figures/binding-site.svg)

This project investigates whether **hybrid training** - combining binding site 
data from large structured-protein databases with IDP-specific annotations from 
DisProt - improves binding site prediction in intrinsically disordered protein 
(IDP) regions. IDPs lack stable 3D structure but play key roles in signaling, 
regulation, and molecular recognition. Because structure-based predictors 
cannot be applied to them, sequence-based approaches are required.

We predict per-residue binding probability for three binding types: 
**protein–protein**, **DNA/RNA**, and **ion** binding sites. Models use 
[ESM-2](https://github.com/facebookresearch/esm) protein language model 
embeddings (1280-dimensional, per-residue) as input - purely sequence-based, 
so applicable to both structured and disordered proteins.

## Table of Contents
- [Research Hypothesis](#research-hypothesis)
- [Results](#results)
- [Demo](#demo)
- [Prediction Scripts](#prediction-scripts)
- [CAID Submission](#caid-submission)
- [Documentation](#documentation)
- [Environment Setup](#environment-setup)
- [Data](#data)
- [Training Overview](#training-overview)
- [Execution Guide](#execution-guide)

---

## Research Hypothesis

> **Hypothesis:** Enhancing IDP binding site prediction by incorporating 
> binding site data from well-structured proteins into training.

Standard approaches train exclusively on DisProt IDP annotations. We 
investigated whether supplementing this limited IDP data with large 
structured-protein binding databases leads to better IDP prediction.

To test this, we trained models under three configurations (referred to 
throughout all documentation as **Phase 1**, **Phase 2**, and **Phase 3**):

| Phase | Training Data | Validation Data | Purpose |
|-------|--------------|-----------------|---------|
| **Phase 1** | Structured only (AHoJ-DB / BioLiP / ScanNet) | Structured val set | Structured-data baseline |
| **Phase 2** | DisProt (IDP) only | DisProt val set | IDP-only baseline |
| **Phase 3 - Hybrid** | Structured + DisProt | DisProt val set | Main approach |

In Phase 3, the model is trained on both data sources but validated on DisProt 
sequences only - ensuring the saved checkpoint is selected for IDP performance, 
which is the prediction target we care about.

---

## Results

Best model performance on the DisProt IDP test set (Phase 3, MLP, optimized 
hyperparameters):

| Binding Type    | AUC    | AUPRC  | MCC    | F1     |
|-----------------|--------|--------|--------|--------|
| Protein-Protein | 0.8394 | 0.6214 | 0.4986 | 0.6460 |
| DNA/RNA         | 0.7126 | 0.5548 | 0.3618 | 0.5881 |
| Ion             | 0.8487 | 0.5931 | 0.4898 | 0.6135 |

Full results including phase comparison, architecture benchmarks, and 
hyperparameter analysis: [docs/3-results_summary.md](docs/3-results_summary.md)

---

## Demo

**Live demo:** [View 3D binding site prediction on AHSA1 (UniProt O95433, DisProt DP04219)](https://amirhosseinmaleki.github.io/idp-binding-site-prediction/demo/binding_site_demo.html)

Run the same prediction locally:
```bash
python predict.py \
  --fasta demo/demo_protein.fasta \
  --binding_type protein \
  --output results/prediction.tsv
```

The demo protein is **AHSA1** (Activator of Hsp90 ATPase, UniProt O95433, 
DisProt entry DP04219, 338 residues), drawn from the protein-protein binding 
test set. Its sequence is in `demo/demo_protein.fasta`. Ground-truth CAID 
labels are in `demo/demo_protein_labels.tsv`, and the expected model output is 
in `demo/expected_output_protein.tsv` for verification - running the command 
above should reproduce those scores. 
Pre-trained model weights are included in `data/` - no additional download is 
needed.

---

## Prediction Scripts

The repository contains three scripts for running the trained models. They cover
two scenarios: a single self-contained command for quick use, and a two-step
pipeline that matches how CAID runs predictions.

| Script | Location | Input | Generates embeddings? | Output |
|--------|----------|-------|----------------------|--------|
| `predict.py` | project root | one sequence (`--sequence` or `--fasta`) | **Yes** (built in) | per-residue TSV |
| `generate_caid_embeddings.py` | project root | multi-sequence FASTA | **Yes** (this is its only job) | one `.npy`/`.h5` per protein |
| `predict_caid.py` | project root | multi-sequence FASTA **+** pre-computed embeddings | No (reads them) | per-protein `.caid` files + `timings.csv` |

**`predict.py` — single-sequence, end-to-end.**
The simplest entry point and the one to use for a quick check or the demo. Give
it one sequence and a binding type; it loads ESM-2, generates the embeddings
itself, runs the model, and writes a per-residue TSV (`position`, `residue`,
`score`, `prediction`). No separate embedding step is needed.

```bash
python predict.py --fasta demo/demo_protein.fasta --binding_type protein --output results/prediction.tsv
```

**`generate_caid_embeddings.py` — batch embedding generation.**
Takes a multi-sequence FASTA and writes one ESM-2 embedding file per protein
(`<protein_id>.npy` or `.h5`, shape `(L, 1280)`, layer 33). This is the step
that produces the embeddings `predict_caid.py` expects. Run it once to prepare a
folder of embeddings for a whole test set.

```bash
python generate_caid_embeddings.py --fasta sequences.fasta --output_dir embeddings/
```

**`predict_caid.py` — CAID submission entry point.**
The script CAID runs inside the Docker container. It does **not** generate
embeddings - CAID pre-computes them and mounts the folder at runtime - so it
reads the pre-computed `.npy`/`.h5` files (the same ones
`generate_caid_embeddings.py` produces), runs all three binding-type models, and
writes per-protein `.caid` files plus `timings.csv`.

```bash
python predict_caid.py --fasta sequences.fasta --embeddings_dir embeddings/ --output_dir output/
```

**How they fit together:** `predict.py` is the embedding and prediction steps
fused into one command. `generate_caid_embeddings.py` followed by
`predict_caid.py` is the same two steps split apart, which is what the CAID
pipeline requires because CAID generates the embeddings separately from running
the predictor. To reproduce a CAID-style run locally, run
`generate_caid_embeddings.py` first, then `predict_caid.py` on its output.

---

## CAID Submission

[CAID (Critical Assessment of Intrinsic Disorder Prediction)](https://caid.idpcentral.org/)
is a community benchmark for evaluating predictors of intrinsic disorder and
disorder-related functions. We participate in the **CAID4** binding site track,
the current (this year's) round of the challenge, submitting the predictor as a
Docker image that CAID runs on its own held-out test set.

Throughout this repository, development and evaluation use the **CAID3** data,
since that is the most recent CAID dataset publicly available for building and
validating the predictor. CAID4 uses newly curated test sequences that are not
released to participants in advance; CAID3 is therefore used here as the
development benchmark, and the same predictor is submitted to CAID4 for
independent evaluation on its blind test set. See the official challenge page
for the submission procedure: https://caid.idpcentral.org/challenge#participate

The predictor is available as a Docker image on Docker Hub:

```bash
docker pull amirhmaleki/idp-binding-caid:latest
```

CAID evaluates the predictor by pre-computing ESM-2 embeddings and passing them
to the container. The container accepts a multi-sequence FASTA and a folder of
per-protein embeddings, and writes per-protein .caid output files for all three
binding types plus a timings.csv (per-protein prediction times, required by the
CAID submission format).

```bash
docker run \
  -v /path/to/sequences.fasta:/input/sequences.fasta \
  -v /path/to/embeddings:/embeddings \
  -v /path/to/output:/output \
  amirhmaleki/idp-binding-caid:latest \
    --fasta /input/sequences.fasta \
    --embeddings_dir /embeddings \
    --output_dir /output
```

**Output structure:**
```
output/
    protein/P04637.caid
    dna_rna/P04637.caid
    ion/P04637.caid
    timings.csv
```

**Embeddings format:** one `.npy` or `.h5` file per protein, named after the
protein identifier in the FASTA header (e.g. `>P04637` --> `P04637.npy`),
shape `(L, 1280)` from ESM-2 `esm2_t33_650M_UR50D` layer 33. These are the files
produced by `generate_caid_embeddings.py`.

The container runs on CPU, requires no internet access, and needs no GPU. CPU
inference is the intended and tested path for the CAID submission: the
embeddings are pre-computed by the evaluator, so the container only runs the
lightweight MLP forward pass, which is fast on CPU.

See [`predict_caid.py`](predict_caid.py) for the full interface.

---

## Documentation

Read in this order:

| # | Document | Description |
|---|----------|-------------|
| 1 | [Data Preparation](docs/1-data_preparation.md) | Step-by-step pipeline for all three binding types |
| 2 | [Code Architecture](docs/2-code_architecture.md) | Module descriptions, class references, data flow |
| 3 | [Results Summary](docs/3-results_summary.md) | Formal results following the project specification |

The project specification is included at [docs/specification.pdf](docs/specification.pdf).

---

## Environment Setup

The project uses two separate dependency sets:

- **`requirements.txt`** — full environment for data preparation, embedding
  generation, training, and evaluation (GPU recommended). This is the
  environment installed by the steps below.
- **`requirements-caid.txt`** — minimal CPU-only environment used by the CAID
  Docker image for inference (`torch`, `numpy`, `h5py`). It has loose version
  bounds and is installed only inside the container, not here.

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 12.8+ (required for embedding generation and training)
- 16 GB+ RAM
- 50 GB+ disk space for datasets and embeddings

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/idp-binding-site-prediction.git
cd idp-binding-site-prediction
```

2. Create and activate a conda environment (the project was developed with 
   [Miniconda](https://docs.conda.io/en/latest/miniconda.html)):
```bash
conda create -n protein python=3.8
conda activate protein
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify GPU availability:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

> **Note for SLURM users:** The cluster submission scripts use 
> `source ~/miniconda3/etc/profile.d/conda.sh` followed by 
> `conda activate protein`. Adjust the path if your Miniconda is installed 
> elsewhere.

---

## Data

Data files are not publicly available due to their size, but can be prepared 
from the sources below or provided upon request.

| Source | Role |
|--------|------|
| [AHoJ-DB](https://apoholo.cz/db/archive) | Structured ion binding training data |
| [BioLiP](https://aideepmed.com/BioLiP/download.html) | Structured DNA/RNA binding training data |
| [ScanNet PPBS](https://github.com/jertubiana/ScanNet/tree/main/datasets) | Structured protein–protein binding training data |
| [DisProt](https://disprot.org/download) | IDP binding annotations (training, validation, and test) |
| [UniProt REST API](https://rest.uniprot.org/) | Protein sequences for DisProt entries, retrieved via `src/idp_old/download_sequences.py` |
| [CAID](https://caid.idpcentral.org/) | Official benchmark test set |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | Sequence clustering tool, used to prevent train/test data leakage |

Full preparation pipeline: [docs/1-data_preparation.md](docs/1-data_preparation.md)

---

## Training Overview

Each binding type is trained in three phases (see [Research Hypothesis](#research-hypothesis)).  
A simple four-layer MLP on ESM-2 embeddings is used as the primary model, chosen 
after benchmarking against Bi-LSTM, Bi-GRU, and 1D CNN alternatives - all of 
which underperformed while training 2–6× slower.

Two multi-task variants were also tested: one with a single shared encoder and 
three task-specific output heads (one per binding type), and one with a custom 
batch sampler enforcing equal representation from each binding type per batch. 
Neither variant outperformed the per-task individual models.

---

## Execution Guide

### Training

Replace `ion` / `dna_rna` / `protein` as needed. Each `.py` script has a 
corresponding `.sh` SLURM submission file.

**Phase 1 - Train on structured data only:**
```bash
python src/base_codes/train_ion_phase1.py
python src/base_codes/train_dna_rna_phase1.py
python src/base_codes/train_phase1_esm.py       # protein-protein
```

**Phase 2 - Train on DisProt (IDP) data only:**
```bash
python src/base_codes/train_ion_phase2.py
python src/base_codes/train_dna_rna_phase2.py
python src/base_codes/train_phase2_esm.py
```

**Phase 3 - Hybrid training, validated on DisProt only (main result):**
```bash
python src/training_scripts/train_ion_hybrid_idp_val_optimized.py
python src/training_scripts/train_dna_rna_hybrid_idp_val_optimized.py
python src/training_scripts/train_protein_hybrid_idp_val_optimized.py
```

**Additional experiments (optional):**
```bash
# Single model predicting all three binding types simultaneously
# (two variants: equal task weight per batch vs. natural data proportions)
python src/training_scripts/train_multitask_balanced.py
python src/training_scripts/train_multitask_unified.py

# Two-stage approach: pre-train on structured data, then adapt to DisProt
python src/training_scripts/train_finetune.py
```

> **Note:** Embedding generation must be completed before any training script 
> is run - see [docs/1-data_preparation.md](docs/1-data_preparation.md). All 
> scripts are designed for a SLURM cluster. To run locally, update the data 
> paths in `config.yaml`.

### Evaluation
```bash
python src/evaluate_scripts/evaluate_ion_all_phases.py
python src/evaluate_scripts/evaluate_protein_comparison.py
python src/architecture_tests/architecture_test_ion.py   # architecture comparison
```

### Hyperparameter Tuning
```bash
python src/parameter_testing/grid_search_protein.py
```