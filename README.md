# IDP Binding Site Prediction

This research project focuses on predicting protein-protein, DNA/RNA, and ion binding sites within intrinsically disordered protein (IDP) regions. IDPs are flexible protein segments that lack a fixed three-dimensional structure but play crucial roles in cellular interactions, including signaling, regulation, and molecular recognition. The system uses deep learning to identify amino acid residues that participate in binding interactions, addressing a critical gap in understanding IDP functionality.

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Demo](#demo)
- [Documentation](#documentation)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Data](#data)
- [Code Architecture](#code-architecture)
- [Execution Guide](#execution-guide)
- [License](#license)


## Overview

![Binding site prediction on a disordered protein sequence](figures/binding-site.svg)

IDPs lack stable 3D structure, which means structure-based binding site predictors cannot be applied to them. Existing sequence-based models are typically trained on structured proteins, leaving a gap in IDP-specific prediction. This project investigates whether that gap can be closed by **hybrid training** - combining large structured-protein binding databases (AHoJ-DB, BioLiP, ScanNet) with IDP-specific annotations from DisProt, while using DisProt-only validation to ensure the saved model performs best on disordered regions.

Three binding types are addressed: protein-protein, DNA/RNA, and ion binding. For each, models are trained across three phases - structured data only, IDP data only, and hybrid - and evaluated on a held-out DisProt test set to measure IDP prediction quality directly.

The core finding is that hybrid training consistently outperforms both single-source alternatives: it matches structured-only performance on structured test sets while achieving significantly better IDP prediction than IDP-only training, which is limited by DisProt's small size.

[ESM-2](https://github.com/facebookresearch/esm) protein language model embeddings (1280-dimensional, per-residue) are used as input features, chosen because they operate purely on sequence and thus apply equally to structured and disordered proteins. Architecture and hyperparameter experiments were conducted to confirm the chosen MLP setup is well-suited for this input type and to rule out that results were an artefact of a suboptimal model.

## Results

Best model performance on the DisProt IDP test set (hybrid training, MLP, optimized hyperparameters):

| Binding Type    | AUC    | AUPRC  | MCC    | F1     |
|-----------------|--------|--------|--------|--------|
| Protein-Protein | 0.8394 | 0.6214 | 0.4986 | 0.6460 |
| DNA/RNA         | 0.7126 | 0.5548 | 0.3618 | 0.5881 |
| Ion             | 0.8487 | 0.5931 | 0.4898 | 0.6135 |

Full results including phase comparison, architecture benchmarks, and hyperparameter tuning analysis: [docs/results_summary.md](docs/results_summary.md)

## Demo

Run a prediction on a single protein sequence:
```bash
python predict.py \
  --sequence "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQASHLGGAVVGGSNNQQNYPPQGSTSNSTYGSSRNMQDIVPNDSRSRPQHSMSRHNPQNSSSTFAFAQNHFQSSDAPGATNSSSNSSTNNNSSSVSGSGRNMQDIVPNDS" \
  --binding_type protein \
  --output results/prediction.tsv
```

This sequence is human alpha-synuclein (UniProt P37840), a well-characterized IDP. The model predicts per-residue binding probabilities.

> **Note:** Pre-trained model weights and a full visualization example are provided in [demo/](demo/).

## Documentation

| Document | Description |
|----------|-------------|
| [Data Preparation](docs/data_preparation.md) | Step-by-step pipeline for all three binding types |
| [Code Architecture](docs/code_architecture.md) | Detailed module descriptions and class references |
| [Results Summary](docs/results_summary.md) | Formal results document following project specification |

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training; CPU possible but slow)
- 16GB+ RAM
- 50GB+ disk space for datasets and models

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/idp-binding-site-prediction.git
   cd idp-binding-site-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv biotite_venv
   source biotite_venv/bin/activate  # On Windows: biotite_venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA 12.8+ (RTX 30-series or newer recommended)
- **CPU**: Multi-core processor for data preprocessing
## Quick Start

1. Follow the [Environment Setup](#environment-setup) above.

2. Download the pre-trained model weights (see [Data](#data) for links).

3. Run a prediction on a protein sequence:
```bash
   python predict.py --sequence "MSEQNNTEMTFQIQRIYTKDI..." --binding_type ion
```
   This outputs per-residue binding probabilities for the input sequence.

> **Note:** A full demo with an example sequence and expected output is provided in [Demo](#demo).

## Data

The project uses multiple datasets for training and evaluation. Data files are currently not publicly available but can be prepared from the following sources.

### Dataset Sources
- **[AHoJ-DB](https://apoholo.cz/db/archive)**: Ion-protein binding data (used for ion binding sites)
- **[BioLiP](https://aideepmed.com/BioLiP/download.html)**: DNA/RNA-protein binding data from PDB (used for nucleic acid binding sites)
- **[ScanNet](https://github.com/jertubiana/ScanNet/tree/main/datasets)**: Protein-protein interaction data (used for protein binding sites)
- **[DisProt](https://disprot.org/download)**: Intrinsically disordered protein annotations (used for IDP-specific training across all binding types)
- **UniProt**: Protein sequences retrieved programmatically via the [UniProt REST API](https://rest.uniprot.org/) using `src/idp_old/download_sequences.py`
- **[CAID3](https://caid.idpcentral.org/)**: Official benchmark test set used for final model evaluation and comparison with published methods
- **[MMseqs2](https://github.com/soedinglab/MMseqs2)**: Sequence clustering tool used to prevent train/test overlap (sequences with >10% identity separated across splits)

### Data Preparation
Full step-by-step pipeline for all three binding types: [docs/data_preparation.md](docs/data_preparation.md)

## Code Architecture

The project is organized into the following top-level modules:

| Module | Responsibility |
|--------|---------------|
| `src/prepare_data/` | Data preprocessing, clustering, and embedding generation |
| `src/staticp_old/` | AHoJ-DB ion binding site extraction pipeline |
| `src/idp_old/` | DisProt data collection and UniProt sequence download |
| `src/other_codes/` | Ion dataset utilities and clustering |
| `src/base_codes/` | Phase 1–3 training scripts and SLURM submission files |
| `src/training_scripts/` | Hybrid and multi-task training |
| `src/architecture_tests/` | MLP vs CNN vs LSTM vs GRU comparison |
| `src/parameter_testing/` | Hyperparameter grid search |
| `src/optimal_epoch_testing/` | Epoch count optimisation |
| `src/evaluate_scripts/` | Model evaluation and phase comparison |
| `src/without_embedding/` | Sequence-only baseline (no ESM-2) |

Full module descriptions, class references, and data flow diagram: [docs/code_architecture.md](docs/code_architecture.md)

## Execution Guide

### Training Models

#### Phase 1: Structured Datasets
Train baseline models on high-quality curated data:
```bash
# Ion binding
python src/base_codes/train_ion_phase1.py

# DNA/RNA binding
python src/base_codes/train_dna_rna_phase1.py

# Protein binding
python src/base_codes/train_protein_phase1.py
```

#### Phase 2: IDP Expansion
Fine-tune on DisProt for IDP characteristics:
```bash
# Continue from Phase 1 models
python src/base_codes/train_ion_phase2.py
python src/base_codes/train_dna_rna_phase2.py
python src/base_codes/train_protein_phase2.py
```

#### Phase 3: Hybrid Optimization
Combine all data with IDP validation:
```bash
python src/base_codes/train_ion_phase3.py
python src/base_codes/train_dna_rna_phase3.py
python src/base_codes/train_protein_phase3.py
```

#### Advanced Training
- **Multi-task learning (unified)**: `python src/training_scripts/train_multitask_unified.py`
- **Multi-task learning (balanced)**: `python src/training_scripts/train_multitask_balanced.py`
- **Hybrid IDP validation**: `python src/training_scripts/train_ion_hybrid_idp_val.py`
- **Fine-tuning**: `python src/training_scripts/train_finetune.py`

### Evaluating Models
Run comprehensive evaluations:
```bash
# Evaluate all phases for ion binding
python src/evaluate_scripts/evaluate_ion_all_phases.py

# Compare architectures
python src/architecture_tests/architecture_test_ion.py

# Performance comparison
python src/evaluate_scripts/evaluate_ion_comparison.py
```

### Hyperparameter Tuning
```bash
# Grid search for protein binding
python src/parameter_testing/grid_search_protein.py
```

### Notes
- Ensure data is prepared and embeddings generated before running any training script. See [docs/data_preparation.md](docs/data_preparation.md).
- All training scripts are designed for a SLURM cluster. Each `.py` script has a corresponding `.sh` submission file. Adjust paths if running locally.
- Models are saved as `.pt` files in the `data/` directory. Pre-trained weights can be downloaded from [link to be added].
- GPU is required for efficient training; adjust batch sizes for memory constraints.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
