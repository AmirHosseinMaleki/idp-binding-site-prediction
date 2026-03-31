# IDP Binding Site Prediction

This research project focuses on predicting protein-protein, DNA/RNA, and ion binding sites within intrinsically disordered protein (IDP) regions. IDPs are flexible protein segments that lack a fixed three-dimensional structure but play crucial roles in cellular interactions, including signaling, regulation, and molecular recognition. The system uses deep learning to identify amino acid residues that participate in binding interactions, addressing a critical gap in understanding IDP functionality.

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Data](#data)
- [Code Architecture](#code-architecture)
- [Execution Guide](#execution-guide)
- [License](#license)

## Overview

The project employs a progressive multi-phase training strategy, starting with high-quality structured datasets and adapting to IDP-specific characteristics. It compares multiple neural architectures (MLP, 1D CNN, Bi-LSTM, Bi-GRU) and implements multi-task learning to capture shared patterns across different binding types. Key innovations include sophisticated class weighting to handle severe imbalance (up to 66:1 negative-to-positive ratio) and hybrid IDP-validation approaches to ensure performance on disordered regions.

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
- **AHoJ-DB**: Ion-protein binding data (used for ion binding sites)
- **BioLiP**: DNA/RNA-protein binding data from PDB (used for nucleic acid binding sites)
- **ScanNet**: Protein-protein interaction data (used for protein binding sites)
- **DisProt**: Intrinsically disordered protein annotations (used for IDP-specific training across all binding types)
- **CAID3**: Official benchmark test set used for final model evaluation and comparison with published methods
- **MMseqs2**: Sequence clustering tool used to prevent train/test overlap (sequences with >10% identity separated across splits)

### Data Preparation
Data preparation involves several steps:

1. **Download raw data** from respective sources (AHoJ-DB, BioLiP, DisProt, ScanNet).

2. **Extract binding sites**:
   - For ions: Parse AHoJ-DB annotations to identify ion-binding residues.
   - For DNA/RNA: Process BioLiP entries to extract nucleic acid contact sites.
   - For proteins: Use ScanNet interaction data.
   - For IDPs: Filter DisProt sequences and annotations.

3. **Generate embeddings**:
   - Use ESM-2 model to create 1280-dimensional embeddings for protein sequences.
   - Scripts: `src/prepare_data/generate_embeddings.py`

4. **Split datasets**:
   - Apply CAID3 clustering to avoid train/test overlap.
   - Create train/val/test splits with balanced IDP representation.

5. **Preprocess for training**:
   - Handle class imbalance with appropriate weighting.
   - Prepare input tensors for different model architectures.

## Code Architecture

The project is organized into the following modules:

### `src/prepare_data/`
- Data preprocessing and embedding generation
- Scripts: `generate_embeddings.py`, `process_biolip.py`, `split_disprot.py`
- Responsibility: Convert raw datasets into training-ready format

### `src/base_codes/`
- Core training scripts for Phase 1-3
- Scripts: `train_ion_phase1.py`, `train_dna_rna_phase2.py`, etc.
- Responsibility: Implement progressive training strategy

### `src/training_scripts/`
- Advanced training approaches
- Scripts: `train_*_hybrid_idp_val.py`, `train_multitask_*.py`
- Responsibility: Multi-task learning and IDP-focused validation

### `src/evaluate_scripts/`
- Model evaluation and comparison
- Scripts: `evaluate_*_all_phases.py`, `evaluate_*_comparison.py`
- Responsibility: Comprehensive performance assessment

### `src/architecture_tests/`
- Architecture benchmarking
- Scripts: `architecture_test_ion.py`, `architecture_test_dna_rna.py`
- Responsibility: Compare MLP, CNN, LSTM, GRU performance

### `src/without_embedding/`
- Alternative sequence-based approach
- Scripts: `train_phase1.py`, `train_phase2.py`, etc.
- Responsibility: Direct sequence processing without pre-computed embeddings

### `src/parameter_testing/`
- Hyperparameter optimization
- Scripts: `grid_search_protein.py`
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
- Ensure data is prepared and embeddings generated before training.
- Models are saved as `.pt` files in the `data/` directory.
- GPU is required for efficient training; adjust batch sizes for memory constraints.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
