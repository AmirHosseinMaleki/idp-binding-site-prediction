"""
predict_caid.py — Generate CAID-format binding site predictions

Usage:
    python predict_caid.py --fasta caid_test.fasta --model_dir /path/to/models/ --output_dir predictions/

Input:
    FASTA file with protein sequences (DisProt IDs as headers, e.g. >DP00001)

Output:
    Three .caid files (one per binding type) in the output directory:
        predictions/binding-protein/YourMethod.caid
        predictions/binding-dna_rna/YourMethod.caid
        predictions/binding-ion/YourMethod.caid

CAID format per sequence block:
    >DP00001
    1    M    0.234    0
    2    S    0.891    1
    ...

Columns: position (1-indexed), residue, score (0-1), binary label (0 or 1)
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import esm

from src.utils.config import load_config, get_model_path, base_parser
# ─────────────────────────────────────────────────────────────────────────────
# Model architecture (must match the trained weights exactly)
# ─────────────────────────────────────────────────────────────────────────────

class BindingNet(nn.Module):
    """Four-layer MLP operating on 1280-dim ESM-2 embeddings."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# FASTA reader
# ─────────────────────────────────────────────────────────────────────────────

def read_fasta(fasta_path):
    """
    Returns a list of (header, sequence) tuples.
    Header is the raw line without '>'.
    """
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:]  # strip '>'
                current_seq = []
            else:
                current_seq.append(line)

    if current_header is not None:
        sequences.append((current_header, "".join(current_seq)))

    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# ESM-2 embedding
# ─────────────────────────────────────────────────────────────────────────────

def load_esm2(device):
    """Load ESM-2 650M model and batch converter."""
    print("Loading ESM-2 model (esm2_t33_650M_UR50D)...")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    esm_model = esm_model.to(device)
    print(f"  ESM-2 loaded on {device}")
    return esm_model, batch_converter


def embed_sequence(sequence, esm_model, batch_converter, device, max_len=2000):
    """
    Returns per-residue embeddings as a numpy array of shape (L, 1280).
    Returns None if the sequence is too long.
    """
    if len(sequence) > max_len:
        return None

    data = [("protein", sequence)]
    with torch.no_grad():
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        results = esm_model(tokens, repr_layers=[33])
        # Shape: (1, L+2, 1280) → strip BOS/EOS → (L, 1280)
        embeddings = results["representations"][33][0, 1:-1, :].cpu().numpy()
    torch.cuda.empty_cache()
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_binding(embeddings, model, threshold, device, batch_size=2048):
    """
    Run embeddings through BindingNet in batches.
    Returns (scores, binary_labels) as numpy arrays of shape (L,).
    """
    model.eval()
    all_scores = []

    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    n = len(emb_tensor)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = emb_tensor[start : start + batch_size].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)

    scores = np.concatenate(all_scores)
    binary = (scores >= threshold).astype(int)
    return scores, binary


# ─────────────────────────────────────────────────────────────────────────────
# CAID output writer
# ─────────────────────────────────────────────────────────────────────────────

def write_caid_block(f, header, sequence, scores, binary_labels):
    """
    Write one protein block in CAID format to an already-open file.

    >header
    1    M    0.234    0
    2    S    0.891    1
    ...
    """
    f.write(f">{header}\n")
    for pos, (aa, score, label) in enumerate(zip(sequence, scores, binary_labels), start=1):
        f.write(f"{pos}\t{aa}\t{score:.3f}\t{label}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate CAID-format binding site predictions for all three binding types."
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to input FASTA file with protein sequences.",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help=(
            "Directory containing the three model weight files:\n"
            "  protein_hybrid_idpval_model.pt\n"
            "  dna_rna_hybrid_idpval_model.pt\n"
            "  ion_hybrid_idpval_model.pt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="predictions",
        help="Directory where CAID output files will be written (default: predictions/).",
    )
    parser.add_argument(
        "--method_name",
        default="HybridIDP",
        help="Name used for output .caid filenames (default: HybridIDP).",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2000,
        help="Maximum sequence length; longer sequences are skipped (default: 2000).",
    )
    # Thresholds from results_summary.md (optimised on DisProt validation set)
    parser.add_argument("--threshold_protein", type=float, default=0.60)
    parser.add_argument("--threshold_dna_rna",  type=float, default=0.40)
    parser.add_argument("--threshold_ion",       type=float, default=0.15)

    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Read FASTA ────────────────────────────────────────────────────────────
    print(f"\nReading sequences from: {args.fasta}")
    sequences = read_fasta(args.fasta)
    print(f"  Found {len(sequences)} sequences")

    # ── Load ESM-2 ────────────────────────────────────────────────────────────
    esm_model, batch_converter = load_esm2(device)

    # ── Load binding models ───────────────────────────────────────────────────
    models_config = [
        {
            "name":       "protein",
            "filename":   "protein_hybrid_idpval_model.pt",
            "threshold":  args.threshold_protein,
            "subdir":     "binding-protein",
        },
        {
            "name":       "dna_rna",
            "filename":   "dna_rna_hybrid_idpval_model.pt",
            "threshold":  args.threshold_dna_rna,
            "subdir":     "binding-dna_rna",
        },
        {
            "name":       "ion",
            "filename":   "ion_hybrid_idpval_model.pt",
            "threshold":  args.threshold_ion,
            "subdir":     "binding-ion",
        },
    ]

    print("\nLoading binding models...")
    for cfg in models_config:
        path = os.path.join(args.model_dir, cfg["filename"])
        if not os.path.exists(path):
            print(f"  ERROR: model file not found: {path}")
            sys.exit(1)
        model = BindingNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        cfg["model"] = model
        print(f"  Loaded {cfg['name']} model  (threshold={cfg['threshold']})")

    # ── Create output directories and open output files ───────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_files = {}
    for cfg in models_config:
        subdir = os.path.join(args.output_dir, cfg["subdir"])
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"{args.method_name}.caid")
        output_files[cfg["name"]] = open(out_path, "w")
        print(f"  Output: {out_path}")

    # ── Process sequences ─────────────────────────────────────────────────────
    print(f"\nProcessing {len(sequences)} sequences...")
    skipped = 0

    for i, (header, sequence) in enumerate(sequences, start=1):
        print(f"  [{i}/{len(sequences)}] {header}  (len={len(sequence)})", end="")

        if len(sequence) > args.max_len:
            print(f"  — SKIPPED (>{args.max_len} residues)")
            skipped += 1
            continue

        # Generate ESM-2 embeddings
        try:
            embeddings = embed_sequence(
                sequence, esm_model, batch_converter, device, args.max_len
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  — SKIPPED (GPU OOM)")
                skipped += 1
                continue
            raise

        if embeddings is None:
            print(f"  — SKIPPED (embedding failed)")
            skipped += 1
            continue

        # Run all three models and write output
        for cfg in models_config:
            scores, binary = predict_binding(
                embeddings, cfg["model"], cfg["threshold"], device
            )
            write_caid_block(
                output_files[cfg["name"]], header, sequence, scores, binary
            )

        # Count predicted binding residues for a quick sanity check
        prot_scores, prot_bin = predict_binding(
            embeddings, models_config[0]["model"],
            models_config[0]["threshold"], device
        )
        print(f"  — done  (protein: {prot_bin.sum()}/{len(sequence)} predicted binding)")

    # ── Close files ───────────────────────────────────────────────────────────
    for f in output_files.values():
        f.close()

    print(f"\nDone.")
    print(f"  Processed: {len(sequences) - skipped}")
    print(f"  Skipped:   {skipped}")
    print(f"\nOutput files:")
    for cfg in models_config:
        subdir = os.path.join(args.output_dir, cfg["subdir"])
        out_path = os.path.join(subdir, f"{args.method_name}.caid")
        print(f"  {out_path}")


if __name__ == "__main__":
    main()