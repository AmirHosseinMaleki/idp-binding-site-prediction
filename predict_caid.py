"""
predict_caid.py — CAID submission entry point.

CAID will pre-compute ESM-2 embeddings and call this script directly.
No internet access, no ESM-2, no GPU required.

Usage:
    python predict_caid.py \
        --embedding_file embeddings.npy \
        --binding_type protein \
        --output prediction.tsv

    python predict_caid.py \
        --embedding_file embeddings.h5 \
        --binding_type ion \
        --output prediction.tsv \
        --h5_key representations

Arguments:
    --embedding_file  Pre-computed ESM-2 embeddings (.npy or .h5), shape (L, 1280)  [required]
    --binding_type    One of: protein, dna_rna, ion                                  [required]
    --output          Output TSV path (default: prediction.tsv)
    --model_dir       Directory containing trained .pt files (default: models/)
    --threshold       Override the default decision threshold (optional)
    --h5_key          Dataset key inside an HDF5 file (default: representations)

Output TSV columns:
    position    1-indexed residue number
    score       binding probability (0.0 – 1.0)
    prediction  binary call at the chosen threshold (0 or 1)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class BindingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BINDING_CONFIG = {
    "protein": {"model_file": "protein_hybrid_idpval_model.pt", "threshold": 0.60},
    "dna_rna": {"model_file": "dna_rna_hybrid_idpval_model.pt", "threshold": 0.40},
    "ion":     {"model_file": "ion_hybrid_idpval_model.pt",     "threshold": 0.15},
}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_npy(path):
    emb = np.load(path)
    if emb.ndim != 2 or emb.shape[1] != 1280:
        raise ValueError(f"Expected shape (L, 1280), got {emb.shape}")
    return emb.astype(np.float32)


def load_h5(path, key):
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py is required for .h5 files.  pip install h5py")
        sys.exit(1)
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not in {path}. Available: {list(f.keys())}")
        emb = f[key][()]
    if emb.ndim != 2 or emb.shape[1] != 1280:
        raise ValueError(f"Expected shape (L, 1280), got {emb.shape}")
    return emb.astype(np.float32)


def load_embeddings(path, h5_key):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return load_npy(path)
    elif ext in (".h5", ".hdf5"):
        return load_h5(path, h5_key)
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Use .npy or .h5")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(embeddings, model, threshold, device, batch_size=2048):
    model.eval()
    parts = []
    t = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(t), batch_size):
            logits = model(t[i: i + batch_size].to(device))
            parts.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(parts)
    return scores, (scores >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def read_fasta(path):
    """Return (header, sequence) for the first entry in a FASTA file."""
    header, seq_lines = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:]
            else:
                seq_lines.append(line)
    if header is None:
        raise ValueError(f"No FASTA entries found in {path}")
    return header, "".join(seq_lines)


def normalise_sequence(seq):
    """
    Handle IUPAC ambiguous residue codes.
    B, Z, U, O, X are in ESM-2's vocabulary natively.
    J (Leu/Ile) is not — map to L (standard convention).
    """
    seq = seq.upper().strip()
    if "J" in seq:
        n_j = seq.count("J")
        seq = seq.replace("J", "L")
        print(f"  Note: replaced {n_j} J residue(s) with L (Leu/Ile ambiguity)")
    return seq


def main():
    parser = argparse.ArgumentParser(
        description="IDP binding site predictor — CAID submission interface."
    )
    parser.add_argument("--embedding_file", required=True,
                        help="Pre-computed ESM-2 embeddings (.npy or .h5, shape L×1280).")
    parser.add_argument("--fasta", default=None,
                        help="Optional FASTA file. Used only to annotate output residues.")
    parser.add_argument("--binding_type", required=True, choices=["protein", "dna_rna", "ion"])
    parser.add_argument("--output",    default="prediction.tsv")
    parser.add_argument("--model_dir", default="data",
                        help="Directory containing .pt weight files (default: data/).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the default decision threshold.")
    parser.add_argument("--h5_key",   default="representations",
                        help="Dataset key inside an HDF5 file (default: representations).")
    args = parser.parse_args()

    cfg       = BINDING_CONFIG[args.binding_type]
    threshold = args.threshold if args.threshold is not None else cfg["threshold"]
    model_path = os.path.join(args.model_dir, cfg["model_file"])

    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {os.path.abspath(model_path)}")
        print("Use --model_dir to point to the directory containing .pt files.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── optional sequence ─────────────────────────────────────────────────────
    sequence = None
    if args.fasta:
        _, raw = read_fasta(args.fasta)
        sequence = normalise_sequence(raw)
        print(f"Sequence     : {len(sequence)} residues (from {args.fasta})")

    print(f"Binding type : {args.binding_type}")
    print(f"Model        : {model_path}")
    print(f"Threshold    : {threshold}")
    print(f"Device       : {device}")

    print(f"\nLoading embeddings: {args.embedding_file}")
    try:
        embeddings = load_embeddings(args.embedding_file, args.h5_key)
    except (ValueError, KeyError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"  Shape: {embeddings.shape}")

    # warn if sequence and embedding lengths disagree
    if sequence is not None and len(sequence) != len(embeddings):
        print(f"WARNING: sequence length ({len(sequence)}) != embedding length "
              f"({len(embeddings)}). Residue annotation will be skipped.")
        sequence = None

    model = BindingNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    scores, binary = predict(embeddings, model, threshold, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        if sequence is not None:
            f.write("position\tresidue\tscore\tprediction\n")
            for i, (aa, s, b) in enumerate(zip(sequence, scores, binary), start=1):
                f.write(f"{i}\t{aa}\t{s:.4f}\t{b}\n")
        else:
            f.write("position\tscore\tprediction\n")
            for i, (s, b) in enumerate(zip(scores, binary), start=1):
                f.write(f"{i}\t{s:.4f}\t{b}\n")

    print(f"\nDone. {int(binary.sum())}/{len(scores)} residues predicted as binding.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()