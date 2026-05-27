"""
predict.py — Per-residue binding site prediction on a single protein sequence.

Usage:
    python predict.py --sequence "MSEQNN..." --binding_type protein --output results/prediction.tsv
    python predict.py --fasta demo/demo_protein.fasta --binding_type ion --output results/prediction.tsv

Arguments:
    --sequence      Amino acid sequence string (use this OR --fasta)
    --fasta         Path to a FASTA file (use this OR --sequence)
    --binding_type  One of: protein, dna_rna, ion
    --output        Path to write the TSV output (default: prediction.tsv)
    --model_dir     Directory containing .pt model files (default: data/)
    --threshold     Override the default decision threshold (optional)

Output TSV columns:
    position    1-indexed residue position
    residue     amino acid letter
    score       binding probability (0–1)
    prediction  binary label at the decision threshold (0 or 1)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Model — must match trained weights exactly
# ─────────────────────────────────────────────────────────────────────────────

class BindingNet(nn.Module):
    """Four-layer MLP on 1280-dim ESM-2 per-residue embeddings."""
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
# Per-binding-type configuration
# ─────────────────────────────────────────────────────────────────────────────

BINDING_CONFIG = {
    "protein": {
        "model_file": "protein_hybrid_idpval_model.pt",
        "threshold":  0.60,
        "description": "Protein-protein binding",
    },
    "dna_rna": {
        "model_file": "dna_rna_hybrid_idpval_model.pt",
        "threshold":  0.40,
        "description": "DNA/RNA binding",
    },
    "ion": {
        "model_file": "ion_hybrid_idpval_model.pt",
        "threshold":  0.15,
        "description": "Ion binding",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FASTA reader
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
                    break          # only read the first entry
                header = line[1:]
            else:
                seq_lines.append(line)
    if header is None:
        raise ValueError(f"No FASTA entries found in {path}")
    return header, "".join(seq_lines)


# ─────────────────────────────────────────────────────────────────────────────
# ESM-2 embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_sequence(sequence, device, max_len=2000):
    """
    Generate per-residue ESM-2 embeddings.
    Returns numpy array of shape (L, 1280).
    """
    try:
        import esm as esm_lib
    except ImportError:
        print("ERROR: the 'esm' package is not installed.")
        print("Install it with:  pip install fair-esm")
        sys.exit(1)

    if len(sequence) > max_len:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds maximum {max_len}. "
            "Longer sequences are skipped during training and are not supported."
        )

    print("Loading ESM-2 model (esm2_t33_650M_UR50D)...")
    esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval().to(device)
    print(f"  ESM-2 loaded on {device}")

    data = [("protein", sequence)]
    with torch.no_grad():
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        results = esm_model(tokens, repr_layers=[33])
        # Strip BOS and EOS tokens → shape (L, 1280)
        embeddings = results["representations"][33][0, 1:-1, :].cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(embeddings, model, threshold, device, batch_size=2048):
    """
    Run embeddings through BindingNet in batches.
    Returns (scores, binary_predictions) as numpy arrays of shape (L,).
    """
    model.eval()
    scores_list = []
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, len(emb_tensor), batch_size):
            batch = emb_tensor[start : start + batch_size].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            scores_list.append(probs)

    scores = np.concatenate(scores_list)
    binary = (scores >= threshold).astype(int)
    return scores, binary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict per-residue binding sites for a single protein sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sequence", type=str,
        help="Amino acid sequence string.",
    )
    input_group.add_argument(
        "--fasta", type=str,
        help="Path to a FASTA file (first entry is used).",
    )

    parser.add_argument(
        "--binding_type", required=True,
        choices=["protein", "dna_rna", "ion"],
        help="Binding type to predict: protein, dna_rna, or ion.",
    )
    parser.add_argument(
        "--output", default="prediction.tsv",
        help="Output TSV file path (default: prediction.tsv).",
    )
    parser.add_argument(
        "--model_dir", default="data",
        help="Directory containing model .pt files (default: data/).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the default decision threshold (optional).",
    )

    args = parser.parse_args()

    # ── Get sequence ──────────────────────────────────────────────────────────
    if args.fasta:
        print(f"Reading sequence from: {args.fasta}")
        protein_id, sequence = read_fasta(args.fasta)
        print(f"  Protein: {protein_id}")
    else:
        sequence = args.sequence
        protein_id = "input_sequence"

    sequence = sequence.upper().strip()

    # J (Leu/Ile) is not in ESM-2's vocabulary — map to L.
    # All other IUPAC ambiguous codes (B, Z, U, O, X) are handled natively.
    if "J" in sequence:
        n_j = sequence.count("J")
        sequence = sequence.replace("J", "L")
        print(f"  Note: replaced {n_j} J residue(s) with L (Leu/Ile ambiguity)")

    print(f"  Length:  {len(sequence)} residues")
    print(f"  Length:  {len(sequence)} residues")

    # ── Binding type config ───────────────────────────────────────────────────
    cfg = BINDING_CONFIG[args.binding_type]
    threshold = args.threshold if args.threshold is not None else cfg["threshold"]

    model_path = os.path.join(args.model_dir, cfg["model_file"])
    if not os.path.exists(model_path):
        print(f"\nERROR: model file not found: {model_path}")
        print(f"Expected: {model_path}")
        print("Make sure model weights are in the data/ directory.")
        sys.exit(1)

    print(f"\nBinding type : {cfg['description']}")
    print(f"Model file   : {model_path}")
    print(f"Threshold    : {threshold}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("\nGenerating ESM-2 embeddings...")
    try:
        embeddings = embed_sequence(sequence, device)
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print(f"  Embeddings shape: {embeddings.shape}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading binding model...")
    model = BindingNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("  Model loaded.")

    # ── Predict ───────────────────────────────────────────────────────────────
    print("\nRunning prediction...")
    scores, binary = predict(embeddings, model, threshold, device)

    n_binding = binary.sum()
    print(f"  Predicted binding residues: {n_binding} / {len(sequence)} ({100*n_binding/len(sequence):.1f}%)")

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w") as f:
        f.write("position\tresidue\tscore\tprediction\n")
        for pos, (aa, score, label) in enumerate(zip(sequence, scores, binary), start=1):
            f.write(f"{pos}\t{aa}\t{score:.4f}\t{label}\n")

    print(f"\nOutput written to: {args.output}")
    print("Columns: position, residue, score (0-1), prediction (0/1 at threshold)")


if __name__ == "__main__":
    main()