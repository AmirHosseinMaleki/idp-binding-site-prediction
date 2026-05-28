"""
predict_caid.py - CAID submission entry point.

Reads a multi-sequence FASTA, loads pre-computed ESM-2 embeddings from a
folder (one file per protein), runs all three binding-type predictors, and
writes per-protein .caid files plus a timings.csv.

Usage:
    python predict_caid.py \
        --fasta sequences.fasta \
        --embeddings_dir /path/to/embeddings \
        --output_dir /path/to/output \
        --model_dir data/

CAID will mount the embeddings folder at runtime.  The folder must contain
one file per protein named <protein_id>.npy or <protein_id>.h5, where
<protein_id> matches the header in the FASTA file (e.g. ">P04637" → "P04637.npy").

Output structure:
    <output_dir>/
        protein/
            P04637.caid
            Q15648.caid
            ...
        dna_rna/
            P04637.caid
            ...
        ion/
            P04637.caid
            ...
        timings.csv

.caid file format (per CAID specification):
    >P04637
    1	M	0.8921	1
    2	E	0.8134	1
    ...

timings.csv format (per CAID specification):
    # Running predictor, started <timestamp>
    sequence,milliseconds
    P04637,1234
    ...
"""

import argparse
import os
import sys
import time
import datetime

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
# Config - all three binding types
# ─────────────────────────────────────────────────────────────────────────────

BINDING_CONFIG = {
    "protein": {
        "model_file": "protein_hybrid_idpval_model.pt",
        "threshold":  0.60,
    },
    "dna_rna": {
        "model_file": "dna_rna_hybrid_idpval_model.pt",
        "threshold":  0.40,
    },
    "ion": {
        "model_file": "ion_hybrid_idpval_model.pt",
        "threshold":  0.15,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FASTA reader - handles multiple sequences
# ─────────────────────────────────────────────────────────────────────────────

def read_fasta(path):
    """
    Read all sequences from a FASTA file.
    Returns list of (protein_id, sequence) tuples.
    Protein ID is everything after '>' up to the first whitespace.
    """
    proteins = []
    current_id, current_seq = None, []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    proteins.append((current_id, "".join(current_seq)))
                # take only the first word as the ID (e.g. ">P04637 human p53" → "P04637")
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())

    if current_id is not None:
        proteins.append((current_id, "".join(current_seq)))

    return proteins


# ─────────────────────────────────────────────────────────────────────────────
# Sequence normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_sequence(seq):
    """
    Handle IUPAC ambiguous residue codes.
    B, Z, U, O, X are in ESM-2's vocabulary natively.
    J (Leu/Ile) is not - map to L (standard convention).
    """
    seq = seq.upper().strip()
    if "J" in seq:
        seq = seq.replace("J", "L")
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# Embedding loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_npy(path):
    emb = np.load(path)
    if emb.ndim != 2 or emb.shape[1] != 1280:
        raise ValueError(f"Expected shape (L, 1280), got {emb.shape}")
    return emb.astype(np.float32)


def load_h5(path):
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py is required for .h5 files.  pip install h5py")
        sys.exit(1)
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if len(keys) == 0:
            raise ValueError(f"No datasets found in {path}")
        # try common key names, fall back to first available
        for candidate in ["representations", "embeddings", keys[0]]:
            if candidate in f:
                emb = f[candidate][()]
                break
    if emb.ndim != 2 or emb.shape[1] != 1280:
        raise ValueError(f"Expected shape (L, 1280), got {emb.shape}")
    return emb.astype(np.float32)


def find_embedding(embeddings_dir, protein_id):
    """
    Look for <protein_id>.npy or <protein_id>.h5 in embeddings_dir.
    Returns (path, format) or raises FileNotFoundError.
    """
    for ext in (".npy", ".h5", ".hdf5"):
        path = os.path.join(embeddings_dir, protein_id + ext)
        if os.path.exists(path):
            return path, ext
    raise FileNotFoundError(
        f"No embedding found for '{protein_id}' in {embeddings_dir}. "
        f"Expected: {protein_id}.npy or {protein_id}.h5"
    )


def load_embedding(embeddings_dir, protein_id):
    path, ext = find_embedding(embeddings_dir, protein_id)
    if ext == ".npy":
        return load_npy(path)
    else:
        return load_h5(path)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction(embeddings, model, threshold, device, batch_size=2048):
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
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def write_caid(output_path, protein_id, sequence, scores, binary):
    """
    Write a .caid file in the CAID-specified format:
        >P04637
        1	M	0.8921	1
        2	E	0.8134	1
        ...
    """
    with open(output_path, "w") as f:
        f.write(f">{protein_id}\n")
        for i, (aa, score, label) in enumerate(zip(sequence, scores, binary), start=1):
            f.write(f"{i}\t{aa}\t{score:.4f}\t{label}\n")


def write_timings(output_path, timings, start_time):
    """
    Write timings.csv in the CAID-specified format.
    timings: list of (protein_id, milliseconds)
    """
    timestamp = datetime.datetime.fromtimestamp(start_time).strftime(
        "%a %b %d %H:%M:%S %Z %Y"
    )
    with open(output_path, "w") as f:
        f.write(f"# Running predictor, started {timestamp}\n")
        f.write("sequence,milliseconds\n")
        for protein_id, ms in timings:
            f.write(f"{protein_id},{ms}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "IDP binding site predictor - CAID submission interface.\n"
            "Processes all sequences in a FASTA file and writes per-protein\n"
            ".caid files for protein, dna_rna, and ion binding."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fasta", required=True,
        help="Input FASTA file containing one or more protein sequences.",
    )
    parser.add_argument(
        "--embeddings_dir", required=True,
        help=(
            "Directory containing pre-computed ESM-2 embeddings. "
            "One file per protein, named <protein_id>.npy or <protein_id>.h5."
        ),
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Directory to write .caid output files and timings.csv (default: output/).",
    )
    parser.add_argument(
        "--model_dir", default="data",
        help="Directory containing trained .pt weight files (default: data/).",
    )

    args = parser.parse_args()

    # ── validate model files exist ────────────────────────────────────────────
    for binding_type, cfg in BINDING_CONFIG.items():
        model_path = os.path.join(args.model_dir, cfg["model_file"])
        if not os.path.exists(model_path):
            print(f"ERROR: model not found: {os.path.abspath(model_path)}")
            print("Use --model_dir to point to the directory containing .pt files.")
            sys.exit(1)

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load all three models once ────────────────────────────────────────────
    print("Loading models...")
    models = {}
    for binding_type, cfg in BINDING_CONFIG.items():
        model_path = os.path.join(args.model_dir, cfg["model_file"])
        m = BindingNet().to(device)
        m.load_state_dict(torch.load(model_path, map_location=device))
        m.eval()
        models[binding_type] = m
        print(f"  {binding_type}: {model_path}")

    # ── create output directories ─────────────────────────────────────────────
    for binding_type in BINDING_CONFIG:
        os.makedirs(os.path.join(args.output_dir, binding_type), exist_ok=True)

    # ── read FASTA ────────────────────────────────────────────────────────────
    print(f"\nReading FASTA: {args.fasta}")
    proteins = read_fasta(args.fasta)
    print(f"  {len(proteins)} sequence(s) found.")

    # ── process each protein ──────────────────────────────────────────────────
    timings = []
    run_start = time.time()

    for protein_id, raw_sequence in proteins:
        sequence = normalise_sequence(raw_sequence)
        print(f"\n[{protein_id}] length={len(sequence)}")

        t_start = time.time()

        # load embedding
        try:
            embeddings = load_embedding(args.embeddings_dir, protein_id)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} - skipping.")
            continue
        except ValueError as e:
            print(f"  WARNING: bad embedding for {protein_id}: {e} - skipping.")
            continue

        # sanity check: embedding length should match sequence length
        if len(embeddings) != len(sequence):
            print(
                f"  WARNING: embedding length ({len(embeddings)}) != "
                f"sequence length ({len(sequence)}) - skipping."
            )
            continue

        # run all three predictors and write output
        for binding_type, cfg in BINDING_CONFIG.items():
            scores, binary = run_prediction(
                embeddings, models[binding_type], cfg["threshold"], device
            )
            out_path = os.path.join(args.output_dir, binding_type, f"{protein_id}.caid")
            write_caid(out_path, protein_id, sequence, scores, binary)
            n_binding = int(binary.sum())
            print(f"  {binding_type}: {n_binding}/{len(sequence)} binding → {out_path}")

        elapsed_ms = int((time.time() - t_start) * 1000)
        timings.append((protein_id, elapsed_ms))
        print(f"  Time: {elapsed_ms} ms")

    # ── write timings ─────────────────────────────────────────────────────────
    timings_path = os.path.join(args.output_dir, "timings.csv")
    write_timings(timings_path, timings, run_start)
    print(f"\nTimings written to: {timings_path}")
    print(f"Done. Processed {len(timings)}/{len(proteins)} sequences.")


if __name__ == "__main__":
    main()