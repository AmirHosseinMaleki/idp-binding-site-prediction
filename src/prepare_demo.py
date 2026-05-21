"""
Run this from your repo root:
    python prepare_demo.py

It reads your DisProt protein binding test set, picks a demo protein,
and writes:
    demo/demo_protein.fasta
    demo/demo_protein_labels.tsv   <- ground truth for reference

Then run predict.py on the output:
    python predict.py \
        --sequence "$(cat demo/demo_protein.fasta | tail -n1)" \
        --binding_type protein \
        --output demo/expected_output_protein.tsv
"""

import pandas as pd
import os

os.makedirs("demo", exist_ok=True)

# ── Load test set ────────────────────────────────────────────────────────────
# Adjust this path if yours is different
TSV_PATH = "/home/malekia/idp-binding-site-prediction/data/ScanNet/datasets/PPBS/scannet_test.csv"

print(f"Loading: {TSV_PATH}")
df = pd.read_csv(TSV_PATH)
print(f"Total proteins in test set: {len(df)}")

# ── Pick a good demo candidate ───────────────────────────────────────────────
# Criteria:
#   - Sequence length between 80 and 400 (short enough to run fast, long enough to be interesting)
#   - Between 15% and 60% binding residues (clearly visible binding regions)

def positive_rate(row):
    # scannet_test.csv uses 'annotation' column: a string of 0s and 1s e.g. "00110001"
    ann = str(row["annotation"])
    return ann.count("1") / len(ann)

df["seq_len"]  = df["sequence"].str.len()
df["pos_rate"] = df.apply(positive_rate, axis=1)

candidates = df[
    (df["seq_len"]  >= 80)  &
    (df["seq_len"]  <= 400) &
    (df["pos_rate"] >= 0.15) &
    (df["pos_rate"] <= 0.60)
].copy()

print(f"Candidates matching criteria: {len(candidates)}")

if len(candidates) == 0:
    # Relax constraints if nothing matches
    candidates = df[
        (df["seq_len"] >= 50) &
        (df["seq_len"] <= 600)
    ].copy()
    print("Relaxed constraints, candidates:", len(candidates))

# Sort by: prefer moderate length and good positive rate
candidates["score"] = (
    abs(candidates["pos_rate"] - 0.30) +   # penalise deviation from 30% positive
    candidates["seq_len"] / 1000           # slightly prefer shorter
)
best = candidates.sort_values("score").iloc[0]

protein_id = best.get("protein_id", "demo_protein")
sequence   = best["sequence"]
labels     = best["annotation"]
seq_len    = best["seq_len"]
pos_rate   = best["pos_rate"]
n_binding  = int(pos_rate * seq_len)

print(f"\n── Selected protein ─────────────────────────────")
print(f"  ID         : {protein_id}")
print(f"  Length     : {seq_len} residues")
print(f"  Binding    : {n_binding} residues ({pos_rate*100:.1f}%)")
print(f"  Sequence   : {sequence[:60]}...")

# ── Write FASTA ──────────────────────────────────────────────────────────────
fasta_path = "demo/demo_protein.fasta"
with open(fasta_path, "w") as f:
    f.write(f">{protein_id}\n")
    # wrap at 60 chars
    for i in range(0, len(sequence), 60):
        f.write(sequence[i:i+60] + "\n")
print(f"\n✓ Wrote {fasta_path}")

# ── Write ground truth labels (for reference / comparison) ───────────────────
labels_path = "/home/malekia/idp-binding-site-prediction/demo/demo_protein_labels.tsv"
label_list  = [int(x) for x in str(labels)]
with open(labels_path, "w") as f:
    f.write("residue_index\tamino_acid\ttrue_label\n")
    for i, (aa, lbl) in enumerate(zip(sequence, label_list), start=1):
        f.write(f"{i}\t{aa}\t{lbl}\n")
print(f" Wrote {labels_path}  (ground truth - binding=1, non-binding=0)")