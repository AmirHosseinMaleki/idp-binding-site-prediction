import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("/home/malekia/idp-binding-site-prediction/figures", exist_ok=True)

# ── Shared style ────────────────────────────────────────────────────────────
COLORS = {
    "phase1": "#5B9BD5",
    "phase2": "#ED7D31",
    "phase3": "#70AD47",
    "mlp":    "#5B9BD5",
    "bigru":  "#ED7D31",
    "cnn":    "#A5A5A5",
    "bilstm": "#FFC000",
}

def style_ax(ax):
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def add_bar_labels(ax, bars, fmt="{:.3f}", offset=0.004, fontsize=8.5):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color="#333333")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Phase comparison (DisProt AUC)
# ════════════════════════════════════════════════════════════════════════════
binding_types = ["Protein-Protein", "DNA/RNA", "Ion"]
phase1 = [0.7187, 0.6180, 0.8111]
phase2 = [0.8396, 0.6957, 0.7661]
phase3 = [0.8404, 0.7121, 0.8581]

x = np.arange(len(binding_types))
w = 0.24

fig, ax = plt.subplots(figsize=(9, 5.5))

b1 = ax.bar(x - w, phase1, w, label="Phase 1 — Structured only",
            color=COLORS["phase1"], edgecolor="white", linewidth=0.7)
b2 = ax.bar(x,     phase2, w, label="Phase 2 — IDP only",
            color=COLORS["phase2"], edgecolor="white", linewidth=0.7)
b3 = ax.bar(x + w, phase3, w, label="Phase 3 — Hybrid (IDP val.)",
            color=COLORS["phase3"], edgecolor="white", linewidth=0.7)

for bars in [b1, b2, b3]:
    add_bar_labels(ax, bars)

ax.set_ylabel("AUC (DisProt test set)", fontsize=12)
ax.set_title("Phase Comparison — DisProt IDP Test Set AUC",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xticks(x)
ax.set_xticklabels(binding_types, fontsize=11)
ax.set_ylim(0.55, 0.93)
ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc")
style_ax(ax)
plt.tight_layout()
plt.savefig("/home/malekia/idp-binding-site-prediction/figures/phase_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓  figures/phase_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Architecture comparison (AUC + F1, protein-protein task)
# ════════════════════════════════════════════════════════════════════════════
architectures = ["MLP\n(selected)", "Bi-GRU", "1D CNN", "Bi-LSTM"]
auc_vals = [0.8428, 0.8283, 0.8265, 0.8214]
f1_vals  = [0.6517, 0.6323, 0.6292, 0.6234]
bar_colors = [COLORS["mlp"], COLORS["bigru"], COLORS["cnn"], COLORS["bilstm"]]

x = np.arange(len(architectures))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))

b_auc = ax.bar(x - w/2, auc_vals, w, label="AUC",
               color=[c + "cc" for c in bar_colors],   # slight transparency trick via hex
               edgecolor="white", linewidth=0.7)
b_f1  = ax.bar(x + w/2, f1_vals,  w, label="F1",
               color=bar_colors,
               edgecolor="white", linewidth=0.7,
               hatch="////", alpha=0.85)

# Use solid colors — patch the transparency manually
for i, bar in enumerate(b_auc):
    bar.set_facecolor(bar_colors[i])
    bar.set_alpha(0.65)

add_bar_labels(ax, b_auc, offset=0.003)
add_bar_labels(ax, b_f1,  offset=0.003)

# Highlight MLP bars with a border
for bar in list(b_auc)[:1] + list(b_f1)[:1]:
    bar.set_edgecolor("#222222")
    bar.set_linewidth(1.6)
    bar.set_alpha(1.0)

ax.set_ylabel("Score (DisProt test set)", fontsize=12)
ax.set_title("Architecture Comparison — Protein-Protein Binding",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xticks(x)
ax.set_xticklabels(architectures, fontsize=11)
ax.set_ylim(0.58, 0.90)
ax.legend(fontsize=11, framealpha=0.9, edgecolor="#cccccc")
style_ax(ax)

# Annotation pointing out winner
ax.annotate("Best architecture", xy=(0 - w/2, 0.8428), xytext=(-0.35, 0.875),
            fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

plt.tight_layout()
plt.savefig("/home/malekia/idp-binding-site-prediction/figures/architecture_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓  figures/architecture_comparison.png")
print("\nDone. Copy both PNGs into your repo's figures/ folder.")