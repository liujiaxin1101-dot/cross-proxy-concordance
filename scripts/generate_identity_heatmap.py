"""
generate_identity_heatmap.py — Generate subject × specification concordance heatmap.
Output: Supplementary Fig. SX — Identity fragility across 48 specifications.
22 subjects (rows) × 48 specifications (columns, sorted by descending ρ).
Green = concordant (K and D agree), Red = discordant (K and D disagree).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import product
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "features_raw.csv")
SPEC_CSV = os.path.join(PROJECT_ROOT, "data", "spec_curve_data.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "identity_matrix.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 16,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

df = pd.read_csv(FEATURES_CSV)

kin_vars_all = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kinetics_vars_all = ["peak_vgrf_bw", "loading_rate_bw_s"]

# ── Specification dimensions (exactly matching spec_curve.py) ──
tasks = OrderedDict([
    ("Pooled (CMJ+DJ)", df),
    ("CMJ only", df[df["trial_type"] == "CMJ"]),
    ("DJ only", df[df["trial_type"] == "DJ"]),
])

dir_specs = OrderedDict([
    ("Pristine", {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}),
    ("Reversed (all +1)", {"hip_flex": 1, "knee_valg": 1, "trunk_lean": 1, "ankle_angle_sagittal": -1}),
])

k_components = OrderedDict([
    ("All 4 (HF+KV+TL+AD)", ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]),
    ("KV only", ["knee_valg"]),
    ("AD only", ["ankle_angle_sagittal"]),
    ("KV+AD", ["knee_valg", "ankle_angle_sagittal"]),
])

d_components = OrderedDict([
    ("Both (GRF+LR)", ["peak_vgrf_bw", "loading_rate_bw_s"]),
    ("Peak GRF only", ["peak_vgrf_bw"]),
])

# ── Build specification list ──
spec_list = []
for (task_name, data), (dir_name, kin_dir), (k_name, k_vars), (d_name, d_vars) in \
    product(tasks.items(), dir_specs.items(), k_components.items(), d_components.items()):

    if len(data["trial_type"].unique()) <= 1:
        task_tag = data["trial_type"].iloc[0]
    else:
        task_tag = "Pooled"

    spec_list.append({
        "task": task_name,
        "task_tag": task_tag,
        "direction": dir_name,
        "k_components": k_name,
        "d_components": d_name,
        "task_data": data,
        "kin_dir": dict(kin_dir),
        "k_vars": list(k_vars),
        "d_vars": list(d_vars),
    })

print(f"Total specifications: {len(spec_list)}")

# ── Compute ρ, κ, and subject-level concordance for each spec ──
results = []
identity_matrix = {}  # {spec_id: {subject: is_concordant}}

for i, spec in enumerate(spec_list):
    data = spec["task_data"]
    kin_dir = spec["kin_dir"]
    k_vars = spec["k_vars"]
    d_vars = spec["d_vars"]

    # Z-score + direction correction
    dfc = data.copy()
    for c in k_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
        if kin_dir[c] == -1:
            dfc[f"z_{c}"] = -dfc[f"z_{c}"]
    dfc["K_raw"] = dfc[[f"z_{c}" for c in k_vars]].mean(axis=1)
    for c in d_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
    dfc["D_raw"] = dfc[[f"z_{c}" for c in d_vars]].mean(axis=1)

    # Subject-level aggregation + second z-score
    sub = dfc.groupby("subject").agg(
        K=("K_raw", "mean"), D=("D_raw", "mean")
    ).reset_index()
    sub["K"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
    sub["D"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

    # Median split
    md_k, md_d = sub["K"].median(), sub["D"].median()
    kh = (sub["K"] >= md_k).astype(int)
    dh = (sub["D"] >= md_d).astype(int)

    # Record subject-level concordance
    sub["is_concordant"] = (kh == dh).astype(int)
    identity_matrix[i] = dict(zip(sub["subject"], sub["is_concordant"]))

    # Spearman ρ and Cohen's κ
    rho_obs, _ = stats.spearmanr(sub["K"], sub["D"])
    kappa_obs = cohen_kappa_score(kh, dh)
    conflict_pct = (kh != dh).mean() * 100

    results.append({
        "spec_id": i,
        "task": spec["task"],
        "task_tag": spec["task_tag"],
        "direction": spec["direction"],
        "k_components": spec["k_components"],
        "d_components": spec["d_components"],
        "n": len(sub),
        "rho": float(rho_obs),
        "kappa": float(kappa_obs),
        "conflict_pct": float(conflict_pct),
    })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("rho", ascending=False).reset_index(drop=True)

# ── Build subject × specification concordance matrix ──
subjects_sorted = sorted(identity_matrix[0].keys())
spec_order = df_results["spec_id"].tolist()  # sorted by descending ρ

# Build label for each spec column
spec_labels = []
for _, row in df_results.iterrows():
    label = f"{row['task_tag'][:3]}|{row['direction'][:3]}|{row['k_components'][:12]}|{row['d_components'][:6]}"
    spec_labels.append(label)

# Build matrix: rows = subjects, cols = specs (ordered by descending ρ)
matrix = np.zeros((len(subjects_sorted), len(spec_order)), dtype=int)
for col_idx, spec_id in enumerate(spec_order):
    for row_idx, subj in enumerate(subjects_sorted):
        matrix[row_idx, col_idx] = identity_matrix[spec_id][subj]

# ── Save CSV ──
df_matrix = pd.DataFrame(matrix, index=subjects_sorted, columns=spec_order)
df_matrix.index.name = "subject"
df_matrix.to_csv(OUT_CSV)
print(f"Saved identity matrix ({len(subjects_sorted)}×{len(spec_order)}) to {OUT_CSV}")

# ── Summary stats ──
concordant_count = matrix.sum(axis=0)  # per spec
print(f"Concordant subjects per spec: min={concordant_count.min()}, max={concordant_count.max()}, "
      f"median={np.median(concordant_count):.1f}")

# ── PLOT: Heatmap ──
fig, ax = plt.subplots(figsize=(24, 7))

# Use green/red colormap
cmap = matplotlib.colors.ListedColormap(['#d62728', '#2ca02c'])  # red for discordant(0), green for concordant(1)

im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

# Axes
ax.set_yticks(range(len(subjects_sorted)))
ax.set_yticklabels(subjects_sorted, fontsize=12)
ax.set_ylabel("Subject", fontsize=14)

# X-axis: spec labels — only show every 4th label to avoid crowding
n_specs = len(spec_order)
tick_positions = list(range(0, n_specs, 4))
ax.set_xticks(tick_positions)
ax.set_xticklabels([spec_labels[i] for i in tick_positions], fontsize=11, rotation=90)
ax.set_xlabel("Specification (sorted by descending ρ)", fontsize=14)

# Task divider lines
task_tags = df_results["task_tag"].values
prev = task_tags[0]
for i in range(1, n_specs):
    if task_tags[i] != prev:
        ax.axvline(x=i - 0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        prev = task_tags[i]

ax.set_title("Fig. SX. Identity fragility across 48 specifications\n"
             "Green = K–D concordant; Red = K–D discordant", fontsize=11, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', label='Concordant (K and D agree)'),
    Patch(facecolor='#d62728', label='Discordant (K and D disagree)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

plt.tight_layout()

png_path = os.path.join(FIG_DIR, "SX_identity_heatmap.png")
pdf_path = os.path.join(FIG_DIR, "SX_identity_heatmap.pdf")
fig.savefig(png_path)
fig.savefig(pdf_path)
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
plt.close()

print("\nDone.")
