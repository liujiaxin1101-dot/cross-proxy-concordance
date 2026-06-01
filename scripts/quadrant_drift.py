"""
quadrant_drift.py 鈥?Quadrant drift visualization across sensitivity conditions.
Shows how each subject's K-Score/D-Score classification shifts under S1-S8.
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
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

df = pd.read_csv(FEATURES_CSV)
kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir_pristine = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kin_dir_reversed = {"hip_flex": 1, "knee_valg": 1, "trunk_lean": 1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]


def compute_condition(data, kin_dir=None, weights_kin=None, weights_kinetics=None):
    kin_d = kin_dir if kin_dir is not None else kin_dir_pristine
    wk = weights_kin or {c: 1 for c in kin_vars}
    wd = weights_kinetics or {c: 1 for c in kinetics_vars}
    wsum_k = sum(wk.values())
    wsum_d = sum(wd.values())

    dfc = data.copy()
    for c in kin_vars:
        z = (dfc[c] - dfc[c].mean()) / dfc[c].std()
        if kin_d[c] == -1:
            z = -z
        dfc[f"z_{c}"] = z
    dfc["K_raw"] = sum(wk[c] * dfc[f"z_{c}"] for c in kin_vars) / wsum_k
    for c in kinetics_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
    dfc["D_raw"] = sum(wd[c] * dfc[f"z_{c}"] for c in kinetics_vars) / wsum_d

    sub = dfc.groupby("subject").agg(K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
    sub["K"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
    sub["D"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()
    md_k, md_d = sub["K"].median(), sub["D"].median()
    sub["K_hi"] = (sub["K"] >= md_k).astype(int)
    sub["D_hi"] = (sub["D"] >= md_d).astype(int)
    sub["quadrant"] = "LL"  # K-low, D-low
    sub.loc[(sub["K_hi"] == 1) & (sub["D_hi"] == 0), "quadrant"] = "HL"  # K-high, D-low
    sub.loc[(sub["K_hi"] == 0) & (sub["D_hi"] == 1), "quadrant"] = "LH"  # K-low, D-high
    sub.loc[(sub["K_hi"] == 1) & (sub["D_hi"] == 1), "quadrant"] = "HH"  # K-high, D-high
    kappa = cohen_kappa_score(sub["K_hi"], sub["D_hi"])
    return sub, kappa


conditions = OrderedDict()
conditions["Primary (IC=20N, equal weights)"] = lambda d: compute_condition(d)
conditions["CMJ only"] = lambda d: compute_condition(d[d["trial_type"] == "CMJ"])
conditions["DJ only"] = lambda d: compute_condition(d[d["trial_type"] == "DJ"])
conditions["Knee valgus 脳2"] = lambda d: compute_condition(d, weights_kin={
    "hip_flex": 1, "knee_valg": 2, "trunk_lean": 1, "ankle_angle_sagittal": 1})
conditions["K-Score = knee valgus only"] = lambda d: compute_condition(d, weights_kin={
    "hip_flex": 0, "knee_valg": 1, "trunk_lean": 0, "ankle_angle_sagittal": 0})
conditions["Direction reversed\n(hip/trunk = risk up)"] = lambda d: compute_condition(
    d, kin_dir=kin_dir_reversed)
conditions["Excluding sub24"] = lambda d: compute_condition(d[d["subject"] != "sub24"])

results = OrderedDict()
for label, fn in conditions.items():
    sub, kappa = fn(df)
    results[label] = {"sub": sub, "kappa": kappa}

# Quadrant mapping: LL=0, HL=1, LH=2, HH=3
qmap = {"LL": 0, "HL": 1, "LH": 2, "HH": 3}

# Build drift matrix: subjects 脳 conditions
sub_ids = sorted(results[list(results.keys())[0]]["sub"]["subject"].unique())
cond_labels = list(results.keys())
drift_matrix = np.full((len(sub_ids), len(cond_labels)), -1, dtype=int)

for j, label in enumerate(cond_labels):
    sub_df = results[label]["sub"]
    qlookup = dict(zip(sub_df["subject"], sub_df["quadrant"]))
    for i, sid in enumerate(sub_ids):
        drift_matrix[i, j] = qmap.get(sid, -1)

# Determine which subjects change quadrants
primary_q = drift_matrix[:, 0]
n_switches = np.sum(drift_matrix != primary_q[:, np.newaxis], axis=1)
# Sort by stability (fewest switches at top)
sort_idx = np.argsort(n_switches)
drift_sorted = drift_matrix[sort_idx]
sub_ids_sorted = [sub_ids[i] for i in sort_idx]

quadrant_names = ["K鈫?D鈫?, "K鈫?D鈫?, "K鈫?D鈫?, "K鈫?D鈫?]
quadrant_colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# LEFT: Drift matrix (heatmap)
ax = axes[0]
im = ax.imshow(drift_sorted, aspect='auto', cmap=plt.matplotlib.colors.ListedColormap(quadrant_colors),
               vmin=-0.5, vmax=3.5, interpolation='nearest')

ax.set_yticks(range(len(sub_ids_sorted)))
ax.set_yticklabels([s.replace("sub", "") for s in sub_ids_sorted], fontsize=7)
ax.set_xticks(range(len(cond_labels)))
ax.set_xticklabels(cond_labels, rotation=35, ha='right', fontsize=7.5)
ax.set_title("Subject Classification Drift Across Conditions\n(each row = one subject, each column = one sensitivity condition)",
             fontsize=11, fontweight='bold', pad=12)

cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.6, pad=0.02)
cbar.set_ticklabels(quadrant_names)
cbar.set_label("Quadrant (K-Score / D-Score)", fontsize=9)

# Add drift count as text
for i in range(len(sub_ids_sorted)):
    n_s = n_switches[sort_idx[i]]
    if n_s > 0:
        ax.text(len(cond_labels) + 0.4, i, f"螖{n_s}", va='center', fontsize=6.5,
                color='#666', fontstyle='italic')

# Add kappa annotation at bottom
for j, label in enumerate(cond_labels):
    k = results[label]["kappa"]
    ax.text(j, len(sub_ids_sorted) + 0.6, f"魏={k:.3f}", ha='center', fontsize=7,
            fontweight='bold', color='#333')

# RIGHT: Bar chart 鈥?number of subjects changing quadrant per condition
ax2 = axes[1]
switch_counts = np.sum(drift_matrix != primary_q[:, np.newaxis], axis=0)
primary_kappa = results[list(results.keys())[0]]["kappa"]
deltas = [results[l]["kappa"] - primary_kappa for l in cond_labels]

colors_bar = ['#2c7fb8'] + ['#7bccc4'] * 2 + ['#fdae61'] * 2 + ['#d73027'] * 2
x_idx = np.arange(len(cond_labels))
width = 0.35

bars1 = ax2.bar(x_idx, switch_counts, width, color=colors_bar, edgecolor='white',
                label='Subjects switching quadrant')
ax2.set_xticks(x_idx)
ax2.set_xticklabels([l.replace('\n', ' ') for l in cond_labels], rotation=35, ha='right', fontsize=7.5)
ax2.set_ylabel("Number of subjects (out of 22)", fontsize=10)
ax2.set_title("Impact of Each Perturbation:\nSubjects Changing Classification & 螖魏", fontsize=11, fontweight='bold')

ax2_twin = ax2.twinx()
ax2_twin.plot(x_idx, deltas, 'ko-', lw=2, markersize=8, label='螖魏 from primary', zorder=5)
ax2_twin.set_ylabel("螖魏 from primary condition", fontsize=10)
ax2_twin.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.5)

for bar, val in zip(bars1, switch_counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

for j, dk in enumerate(deltas):
    ax2_twin.text(j, dk + (0.03 if dk >= 0 else -0.08), f"{dk:+.0f}",
                 ha='center', fontsize=7.5, fontweight='bold', color='black')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

ax2.spines['top'].set_visible(False)
ax2_twin.spines['top'].set_visible(False)

fig.suptitle("Quadrant Drift Analysis: How Methodological Choices Reshape Subject Classification",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

out_png = os.path.join(FIG_DIR, "08_quadrant_drift.png")
out_pdf = os.path.join(FIG_DIR, "08_quadrant_drift.pdf")
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
print("Done.")
