import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

df = pd.read_csv(FEATURES_CSV)

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

quadrant_colors = {"Agree": "#4CAF50", "K+D-": "#FF5722", "K-D+": "#2196F3"}

results = {}
for task_label in ["CMJ", "DJ"]:
    dft = df[df["trial_type"] == task_label].copy()
    for c in kin_vars:
        z = (dft[c] - dft[c].mean()) / dft[c].std()
        if kin_dir[c] == -1:
            z = -z
        dft[f"z_{c}"] = z
    dft["K_raw"] = dft[[f"z_{c}" for c in kin_vars]].mean(axis=1)
    for c in kinetics_vars:
        dft[f"z_{c}"] = (dft[c] - dft[c].mean()) / dft[c].std()
    dft["D_raw"] = dft[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    sub = dft.groupby("subject").agg(K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
    sub["K"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
    sub["D"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

    md_k, md_d = sub["K"].median(), sub["D"].median()
    sub["K_hi"] = (sub["K"] >= md_k).astype(int)
    sub["D_hi"] = (sub["D"] >= md_d).astype(int)
    sub["concordant"] = (sub["K_hi"] == sub["D_hi"])

    sub["quadrant"] = "Agree"
    sub.loc[(sub["K_hi"] == 1) & (sub["D_hi"] == 0), "quadrant"] = "K+D-"
    sub.loc[(sub["K_hi"] == 0) & (sub["D_hi"] == 1), "quadrant"] = "K-D+"

    rho, _ = stats.spearmanr(sub["K"], sub["D"])
    n = len(sub)
    se = 1.0 / np.sqrt(n - 3)
    ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se)
    ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se)
    agree = sub["concordant"].mean() * 100

    results[task_label] = {
        "sub": sub, "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "agree": agree, "md_k": md_k, "md_d": md_d,
    }

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

for ax_idx, task_label in enumerate(["CMJ", "DJ"]):
    ax = axes[ax_idx]
    r = results[task_label]
    sub = r["sub"]
    marker = 'o' if task_label == "CMJ" else '^'

    for q in ["Agree", "K+D-", "K-D+"]:
        grp = sub[sub["quadrant"] == q]
        ax.scatter(grp["K"], grp["D"], c=quadrant_colors[q], s=100,
                   marker=marker, edgecolors='k', linewidth=0.6, zorder=3,
                   label=f'{q} (n={len(grp)})')

    for _, row in sub.iterrows():
        ax.annotate(row["subject"].replace("sub", ""), (row["K"], row["D"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.axhline(r["md_d"], color='#999', ls=':', lw=1, alpha=0.5)
    ax.axvline(r["md_k"], color='#999', ls=':', lw=1, alpha=0.5)
    ax.axhline(0, color='grey', ls='--', lw=0.6, alpha=0.4)
    ax.axvline(0, color='grey', ls='--', lw=0.6, alpha=0.4)

    ax.set_xlabel("K-Score (z)")
    ax.set_ylabel("D-Score (z)")
    ax.set_title(f"{task_label}\n" + r"$\rho$ = " + f"{r['rho']:.3f}, "
                 + f"95% CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], "
                 + f"{r['agree']:.0f}% agree")
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(sub["K"].min() - 0.6, sub["K"].max() + 0.6)
    ax.set_ylim(sub["D"].min() - 0.6, sub["D"].max() + 0.6)

fig.suptitle("K-Score vs.\ D-Score by Landing Task Type", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

out_png = os.path.join(FIG_DIR, "05_paper_figure.png")
out_pdf = os.path.join(FIG_DIR, "05_paper_figure.pdf")
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
print("Done.")
