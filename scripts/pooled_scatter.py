"""
pooled_scatter.py �?Generate pooled K-Score vs D-Score scatter plot
Uses pooled (CMJ+DJ) z-scores, one point per subject.
"""
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
    'font.family': 'Arial', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

df = pd.read_csv(FEATURES_CSV)

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

# ── Pooled z-scores (ALL CMJ+DJ trials together) ──
for c in kin_vars:
    z = (df[c] - df[c].mean()) / df[c].std()
    df[f"z_{c}"] = z if kin_dir[c] == 1 else -z
df["K_raw"] = df[[f"z_{c}" for c in kin_vars]].mean(axis=1)

for c in kinetics_vars:
    df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
df["D_raw"] = df[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

# ── Subject-level aggregation + second z-score ──
sub = df.groupby("subject").agg(K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
sub["K"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
sub["D"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

n = len(sub)
md_k, md_d = sub["K"].median(), sub["D"].median()
sub["K_hi"] = (sub["K"] >= md_k).astype(int)
sub["D_hi"] = (sub["D"] >= md_d).astype(int)
sub["concordant"] = (sub["K_hi"] == sub["D_hi"])

sub["quadrant"] = "Agree"
sub.loc[(sub["K_hi"] == 1) & (sub["D_hi"] == 0), "quadrant"] = "K+D-"
sub.loc[(sub["K_hi"] == 0) & (sub["D_hi"] == 1), "quadrant"] = "K-D+"

rho, _ = stats.spearmanr(sub["K"], sub["D"])
se = 1.0 / np.sqrt(n - 3)
ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se)
ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se)
agree_pct = sub["concordant"].mean() * 100

print(f"Pooled: N={n}, ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], agree={agree_pct:.1f}%")
print(f"Conflicts: {(~sub['concordant']).sum()}/{n}")
for _, row in sub[~sub["concordant"]].iterrows():
    print(f"  {row['subject']}  K={row['K']:+.2f} D={row['D']:+.2f}  {row['quadrant']}")

# ── Plot ──
quadrant_colors = {"Agree": "#4CAF50", "K+D-": "#FF5722", "K-D+": "#2196F3"}

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

for q in ["Agree", "K+D-", "K-D+"]:
    grp = sub[sub["quadrant"] == q]
    ax.scatter(grp["K"], grp["D"], c=quadrant_colors[q], s=160,
               edgecolors='k', linewidth=0.6, zorder=3, label=f'{q} (n={len(grp)})')

for _, row in sub.iterrows():
    ax.annotate(row["subject"].replace("sub", ""), (row["K"], row["D"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)

ax.axhline(md_d, color='#999', ls=':', lw=1, alpha=0.5)
ax.axvline(md_k, color='#999', ls=':', lw=1, alpha=0.5)
ax.axhline(0, color='grey', ls='--', lw=0.6, alpha=0.4)
ax.axvline(0, color='grey', ls='--', lw=0.6, alpha=0.4)

ax.set_xlabel("K-Score (z, pooled CMJ+DJ)")
ax.set_ylabel("D-Score (z, pooled CMJ+DJ)")
ax.set_title(f"Pooled K-Score vs. D-Score (N={n})\n"
             + r"$\bf{\rho = " + f"{rho:.3f}" + r"}$, 95% CI "
             + f"[{ci_lo:.3f}, {ci_hi:.3f}], "
             + f"{agree_pct:.0f}% agree")
ax.legend(fontsize=8, loc='lower right')
pad = 0.6
ax.set_xlim(sub["K"].min() - pad, sub["K"].max() + pad)
ax.set_ylim(sub["D"].min() - pad, sub["D"].max() + pad)
ax.set_aspect('equal')

plt.tight_layout()
out_png = os.path.join(FIG_DIR, "10_pooled_scatter.png")
out_pdf = os.path.join(FIG_DIR, "10_pooled_scatter.pdf")
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
print("Done.")
