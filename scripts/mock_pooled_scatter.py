"""
mock_pooled_scatter.py �?Mock preview of pooled K vs D scatter plot
"""
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ── Mock: 22 subjects, pooled K and D with ρ �?0.56 ──
np.random.seed(42)
n = 22
rho_target = 0.56
cov = np.array([[1.0, rho_target], [rho_target, 1.0]])
K, D = np.random.multivariate_normal([0, 0], cov, size=n).T

sub_ids = [f"sub{i:02d}" for i in range(1, 23)]
md_k, md_d = np.median(K), np.median(D)

K_hi = (K >= md_k).astype(int)
D_hi = (D >= md_d).astype(int)

quadrant = np.full(n, "Agree", dtype=object)
quadrant[(K_hi == 1) & (D_hi == 0)] = "K+D-"
quadrant[(K_hi == 0) & (D_hi == 1)] = "K-D+"

colors = {"Agree": "#4CAF50", "K+D-": "#FF5722", "K-D+": "#2196F3"}

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

for q in ["Agree", "K+D-", "K-D+"]:
    mask = quadrant == q
    ax.scatter(K[mask], D[mask], c=colors[q], s=140, edgecolors='k',
               linewidth=0.6, zorder=3, label=f'{q} (n={mask.sum()})')

for i in range(n):
    ax.annotate(sub_ids[i].replace("sub", ""), (K[i], D[i]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)

ax.axhline(md_d, color='#999', ls=':', lw=1, alpha=0.5, label=f'D median')
ax.axvline(md_k, color='#999', ls=':', lw=1, alpha=0.5, label=f'K median')
ax.axhline(0, color='grey', ls='--', lw=0.6, alpha=0.4)
ax.axvline(0, color='grey', ls='--', lw=0.6, alpha=0.4)

agree_pct = (quadrant == "Agree").mean() * 100
ax.set_xlabel("K-Score (z, pooled CMJ+DJ)")
ax.set_ylabel("D-Score (z, pooled CMJ+DJ)")
ax.set_title(f"Pooled K-Score vs. D-Score (N={n})\n"
             + r"$\bf{\rho \approx 0.56}$, "
             + f"{agree_pct:.0f}% agree (median split)\n"
             + "[MOCK DATA �?for preview only]",
             fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(K.min() - 0.6, K.max() + 0.6)
ax.set_ylim(D.min() - 0.6, D.max() + 0.6)
ax.set_aspect('equal')

plt.tight_layout()
out = os.path.join(FIG_DIR, "MOCK_pooled_scatter.png")
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
plt.close()
print("Done.")
