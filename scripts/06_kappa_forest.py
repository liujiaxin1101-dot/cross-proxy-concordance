import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

conditions = [
    ("Primary (IC=20N, equal weights)", 0.636, 0.314, 0.959, "#2c7fb8"),
    ("CMJ only", 0.455, 0.082, 0.827, "#7bccc4"),
    ("DJ only", 0.273, -0.129, 0.675, "#7bccc4"),
    ("IC = 50 N", 0.636, 0.314, 0.959, "#a8ddb5"),
    ("IC = kinematic (ankle velocity)", 0.091, -0.325, 0.507, "#a8ddb5"),
    ("Knee valgus ×2", 0.455, 0.082, 0.827, "#fdae61"),
    ("Loading rate ×2", 0.636, 0.314, 0.959, "#fdae61"),
    ("K-Score = knee valgus only", 0.273, -0.129, 0.675, "#fdae61"),
    ("D-Score = peak GRF only", 0.636, 0.314, 0.959, "#fdae61"),
    ("Excluding sub24", 0.618, 0.282, 0.954, "#d73027"),
    ("Direction incorrect (hip/trunk = risk�?", 0.091, -0.325, 0.507, "#d73027"),
]

n = len(conditions)
fig, ax = plt.subplots(figsize=(10, 5.5))

y_positions = np.arange(n)[::-1]
kappa_vals = np.array([c[1] for c in conditions])
ci_los = np.array([c[2] for c in conditions])
ci_his = np.array([c[3] for c in conditions])
colors = [c[4] for c in conditions]
labels = [c[0] for c in conditions]

ci_err_low = kappa_vals - ci_los
ci_err_high = ci_his - kappa_vals

for i in range(n):
    ax.errorbar(kappa_vals[i], y_positions[i],
                xerr=[[ci_err_low[i]], [ci_err_high[i]]],
                fmt='o', color=colors[i], capsize=4, capthick=1.5,
                markersize=7, markeredgecolor='white', markeredgewidth=1.2,
                elinewidth=2, zorder=3)

ax.axvline(x=0, color='grey', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
ax.axvline(x=0.20, color='#d9d9d9', linestyle=':', linewidth=1, zorder=1)
ax.axvline(x=0.40, color='#d9d9d9', linestyle=':', linewidth=1, zorder=1)
ax.axvline(x=0.60, color='#d9d9d9', linestyle=':', linewidth=1, zorder=1)
ax.axvline(x=0.80, color='#d9d9d9', linestyle=':', linewidth=1, zorder=1)

ax.text(0.10, n - 0.65, 'Slight', fontsize=9, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.30, n - 0.65, 'Fair', fontsize=9, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.50, n - 0.65, 'Moderate', fontsize=9, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.70, n - 0.65, 'Substantial', fontsize=9, color='grey', fontstyle='italic', clip_on=False)

ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Cohen's κ", fontsize=12)
ax.set_xlim(-0.45, 1.10)
ax.set_title("Classification agreement (κ) across methodological conditions",
             fontsize=13, fontweight='bold', pad=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i in range(n):
    if kappa_vals[i] > 0:
        ax.text(kappa_vals[i] + 0.015, y_positions[i],
                f"{kappa_vals[i]:.3f}", va='center', fontsize=8.5, fontweight='bold')
    else:
        ax.text(kappa_vals[i] - 0.10, y_positions[i],
                f"{kappa_vals[i]:.3f}", va='center', fontsize=8.5, fontweight='bold')

pos_y = -0.3
ax.text(0.02, pos_y, 'Poor', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.18, pos_y, 'Slight', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.30, pos_y, 'Fair', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.48, pos_y, 'Moderate', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.68, pos_y, 'Substantial', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.85, pos_y, 'Almost perfect', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)

ax.text(-0.42, pos_y, 'Landis & Koch (1977):', fontsize=7.5, color='grey', fontweight='bold', clip_on=False)

fig.subplots_adjust(bottom=0.13)
out_png = os.path.join(PROJECT_ROOT, "figures", "06_kappa_forest.png")
out_pdf = os.path.join(PROJECT_ROOT, "figures", "06_kappa_forest.pdf")
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
