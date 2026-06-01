"""
combinatorial_visual.py �?Visualize κ and ρ across single-factor and combinatorial perturbations.
9 conditions: Primary, S5, S1b, S4c, C1, C2, C3, C4.
"""
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ── Data ──
labels = [
    "Primary\n(pooled, equal wts)",
    "S5\n(direction reversed)",
    "S1b\n(DJ only)",
    "S4c\n(KV only)",
    "C1\nS5 × S1b",
    "C2\nS5 × S4c",
    "C3\nS1b × S4c",
    "C4\nS5 × S1b × S4c",
]

kappa_vals   = [0.636, 0.091, 0.273, 0.273, 0.273, 0.273, 0.091, 0.091]
kappa_ci_lo  = [0.273, -0.320, -0.273, -0.128, -0.128, -0.144, -0.316, -0.316]
kappa_ci_hi  = [0.909, 0.472, 0.636, 0.636, 0.637, 0.636, 0.472, 0.468]

rho_vals     = [0.558, 0.460, 0.300, 0.440, 0.359, 0.440, 0.396, 0.396]
rho_ci_lo    = [0.199, -0.027, -0.164, -0.006, -0.132, -0.019, -0.081, -0.082]
rho_ci_hi    = [0.810, 0.818, 0.691, 0.767, 0.725, 0.777, 0.746, 0.742]

n = len(labels)
x = np.arange(n)

# Color scheme
single_colors = ['#2c7fb8', '#d73027', '#7bccc4', '#fdae61', 
                 '#d73027', '#d73027', '#fdae61', '#d73027']

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                          gridspec_kw={'height_ratios': [1.15, 1]})

# ── Panel A: Cohen's κ ──
ax_k = axes[0]
kw = 0.32
for i in range(n):
    lo = kappa_vals[i] - kappa_ci_lo[i]
    hi = kappa_ci_hi[i] - kappa_vals[i]
    ax_k.errorbar(x[i], kappa_vals[i], yerr=[[lo], [hi]], fmt='o',
                  color=single_colors[i], capsize=5, capthick=1.8,
                  markersize=11, markeredgecolor='white', markeredgewidth=1.3,
                  elinewidth=2, zorder=4)

for i in range(n):
    ax_k.text(x[i], kappa_vals[i] + 0.06, f"{kappa_vals[i]:.3f}",
              ha='center', fontsize=8.5, fontweight='bold', color='#333', zorder=5)

ax_k.axhline(0, color='grey', ls='--', lw=1, alpha=0.5, zorder=1)
for lvl, lbl in [(0.20, 'Slight'), (0.40, 'Fair'), (0.60, 'Moderate'), (0.80, 'Substantial')]:
    ax_k.axhline(lvl, color='#d9d9d9', ls=':', lw=0.8, zorder=1)
    ax_k.text(-0.22, lvl, lbl, fontsize=7, va='center', color='#999',
              fontstyle='italic')

ax_k.set_ylabel("Cohen's κ", fontsize=11)
ax_k.set_title("Perturbation Sensitivity: κ and ρ Across Single-Factor and Combinatorial Conditions",
               fontsize=12, fontweight='bold', pad=14)
ax_k.set_ylim(-0.45, 1.08)
ax_k.spines['top'].set_visible(False)
ax_k.spines['right'].set_visible(False)
ax_k.tick_params(axis='x', bottom=False)

# Shade regions: left=primary, middle=singles, right=combinatorial
ax_k.axvspan(-0.55, 0.55, alpha=0.04, color='#2c7fb8', zorder=0)
ax_k.axvspan(0.55, 3.55, alpha=0.04, color='#fdae61', zorder=0)
ax_k.axvspan(3.55, 7.55, alpha=0.04, color='#999', zorder=0)

ax_k.text(0, -0.38, 'Primary', ha='center', fontsize=7.5, fontweight='bold', color='#2c7fb8')
ax_k.text(2, -0.38, 'Single-factor', ha='center', fontsize=7.5, fontweight='bold', color='#fdae61')
ax_k.text(5.5, -0.38, 'Combinatorial', ha='center', fontsize=7.5, fontweight='bold', color='#666')


# ── Panel B: Spearman's ρ ──
ax_r = axes[1]
for i in range(n):
    lo_r = rho_vals[i] - rho_ci_lo[i]
    hi_r = rho_ci_hi[i] - rho_vals[i]
    ax_r.errorbar(x[i], rho_vals[i], yerr=[[lo_r], [hi_r]], fmt='o',
                  color=single_colors[i], capsize=5, capthick=1.8,
                  markersize=11, markeredgecolor='white', markeredgewidth=1.3,
                  elinewidth=2, zorder=4)

for i in range(n):
    ax_r.text(x[i], rho_vals[i] + 0.055, f"{rho_vals[i]:.3f}",
              ha='center', fontsize=8.5, fontweight='bold', color='#333', zorder=5)

ax_r.axhline(0, color='grey', ls='--', lw=1, alpha=0.5, zorder=1)
for lvl, lbl in [(0.30, 'Med.'), (0.50, 'Large')]:
    ax_r.axhline(lvl, color='#d9d9d9', ls=':', lw=0.8, zorder=1)
    ax_r.text(-0.22, lvl, f'{lbl} (Cohen)', fontsize=7, va='center', color='#999',
              fontstyle='italic')

ax_r.set_ylabel("Spearman's ρ", fontsize=11)
ax_r.set_ylim(-0.25, 1.02)
ax_r.set_xticks(x)
ax_r.set_xticklabels(labels, fontsize=7.8)
ax_r.spines['top'].set_visible(False)
ax_r.spines['right'].set_visible(False)

# Re-apply shading to bottom panel
ax_r.axvspan(-0.55, 0.55, alpha=0.04, color='#2c7fb8', zorder=0)
ax_r.axvspan(0.55, 3.55, alpha=0.04, color='#fdae61', zorder=0)
ax_r.axvspan(3.55, 7.55, alpha=0.04, color='#999', zorder=0)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2c7fb8', alpha=0.5, label='Primary'),
    mpatches.Patch(facecolor='#d73027', alpha=0.5, label='Direction reversed / combinatorial'),
    mpatches.Patch(facecolor='#7bccc4', alpha=0.5, label='Task (DJ only)'),
    mpatches.Patch(facecolor='#fdae61', alpha=0.5, label='Weight (KV only)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.03, 1, 1])

out_png = os.path.join(FIG_DIR, "09_combinatorial_panel.png")
out_pdf = os.path.join(FIG_DIR, "09_combinatorial_panel.pdf")
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
print("Done.")
