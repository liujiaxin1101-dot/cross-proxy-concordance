"""
spec_curve_plot.py �?Plot the Specification Curve
Based on Simonsohn et al. (2020) Fig 2-3
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SPEC_CSV = os.path.join(PROJECT_ROOT, "data", r"spec_curve_data.csv")
JOINT_JSON = os.path.join(PROJECT_ROOT, "data", r"spec_curve_joint.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(SPEC_CSV)
with open(JOINT_JSON) as f:
    joint = json.load(f)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Colors for specification dimensions ──
task_colors = {
    "Pooled (CMJ+DJ)": "#2E7D32",
    "CMJ only": "#1976D2",
    "DJ only": "#E64A19",
}
dir_colors = {
    "Pristine": "#7B1FA2",
    "Reversed (all +1)": "#F57C00",
}
component_colors = {
    "All 4 (HF+KV+TL+AD)": "#455A64",
    "KV only": "#C2185B",
    "AD only": "#00796B",
    "KV+AD": "#8E24AA",
}

n_specs = len(df)

# ── Figure layout: upper panel (spec curve) + lower panel (specification panel) ──
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.12)

ax_curve = fig.add_subplot(gs[0])
ax_spec = fig.add_subplot(gs[1])

# ── Upper panel: Specification curve (ρ with bootstrap CI) ──
x = np.arange(n_specs)
rho_vals = df["rho_boot_median"].values
ci_lo = df["rho_ci_lo_boot"].values
ci_hi = df["rho_ci_hi_boot"].values

for i in range(n_specs):
    color = task_colors.get(df.iloc[i]["task"], "#999")
    ax_curve.plot([i, i], [ci_lo[i], ci_hi[i]], color=color, lw=0.8, alpha=0.6)
    ax_curve.scatter(i, rho_vals[i], s=12, c=color, edgecolors='k', linewidth=0.3, zorder=3)

ax_curve.axhline(y=0, color='grey', ls='--', lw=0.7, alpha=0.5)
ax_curve.axhline(y=joint["observed_median_rho"], color='#D32F2F', ls='-', lw=1.2, alpha=0.8,
                  label=f"Median ρ = {joint['observed_median_rho']:.3f}")

# Shade bottom portion
ax_curve.fill_between([-0.5, n_specs - 0.5], joint["null_rho_95ci"][0], joint["null_rho_95ci"][1],
                       color='grey', alpha=0.08, label=f"Null 95% CI [{joint['null_rho_95ci'][0]:.3f}, {joint['null_rho_95ci'][1]:.3f}]")

ax_curve.set_xlim(-0.5, n_specs - 0.5)
ax_curve.set_ylim(-1.05, 1.05)
ax_curve.set_ylabel("Spearman ρ\n(bootstrap median)", fontweight='bold')
ax_curve.tick_params(labelbottom=False)

# Legend for task
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=v, markersize=8, label=k, markeredgecolor='k', markeredgewidth=0.5)
    for k, v in task_colors.items()
]
legend_elements.append(Line2D([0], [0], color='#D32F2F', lw=1.5, label=f"Median ρ = {joint['observed_median_rho']:.3f}"))
legend_elements.append(Patch(facecolor='grey', alpha=0.08, label=f"Null 95% CI"))
ax_curve.legend(handles=legend_elements, fontsize=7, loc='upper right', ncol=1,
                title=f"Joint inference: p(ρ>0) = {joint['p_rho_one_sided']:.3f}",
                title_fontsize=7)

# ── Lower panel: Specification dimensions ──
# Row 1: Task
# Row 2: Direction
# Row 3: K-Score components

task_map = {v: i for i, v in enumerate(task_colors.keys())}
dir_map = {v: i for i, v in enumerate(dir_colors.keys())}
comp_map_k = {v: i for i, v in enumerate(component_colors.keys())}
d_comp_map = {"Both (GRF+LR)": 0, "Peak GRF only": 1}

n_rows = 4
bar_height = 0.7

# Row 1: Task
for i in range(n_specs):
    task_name = df.iloc[i]["task"]
    y_pos = task_map[task_name]
    ax_spec.barh(y_pos, 1.0, left=i, height=bar_height, color=task_colors[task_name],
                 edgecolor='white', linewidth=0.1)

# Row 2: Direction
row2_offset = 3
for i in range(n_specs):
    dir_name = df.iloc[i]["direction"]
    y_pos = row2_offset + dir_map[dir_name]
    ax_spec.barh(y_pos, 1.0, left=i, height=bar_height, color=dir_colors[dir_name],
                 edgecolor='white', linewidth=0.1)

# Row 3: K-Score components
row3_offset = 5
for i in range(n_specs):
    k_name = df.iloc[i]["k_components"]
    y_pos = row3_offset + comp_map_k[k_name]
    ax_spec.barh(y_pos, 1.0, left=i, height=bar_height, color=component_colors[k_name],
                 edgecolor='white', linewidth=0.1)

# Row 4: D-Score components
row4_offset = 9
d_comp_colors = {"Both (GRF+LR)": "#37474F", "Peak GRF only": "#78909C"}
for i in range(n_specs):
    d_name = df.iloc[i]["d_components"]
    y_pos = row4_offset + d_comp_map[d_name]
    ax_spec.barh(y_pos, 1.0, left=i, height=bar_height, color=d_comp_colors[d_name],
                 edgecolor='white', linewidth=0.1)

ax_spec.set_xlim(-0.5, n_specs - 0.5)
ax_spec.set_ylim(-0.5, 11.5)

# Y-axis labels
ytick_positions = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
ytick_labels = [
    "Pooled\n(CMJ+DJ)", "CMJ\nonly", "DJ\nonly",
    "Pristine", "Reversed\n(all +1)",
    "All 4\n(HF+KV+TL+AD)", "KV\nonly", "AD\nonly", "KV\n+AD",
    "Both\n(GRF+LR)", "Peak GRF\nonly",
]

ax_spec.set_yticks(ytick_positions)
ax_spec.set_yticklabels(ytick_labels, fontsize=7)

# Section labels on right
for label_text, y_pos in [
    ("TASK", 1.0), ("DIRECTION", 4.0), ("K-SCORE\nCOMPONENTS", 7.0), ("D-SCORE\nCOMPONENTS", 10.0)
]:
    ax_spec.text(n_specs + 1.5, y_pos, label_text, fontsize=7.5, fontweight='bold',
                 ha='left', va='center', color='#333')

ax_spec.set_xlabel("Specification (ranked by descending ρ)", fontsize=9)

# Grid lines between dimension groups
for y_line in [2.8, 4.8, 8.8]:
    ax_spec.axhline(y=y_line, color='black', lw=0.5, alpha=0.3)

ax_spec.spines['top'].set_visible(False)
ax_spec.spines['right'].set_visible(False)
ax_spec.tick_params(axis='y', length=0)

fig.suptitle("Specification Curve: Cross-Proxy Consistency (K-Score vs. D-Score)",
             fontsize=13, fontweight='bold', y=0.98)

# ── Annotation text ──
joint_text = (
    f"Joint inference across {n_specs} specifications:\n"
    f"  Median ρ = {joint['observed_median_rho']:.3f}, "
    f"one-sided p(ρ > 0) = {joint['p_rho_one_sided']:.3f}\n"
    f"  {joint['n_rho_ci_above_zero']}/{n_specs} specs ({joint['n_rho_ci_above_zero']/n_specs*100:.0f}%) "
    f"have bootstrap CI entirely above zero"
)
fig.text(0.12, 0.01, joint_text, fontsize=7.5, fontfamily='monospace', color='#555',
         va='bottom')

plt.tight_layout(rect=[0, 0.05, 1, 0.96])

out_png = os.path.join(FIG_DIR, "09_spec_curve.png")
out_pdf = os.path.join(FIG_DIR, "09_spec_curve.pdf")
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()

# ── Joint inference null distribution plot ──
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))

# Simplified: generate null distribution
np.random.seed(42)
n_jb = 50000
rho_v = df["rho_boot_median"].values
n_s = len(rho_v)
null_meds = np.array([np.median(rho_v * np.random.choice([-1, 1], size=n_s)) for _ in range(n_jb)])

ax2.hist(null_meds, bins=80, color='#90CAF9', edgecolor='#1565C0', alpha=0.7, density=True)
ax2.axvline(x=joint["observed_median_rho"], color='#D32F2F', lw=2.5, ls='-',
            label=f"Observed median ρ = {joint['observed_median_rho']:.3f}")
ax2.axvline(x=0, color='grey', ls='--', lw=1, alpha=0.6)

# p-value annotation
ax2.text(0.98, 0.95,
         f"One-sided p = {joint['p_rho_one_sided']:.3f}\n"
         f"Two-sided p = {joint['p_rho_two_sided']:.3f}",
         transform=ax2.transAxes, ha='right', va='top',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.set_xlabel("Median Spearman ρ across all specifications", fontweight='bold')
ax2.set_ylabel("Density", fontweight='bold')
ax2.set_title("Joint Inference: Null Distribution of Median ρ\n(50,000 sign-flip permutations)",
              fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')

plt.tight_layout()
out_png2 = os.path.join(FIG_DIR, "09_joint_inference.png")
out_pdf2 = os.path.join(FIG_DIR, "09_joint_inference.pdf")
fig2.savefig(out_png2, dpi=300, bbox_inches='tight', facecolor='white')
fig2.savefig(out_pdf2, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png2}")
print(f"Saved: {out_pdf2}")
plt.close()
print("Done.")
