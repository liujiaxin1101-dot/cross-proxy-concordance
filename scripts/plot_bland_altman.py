import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "figures", "07_bland_altman.pdf")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "figures", "07_bland_altman.png")

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} records from {df.subject.nunique()} subjects")

for col in kin_vars:
    zcol = f"z_{col}"
    df[zcol] = (df[col] - df[col].mean()) / df[col].std()
    if kin_dir[col] == -1:
        df[zcol] = -df[zcol]

df["KScore_raw"] = df[[f"z_{c}" for c in kin_vars]].mean(axis=1)

for col in kinetics_vars:
    zcol = f"z_{col}"
    df[zcol] = (df[col] - df[col].mean()) / df[col].std()

df["DScore_raw"] = df[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

sub_agg = df.groupby("subject").agg(
    KScore_raw_mean=("KScore_raw", "mean"),
    DScore_raw_mean=("DScore_raw", "mean"),
).reset_index()

diffs = sub_agg["KScore_raw_mean"].values - sub_agg["DScore_raw_mean"].values
means = (sub_agg["KScore_raw_mean"].values + sub_agg["DScore_raw_mean"].values) / 2.0

bias = np.mean(diffs)
sd_diff = np.std(diffs, ddof=1)
loa_lo = bias - 1.96 * sd_diff
loa_hi = bias + 1.96 * sd_diff

print(f"\nBland-Altman results (pre-second-z subject means):")
print(f"  Bias = {bias:.3f}")
print(f"  SD of differences = {sd_diff:.3f}")
print(f"  95% LoA = [{loa_lo:.3f}, {loa_hi:.3f}]")

ci_bias = 1.96 * sd_diff / np.sqrt(len(diffs))
print(f"  95% CI of bias = [{bias - ci_bias:.3f}, {bias + ci_bias:.3f}]")

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(means, diffs, c='#2c3e50', s=60, alpha=0.8, edgecolors='white', linewidth=0.5,
           zorder=3)

ax.axhline(y=bias, color='#e74c3c', linestyle='-', linewidth=1.2, label=f'Bias = {bias:.3f}')
ax.axhline(y=loa_lo, color='#3498db', linestyle='--', linewidth=1.0,
           label=f'95% LoA = [{loa_lo:.2f}, {loa_hi:.2f}]')
ax.axhline(y=loa_hi, color='#3498db', linestyle='--', linewidth=1.0)
ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

for i, sub in enumerate(sub_agg["subject"]):
    if abs(diffs[i] - bias) > 1.5 * sd_diff:
        ax.annotate(sub.replace("sub", ""), (means[i], diffs[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7,
                    color='#e74c3c', fontweight='bold')

ax.set_xlabel('Mean of K-Score$_{raw}$ and D-Score$_{raw}$', fontsize=12)
ax.set_ylabel('Difference (K-Score$_{raw}$ $-$ D-Score$_{raw}$)', fontsize=12)
ax.set_title('Bland-Altman Plot: K-Score vs. D-Score\n(pre-second-$z$ subject means, $N = 22$)', fontsize=13)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(OUTPUT_PDF, dpi=300)
fig.savefig(OUTPUT_PNG, dpi=300)
print(f"\nSaved to {OUTPUT_PDF} and {OUTPUT_PNG}")
plt.close()
