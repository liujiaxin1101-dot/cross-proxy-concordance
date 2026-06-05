"""
Upgrade D-Score from 2 force-plate components to 4 multi-dimensional components.

Original D-Score (2D): peak_vgrf_bw + loading_rate_bw_s
Upgraded D-Score (4D): original 2 + knee_moment_avg_Nmkg + hip_moment_avg_Nmkg

All components are in the risk direction (higher = greater injury risk).
"""

import numpy as np
import pandas as pd
from scipy import stats
import json
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "features_raw.csv")
MOMENTS_CSV = os.path.join(PROJECT_ROOT, "data", "joint_moments.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")

# Load data
df = pd.read_csv(FEATURES_CSV)
df_moments = pd.read_csv(MOMENTS_CSV)

print(f"Features: {len(df)} records, Moments: {len(df_moments)} records")

# Merge on subject, trial_type, trial_num
df = df.merge(
    df_moments[["subject", "trial_type", "trial_num",
                "knee_moment_avg_Nmkg", "hip_moment_avg_Nmkg"]],
    on=["subject", "trial_type", "trial_num"],
    how="inner",
)
print(f"Merged: {len(df)} records")
print(f"Subjects: {sorted(df['subject'].unique())}")

# ==============================================================================
# Component definitions
# ==============================================================================
kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}

# Original D-Score (2 components)
dscore_orig_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]
dscore_orig_dir = {"peak_vgrf_bw": 1, "loading_rate_bw_s": 1}

# Upgraded D-Score (4 components)
dscore_upgraded_vars = dscore_orig_vars + ["knee_moment_avg_Nmkg", "hip_moment_avg_Nmkg"]
dscore_upgraded_dir = {
    "peak_vgrf_bw": 1,
    "loading_rate_bw_s": 1,
    "knee_moment_avg_Nmkg": 1,  # higher knee moment = higher risk
    "hip_moment_avg_Nmkg": 1,   # higher hip moment = higher risk
}

print(f"\n=== ORIGINAL D-Score: {dscore_orig_vars}")
print(f"=== UPGRADED D-Score: {dscore_upgraded_vars}")

# ==============================================================================
# K-Score computation (unchanged)
# ==============================================================================
all_z = {}
for col in kin_vars:
    all_z[col] = (df[col] - df[col].mean()) / df[col].std()
    if kin_dir[col] == -1:
        all_z[col] = -all_z[col]

kin_z_cols = []
for col in kin_vars:
    zcol = f"z_{col}"
    kin_z_cols.append(zcol)
    df[zcol] = all_z[col]

df["KScore_raw"] = df[kin_z_cols].mean(axis=1)

# ==============================================================================
# Original D-Score (2D)
# ==============================================================================
for col in dscore_orig_vars:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
    if dscore_orig_dir.get(col, 1) == -1:
        df[f"z_{col}"] = -df[f"z_{col}"]

df["DScore_orig_raw"] = df[[f"z_{c}" for c in dscore_orig_vars]].mean(axis=1)

# ==============================================================================
# Upgraded D-Score (4D)
# ==============================================================================
for col in dscore_upgraded_vars:
    df[f"z_up_{col}"] = (df[col] - df[col].mean()) / df[col].std()
    if dscore_upgraded_dir.get(col, 1) == -1:
        df[f"z_up_{col}"] = -df[f"z_up_{col}"]

df["DScore_upgraded_raw"] = df[[f"z_up_{c}" for c in dscore_upgraded_vars]].mean(axis=1)

# ==============================================================================
# Subject-level aggregation
# ==============================================================================
sub_agg = df.groupby("subject").agg(
    KScore_raw_mean=("KScore_raw", "mean"),
    DScore_orig_mean=("DScore_orig_raw", "mean"),
    DScore_upgraded_mean=("DScore_upgraded_raw", "mean"),
    KScore_std=("KScore_raw", "std"),
    DScore_orig_std=("DScore_orig_raw", "std"),
    DScore_upgraded_std=("DScore_upgraded_raw", "std"),
    n_trials=("trial_type", "count"),
).reset_index()

# Second z-score at subject level
sub_agg["KScore"] = (sub_agg["KScore_raw_mean"] - sub_agg["KScore_raw_mean"].mean()) / sub_agg["KScore_raw_mean"].std()
sub_agg["DScore_orig"] = (sub_agg["DScore_orig_mean"] - sub_agg["DScore_orig_mean"].mean()) / sub_agg["DScore_orig_mean"].std()
sub_agg["DScore_upgraded"] = (sub_agg["DScore_upgraded_mean"] - sub_agg["DScore_upgraded_mean"].mean()) / sub_agg["DScore_upgraded_mean"].std()

print(f"\n{'='*70}")
print(f"SUBJECT-LEVEL SCORES (N={len(sub_agg)})")
print(f"{'='*70}")
print(sub_agg[["subject", "KScore", "DScore_orig", "DScore_upgraded", "n_trials"]].to_string(index=False))

# ==============================================================================
# Spearman correlation: K-Score vs Original D-Score
# ==============================================================================
print(f"\n{'='*70}")
print(f"H1: Spearman correlation: K-Score vs D-Score")
print(f"{'='*70}")

rho_orig, p_orig = stats.spearmanr(sub_agg["KScore"], sub_agg["DScore_orig"])
rho_up, p_up = stats.spearmanr(sub_agg["KScore"], sub_agg["DScore_upgraded"])

n = len(sub_agg)
se = 1.0 / np.sqrt(n - 3)

ci_lo_orig = np.tanh(np.arctanh(rho_orig) - 1.96 * se)
ci_hi_orig = np.tanh(np.arctanh(rho_orig) + 1.96 * se)
ci_lo_up = np.tanh(np.arctanh(rho_up) - 1.96 * se)
ci_hi_up = np.tanh(np.arctanh(rho_up) + 1.96 * se)

print(f"  Original D-Score (2D):  rho = {rho_orig:.3f}, 95% CI [{ci_lo_orig:.3f}, {ci_hi_orig:.3f}], p = {p_orig:.4f}")
print(f"  Upgraded D-Score (4D):  rho = {rho_up:.3f}, 95% CI [{ci_lo_up:.3f}, {ci_hi_up:.3f}], p = {p_up:.4f}")

# ==============================================================================
# Cohen's Kappa: classification agreement
# ==============================================================================
print(f"\n{'='*70}")
print(f"H2: Classification agreement (median split)")
print(f"{'='*70}")

from sklearn.metrics import cohen_kappa_score

median_k = sub_agg["KScore"].median()
median_d_orig = sub_agg["DScore_orig"].median()
median_d_up = sub_agg["DScore_upgraded"].median()

sub_agg["K_hi"] = (sub_agg["KScore"] >= median_k).astype(int)
sub_agg["D_orig_hi"] = (sub_agg["DScore_orig"] >= median_d_orig).astype(int)
sub_agg["D_up_hi"] = (sub_agg["DScore_upgraded"] >= median_d_up).astype(int)

kappa_orig = cohen_kappa_score(sub_agg["K_hi"], sub_agg["D_orig_hi"])
kappa_up = cohen_kappa_score(sub_agg["K_hi"], sub_agg["D_up_hi"])

agree_orig = (sub_agg["K_hi"] == sub_agg["D_orig_hi"]).mean()
agree_up = (sub_agg["K_hi"] == sub_agg["D_up_hi"]).mean()

# Kappa CI
po_orig = (sub_agg["K_hi"] == sub_agg["D_orig_hi"]).sum() / n
po_up = (sub_agg["K_hi"] == sub_agg["D_up_hi"]).sum() / n
pe = 0.5
if pe < 1.0:
    se_kappa_orig = np.sqrt(po_orig * (1 - po_orig) / (n * (1 - pe) ** 2))
    se_kappa_up = np.sqrt(po_up * (1 - po_up) / (n * (1 - pe) ** 2))
else:
    se_kappa_orig = se_kappa_up = np.nan

kappa_ci_lo_orig = kappa_orig - 1.96 * se_kappa_orig
kappa_ci_hi_orig = kappa_orig + 1.96 * se_kappa_orig
kappa_ci_lo_up = kappa_up - 1.96 * se_kappa_up
kappa_ci_hi_up = kappa_up + 1.96 * se_kappa_up

print(f"  Original D-Score (2D):  kappa = {kappa_orig:.3f}, 95% CI [{kappa_ci_lo_orig:.3f}, {kappa_ci_hi_orig:.3f}], agree = {agree_orig*100:.1f}%")
print(f"  Upgraded D-Score (4D):  kappa = {kappa_up:.3f}, 95% CI [{kappa_ci_lo_up:.3f}, {kappa_ci_hi_up:.3f}], agree = {agree_up*100:.1f}%")

# ==============================================================================
# Conflict analysis
# ==============================================================================
print(f"\n{'='*70}")
print(f"H3: Conflict analysis")
print(f"{'='*70}")

sub_agg["conflict_orig"] = (sub_agg["K_hi"] != sub_agg["D_orig_hi"]).astype(int)
sub_agg["conflict_up"] = (sub_agg["K_hi"] != sub_agg["D_up_hi"]).astype(int)

n_conflict_orig = sub_agg["conflict_orig"].sum()
n_conflict_up = sub_agg["conflict_up"].sum()
pct_orig = n_conflict_orig / n * 100
pct_up = n_conflict_up / n * 100

print(f"  Original (2D): {n_conflict_orig}/{n} conflicts ({pct_orig:.1f}%)")
print(f"  Upgraded (4D): {n_conflict_up}/{n} conflicts ({pct_up:.1f}%)")

print(f"\n  Original confusion matrix:")
print(pd.crosstab(sub_agg["K_hi"], sub_agg["D_orig_hi"]))
print(f"\n  Upgraded confusion matrix:")
print(pd.crosstab(sub_agg["K_hi"], sub_agg["D_up_hi"]))

# ==============================================================================
# Original vs Upgraded D-Score correlation
# ==============================================================================
rho_dd, p_dd = stats.spearmanr(sub_agg["DScore_orig"], sub_agg["DScore_upgraded"])
print(f"\n{'='*70}")
print(f"D-Score correlation: Original (2D) vs Upgraded (4D)")
print(f"{'='*70}")
print(f"  Spearman rho = {rho_dd:.3f}, p = {p_dd:.4f}")

# ==============================================================================
# Component contributions
# ==============================================================================
print(f"\n{'='*70}")
print(f"Component contributions to Upgraded D-Score")
print(f"{'='*70}")
print(f"\nCorrelation of each component with Upgraded D-Score (subject level):")
for col in dscore_upgraded_vars:
    sub_mean = df.groupby("subject")[col].mean()
    sub_scores = sub_agg.set_index("subject")["DScore_upgraded"]
    common = sub_mean.index.intersection(sub_scores.index)
    r, p = stats.spearmanr(sub_mean[common], sub_scores[common])
    print(f"  {col:30s}: rho = {r:+.3f}, p = {p:.4f}")

# ==============================================================================
# Bland-Altman: K-Score vs Upgraded D-Score
# ==============================================================================
ba_bias = (sub_agg["KScore_raw_mean"].mean() - sub_agg["DScore_upgraded_mean"].mean())
ba_diffs = sub_agg["KScore_raw_mean"].values - sub_agg["DScore_upgraded_mean"].values
ba_sd = np.std(ba_diffs, ddof=0)
ba_loa_lo = ba_bias - 1.96 * ba_sd
ba_loa_hi = ba_bias + 1.96 * ba_sd

print(f"\n{'='*70}")
print(f"Bland-Altman (K-Score vs Upgraded D-Score, pre-second-z)")
print(f"{'='*70}")
print(f"  Bias = {ba_bias:.3f}")
print(f"  SD of differences = {ba_sd:.3f}")
print(f"  LoA = [{ba_loa_lo:.3f}, {ba_loa_hi:.3f}]")

# ==============================================================================
# Save enhanced features
# ==============================================================================
enhanced_csv = os.path.join(OUTPUT_DIR, "features_enhanced.csv")
df.to_csv(enhanced_csv, index=False)
print(f"\nEnhanced features saved to {enhanced_csv}")

# ==============================================================================
# Save results JSON
# ==============================================================================
results = {
    "date": "2026-06-03",
    "n_subjects": n,
    "n_records": len(df),
    "dscore_components_original": dscore_orig_vars,
    "dscore_components_upgraded": dscore_upgraded_vars,
    "spearman_rho_original": float(rho_orig),
    "spearman_rho_upgraded": float(rho_up),
    "spearman_ci_original": [float(ci_lo_orig), float(ci_hi_orig)],
    "spearman_ci_upgraded": [float(ci_lo_up), float(ci_hi_up)],
    "cohens_kappa_original": float(kappa_orig),
    "cohens_kappa_upgraded": float(kappa_up),
    "kappa_ci_original": [float(kappa_ci_lo_orig), float(kappa_ci_hi_orig)],
    "kappa_ci_upgraded": [float(kappa_ci_lo_up), float(kappa_ci_hi_up)],
    "percent_agreement_original": float(agree_orig),
    "percent_agreement_upgraded": float(agree_up),
    "n_conflicts_original": int(n_conflict_orig),
    "n_conflicts_upgraded": int(n_conflict_up),
    "conflict_pct_original": float(pct_orig),
    "conflict_pct_upgraded": float(pct_up),
    "dscore_correlation_orig_vs_upgraded": float(rho_dd),
    "bland_altman_bias_upgraded": float(ba_bias),
    "bland_altman_sd_diff_upgraded": float(ba_sd),
    "bland_altman_loa_upgraded": [float(ba_loa_lo), float(ba_loa_hi)],
}

out_path = os.path.join(OUTPUT_DIR, "results_upgraded_dscore.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {out_path}")

print(f"\n{'='*70}")
print(f"SUMMARY: D-Score Upgrade (2D → 4D with joint moments)")
print(f"{'='*70}")
print(f"  Original (2D):  rho={rho_orig:.3f}, kappa={kappa_orig:.3f}, conflict={pct_orig:.0f}%")
print(f"  Upgraded (4D):  rho={rho_up:.3f}, kappa={kappa_up:.3f}, conflict={pct_up:.0f}%")
