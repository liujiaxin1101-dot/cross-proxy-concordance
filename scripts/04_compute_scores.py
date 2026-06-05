import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
import json

FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "features_raw.csv")

df = pd.read_csv(FEATURES_CSV)

print(f"Loaded {len(df)} records")
print(f"Subjects: {sorted(df['subject'].unique())}")
print(f"Trial types: {df['trial_type'].value_counts().to_dict()}")

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]

kin_dir = {
    "hip_flex": -1,
    "knee_valg": 1,
    "trunk_lean": -1,
    "ankle_angle_sagittal": -1,
}

kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]
kin_dir_kin = {"peak_vgrf_bw": 1, "loading_rate_bw_s": 1}

all_z = {}
for col in kin_vars:
    all_z[col] = (df[col] - df[col].mean()) / df[col].std()
    if kin_dir[col] == -1:
        all_z[col] = -all_z[col]

kin_z_cols = []
for i, col in enumerate(kin_vars):
    zcol = f"z_{col}"
    kin_z_cols.append(zcol)
    df[zcol] = all_z[col]

df["KScore_raw"] = df[kin_z_cols].mean(axis=1)

for col in kinetics_vars:
    all_z[col] = (df[col] - df[col].mean()) / df[col].std()

kin_z_cols_d = []
for i, col in enumerate(kinetics_vars):
    zcol = f"z_{col}"
    kin_z_cols_d.append(zcol)
    df[zcol] = all_z[col]

df["DScore_raw"] = df[kin_z_cols_d].mean(axis=1)

sub_agg = df.groupby("subject").agg(
    KScore_raw_mean=("KScore_raw", "mean"),
    DScore_raw_mean=("DScore_raw", "mean"),
    KScore_std=("KScore_raw", "std"),
    DScore_std=("DScore_raw", "std"),
    n_trials=("trial_type", "count"),
).reset_index()

ba_bias = (sub_agg["KScore_raw_mean"].mean() - sub_agg["DScore_raw_mean"].mean())
ba_diffs = sub_agg["KScore_raw_mean"].values - sub_agg["DScore_raw_mean"].values
ba_sd = np.std(ba_diffs, ddof=0)
ba_loa_lo = ba_bias - 1.96 * ba_sd
ba_loa_hi = ba_bias + 1.96 * ba_sd

sub_agg["KScore"] = (sub_agg["KScore_raw_mean"] - sub_agg["KScore_raw_mean"].mean()) / sub_agg["KScore_raw_mean"].std()
sub_agg["DScore"] = (sub_agg["DScore_raw_mean"] - sub_agg["DScore_raw_mean"].mean()) / sub_agg["DScore_raw_mean"].std()

print(f"\n=== SUBJECT-LEVEL SCORES (N={len(sub_agg)}) ===")
print(sub_agg[["subject", "KScore", "DScore", "n_trials"]].to_string(index=False))

rho, p_rho = stats.spearmanr(sub_agg["KScore"], sub_agg["DScore"])
print(f"\n=== H1: Spearman correlation ===")
print(f"  rho = {rho:.3f}")
n = len(sub_agg)
se = 1.0 / np.sqrt(n - 3)
z = np.arctanh(rho)
ci_lo = np.tanh(z - 1.96 * se)
ci_hi = np.tanh(z + 1.96 * se)
print(f"  95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]")

median_k = sub_agg["KScore"].median()
median_d = sub_agg["DScore"].median()
sub_agg["K_hi"] = (sub_agg["KScore"] >= median_k).astype(int)
sub_agg["D_hi"] = (sub_agg["DScore"] >= median_d).astype(int)

from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(sub_agg["K_hi"], sub_agg["D_hi"])
agreement = (sub_agg["K_hi"] == sub_agg["D_hi"]).mean()

print(f"\n=== H2: Classification agreement ===")
print(f"  Cohen's kappa = {kappa:.3f}")
print(f"  Percent agreement = {agreement*100:.1f}%")

sub_agg["conflict"] = (sub_agg["K_hi"] != sub_agg["D_hi"]).astype(int)
n_conflict = sub_agg["conflict"].sum()
conflict_pct = n_conflict / len(sub_agg) * 100

print(f"\n=== H3: Conflict analysis ===")
print(f"  Conflicts: {n_conflict}/{len(sub_agg)} ({conflict_pct:.1f}%)")
conflict_ci_lo = stats.binom.ppf(0.025, len(sub_agg), conflict_pct / 100)
conflict_ci_hi = stats.binom.ppf(0.975, len(sub_agg), conflict_pct / 100)

n_agree = int((sub_agg["K_hi"] == sub_agg["D_hi"]).sum())
n_total = len(sub_agg)
po = n_agree / n_total
pe = 0.5
if pe < 1.0:
    kappa_se = np.sqrt(po * (1 - po) / (n_total * (1 - pe) ** 2))
else:
    kappa_se = np.nan
kappa_ci_lo = kappa - 1.96 * kappa_se
kappa_ci_hi = kappa + 1.96 * kappa_se

print(f"  Kappa = {kappa:.3f}, 95% CI [{kappa_ci_lo:.3f}, {kappa_ci_hi:.3f}]")

table = pd.crosstab(sub_agg["K_hi"], sub_agg["D_hi"])
print(f"\n  Confusion matrix:\n{table}")

n_00 = table.values[0, 0] if table.shape == (2, 2) else 0
n_01 = table.values[0, 1] if table.shape == (2, 2) else 0
n_10 = table.values[1, 0] if table.shape == (2, 2) else 0
n_11 = table.values[1, 1] if table.shape == (2, 2) else 0
if n_01 + n_10 > 0:
    mcnemar_stat = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
else:
    mcnemar_stat, mcnemar_p = 0.0, 1.0
print(f"  McNemar chi2 = {mcnemar_stat:.3f}, p = {mcnemar_p:.3f}")

print(f"\n=== Bland-Altman (pre-second-z subject means) ===")
print(f"  Bias = {ba_bias:.3f}")
print(f"  SD of differences = {ba_sd:.3f}")
print(f"  LoA = [{ba_loa_lo:.3f}, {ba_loa_hi:.3f}]")
print(f"  (Reported on pre-second-z means to avoid bias=0 by construction)")

print(f"\n=== POWER ===")
print(f"  Observed rho = {rho:.3f}")
print(f"  CI lower bound = {ci_lo:.3f}")
if ci_lo > 0:
    for target_power in [0.80, 0.90]:
        from scipy.stats import norm
        za = norm.ppf(0.975)
        zb = norm.ppf(target_power)
        n_needed = ((za + zb) / np.arctanh(ci_lo)) ** 2 + 3
        print(f"  N needed for {target_power*100:.0f}% power (using CI lower): {int(np.ceil(n_needed))}")
else:
    print(f"  CI lower bound {ci_lo:.3f} <= 0, cannot compute N")

output = {
    "date": "2026-05-20",
    "n_subjects": len(sub_agg),
    "n_records": len(df),
    "spearman_rho": float(rho),
    "spearman_ci_lo": float(ci_lo),
    "spearman_ci_hi": float(ci_hi),
    "cohens_kappa": float(kappa),
    "kappa_ci_lo": float(kappa_ci_lo),
    "kappa_ci_hi": float(kappa_ci_hi),
    "percent_agreement": float(agreement),
    "n_conflicts": int(n_conflict),
    "conflict_pct": float(conflict_pct),
    "bland_altman_bias": float(ba_bias),
    "bland_altman_sd_diff": float(ba_sd),
    "bland_altman_loa": [float(ba_loa_lo), float(ba_loa_hi)],
}

out_path = os.path.join(PROJECT_ROOT, "data", r"results_summary.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")

print(f"\n{'='*70}")
print(f"CMJ-only vs DJ-only ANALYSIS")
print(f"{'='*70}")

for task_label, task_mask in [("CMJ", df["trial_type"] == "CMJ"), ("DJ", df["trial_type"] == "DJ")]:
    df_task = df[task_mask].copy()
    for col in kin_vars:
        df_task[f"z_{col}"] = (df_task[col] - df_task[col].mean()) / df_task[col].std()
        if kin_dir[col] == -1:
            df_task[f"z_{col}"] = -df_task[f"z_{col}"]
    df_task["KScore_raw"] = df_task[[f"z_{c}" for c in kin_vars]].mean(axis=1)
    for col in kinetics_vars:
        df_task[f"z_{col}"] = (df_task[col] - df_task[col].mean()) / df_task[col].std()
    df_task["DScore_raw"] = df_task[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    sub_task = df_task.groupby("subject").agg(
        KScore=("KScore_raw", "mean"), DScore=("DScore_raw", "mean")
    ).reset_index()
    sub_task["KScore"] = (sub_task["KScore"] - sub_task["KScore"].mean()) / sub_task["KScore"].std()
    sub_task["DScore"] = (sub_task["DScore"] - sub_task["DScore"].mean()) / sub_task["DScore"].std()

    n_t = len(sub_task)
    rho_t, _ = stats.spearmanr(sub_task["KScore"], sub_task["DScore"])
    se_t = 1.0 / np.sqrt(n_t - 3)
    ci_lo_t = np.tanh(np.arctanh(rho_t) - 1.96 * se_t)
    ci_hi_t = np.tanh(np.arctanh(rho_t) + 1.96 * se_t)

    md_k = sub_task["KScore"].median()
    md_d = sub_task["DScore"].median()
    kh_t = (sub_task["KScore"] >= md_k).astype(int)
    dh_t = (sub_task["DScore"] >= md_d).astype(int)
    kappa_t = cohen_kappa_score(kh_t, dh_t)
    conflict_t = (kh_t != dh_t).mean()

    print(f"\n  {task_label}: N={n_t}")
    print(f"    Spearman rho = {rho_t:.3f}, 95% CI [{ci_lo_t:.3f}, {ci_hi_t:.3f}]")
    print(f"    Cohen's kappa = {kappa_t:.3f}, Conflict = {conflict_t*100:.0f}%")
