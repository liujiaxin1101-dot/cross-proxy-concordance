"""
Bootstrap confidence intervals for Spearman rho and Cohen's kappa.
N=22 is too small for analytic CI formulas (Fisher z, SE_kappa).
Uses 10,000 bootstrap resamples with seed=42 for reproducibility.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")
N_BOOT = 10000
SEED = 42

rng = np.random.default_rng(SEED)

df = pd.read_csv(FEATURES_CSV)

# --- Primary analysis: pooled CMJ+DJ, equal weights, protective conventions ---
kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

for col in kin_vars:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
    if kin_dir[col] == -1:
        df[f"z_{col}"] = -df[f"z_{col}"]
df["KScore_raw"] = df[[f"z_{c}" for c in kin_vars]].mean(axis=1)

for col in kinetics_vars:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
df["DScore_raw"] = df[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

sub = df.groupby("subject").agg(
    KScore_raw=("KScore_raw", "mean"),
    DScore_raw=("DScore_raw", "mean"),
).reset_index()
sub["KScore"] = (sub["KScore_raw"] - sub["KScore_raw"].mean()) / sub["KScore_raw"].std()
sub["DScore"] = (sub["DScore_raw"] - sub["DScore_raw"].mean()) / sub["DScore_raw"].std()

# --- Bootstrap Spearman rho ---
rho_boot = []
for _ in range(N_BOOT):
    idx = rng.choice(len(sub), size=len(sub), replace=True)
    s = sub.iloc[idx]
    rho_b, _ = stats.spearmanr(s["KScore"], s["DScore"])
    rho_boot.append(rho_b)

rho_pt = np.median(rho_boot)
rho_ci_lo = np.percentile(rho_boot, 2.5)
rho_ci_hi = np.percentile(rho_boot, 97.5)

print(f"=== Bootstrap Spearman rho (N={len(sub)}, {N_BOOT} resamples) ===")
print(f"  Point estimate (median): {rho_pt:.3f}")
print(f"  95% CI (percentile):     [{rho_ci_lo:.3f}, {rho_ci_hi:.3f}]")

# Analytic CI for comparison
n = len(sub)
se = 1.0 / np.sqrt(n - 3)
rho_obs, _ = stats.spearmanr(sub["KScore"], sub["DScore"])
z_val = np.arctanh(rho_obs)
rho_an_lo = np.tanh(z_val - 1.96 * se)
rho_an_hi = np.tanh(z_val + 1.96 * se)
print(f"  Analytic CI (Fisher z):  [{rho_an_lo:.3f}, {rho_an_hi:.3f}]  (for comparison, may be unreliable at N=22)")
print()

# --- Bootstrap Cohen's kappa ---
median_k = sub["KScore"].median()
median_d = sub["DScore"].median()
sub["K_hi"] = (sub["KScore"] >= median_k).astype(int)
sub["D_hi"] = (sub["DScore"] >= median_d).astype(int)

kappa_boot = []
for _ in range(N_BOOT):
    idx = rng.choice(len(sub), size=len(sub), replace=True)
    s = sub.iloc[idx]
    kappa_b = cohen_kappa_score(s["K_hi"], s["D_hi"])
    kappa_boot.append(kappa_b)

kappa_pt = np.median(kappa_boot)
kappa_ci_lo = np.percentile(kappa_boot, 2.5)
kappa_ci_hi = np.percentile(kappa_boot, 97.5)

print(f"=== Bootstrap Cohen's kappa (N={len(sub)}, {N_BOOT} resamples) ===")
print(f"  Point estimate (median): {kappa_pt:.3f}")
print(f"  95% CI (percentile):     [{kappa_ci_lo:.3f}, {kappa_ci_hi:.3f}]")

# Analytic CI for comparison
n_agree = int((sub["K_hi"] == sub["D_hi"]).sum())
po = n_agree / len(sub)
pe = 0.5
kappa_obs = cohen_kappa_score(sub["K_hi"], sub["D_hi"])
kappa_se = np.sqrt(po * (1 - po) / (len(sub) * (1 - pe) ** 2))
kappa_an_lo = kappa_obs - 1.96 * kappa_se
kappa_an_hi = kappa_obs + 1.96 * kappa_se
print(f"  Analytic CI (SE formula): [{kappa_an_lo:.3f}, {kappa_an_hi:.3f}]  (for comparison, may be unreliable at N=22)")
print()

# --- Bootstrap ρ for CMJ and DJ separately ---
for task_label in ["CMJ", "DJ"]:
    df_task = df[df["trial_type"] == task_label].copy()
    for col in kin_vars:
        df_task[f"z_{col}"] = (df_task[col] - df_task[col].mean()) / df_task[col].std()
        if kin_dir[col] == -1:
            df_task[f"z_{col}"] = -df_task[f"z_{col}"]
    df_task["KScore_raw"] = df_task[[f"z_{c}" for c in kin_vars]].mean(axis=1)
    for col in kinetics_vars:
        df_task[f"z_{col}"] = (df_task[col] - df_task[col].mean()) / df_task[col].std()
    df_task["DScore_raw"] = df_task[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    sub_t = df_task.groupby("subject").agg(
        KScore=("KScore_raw", "mean"), DScore=("DScore_raw", "mean")
    ).reset_index()
    sub_t["KScore"] = (sub_t["KScore"] - sub_t["KScore"].mean()) / sub_t["KScore"].std()
    sub_t["DScore"] = (sub_t["DScore"] - sub_t["DScore"].mean()) / sub_t["DScore"].std()

    md_k = sub_t["KScore"].median(); md_d = sub_t["DScore"].median()
    kh = (sub_t["KScore"] >= md_k).astype(int)
    dh = (sub_t["DScore"] >= md_d).astype(int)

    rng2 = np.random.default_rng(SEED)
    rho_t_boot = []
    kappa_t_boot = []
    for _ in range(N_BOOT):
        idx = rng2.choice(len(sub_t), size=len(sub_t), replace=True)
        s = sub_t.iloc[idx]
        rho_t_boot.append(stats.spearmanr(s["KScore"], s["DScore"])[0])
        kappa_t_boot.append(cohen_kappa_score(
            (s["KScore"] >= s["KScore"].median()).astype(int),
            (s["DScore"] >= s["DScore"].median()).astype(int)))

    print(f"=== {task_label}: Bootstrap (N={len(sub_t)}, {N_BOOT} resamples) ===")
    print(f"  Spearman rho: {np.median(rho_t_boot):.3f} [{np.percentile(rho_t_boot, 2.5):.3f}, {np.percentile(rho_t_boot, 97.5):.3f}]")
    print(f"  Cohen kappa:  {np.median(kappa_t_boot):.3f} [{np.percentile(kappa_t_boot, 2.5):.3f}, {np.percentile(kappa_t_boot, 97.5):.3f}]")
    print()

print("=== SUMMARY FOR PAPER (use these in tables.tex) ===")
print(f"Primary pooled:  rho = {rho_pt:.3f} [{rho_ci_lo:.3f}, {rho_ci_hi:.3f}],  kappa = {kappa_pt:.3f} [{kappa_ci_lo:.3f}, {kappa_ci_hi:.3f}]")
print(f"  (cf. analytic: rho = {rho_obs:.3f} [{rho_an_lo:.3f}, {rho_an_hi:.3f}],  kappa = {kappa_obs:.3f} [{kappa_an_lo:.3f}, {kappa_an_hi:.3f}])")
