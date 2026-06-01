import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")

df = pd.read_csv(FEATURES_CSV)

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

def compute_scores(data):
    dfc = data.copy()
    for col in kin_vars:
        dfc[f"z_{col}"] = (dfc[col] - dfc[col].mean()) / dfc[col].std()
        if kin_dir[col] == -1:
            dfc[f"z_{col}"] = -dfc[f"z_{col}"]
    dfc["KScore_raw"] = dfc[[f"z_{c}" for c in kin_vars]].mean(axis=1)

    for col in kinetics_vars:
        dfc[f"z_{col}"] = (dfc[col] - dfc[col].mean()) / dfc[col].std()
    dfc["DScore_raw"] = dfc[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    sub = dfc.groupby("subject").agg(
        KScore=("KScore_raw", "mean"), DScore=("DScore_raw", "mean"),
    ).reset_index()

    sub["KScore"] = (sub["KScore"] - sub["KScore"].mean()) / sub["KScore"].std()
    sub["DScore"] = (sub["DScore"] - sub["DScore"].mean()) / sub["DScore"].std()

    return sub

sub_full = compute_scores(df)
sub_excl = compute_scores(df[df["subject"] != "sub24"])

scenarios = [
    ("Full (N=22)", sub_full),
    ("Excl sub24 (N=21)", sub_excl),
]

print("=" * 70)
print("SENSITIVITY ANALYSIS 鈥?Excluding sub24")
print("=" * 70)

for label, sub in scenarios:
    n = len(sub)
    median_k = sub["KScore"].median()
    median_d = sub["DScore"].median()
    k_hi = (sub["KScore"] >= median_k).astype(int)
    d_hi = (sub["DScore"] >= median_d).astype(int)
    conflict = (k_hi != d_hi).astype(int)

    rho, p_rho = stats.spearmanr(sub["KScore"], sub["DScore"])
    se = 1.0 / np.sqrt(n - 3)
    z_val = np.arctanh(rho)
    ci_lo = np.tanh(z_val - 1.96 * se)
    ci_hi = np.tanh(z_val + 1.96 * se)

    kappa = cohen_kappa_score(k_hi, d_hi)
    agree = (k_hi == d_hi).mean()
    n_conflict = conflict.sum()
    pct = conflict.mean() * 100

    n_kp_dm = int(sum((k_hi == 1) & (d_hi == 0)))
    n_km_dp = int(sum((k_hi == 0) & (d_hi == 1)))

    if n_kp_dm + n_km_dp > 0:
        mcnemar_stat = (abs(n_kp_dm - n_km_dp) - 1) ** 2 / (n_kp_dm + n_km_dp)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
    else:
        mcnemar_stat, mcnemar_p = 0.0, 1.0

    print(f"\n--- {label} ---")
    print(f"  Spearman rho        = {rho:.3f},  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Cohen's kappa       = {kappa:.3f}")
    print(f"  Agreement           = {agree*100:.1f}%")
    print(f"  Conflicts           = {n_conflict}/{n} ({pct:.0f}%)")
    print(f"    K+D- = {n_kp_dm},  K-D+ = {n_km_dp}")
    print(f"  McNemar chi2        = {mcnemar_stat:.3f}, p = {mcnemar_p:.3f}")

delta_rho = scenarios[1][1]["KScore"].corr(scenarios[1][1]["DScore"], method='spearman') - \
            scenarios[0][1]["KScore"].corr(scenarios[0][1]["DScore"], method='spearman')

print(f"\n{'='*70}")
print(f"DELTA (Excl - Full)")
print(f"  Delta rho   = {delta_rho:+.3f}")
kappa_full = cohen_kappa_score(
    (scenarios[0][1]["KScore"] >= scenarios[0][1]["KScore"].median()).astype(int),
    (scenarios[0][1]["DScore"] >= scenarios[0][1]["DScore"].median()).astype(int))
kappa_excl = cohen_kappa_score(
    (scenarios[1][1]["KScore"] >= scenarios[1][1]["KScore"].median()).astype(int),
    (scenarios[1][1]["DScore"] >= scenarios[1][1]["DScore"].median()).astype(int))
print(f"  Delta kappa = {kappa_excl - kappa_full:+.3f}")

print(f"\nCONCLUSION: Removing sub24 shifts rho by {delta_rho:+.3f} and kappa by "
      f"{kappa_excl - kappa_full:+.3f}. The effect is { 'substantial' if abs(delta_rho) > 0.1 else 'modest' if abs(delta_rho) > 0.05 else 'minor' }. "
      f"The qualitative conclusion ({ 'does' if kappa_excl < 0.60 and scenarios[1][1]['KScore'].corr(scenarios[1][1]['DScore'], method='spearman') < 0.70 else 'may' } not change).")
