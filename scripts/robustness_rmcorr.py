import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")

df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} trial-level records from {df['subject'].nunique()} subjects")

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

for c in kin_vars:
    df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
    if kin_dir[c] == -1:
        df[f"z_{c}"] = -df[f"z_{c}"]
df["K_trial"] = df[[f"z_{c}" for c in kin_vars]].mean(axis=1)

for c in kinetics_vars:
    df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
df["D_trial"] = df[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

def rm_corr(x, y, subject):
    subjects = np.unique(subject)
    x_demean = np.zeros(len(x))
    y_demean = np.zeros(len(y))
    for s in subjects:
        mask = subject == s
        x_demean[mask] = x[mask] - x[mask].mean()
        y_demean[mask] = y[mask] - y[mask].mean()
    num = np.sum(x_demean * y_demean)
    den = np.sqrt(np.sum(x_demean**2) * np.sum(y_demean**2))
    if den < 1e-12:
        return np.nan
    return num / den

def rm_corr_ci(r, n_subjects, n_total, alpha=0.05):
    k = n_total / n_subjects
    se = np.sqrt((1 - r**2) / (n_total - n_subjects - 1))
    se_adj = se * np.sqrt(k / (k - 1)) if k > 1 else se
    zr = np.arctanh(r)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(zr - z_crit * se_adj), np.tanh(zr + z_crit * se_adj)

n_sub = df['subject'].nunique()
n_total = len(df)

r_rm = rm_corr(df["K_trial"].values, df["D_trial"].values, df["subject"].values)
ci_lo, ci_hi = rm_corr_ci(r_rm, n_sub, n_total)

print(f"\n=== rmcorr (repeated measures correlation) ===")
print(f"  r_rm = {r_rm:.3f},  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  N_subjects = {n_sub}, N_trials = {n_total}")
print(f"  avg k = {n_total/n_sub:.1f} trials per subject")

sub = df.groupby("subject").agg(K=("K_trial", "mean"), D=("D_trial", "mean")).reset_index()
sub["K_z"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
sub["D_z"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

r_sub, _ = stats.spearmanr(sub["K_z"], sub["D_z"])
n_s = len(sub)
se_sub = 1 / np.sqrt(n_s - 3)
ci_lo_sub = np.tanh(np.arctanh(r_sub) - 1.96 * se_sub)
ci_hi_sub = np.tanh(np.arctanh(r_sub) + 1.96 * se_sub)

print(f"\n=== Subject-mean Spearman (for comparison) ===")
print(f"  rho_sub = {r_sub:.3f},  95% CI [{ci_lo_sub:.3f}, {ci_hi_sub:.3f}]")
print(f"  N = {n_s}")

print(f"\n=== COMPARISON ===")
print(f"  rmcorr    = {r_rm:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]  (accounts for within-subject dependence)")
print(f"  Spearman  = {r_sub:.3f}  [{ci_lo_sub:.3f}, {ci_hi_sub:.3f}]  (subject means, treats subjects as independent)")
print(f"  Delta     = {r_rm - r_sub:+.3f}")

if abs(r_rm - r_sub) < 0.1:
    print(f"  VERDICT: rmcorr and subject-mean Spearman produce similar estimates. "
          f"Non-independence adjustment does not qualitatively change the conclusion.")
else:
    print(f"  VERDICT: rmcorr and subject-mean Spearman differ by >0.10. "
          f"Non-independence warrants discussion.")

naive_r = np.corrcoef(df["K_trial"], df["D_trial"])[0, 1]
print(f"\n  Naive Pearson (all 132 trials as independent) = {naive_r:.3f}")
print(f"  This is biased because it ignores within-subject clustering.")
print(f"  rmcorr removes this bias by partialling out subject means.")
