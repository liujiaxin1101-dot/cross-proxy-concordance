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

print("=" * 72)
print("STEP 0: INPUT DATA")
print("=" * 72)
print(f"  Records: {len(df)}")
print(f"  Subjects: {df['subject'].nunique()}")
print(f"  CMJ trials: {(df['trial_type']=='CMJ').sum()}")
print(f"  DJ trials:  {(df['trial_type']=='DJ').sum()}")
print()

for col in kin_vars + kinetics_vars:
    vals = df[col]
    print(f"  {col:20s}  mean={vals.mean():10.4f}  std={vals.std():10.4f}  "
          f"min={vals.min():10.4f}  max={vals.max():10.4f}")

print()
print("=" * 72)
print("STEP 1: Z-SCORE NORMALIZATION (trial-level, N=132)")
print("=" * 72)

for col in kin_vars:
    raw = df[col].values
    mu, sigma = raw.mean(), raw.std()
    z = (raw - mu) / sigma
    if kin_dir[col] == -1:
        z = -z
    print(f"  {col:20s}  mean_raw={mu:.4f}  std_raw={sigma:.4f}  "
          f"direction={kin_dir[col]:+d}  z_mean={z.mean():.4f}  z_std={z.std():.4f}")

for col in kinetics_vars:
    raw = df[col].values
    mu, sigma = raw.mean(), raw.std()
    z = (raw - mu) / sigma
    print(f"  {col:20s}  mean_raw={mu:.4f}  std_raw={sigma:.4f}  "
          f"direction=+1  z_mean={z.mean():.4f}  z_std={z.std():.4f}")

all_z_kin = {}
for col in kin_vars:
    raw = df[col].values
    z = (raw - raw.mean()) / raw.std()
    if kin_dir[col] == -1:
        z = -z
    all_z_kin[col] = z
    df[f"z_{col}"] = z

all_z_dyn = {}
for col in kinetics_vars:
    raw = df[col].values
    z = (raw - raw.mean()) / raw.std()
    all_z_dyn[col] = z
    df[f"z_{col}"] = z

df["KScore_raw"] = df[[f"z_{c}" for c in kin_vars]].mean(axis=1)
df["DScore_raw"] = df[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

print()
print("=" * 72)
print("STEP 2: TRIAL-LEVEL RAW SCORES (N=132)")
print("=" * 72)
print(f"  KScore_raw: mean={df['KScore_raw'].mean():.6f}  std={df['KScore_raw'].std():.6f}")
print(f"  DScore_raw: mean={df['DScore_raw'].mean():.6f}  std={df['DScore_raw'].std():.6f}")

sub_agg = df.groupby("subject").agg(
    KScore_raw_mean=("KScore_raw", "mean"),
    DScore_raw_mean=("DScore_raw", "mean"),
    KScore_raw_std=("KScore_raw", "std"),
    DScore_raw_std=("DScore_raw", "std"),
    n_trials=("trial_type", "count"),
).reset_index()

print()
print("=" * 72)
print("STEP 3: SUBJECT-MEAN SCORES (raw, before 2nd z-score)")
print("=" * 72)
print(f"  {'subject':>7s}  {'K_raw':>8s}  {'K_std':>8s}  {'D_raw':>8s}  {'D_std':>8s}  n")
print(f"  {'-'*52}")
for _, row in sub_agg.iterrows():
    print(f"  {row['subject']:>7s}  {row['KScore_raw_mean']:8.4f}  {row['KScore_raw_std']:8.4f}  "
          f"{row['DScore_raw_mean']:8.4f}  {row['DScore_raw_std']:8.4f}  {int(row['n_trials'])}")

K_raw_mean = sub_agg["KScore_raw_mean"].mean()
K_raw_std = sub_agg["KScore_raw_mean"].std()
D_raw_mean = sub_agg["DScore_raw_mean"].mean()
D_raw_std = sub_agg["DScore_raw_mean"].std()

print(f"\n  Mean of subject-level KScore_raw: {K_raw_mean:.6f}")
print(f"  Std  of subject-level KScore_raw: {K_raw_std:.6f}")
print(f"  Mean of subject-level DScore_raw: {D_raw_mean:.6f}")
print(f"  Std  of subject-level DScore_raw: {D_raw_std:.6f}")

sub_agg["KScore"] = (sub_agg["KScore_raw_mean"] - K_raw_mean) / K_raw_std
sub_agg["DScore"] = (sub_agg["DScore_raw_mean"] - D_raw_mean) / D_raw_std

print()
print("=" * 72)
print("STEP 4: FINAL SUBJECT-LEVEL SCORES (after 2nd z-score)")
print("=" * 72)
print(f"  KScore: mean={sub_agg['KScore'].mean():.6f}  std={sub_agg['KScore'].std():.6f}")
print(f"  DScore: mean={sub_agg['DScore'].mean():.6f}  std={sub_agg['DScore'].std():.6f}")
print(f"  median K = {sub_agg['KScore'].median():.6f}")
print(f"  median D = {sub_agg['DScore'].median():.6f}")
print()
print(f"  {'subject':>7s}  {'KScore':>8s}  {'DScore':>8s}  {'diff':>8s}")
print(f"  {'-'*38}")
for _, row in sub_agg.iterrows():
    diff = row["KScore"] - row["DScore"]
    print(f"  {row['subject']:>7s}  {row['KScore']:8.4f}  {row['DScore']:8.4f}  {diff:+8.4f}")

print()
print("=" * 72)
print("STEP 5: SPEARMAN RHO")
print("=" * 72)

ks = sub_agg["KScore"].values
ds = sub_agg["DScore"].values
n = len(ks)

rank_k = stats.rankdata(ks)
rank_d = stats.rankdata(ds)

print(f"  N = {n}")
print(f"  {'subject':>7s}  {'KScore':>8s}  {'rank_K':>6s}  {'DScore':>8s}  {'rank_D':>6s}  {'d':>6s}  {'d^2':>8s}")
print(f"  {'-'*60}")

sum_d2 = 0.0
for i, row in sub_agg.iterrows():
    d = rank_k[i] - rank_d[i]
    d2 = d * d
    sum_d2 += d2
    print(f"  {row['subject']:>7s}  {row['KScore']:8.4f}  {int(rank_k[i]):6d}  "
          f"{row['DScore']:8.4f}  {int(rank_d[i]):6d}  {int(d):+6d}  {d2:8.1f}")

rho_manual = 1.0 - (6.0 * sum_d2) / (n * (n * n - 1.0))
rho_scipy, p_rho = stats.spearmanr(ks, ds)

print(f"\n  Sum d^2 = {sum_d2:.1f}")
print(f"  rho (manual, no ties correction) = 1 - 6*{sum_d2:.1f} / ({n}*({n}^2-1)) = {rho_manual:.6f}")
print(f"  rho (scipy, with ties correction) = {rho_scipy:.6f}")
print(f"  p = {p_rho:.6f}")

se = 1.0 / np.sqrt(n - 3)
z_val = np.arctanh(rho_scipy)
ci_lo = np.tanh(z_val - 1.96 * se)
ci_hi = np.tanh(z_val + 1.96 * se)

print(f"\n  Fisher z transformation:")
print(f"    arctanh(rho) = arctanh({rho_scipy:.6f}) = {z_val:.6f}")
print(f"    SE = 1 / sqrt({n} - 3) = {se:.6f}")
print(f"    z_lo = {z_val:.6f} - 1.96 * {se:.6f} = {z_val - 1.96*se:.6f}")
print(f"    z_hi = {z_val:.6f} + 1.96 * {se:.6f} = {z_val + 1.96*se:.6f}")
print(f"    CI = [tanh({z_val - 1.96*se:.6f}), tanh({z_val + 1.96*se:.6f})]")
print(f"    CI = [{ci_lo:.6f}, {ci_hi:.6f}]")

print()
print("=" * 72)
print("STEP 6: MEDIAN-SPLIT BINARY CLASSIFICATION")
print("=" * 72)

median_k = sub_agg["KScore"].median()
median_d = sub_agg["DScore"].median()

print(f"  median K-Score = {median_k:.6f}")
print(f"  median D-Score = {median_d:.6f}")
print()
print(f"  {'subject':>7s}  {'KScore':>8s}  {'K>=med?':>7s}  {'DScore':>8s}  {'D>=med?':>7s}  {'Agree?':>7s}")
print(f"  {'-'*53}")

kh = (sub_agg["KScore"] >= median_k).astype(int)
dh = (sub_agg["DScore"] >= median_d).astype(int)
agree_vec = (kh == dh)
conflict_vec = (kh != dh)

n_agree = int(agree_vec.sum())
n_total = len(sub_agg)
po = n_agree / n_total

for i, row in sub_agg.iterrows():
    print(f"  {row['subject']:>7s}  {row['KScore']:8.4f}  "
          f"{'HIGH' if kh[i] else 'low':>7s}  "
          f"{row['DScore']:8.4f}  "
          f"{'HIGH' if dh[i] else 'low':>7s}  "
          f"{'YES' if agree_vec[i] else 'NO':>7s}")

print()
print("=" * 72)
print("STEP 7: COHEN'S KAPPA")
print("=" * 72)

kappa = cohen_kappa_score(kh, dh)

table = pd.crosstab(
    pd.Series(kh, name="K_hi").map({1: "High", 0: "Low"}),
    pd.Series(dh, name="D_hi").map({1: "High", 0: "Low"})
)
kh_arr = kh.values if hasattr(kh, 'values') else kh
dh_arr = dh.values if hasattr(dh, 'values') else dh

n_00 = int(((kh_arr == 0) & (dh_arr == 0)).sum())
n_01 = int(((kh_arr == 0) & (dh_arr == 1)).sum())
n_10 = int(((kh_arr == 1) & (dh_arr == 0)).sum())
n_11 = int(((kh_arr == 1) & (dh_arr == 1)).sum())

print(f"  Confusion matrix:")
print(f"              D_Low  D_High")
print(f"    K_Low      {n_00}      {n_01}")
print(f"    K_High     {n_10}      {n_11}")
print()

po = n_agree / n_total
pe = 0.5
kappa_manual = (po - pe) / (1.0 - pe)

print(f"  Observed agreement  po = {n_agree}/{n_total} = {po:.6f}")
print(f"  Expected agreement  pe = 0.5 (by median-split construction)")
print(f"  kappa (manual) = (po - pe) / (1 - pe)")
print(f"             = ({po:.6f} - {pe:.6f}) / (1 - {pe:.6f})")
print(f"             = {po - pe:.6f} / {1.0 - pe:.6f}")
print(f"             = {kappa_manual:.6f}")
print(f"  kappa (sklearn)= {kappa:.6f}")
print()

kappa_se = np.sqrt(po * (1 - po) / (n_total * (1 - pe) ** 2))
kappa_ci_lo = kappa - 1.96 * kappa_se
kappa_ci_hi = kappa + 1.96 * kappa_se

print(f"  kappa SE = sqrt(po(1-po) / (n(1-pe)^2))")
print(f"           = sqrt({po:.6f}*{1-po:.6f} / ({n_total}*{(1-pe)**2:.6f}))")
print(f"           = sqrt({po*(1-po):.6f} / {n_total*(1-pe)**2:.6f})")
print(f"           = {kappa_se:.6f}")
print(f"  kappa 95% CI = [{kappa:.6f} - 1.96*{kappa_se:.6f}, {kappa:.6f} + 1.96*{kappa_se:.6f}]")
print(f"           = [{kappa_ci_lo:.6f}, {kappa_ci_hi:.6f}]")

n_conflict = int(conflict_vec.sum())
print(f"\n  Conflicts: {n_conflict}/{n_total} ({n_conflict/n_total*100:.1f}%)")

if n_01 + n_10 > 0:
    mcnemar_stat = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
else:
    mcnemar_stat, mcnemar_p = 0.0, 1.0

print(f"  McNemar chi2 (Yates continuity corr.) = {mcnemar_stat:.6f}, p = {mcnemar_p:.6f}")

print()
print("=" * 72)
print("STEP 8: BLAND-ALTMAN")
print("=" * 72)

diffs = ks - ds
bias = diffs.mean()
sd_diff = diffs.std(ddof=0)
loa_lo = bias - 1.96 * sd_diff
loa_hi = bias + 1.96 * sd_diff

print(f"  Bias = {bias:.6f}")
print(f"  SD of differences = {sd_diff:.6f}")
print(f"  LoA = [{loa_lo:.6f}, {loa_hi:.6f}]")

print()
print("=" * 72)
print("STEP 9: POWER ANALYSIS")
print("=" * 72)

za = stats.norm.ppf(0.975)
zb = stats.norm.ppf(0.80)
for target_rho in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    n_needed = ((za + zb) / np.arctanh(target_rho)) ** 2 + 3
    print(f"  rho = {target_rho:.2f} -> N needed (80% power) = {int(np.ceil(n_needed))}")

print(f"\n  At current N = {n}:")
rho_50pct = np.tanh(za / np.sqrt(n - 3))
rho_80pct = np.tanh((za + zb) / np.sqrt(n - 3))
print(f"    50% power to detect rho >= {rho_50pct:.4f}")
print(f"    80% power to detect rho >= {rho_80pct:.4f}")
