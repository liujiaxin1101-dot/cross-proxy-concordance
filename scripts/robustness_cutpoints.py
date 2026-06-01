import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")

df = pd.read_csv(FEATURES_CSV)

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

for c in kin_vars:
    df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
    if kin_dir[c] == -1:
        df[f"z_{c}"] = -df[f"z_{c}"]
df["K_raw"] = df[[f"z_{c}" for c in kin_vars]].mean(axis=1)

for c in kinetics_vars:
    df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
df["D_raw"] = df[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

sub = df.groupby("subject").agg(K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
sub["K"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
sub["D"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

n = len(sub)
rho, _ = stats.spearmanr(sub["K"], sub["D"])
se = 1 / np.sqrt(n - 3)
ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se)
ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se)

print("=" * 65)
print("ROBUSTNESS CHECK: Classification cut-point sensitivity")
print("=" * 65)
print(f"\n  Spearman rho = {rho:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}] (invariant to cut-point)")
print(f"  N = {n}\n")

cutpoints = [30, 35, 40, 45, 50, 55, 60, 65, 70]
print(f"  {'Cut':>6s}  {'K-thr':>8s}  {'D-thr':>8s}  {'kappa':>8s}  {'Agree%':>8s}  {'Conflict%':>10s}")
print(f"  {'-'*55}")

kappas = []
for cp in cutpoints:
    k_thr = np.percentile(sub["K"], cp)
    d_thr = np.percentile(sub["D"], cp)
    kh = (sub["K"] >= k_thr).astype(int)
    dh = (sub["D"] >= d_thr).astype(int)
    kappa = cohen_kappa_score(kh, dh)
    kappas.append(kappa)
    agree = (kh == dh).mean() * 100
    conflict = (kh != dh).mean() * 100
    print(f"  {cp:3d}th%  {k_thr:8.3f}  {d_thr:8.3f}  {kappa:8.3f}  {agree:7.1f}%  {conflict:9.1f}%")

print(f"\n  Kappa range across cut-points: [{min(kappas):.3f}, {max(kappas):.3f}]")
