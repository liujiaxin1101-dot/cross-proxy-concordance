import numpy as np
import pandas as pd
from scipy import stats
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


np.random.seed(42)

FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")
df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} records, {df['subject'].nunique()} subjects")

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

for col in kin_vars:
    z = (df[col] - df[col].mean()) / df[col].std()
    df[f"z_{col}"] = z if kin_dir[col] == 1 else -z
df["KScore_raw"] = df[[f"z_{c}" for c in kin_vars]].mean(axis=1)

for col in kinetics_vars:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
df["DScore_raw"] = df[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

sub = df.groupby("subject").agg(
    hip_flex=("hip_flex", "mean"),
    knee_valg=("knee_valg", "mean"),
    trunk_lean=("trunk_lean", "mean"),
    ankle_angle_sagittal=("ankle_angle_sagittal", "mean"),
    peak_vgrf_bw=("peak_vgrf_bw", "mean"),
    loading_rate_bw_s=("loading_rate_bw_s", "mean"),
    KScore_raw=("KScore_raw", "mean"),
    DScore_raw=("DScore_raw", "mean"),
).reset_index()

N = len(sub)
N_BOOT = 10000

def boot_ci(x, y):
    rho, _ = stats.spearmanr(x, y)
    boots = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx = np.random.randint(0, N, size=N)
        b, _ = stats.spearmanr(x.iloc[idx], y.iloc[idx])
        boots[i] = b
    ci_lo = np.percentile(boots, 2.5)
    ci_hi = np.percentile(boots, 97.5)
    return rho, ci_lo, ci_hi

print("\n=== Each K-Score component vs D-Score (subject-means, raw values) ===")
print(f"{'Component':>18}  {'rho':>8}  {'95% CI':>24}")
print("-" * 58)
for comp in kin_vars:
    rho, ci_lo, ci_hi = boot_ci(sub[comp], sub["DScore_raw"])
    print(f"{comp:>18}  {rho:>8.3f}  [{ci_lo:>8.3f}, {ci_hi:>8.3f}]")

rho_k, ci_lo_k, ci_hi_k = boot_ci(sub["KScore_raw"], sub["DScore_raw"])
print(f"{'K-Score (composite)':>18}  {rho_k:>8.3f}  [{ci_lo_k:>8.3f}, {ci_hi_k:>8.3f}]")

print("\n=== Each kinetic component vs K-Score ===")
print(f"{'Component':>18}  {'rho':>8}  {'95% CI':>24}")
print("-" * 58)
for comp in kinetics_vars:
    rho, ci_lo, ci_hi = boot_ci(sub[comp], sub["KScore_raw"])
    print(f"{comp:>18}  {rho:>8.3f}  [{ci_lo:>8.3f}, {ci_hi:>8.3f}]")

# Direction-corrected
print("\n=== Direction-corrected K-Score components vs D-Score ===")
print("(* = multiplied by -1: higher = 'more loading-like')")
print(f"{'Component':>18}  {'rho':>8}  {'95% CI':>24}")
print("-" * 58)
for comp in kin_vars:
    sign = -1 if kin_dir[comp] == -1 else 1
    val = sub[comp] * sign
    label = f"{comp}*" if kin_dir[comp] == -1 else comp
    rho, ci_lo, ci_hi = boot_ci(val, sub["DScore_raw"])
    print(f"{label:>18}  {rho:>8.3f}  [{ci_lo:>8.3f}, {ci_hi:>8.3f}]")

# Task-stratified
print("\n" + "="*70)
print("TASK-STRATIFIED")
print("="*70)

for task_label in ["CMJ", "DJ"]:
    df_task = df[df["trial_type"] == task_label].copy()
    for col in kin_vars:
        z = (df_task[col] - df_task[col].mean()) / df_task[col].std()
        df_task[f"z_{col}"] = z if kin_dir[col] == 1 else -z
    df_task["KScore_raw"] = df_task[[f"z_{c}" for c in kin_vars]].mean(axis=1)
    for col in kinetics_vars:
        df_task[f"z_{col}"] = (df_task[col] - df_task[col].mean()) / df_task[col].std()
    df_task["DScore_raw"] = df_task[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    sub_task = df_task.groupby("subject").agg(
        **{c: (c, "mean") for c in kin_vars},
        **{c: (c, "mean") for c in kinetics_vars},
        KScore_raw=("KScore_raw", "mean"),
        DScore_raw=("DScore_raw", "mean"),
    ).reset_index()

    print(f"\n--- {task_label} (N={len(sub_task)}) ---")
    print(f"{'Component':>18}  {'rho':>8}  {'95% CI':>24}")
    print("-" * 58)
    for comp in kin_vars:
        rho, ci_lo, ci_hi = boot_ci(sub_task[comp], sub_task["DScore_raw"])
        print(f"{comp:>18}  {rho:>8.3f}  [{ci_lo:>8.3f}, {ci_hi:>8.3f}]")
    rho_t, ci_lo_t, ci_hi_t = boot_ci(sub_task["KScore_raw"], sub_task["DScore_raw"])
    print(f"{'K-Score (composite)':>18}  {rho_t:>8.3f}  [{ci_lo_t:>8.3f}, {ci_hi_t:>8.3f}]")

print("\nDone.")
