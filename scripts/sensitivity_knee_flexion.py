import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_with_knee.csv")

df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} records")

kin_dir = {
    "hip_flex": -1,
    "knee_valg": 1,
    "trunk_lean": -1,
    "ankle_angle_sagittal": -1,
    "knee_flex": -1,
}

kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]
kin_dir_kin = {"peak_vgrf_bw": 1, "loading_rate_bw_s": 1}

print("\n=== COLLINEARITY CHECK ===")
sub_means = df.groupby("subject")[["hip_flex", "knee_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]].mean()
rho_hk, _ = stats.spearmanr(sub_means["hip_flex"], sub_means["knee_flex"])
rho_pooled, _ = stats.spearmanr(df["hip_flex"], df["knee_flex"])
print(f"Hip flex vs Knee flex (subject means): rho = {rho_hk:.3f}")
print(f"Hip flex vs Knee flex (trial-level): rho = {rho_pooled:.3f}")


def compute_all_metrics(sub_agg):
    n = len(sub_agg)
    rho, _ = stats.spearmanr(sub_agg["KScore"], sub_agg["DScore"])
    z = np.arctanh(rho)
    se_rho = 1.0 / np.sqrt(n - 3)
    rho_ci_lo = np.tanh(z - 1.96 * se_rho)
    rho_ci_hi = np.tanh(z + 1.96 * se_rho)

    median_k = sub_agg["KScore"].median()
    median_d = sub_agg["DScore"].median()
    sub_agg["K_hi"] = (sub_agg["KScore"] >= median_k).astype(int)
    sub_agg["D_hi"] = (sub_agg["DScore"] >= median_d).astype(int)
    kappa = cohen_kappa_score(sub_agg["K_hi"], sub_agg["D_hi"])
    conflict = (sub_agg["K_hi"] != sub_agg["D_hi"]).mean()

    return rho, rho_ci_lo, rho_ci_hi, kappa, conflict


def run_config(name, kin_vars, kin_dir_config):
    df_config = df.copy()
    for col in kin_vars:
        df_config[f"z_{col}"] = (df_config[col] - df_config[col].mean()) / df_config[col].std()
        if kin_dir_config[col] == -1:
            df_config[f"z_{col}"] = -df_config[f"z_{col}"]
    df_config["KScore_raw"] = df_config[[f"z_{c}" for c in kin_vars]].mean(axis=1)

    for col in kinetics_vars:
        df_config[f"z_{col}"] = (df_config[col] - df_config[col].mean()) / df_config[col].std()
    df_config["DScore_raw"] = df_config[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    sub_agg = df_config.groupby("subject").agg(
        KScore_raw_mean=("KScore_raw", "mean"),
        DScore_raw_mean=("DScore_raw", "mean"),
    ).reset_index()

    sub_agg["KScore"] = (sub_agg["KScore_raw_mean"] - sub_agg["KScore_raw_mean"].mean()) / sub_agg["KScore_raw_mean"].std()
    sub_agg["DScore"] = (sub_agg["DScore_raw_mean"] - sub_agg["DScore_raw_mean"].mean()) / sub_agg["DScore_raw_mean"].std()

    rho, rho_lo, rho_hi, kappa, conflict = compute_all_metrics(sub_agg)
    return rho, rho_lo, rho_hi, kappa, conflict


print("\n=== K-SCORE CONFIGURATION SENSITIVITY ===")
print(f"{'Configuration':<50} {'rho':>7} {'rho_CI':>22} {'kappa':>7} {'conflict':>8}")
print("-" * 100)

configs = [
    ("S0: Current 4-var (no knee flex)", ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"], kin_dir),
    ("S0a: 5-var (+knee flex)", ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal", "knee_flex"], kin_dir),
    ("S0b: 3-var (-ankle dorsi)", ["hip_flex", "knee_valg", "trunk_lean"], kin_dir),
    ("S0c: Replace ankle with knee flex", ["hip_flex", "knee_valg", "trunk_lean", "knee_flex"], kin_dir),
    ("S0d: Knee flex only", ["knee_flex"], kin_dir),
    ("S0e: Hip flex + knee flex + knee valg + trunk", ["hip_flex", "knee_flex", "knee_valg", "trunk_lean"], kin_dir),
]

results = []
for name, kin_vars, dir_config in configs:
    rho, rho_lo, rho_hi, kappa, conflict = run_config(name, kin_vars, dir_config)
    results.append({
        "config": name,
        "rho": rho,
        "rho_ci_lo": rho_lo,
        "rho_ci_hi": rho_hi,
        "kappa": kappa,
        "conflict": conflict,
    })
    print(f"{name:<50} {rho:7.3f} [{rho_lo:7.3f}, {rho_hi:7.3f}] {kappa:7.3f} {conflict:7.1%}")

print(f"\n=== TASK-STRATIFIED (5-var K-Score) ===")

kin_vars_5 = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal", "knee_flex"]

for task_label, task_mask in [("CMJ", df["trial_type"] == "CMJ"), ("DJ", df["trial_type"] == "DJ")]:
    df_task = df[task_mask].copy()
    for col in kin_vars_5:
        df_task[f"z_{col}"] = (df_task[col] - df_task[col].mean()) / df_task[col].std()
        if kin_dir[col] == -1:
            df_task[f"z_{col}"] = -df_task[f"z_{col}"]
    df_task["KScore_raw"] = df_task[[f"z_{c}" for c in kin_vars_5]].mean(axis=1)
    for col in kinetics_vars:
        df_task[f"z_{col}"] = (df_task[col] - df_task[col].mean()) / df_task[col].std()
    df_task["DScore_raw"] = df_task[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    sub_task = df_task.groupby("subject").agg(
        KScore_raw_mean=("KScore_raw", "mean"),
        DScore_raw_mean=("DScore_raw", "mean"),
    ).reset_index()
    sub_task["KScore"] = (sub_task["KScore_raw_mean"] - sub_task["KScore_raw_mean"].mean()) / sub_task["KScore_raw_mean"].std()
    sub_task["DScore"] = (sub_task["DScore_raw_mean"] - sub_task["DScore_raw_mean"].mean()) / sub_task["DScore_raw_mean"].std()

    rho_t, rho_lo_t, rho_hi_t, kappa_t, conflict_t = compute_all_metrics(sub_task)
    print(f"  {task_label}: rho={rho_t:.3f} [{rho_lo_t:.3f}, {rho_hi_t:.3f}], kappa={kappa_t:.3f}, conflict={conflict_t:.1%}")

print(f"\n=== KNEE FLEX vs HIP FLEX CORRELATION DETAIL ===")
for task_label, task_mask in [("Pooled", slice(None)), ("CMJ", df["trial_type"] == "CMJ"), ("DJ", df["trial_type"] == "DJ")]:
    sub_df = df[task_mask] if task_label != "Pooled" else df
    sub_means = sub_df.groupby("subject")[["hip_flex", "knee_flex"]].mean()
    rho_task, p_task = stats.spearmanr(sub_means["hip_flex"], sub_means["knee_flex"])
    print(f"  {task_label}: rho(hip_flex, knee_flex) = {rho_task:.3f} (subject means, n={len(sub_means)})")

df_results = pd.DataFrame(results)
print(f"\n{'='*70}")
print("SUMMARY: delta-rho and delta-kappa relative to baseline (S0)")
print(f"{'='*70}")
baseline = results[0]
for r in results[1:]:
    drho = r["rho"] - baseline["rho"]
    dkappa = r["kappa"] - baseline["kappa"]
    print(f"  {r['config']}: Delta_rho={drho:+.3f}, Delta_kappa={dkappa:+.3f}")
