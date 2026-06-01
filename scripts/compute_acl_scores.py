import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import json
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_combined.csv")

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]


def compute_group_scores(df, group_label, z_pool=None):
    dfc = df.copy()
    for col in kin_vars:
        dfc[f"z_{col}"] = (dfc[col] - dfc[col].mean()) / dfc[col].std()
        if kin_dir[col] == -1:
            dfc[f"z_{col}"] = -dfc[f"z_{col}"]
    dfc["K_raw"] = dfc[[f"z_{col}" for col in kin_vars]].mean(axis=1)
    for col in kinetics_vars:
        dfc[f"z_{col}"] = (dfc[col] - dfc[col].mean()) / dfc[col].std()
    dfc["D_raw"] = dfc[[f"z_{col}" for col in kinetics_vars]].mean(axis=1)

    sub = dfc.groupby("subject").agg(
        K=("K_raw", "mean"), D=("D_raw", "mean"), n=("trial_type", "count")
    ).reset_index()
    sub["Kz"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
    sub["Dz"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

    n = len(sub)
    rho, p_rho = stats.spearmanr(sub["Kz"], sub["Dz"])
    se = 1.0 / np.sqrt(n - 3)
    ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se)
    ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se)

    md_k = sub["Kz"].median()
    md_d = sub["Dz"].median()
    kh = (sub["Kz"] >= md_k).astype(int)
    dh = (sub["Dz"] >= md_d).astype(int)
    kappa = cohen_kappa_score(kh, dh)
    conflict = (kh != dh).mean()

    return {
        "group": group_label, "n": n,
        "rho": float(rho), "rho_ci_lo": float(ci_lo), "rho_ci_hi": float(ci_hi),
        "kappa": float(kappa), "conflict_pct": float(conflict * 100),
        "k_mean": float(sub["K"].mean()), "k_std": float(sub["K"].std()),
        "d_mean": float(sub["D"].mean()), "d_std": float(sub["D"].std()),
    }


def compute_task_stratified(df, group_label):
    results = []
    for task in ["CMJ", "DJ"]:
        dft = df[df["trial_type"] == task].copy()
        if len(dft) < 5:
            continue
        for col in kin_vars:
            dft[f"z_{col}"] = (dft[col] - dft[col].mean()) / dft[col].std()
            if kin_dir[col] == -1:
                dft[f"z_{col}"] = -dft[f"z_{col}"]
        dft["K_raw"] = dft[[f"z_{col}" for col in kin_vars]].mean(axis=1)
        for col in kinetics_vars:
            dft[f"z_{col}"] = (dft[col] - dft[col].mean()) / dft[col].std()
        dft["D_raw"] = dft[[f"z_{col}" for col in kinetics_vars]].mean(axis=1)

        sub = dft.groupby("subject").agg(
            K=("K_raw", "mean"), D=("D_raw", "mean")
        ).reset_index()
        sub["Kz"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
        sub["Dz"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

        n = len(sub)
        rho, _ = stats.spearmanr(sub["Kz"], sub["Dz"])
        se = 1.0 / np.sqrt(n - 3)
        ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se)
        ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se)

        kh = (sub["Kz"] >= sub["Kz"].median()).astype(int)
        dh = (sub["Dz"] >= sub["Dz"].median()).astype(int)
        kappa = cohen_kappa_score(kh, dh)
        conflict = (kh != dh).mean()

        results.append({
            "group": f"{group_label}-{task}", "n": n,
            "rho": float(rho), "rho_ci_lo": float(ci_lo), "rho_ci_hi": float(ci_hi),
            "kappa": float(kappa), "conflict_pct": float(conflict * 100),
        })
    return results


df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} records, {df.subject.nunique()} subjects")
print(f"  Control: {df[df.group == 'Control'].subject.nunique()}")
print(f"  ACL:     {df[df.group == 'ACL'].subject.nunique()}")

all_results = []
all_results.append(compute_group_scores(df, "Pooled (N=43)"))
all_results.append(compute_group_scores(df[df.group == "Control"], "Control (N=22)"))
all_results.append(compute_group_scores(df[df.group == "ACL"], "ACL (N=21)"))

all_results.extend(compute_task_stratified(df, "Pooled"))
all_results.extend(compute_task_stratified(df[df.group == "Control"], "CTL"))
all_results.extend(compute_task_stratified(df[df.group == "ACL"], "ACL"))

print(f"\n{'Group':<25s} {'N':>3s} {'rho':>7s} {'95% CI':>24s} {'kappa':>7s} {'conflict':>9s}")
print("-" * 82)
for r in all_results:
    print(f"  {r['group']:<25s} {r['n']:3d} {r['rho']:7.3f} [{r['rho_ci_lo']:7.3f}, {r['rho_ci_hi']:7.3f}] "
          f"{r['kappa']:7.3f} {r['conflict_pct']:7.1f}%")

acl_out = os.path.join(PROJECT_ROOT, "data", r"acl_results_summary.json")
with open(acl_out, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {acl_out}")
