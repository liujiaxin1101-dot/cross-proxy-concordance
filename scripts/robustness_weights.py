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

weight_schemes = {
    "Equal": {
        "kin": {"hip_flex": 1, "knee_valg": 1, "trunk_lean": 1, "ankle_angle_sagittal": 1},
        "kinetics": {"peak_vgrf_bw": 1, "loading_rate_bw_s": 1},
    },
    "KneeValgus_x2": {
        "kin": {"hip_flex": 1, "knee_valg": 2, "trunk_lean": 1, "ankle_angle_sagittal": 1},
        "kinetics": {"peak_vgrf_bw": 1, "loading_rate_bw_s": 1},
    },
    "LoadingRate_x2": {
        "kin": {"hip_flex": 1, "knee_valg": 1, "trunk_lean": 1, "ankle_angle_sagittal": 1},
        "kinetics": {"peak_vgrf_bw": 1, "loading_rate_bw_s": 2},
    },
    "K-Single_KneeValgus": {
        "kin": {"hip_flex": 0, "knee_valg": 1, "trunk_lean": 0, "ankle_angle_sagittal": 0},
        "kinetics": {"peak_vgrf_bw": 1, "loading_rate_bw_s": 1},
    },
    "D-Single_PeakGRF": {
        "kin": {"hip_flex": 1, "knee_valg": 1, "trunk_lean": 1, "ankle_angle_sagittal": 1},
        "kinetics": {"peak_vgrf_bw": 1, "loading_rate_bw_s": 0},
    },
}


def compute_scores(data, weights_kin, weights_kinetics):
    dfc = data.copy()
    w_sum_k = sum(weights_kin.values())
    w_sum_d = sum(weights_kinetics.values())
    if w_sum_k == 0 or w_sum_d == 0:
        return None

    for c in kin_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
        if kin_dir[c] == -1:
            dfc[f"z_{c}"] = -dfc[f"z_{c}"]
    dfc["K_raw"] = sum(weights_kin[c] * dfc[f"z_{c}"] for c in kin_vars) / w_sum_k

    for c in kinetics_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
    dfc["D_raw"] = sum(weights_kinetics[c] * dfc[f"z_{c}"] for c in kinetics_vars) / w_sum_d

    sub = dfc.groupby("subject").agg(K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
    sub["K"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
    sub["D"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

    n = len(sub)
    rho, _ = stats.spearmanr(sub["K"], sub["D"])
    se = 1 / np.sqrt(n - 3)
    ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se)
    ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se)

    md_k, md_d = sub["K"].median(), sub["D"].median()
    kh = (sub["K"] >= md_k).astype(int)
    dh = (sub["D"] >= md_d).astype(int)
    kappa = cohen_kappa_score(kh, dh)
    conflict = (kh != dh).mean()

    return {"rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi, "kappa": kappa, "conflict": conflict}


print("=" * 70)
print("ROBUSTNESS CHECK: Weight Sensitivity")
print("=" * 70)

results = {}
for name, scheme in weight_schemes.items():
    r = compute_scores(df, scheme["kin"], scheme["kinetics"])
    if r:
        results[name] = r
        print(f"  {name:25s}  rho={r['rho']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
              f"kappa={r['kappa']:.3f}  conflict={r['conflict']*100:.0f}%")

print()

base = results["Equal"]
print(f"  {'Scheme':25s} {'Delta rho':>10s} {'Delta kappa':>12s} {'Delta conflict':>15s}")
print(f"  {'-'*65}")
for name, r in results.items():
    if name == "Equal":
        continue
    dr = r["rho"] - base["rho"]
    dk = r["kappa"] - base["kappa"]
    dc = (r["conflict"] - base["conflict"]) * 100
    print(f"  {name:25s} {dr:+10.3f} {dk:+12.3f} {dc:+13.0f}pp")

print()
max_drho = max(abs(r["rho"] - base["rho"]) for n, r in results.items() if n != "Equal")
max_dkappa = max(abs(r["kappa"] - base["kappa"]) for n, r in results.items() if n != "Equal")
print(f"  Max |Delta rho|   = {max_drho:.3f}")
print(f"  Max |Delta kappa| = {max_dkappa:.3f}")
print(f"  VERDICT: {'Weight changes shift rho by up to ' + str(round(max_drho, 3)) + ' and kappa by up to ' + str(round(max_dkappa, 3)) + '. ' + ('Non-trivial sensitivity 鈥?report in discussion.' if max_drho > 0.10 or max_dkappa > 0.10 else 'Minor sensitivity 鈥?conclusions robust to weight perturbations.')}")
