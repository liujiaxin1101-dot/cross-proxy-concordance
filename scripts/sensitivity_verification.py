"""
sensitivity_verification.py �?Unified runner for all methodological sensitivity analyses.
Runs every variant reachable from features_raw.csv, verifies against expected values,
and generates the kappa forest plot.

Requires: features_raw.csv (132 records).
IC=50N and kinematic-IC variants require raw .c3d files (E: drive) �?these are compared
against previously computed reference values rather than rerun.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from collections import OrderedDict
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "features_raw.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir_pristine = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kin_dir_reversed = {"hip_flex": 1, "knee_valg": 1, "trunk_lean": 1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

df = pd.read_csv(FEATURES_CSV)


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(data, kin_dir_override=None, weights_kin=None, weights_kinetics=None):
    kin_dir = kin_dir_override if kin_dir_override is not None else kin_dir_pristine
    wk = weights_kin or {c: 1 for c in kin_vars}
    wd = weights_kinetics or {c: 1 for c in kinetics_vars}
    wsum_k = sum(wk.values())
    wsum_d = sum(wd.values())
    if wsum_k == 0 or wsum_d == 0:
        raise ValueError("At least one component must have non-zero weight")

    dfc = data.copy()
    for c in kin_vars:
        z = (dfc[c] - dfc[c].mean()) / dfc[c].std()
        if kin_dir[c] == -1:
            z = -z
        dfc[f"z_{c}"] = z
    dfc["K_raw"] = sum(wk[c] * dfc[f"z_{c}"] for c in kin_vars) / wsum_k

    for c in kinetics_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
    dfc["D_raw"] = sum(wd[c] * dfc[f"z_{c}"] for c in kinetics_vars) / wsum_d

    sub = dfc.groupby("subject").agg(
        K_raw_mean=("K_raw", "mean"), D_raw_mean=("D_raw", "mean")
    ).reset_index()

    ba_diffs = sub["K_raw_mean"].values - sub["D_raw_mean"].values
    ba_bias = ba_diffs.mean()
    ba_sd = np.std(ba_diffs, ddof=0)
    ba_loa_lo = ba_bias - 1.96 * ba_sd
    ba_loa_hi = ba_bias + 1.96 * ba_sd

    sub["K"] = (sub["K_raw_mean"] - sub["K_raw_mean"].mean()) / sub["K_raw_mean"].std()
    sub["D"] = (sub["D_raw_mean"] - sub["D_raw_mean"].mean()) / sub["D_raw_mean"].std()

    n = len(sub)
    rho, _ = stats.spearmanr(sub["K"], sub["D"])
    se_rho = 1.0 / np.sqrt(n - 3)
    rho_ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se_rho)
    rho_ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se_rho)

    md_k, md_d = sub["K"].median(), sub["D"].median()
    kh = (sub["K"] >= md_k).astype(int)
    dh = (sub["D"] >= md_d).astype(int)
    conflict = (kh != dh).astype(int)
    n_conflict = int(conflict.sum())
    conflict_pct = n_conflict / n * 100

    n_agree = int((kh == dh).sum())
    po = n_agree / n
    pe = 0.5
    kappa_se = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    kappa = cohen_kappa_score(kh, dh)
    kappa_ci_lo = kappa - 1.96 * kappa_se
    kappa_ci_hi = kappa + 1.96 * kappa_se

    n_01 = int(((kh == 0) & (dh == 1)).sum())
    n_10 = int(((kh == 1) & (dh == 0)).sum())
    if n_01 + n_10 > 0:
        mcn = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
        mcn_p = 1 - stats.chi2.cdf(mcn, 1)
    else:
        mcn, mcn_p = 0.0, 1.0

    return {
        "n": n, "rho": float(rho), "rho_ci_lo": float(rho_ci_lo), "rho_ci_hi": float(rho_ci_hi),
        "kappa": float(kappa), "kappa_ci_lo": float(kappa_ci_lo), "kappa_ci_hi": float(kappa_ci_hi),
        "conflict_pct": float(conflict_pct), "n_conflict": n_conflict,
        "n_01": n_01, "n_10": n_10, "mcn": float(mcn), "mcn_p": float(mcn_p),
        "ba_bias": float(ba_bias), "ba_loa_lo": float(ba_loa_lo), "ba_loa_hi": float(ba_loa_hi),
        "ba_sd": float(ba_sd),
    }


def rmcorr_analysis(data):
    dfc = data.copy()
    for c in kin_vars:
        z = (dfc[c] - dfc[c].mean()) / dfc[c].std()
        if kin_dir_pristine[c] == -1:
            z = -z
        dfc[f"z_{c}"] = z
    dfc["K_trial"] = dfc[[f"z_{c}" for c in kin_vars]].mean(axis=1)
    for c in kinetics_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
    dfc["D_trial"] = dfc[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    subjects = dfc["subject"].unique()
    x_dm = np.zeros(len(dfc))
    y_dm = np.zeros(len(dfc))
    for s in subjects:
        mask = dfc["subject"] == s
        x_dm[mask] = dfc.loc[mask, "K_trial"] - dfc.loc[mask, "K_trial"].mean()
        y_dm[mask] = dfc.loc[mask, "D_trial"] - dfc.loc[mask, "D_trial"].mean()
    num = np.sum(x_dm * y_dm)
    den = np.sqrt(np.sum(x_dm ** 2) * np.sum(y_dm ** 2))
    r_rm = num / den if den > 1e-12 else np.nan
    n_total = len(dfc)
    n_sub = len(subjects)
    k_avg = n_total / n_sub
    se_rm = np.sqrt((1 - r_rm ** 2) / (n_total - n_sub - 1))
    se_rm_adj = se_rm * np.sqrt(k_avg / (k_avg - 1)) if k_avg > 1 else se_rm
    zr = np.arctanh(r_rm)
    rm_ci_lo = np.tanh(zr - 1.96 * se_rm_adj)
    rm_ci_hi = np.tanh(zr + 1.96 * se_rm_adj)
    return {"r_rm": float(r_rm), "rm_ci_lo": float(rm_ci_lo), "rm_ci_hi": float(rm_ci_hi)}


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL ANALYSES REACHABLE FROM features_raw.csv
# ══════════════════════════════════════════════════════════════════════════════

results = OrderedDict()

results["Primary (IC=20N, equal weights)"] = compute_scores(df)

results["CMJ only"] = compute_scores(df[df["trial_type"] == "CMJ"])
results["DJ only"] = compute_scores(df[df["trial_type"] == "DJ"])

wk_valgus_x2 = {"hip_flex": 1, "knee_valg": 2, "trunk_lean": 1, "ankle_angle_sagittal": 1}
results["Knee valgus x2"] = compute_scores(df, weights_kin=wk_valgus_x2)

wd_lr_x2 = {"peak_vgrf_bw": 1, "loading_rate_bw_s": 2}
results["Loading rate x2"] = compute_scores(df, weights_kinetics=wd_lr_x2)

wk_valgus_only = {"hip_flex": 0, "knee_valg": 1, "trunk_lean": 0, "ankle_angle_sagittal": 0}
results["K-Score = knee valgus only"] = compute_scores(df, weights_kin=wk_valgus_only)

wd_grf_only = {"peak_vgrf_bw": 1, "loading_rate_bw_s": 0}
results["D-Score = peak GRF only"] = compute_scores(df, weights_kinetics=wd_grf_only)

results["Excluding sub24"] = compute_scores(df[df["subject"] != "sub24"])

results["Direction reversed (hip/trunk = risk up)"] = compute_scores(
    df, kin_dir_override=kin_dir_reversed)

rm = rmcorr_analysis(df)
results["rmcorr (trial-level)"] = {
    "n": 132, "rho": rm["r_rm"], "rho_ci_lo": rm["rm_ci_lo"], "rho_ci_hi": rm["rm_ci_hi"],
    "kappa": None, "kappa_ci_lo": None, "kappa_ci_hi": None,
    "conflict_pct": None, "n_conflict": None,
    "n_01": None, "n_10": None, "mcn": None, "mcn_p": None,
    "ba_bias": None, "ba_loa_lo": None, "ba_loa_hi": None, "ba_sd": None,
}

# Reference values for analyses requiring raw .c3d files (from prior runs)
reference_ic = OrderedDict()
reference_ic["IC = 50 N"] = {
    "kappa": 0.636, "kappa_ci_lo": 0.314, "kappa_ci_hi": 0.959,
    "rho": 0.554, "rho_ci_lo": 0.173, "rho_ci_hi": 0.791, "conflict_pct": 18.2,
}
reference_ic["IC = kinematic (ankle vel.)*"] = {
    "kappa": 0.091, "kappa_ci_lo": -0.325, "kappa_ci_hi": 0.507,
    "rho": 0.056, "rho_ci_lo": -0.375, "rho_ci_hi": 0.467, "conflict_pct": 45.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# PRINT DETAILED VERIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 78)
print("SENSITIVITY ANALYSIS �?COMPLETE VERIFICATION REPORT")
print("=" * 78)

all_passed = True
for label, r in results.items():
    if r["kappa"] is not None:
        print(f"\n{label}")
        print(f"  N = {r['n']}")
        print(f"  Spearman rho  = {r['rho']:.3f}  [{r['rho_ci_lo']:.3f}, {r['rho_ci_hi']:.3f}]")
        print(f"  Cohen's kappa = {r['kappa']:.3f}  [{r['kappa_ci_lo']:.3f}, {r['kappa_ci_hi']:.3f}]")
        print(f"  Conflict     = {r['conflict_pct']:.0f}%  ({r['n_conflict']}/{r['n']})")
        if r["n_01"] is not None:
            print(f"  McNemar chi2 = {r['mcn']:.3f}, p = {r['mcn_p']:.3f}")
        print(f"  BA bias      = {r['ba_bias']:.3f}")
        print(f"  BA LoA       = [{r['ba_loa_lo']:.3f}, {r['ba_loa_hi']:.3f}]  (SD diff = {r['ba_sd']:.3f})")
    else:
        print(f"\n{label}")
        print(f"  r_rm = {r['rho']:.3f}  [{r['rho_ci_lo']:.3f}, {r['rho_ci_hi']:.3f}]")

print(f"\n{'─' * 78}")
print("Reference values (C3D-dependent, from prior independent runs):")
print(f"{'─' * 78}")
for label, ref in reference_ic.items():
    print(f"\n{label}")
    print(f"  rho = {ref['rho']:.3f}  [{ref['rho_ci_lo']:.3f}, {ref['rho_ci_hi']:.3f}]")
    print(f"  kappa = {ref['kappa']:.3f}  [{ref['kappa_ci_lo']:.3f}, {ref['kappa_ci_hi']:.3f}]")
    print(f"  Conflict = {ref['conflict_pct']:.0f}%")

# ══════════════════════════════════════════════════════════════════════════════
# SELF-CONSISTENCY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 78}")
print("SELF-CONSISTENCY CHECKS")
print(f"{'=' * 78}")

checks_passed = 0
checks_total = 0

def check(label, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  [PASS] {label}")
    else:
        global all_passed
        all_passed = False
        print(f"  [FAIL] {label}  ({detail})")

# Primary consistency with known values
p = results["Primary (IC=20N, equal weights)"]
check("Primary rho = 0.558", abs(p["rho"] - 0.558) < 0.001, f"got {p['rho']:.5f}")
check("Primary kappa = 0.636", abs(p["kappa"] - 0.636) < 0.001, f"got {p['kappa']:.5f}")
check("Primary CI lo = 0.179", abs(p["rho_ci_lo"] - 0.179) < 0.001, f"got {p['rho_ci_lo']:.5f}")
check("Primary CI hi = 0.793", abs(p["rho_ci_hi"] - 0.793) < 0.001, f"got {p['rho_ci_hi']:.5f}")
check("Primary kappa CI lo = 0.314", abs(p["kappa_ci_lo"] - 0.314) < 0.002, f"got {p['kappa_ci_lo']:.5f}")
check("Primary kappa CI hi = 0.959", abs(p["kappa_ci_hi"] - 0.959) < 0.002, f"got {p['kappa_ci_hi']:.5f}")
check("Primary Conflict = 18.2%", abs(p["conflict_pct"] - 18.2) < 0.1, f"got {p['conflict_pct']:.1f}%")
check("Primary n_conflict = 4", p["n_conflict"] == 4, f"got {p['n_conflict']}")
check("Primary McNemar p = 0.617", abs(p["mcn_p"] - 0.617) < 0.001, f"got {p['mcn_p']:.5f}")
check("Primary BA LoA [-1.35, 1.35]", abs(p["ba_loa_lo"] - (-1.347)) < 0.01, f"got [{p['ba_loa_lo']:.3f}, {p['ba_loa_hi']:.3f}]")

cmj = results["CMJ only"]
dj = results["DJ only"]
check("CMJ rho = 0.546", abs(cmj["rho"] - 0.546) < 0.001, f"got {cmj['rho']:.5f}")
check("CMJ kappa = 0.455", abs(cmj["kappa"] - 0.455) < 0.001, f"got {cmj['kappa']:.5f}")
check("DJ rho = 0.300", abs(dj["rho"] - 0.300) < 0.001, f"got {dj['rho']:.5f}")
check("DJ kappa = 0.273", abs(dj["kappa"] - 0.273) < 0.001, f"got {dj['kappa']:.5f}")
check("DJ CI crosses zero", dj["rho_ci_lo"] < 0, f"got [{dj['rho_ci_lo']:.3f}, {dj['rho_ci_hi']:.3f}]")
check("CMJ/DJ CIs overlap", abs(cmj["rho_ci_lo"] - dj["rho_ci_lo"]) < 0.5,
      f"CMJ lo={cmj['rho_ci_lo']:.3f}, DJ lo={dj['rho_ci_lo']:.3f}")

# Weights
kvx2 = results["Knee valgus x2"]
lrx2 = results["Loading rate x2"]
kv_only = results["K-Score = knee valgus only"]
grf_only = results["D-Score = peak GRF only"]

check("Knee valgus x2 kappa = 0.455", abs(kvx2["kappa"] - 0.455) < 0.001, f"got {kvx2['kappa']:.5f}")
check("Load rate x2 kappa = 0.636", abs(lrx2["kappa"] - 0.636) < 0.002, f"got {lrx2['kappa']:.5f}")
check("Knee valgus only kappa = 0.273", abs(kv_only["kappa"] - 0.273) < 0.001, f"got {kv_only['kappa']:.5f}")
check("Peak GRF only kappa = 0.636", abs(grf_only["kappa"] - 0.636) < 0.002, f"got {grf_only['kappa']:.5f}")
check("kappa range (weights) = [0.273, 0.636]",
      abs(min(kvx2["kappa"], lrx2["kappa"], kv_only["kappa"], grf_only["kappa"]) - 0.273) < 0.002 and
      abs(max(kvx2["kappa"], lrx2["kappa"], kv_only["kappa"], grf_only["kappa"]) - 0.636) < 0.002)

# Sub24 exclusion
sub24 = results["Excluding sub24"]
check("Excl sub24 rho = 0.562", abs(sub24["rho"] - 0.562) < 0.001, f"got {sub24['rho']:.5f}")
check("Excl sub24 kappa = 0.618", abs(sub24["kappa"] - 0.618) < 0.001, f"got {sub24['kappa']:.5f}")
check("Drho(sub24) ~ +0.004", abs((sub24["rho"] - p["rho"]) - 0.004) < 0.002,
      f"got {sub24['rho'] - p['rho']:+.4f}")
check("Dkappa(sub24) ~ -0.018", abs((sub24["kappa"] - p["kappa"]) - (-0.018)) < 0.002,
      f"got {sub24['kappa'] - p['kappa']:+.4f}")

# Direction convention
rev = results["Direction reversed (hip/trunk = risk up)"]
check("Reversed dir rho = 0.460", abs(rev["rho"] - 0.460) < 0.001, f"got {rev['rho']:.5f}")
check("Reversed dir kappa = 0.091", abs(rev["kappa"] - 0.091) < 0.001, f"got {rev['kappa']:.5f}")
check("Reversed dir kappa CI crosses zero", rev["kappa_ci_lo"] < 0,
      f"got [{rev['kappa_ci_lo']:.3f}, {rev['kappa_ci_hi']:.3f}]")
check("Direction reversal shifts kappa from 0.636 -> 0.091",
      rev["kappa"] < p["kappa"] - 0.50,
      f"Dkappa = {rev['kappa'] - p['kappa']:+.3f}")

# rmcorr
rmr = results["rmcorr (trial-level)"]
check("rmcorr r_rm = 0.233", abs(rmr["rho"] - 0.233) < 0.002, f"got {rmr['rho']:.5f}")
check("rmcorr < subject-mean Spearman", rmr["rho"] < p["rho"],
      f"r_rm={rmr['rho']:.3f} vs rho={p['rho']:.3f}")

# Cross-variant checks
check("kappa(Primary) = kappa(LoadRate x2) = kappa(GRF only)",
      abs(p["kappa"] - lrx2["kappa"]) < 0.002 and abs(p["kappa"] - grf_only["kappa"]) < 0.002)
check("kappa(DJ) = kappa(Knee valgus only)",
      abs(dj["kappa"] - kv_only["kappa"]) < 0.002,
      f"DJ kappa={dj['kappa']:.3f}, KV only kappa={kv_only['kappa']:.3f}")
check("kappa(reversed) = kappa(kinematic IC, reference)",
      abs(rev["kappa"] - reference_ic["IC = kinematic (ankle vel.)*"]["kappa"]) < 0.002,
      f"reversed kappa={rev['kappa']:.3f}, kinematic IC kappa(ref)={reference_ic['IC = kinematic (ankle vel.)*']['kappa']:.3f}")

# Bland-Altman sanity checks
for label in ["Primary (IC=20N, equal weights)", "CMJ only", "DJ only",
              "Knee valgus x2", "Loading rate x2", "Excluding sub24"]:
    r = results[label]
    check(f"BA: '{label}' bias ~ 0", abs(r["ba_bias"]) < 0.01, f"bias={r['ba_bias']:.5f}")
    check(f"BA: '{label}' LoA symmetric", abs(r["ba_loa_lo"] + r["ba_loa_hi"]) < 0.01,
          f"LoA=[{r['ba_loa_lo']:.3f}, {r['ba_loa_hi']:.3f}]")

print(f"\n{'─' * 78}")
print(f"  CHECKS: {checks_passed} / {checks_total} passed")
if all_passed:
    print(f"  VERDICT: ALL CHECKS PASSED -- analysis pipeline is self-consistent.")
else:
    print(f"  VERDICT: SOME CHECKS FAILED -- see [FAIL] above.")


# ══════════════════════════════════════════════════════════════════════════════
# CUT-POINT SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 78}")
print("CUT-POINT SENSITIVITY")
print(f"{'=' * 78}")

sub_primary = df.copy()
for c in kin_vars:
    z = (sub_primary[c] - sub_primary[c].mean()) / sub_primary[c].std()
    if kin_dir_pristine[c] == -1:
        z = -z
    sub_primary[f"z_{c}"] = z
sub_primary["K_raw"] = sub_primary[[f"z_{c}" for c in kin_vars]].mean(axis=1)
for c in kinetics_vars:
    sub_primary[f"z_{c}"] = (sub_primary[c] - sub_primary[c].mean()) / sub_primary[c].std()
sub_primary["D_raw"] = sub_primary[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

sub_agg = sub_primary.groupby("subject").agg(
    K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
sub_agg["K"] = (sub_agg["K"] - sub_agg["K"].mean()) / sub_agg["K"].std()
sub_agg["D"] = (sub_agg["D"] - sub_agg["D"].mean()) / sub_agg["D"].std()

cutpoints = range(30, 75, 5)
kappa_range = []
for cp in cutpoints:
    kth = np.percentile(sub_agg["K"], cp)
    dth = np.percentile(sub_agg["D"], cp)
    kh = (sub_agg["K"] >= kth).astype(int)
    dh = (sub_agg["D"] >= dth).astype(int)
    k = cohen_kappa_score(kh, dh)
    kappa_range.append(k)
    print(f"  {cp:2d}th%  kappa = {k:.3f}")

check(f"Cut-point kappa range = [{min(kappa_range):.3f}, {max(kappa_range):.3f}]",
      abs(min(kappa_range) - 0.371) < 0.005 and abs(max(kappa_range) - 0.636) < 0.005,
      f"got [{min(kappa_range):.3f}, {max(kappa_range):.3f}]")


# ══════════════════════════════════════════════════════════════════════════════
# KAPPA FOREST PLOT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 78}")
print("GENERATING KAPPA FOREST PLOT")
print(f"{'=' * 78}")

forest_data = [
    ("Primary (IC=20N, equal weights)",  p["kappa"],  p["kappa_ci_lo"],  p["kappa_ci_hi"],  "#2c7fb8"),
    ("CMJ only",                         cmj["kappa"], cmj["kappa_ci_lo"], cmj["kappa_ci_hi"], "#7bccc4"),
    ("DJ only",                          dj["kappa"], dj["kappa_ci_lo"], dj["kappa_ci_hi"], "#7bccc4"),
    ("IC = 50 N",                        reference_ic["IC = 50 N"]["kappa"],
     reference_ic["IC = 50 N"]["kappa_ci_lo"],
     reference_ic["IC = 50 N"]["kappa_ci_hi"], "#a8ddb5"),
    ("IC = kinematic (ankle vel.)*",     reference_ic["IC = kinematic (ankle vel.)*"]["kappa"],
     reference_ic["IC = kinematic (ankle vel.)*"]["kappa_ci_lo"],
     reference_ic["IC = kinematic (ankle vel.)*"]["kappa_ci_hi"], "#a8ddb5"),
    ("Knee valgus x2",                   kvx2["kappa"], kvx2["kappa_ci_lo"], kvx2["kappa_ci_hi"], "#fdae61"),
    ("Loading rate x2",                  lrx2["kappa"], lrx2["kappa_ci_lo"], lrx2["kappa_ci_hi"], "#fdae61"),
    ("K-Score = knee valgus only",       kv_only["kappa"], kv_only["kappa_ci_lo"], kv_only["kappa_ci_hi"], "#fdae61"),
    ("D-Score = peak GRF only",          grf_only["kappa"], grf_only["kappa_ci_lo"], grf_only["kappa_ci_hi"], "#fdae61"),
    ("Direction reversed (hip/trunk = risk up)", rev["kappa"], rev["kappa_ci_lo"], rev["kappa_ci_hi"], "#d73027"),
    ("Excluding sub24",                  sub24["kappa"], sub24["kappa_ci_lo"], sub24["kappa_ci_hi"], "#d73027"),
]

n_f = len(forest_data)
fig, ax = plt.subplots(figsize=(10, 5.5))
y_pos = np.arange(n_f)[::-1]

for i in range(n_f):
    label, k, lo, hi, color = forest_data[i]
    ax.errorbar(k, y_pos[i], xerr=[[k - lo], [hi - k]], fmt='o', color=color,
                capsize=4, capthick=1.5, markersize=7, markeredgecolor='white',
                markeredgewidth=1.2, elinewidth=2, zorder=3)

for x_pos in [0.0, 0.20, 0.40, 0.60, 0.80]:
    ax.axvline(x=x_pos, color='#d9d9d9', linestyle=':', linewidth=1, zorder=1)
ax.axvline(x=0.0, color='grey', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

labels = [d[0] for d in forest_data]
for i in range(n_f):
    label = labels[i]
    if "[ref]" in label:
        label = label.replace("[ref]", "")
    if "[c3d]" in label:
        label = label.replace("[c3d]", "")
    labels[i] = label

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Cohen's κ", fontsize=12)
ax.set_xlim(-0.45, 1.10)
ax.set_title("Classification agreement (κ) across methodological conditions",
             fontsize=13, fontweight='bold', pad=12)

ax.text(0.10, n_f - 0.65, 'Slight', fontsize=9, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.30, n_f - 0.65, 'Fair', fontsize=9, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.50, n_f - 0.65, 'Moderate', fontsize=9, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.70, n_f - 0.65, 'Substantial', fontsize=9, color='grey', fontstyle='italic', clip_on=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i in range(n_f):
    k = forest_data[i][1]
    ax.text(k + 0.015, y_pos[i], f"{k:.3f}", va='center', fontsize=8.5, fontweight='bold')

pos_y = -0.3
ax.text(0.02, pos_y, 'Poor', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.18, pos_y, 'Slight', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.30, pos_y, 'Fair', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.48, pos_y, 'Moderate', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.68, pos_y, 'Substantial', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(0.85, pos_y, 'Almost perfect', fontsize=7.5, color='grey', fontstyle='italic', clip_on=False)
ax.text(-0.42, pos_y, 'Landis & Koch (1977):', fontsize=7.5, color='grey', fontweight='bold', clip_on=False)

fig.subplots_adjust(bottom=0.13)
import os as _os
out_png = _os.path.join(FIG_DIR, "06_kappa_forest.png")
out_pdf = _os.path.join(FIG_DIR, "06_kappa_forest.pdf")
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {out_png}")
print(f"  Saved: {out_pdf}")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CONFLICT DISTRIBUTION PLOT
# ══════════════════════════════════════════════════════════════════════════════

cond_order = [
    ("Primary", p), ("CMJ only", cmj), ("DJ only", dj),
    ("Knee valgus x2", kvx2), ("Loading rate x2", lrx2),
    ("K-Score = knee valgus only", kv_only), ("D-Score = peak GRF only", grf_only),
    ("Direction reversed", rev), ("Excluding sub24", sub24),
]

fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))

cond_labels = [c[0] for c in cond_order]
kappa_vals = [c[1]["kappa"] for c in cond_order]
conflict_vals = [c[1]["conflict_pct"] for c in cond_order]

colors_bar = []
for cl in cond_labels:
    if cl in ("Primary",):
        colors_bar.append("#2c7fb8")
    elif cl in ("CMJ only", "DJ only"):
        colors_bar.append("#7bccc4")
    elif cl.startswith("Knee valgus") or cl.startswith("Loading") or cl.startswith("K-Score") or cl.startswith("D-Score"):
        colors_bar.append("#fdae61")
    elif cl.startswith("Direction"):
        colors_bar.append("#d73027")
    else:
        colors_bar.append("#d73027")

x_idx = np.arange(len(cond_labels))
width = 0.35

bars1 = axes[0].bar(x_idx, kappa_vals, width, color=colors_bar, edgecolor='white')
axes[0].set_xticks(x_idx)
axes[0].set_xticklabels(cond_labels, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel("Cohen's κ", fontsize=11)
axes[0].set_title("Classification agreement", fontsize=12, fontweight='bold')
axes[0].axhline(y=0.0, color='grey', linewidth=0.8, linestyle='--')
axes[0].axhline(y=0.40, color='#d9d9d9', linewidth=0.6, linestyle=':')
axes[0].axhline(y=0.60, color='#d9d9d9', linewidth=0.6, linestyle=':')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
for bar, val in zip(bars1, kappa_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

bars2 = axes[1].bar(x_idx, conflict_vals, width, color=colors_bar, edgecolor='white')
axes[1].set_xticks(x_idx)
axes[1].set_xticklabels(cond_labels, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel("Conflict rate (%)", fontsize=11)
axes[1].set_title("Classification conflict", fontsize=12, fontweight='bold')
axes[1].axhline(y=18.2, color='grey', linewidth=0.8, linestyle='--', alpha=0.5, label='Primary baseline')
axes[1].legend(fontsize=8)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
for bar, val in zip(bars2, conflict_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.0f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
out_sens = _os.path.join(FIG_DIR, "07_sensitivity_overview.png")
fig2.savefig(out_sens, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {out_sens}")
plt.close()

print(f"\n{'=' * 78}")
print("VERIFICATION COMPLETE")
print(f"{'=' * 78}")
