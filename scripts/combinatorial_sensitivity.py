"""
combinatorial_sensitivity.py �?Four combinatorial perturbations:
  C1 = S5 + S1b  : direction reversed + DJ only
  C2 = S5 + S4c  : direction reversed + K-Score = knee valgus only
  C3 = S1b + S4c : DJ only + K-Score = knee valgus only
  C4 = S5 + S1b + S4c : all three combined
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from collections import OrderedDict
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "features_raw.csv")

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir_pristine = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kin_dir_reversed = {"hip_flex": 1, "knee_valg": 1, "trunk_lean": 1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

df = pd.read_csv(FEATURES_CSV)

SEED = 42
N_BOOT = 10000
np.random.seed(SEED)


def compute_scores(data, kin_dir=None, weights_kin=None, weights_kinetics=None):
    kin_d = kin_dir if kin_dir is not None else kin_dir_pristine
    wk = weights_kin or {c: 1 for c in kin_vars}
    wd = weights_kinetics or {c: 1 for c in kinetics_vars}
    wsum_k = sum(wk.values())
    wsum_d = sum(wd.values())
    if wsum_k == 0 or wsum_d == 0:
        raise ValueError("At least one component must have non-zero weight")

    dfc = data.copy()
    for c in kin_vars:
        z = (dfc[c] - dfc[c].mean()) / dfc[c].std()
        if kin_d[c] == -1:
            z = -z
        dfc[f"z_{c}"] = z
    dfc["K_raw"] = sum(wk[c] * dfc[f"z_{c}"] for c in kin_vars) / wsum_k
    for c in kinetics_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
    dfc["D_raw"] = sum(wd[c] * dfc[f"z_{c}"] for c in kinetics_vars) / wsum_d

    sub = dfc.groupby("subject").agg(K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
    sub["K"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
    sub["D"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

    n = len(sub)
    rho, _ = stats.spearmanr(sub["K"], sub["D"])
    se_rho = 1.0 / np.sqrt(n - 3)
    rho_ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se_rho)
    rho_ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se_rho)

    md_k, md_d = sub["K"].median(), sub["D"].median()
    kh = (sub["K"] >= md_k).astype(int)
    dh = (sub["D"] >= md_d).astype(int)
    kappa = cohen_kappa_score(kh, dh)
    conflict = (kh != dh).astype(int)
    n_conflict = int(conflict.sum())
    conflict_pct = n_conflict / n * 100

    n_agree = int((kh == dh).sum())
    po = n_agree / n
    pe = 0.5
    kappa_se = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)) if pe < 1.0 else np.nan
    kappa_ci_lo = kappa - 1.96 * kappa_se
    kappa_ci_hi = kappa + 1.96 * kappa_se

    boots_rho = np.zeros(N_BOOT)
    boots_kappa = np.zeros(N_BOOT)
    idx_arr = np.arange(n)
    for i in range(N_BOOT):
        idx = np.random.choice(idx_arr, size=n, replace=True)
        br, _ = stats.spearmanr(sub["K"].iloc[idx], sub["D"].iloc[idx])
        boots_rho[i] = br
        bkh = kh.iloc[idx].values
        bdh = dh.iloc[idx].values
        boots_kappa[i] = cohen_kappa_score(bkh, bdh)
    rho_ci_lo_b = np.percentile(boots_rho, 2.5)
    rho_ci_hi_b = np.percentile(boots_rho, 97.5)
    kappa_ci_lo_b = np.percentile(boots_kappa, 2.5)
    kappa_ci_hi_b = np.percentile(boots_kappa, 97.5)

    n_01 = int(((kh == 0) & (dh == 1)).sum())
    n_10 = int(((kh == 1) & (dh == 0)).sum())

    return {
        "n": n, "rho": float(rho), "rho_ci_lo_analytic": float(rho_ci_lo),
        "rho_ci_hi_analytic": float(rho_ci_hi),
        "rho_ci_lo_boot": float(rho_ci_lo_b), "rho_ci_hi_boot": float(rho_ci_hi_b),
        "kappa": float(kappa), "kappa_ci_lo": float(kappa_ci_lo),
        "kappa_ci_hi": float(kappa_ci_hi),
        "kappa_ci_lo_boot": float(kappa_ci_lo_b), "kappa_ci_hi_boot": float(kappa_ci_hi_b),
        "conflict_pct": float(conflict_pct), "n_conflict": n_conflict,
        "n_01": n_01, "n_10": n_10,
    }


# ── Primary (for reference) ──
primary = compute_scores(df)
primary_cmj = compute_scores(df[df["trial_type"] == "CMJ"])
primary_dj = compute_scores(df[df["trial_type"] == "DJ"])

# ── Single-factor references ──
s5 = compute_scores(df, kin_dir=kin_dir_reversed)
s4c = compute_scores(df, weights_kin={"hip_flex": 0, "knee_valg": 1, "trunk_lean": 0, "ankle_angle_sagittal": 0})
s1b = compute_scores(df[df["trial_type"] == "DJ"])

# ── Combinatorial ──
# C1 = S5 + S1b
c1 = compute_scores(df[df["trial_type"] == "DJ"], kin_dir=kin_dir_reversed)
# C2 = S5 + S4c
c2 = compute_scores(df, kin_dir=kin_dir_reversed,
                    weights_kin={"hip_flex": 0, "knee_valg": 1, "trunk_lean": 0, "ankle_angle_sagittal": 0})
# C3 = S1b + S4c
c3 = compute_scores(df[df["trial_type"] == "DJ"],
                    weights_kin={"hip_flex": 0, "knee_valg": 1, "trunk_lean": 0, "ankle_angle_sagittal": 0})
# C4 = S5 + S1b + S4c
c4 = compute_scores(df[df["trial_type"] == "DJ"], kin_dir=kin_dir_reversed,
                    weights_kin={"hip_flex": 0, "knee_valg": 1, "trunk_lean": 0, "ankle_angle_sagittal": 0})

results = OrderedDict([
    ("Primary (pooled, IC=20N, equal weights)", primary),
    ("CMJ only", primary_cmj),
    ("DJ only (S1b)", s1b),
    ("Direction reversed (S5)", s5),
    ("K-Score = knee valgus only (S4c)", s4c),
    ("C1: S5 × S1b (reversed dir + DJ)", c1),
    ("C2: S5 × S4c (reversed dir + KV only)", c2),
    ("C3: S1b × S4c (DJ + KV only)", c3),
    ("C4: S5 × S1b × S4c (reversed dir + DJ + KV only)", c4),
])

print("=" * 85)
print("COMBINATORIAL SENSITIVITY ANALYSIS")
print("=" * 85)
print(f"{'Condition':<52s} {'N':>3s}  {'ρ':>7s}  {'ρ boot CI':>24s}  {'κ':>7s}  {'κ boot CI':>24s}  {'Conflict':>9s}")
print("-" * 85)

for label, r in results.items():
    print(f"{label:<52s} {r['n']:3d}  {r['rho']:7.3f}  [{r['rho_ci_lo_boot']:7.3f}, {r['rho_ci_hi_boot']:7.3f}]  "
          f"{r['kappa']:7.3f}  [{r['kappa_ci_lo_boot']:7.3f}, {r['kappa_ci_hi_boot']:7.3f}]  "
          f"{r['conflict_pct']:5.1f}% ({r['n_conflict']:d}/{r['n']:d})")

# ── Key observations ──
print("\n" + "=" * 85)
print("KEY OBSERVATIONS")
print("=" * 85)

for combo_label, combo_r, s1_label, s1_r, s2_label, s2_r in [
    ("C1 (S5×S1b)", c1, "S5 alone", s5, "S1b alone", s1b),
    ("C2 (S5×S4c)", c2, "S5 alone", s5, "S4c alone", s4c),
    ("C3 (S1b×S4c)", c3, "S1b alone", s1b, "S4c alone", s4c),
]:
    dk1 = combo_r["kappa"] - s1_r["kappa"]
    dk2 = combo_r["kappa"] - s2_r["kappa"]
    worst = min(combo_r["kappa"], s1_r["kappa"], s2_r["kappa"])
    print(f"  {combo_label}: κ={combo_r['kappa']:.3f} (vs {s1_label} κ={s1_r['kappa']:.3f}, "
          f"Δ={dk1:+.3f}; vs {s2_label} κ={s2_r['kappa']:.3f}, Δ={dk2:+.3f})")

print(f"  C4 (S5×S1b×S4c): κ={c4['kappa']:.3f}, ρ={c4['rho']:.3f} "
      f"[{c4['rho_ci_lo_boot']:.3f}, {c4['rho_ci_hi_boot']:.3f}] �?triple perturbation: "
      f"{'κ approaches zero' if abs(c4['kappa']) < 0.1 else 'κ remains non-negligible'}")
print(f"  Note: C4 K-Score reduced to a single variable (knee valgus, direction-reversed). "
      f"In this limit, the 'composite proxy' construct is essentially degenerate—its validity "
      f"as a multidimensional postural representation is lost, but the perturbation serves as a "
      f"boundary condition illustration.")

print("\nDone.")
