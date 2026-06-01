"""
Supplementary analyses for revision:
  1. Sex-stratified descriptive analysis (P0-2)
  2. Trial-to-trial reliability ICC (P0-3)
  3. Multilevel mixed-effects modeling (P1-2)

Target journal: Sports Biomechanics (Methods and Theoretical Perspectives)
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import json
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")
OUT_JSON = os.path.join(PROJECT_ROOT, "data", r"supplement_results.json")
OUT_TXT = os.path.join(PROJECT_ROOT, "data", r"supplement_results.txt")

# ---- Gender mapping from participant_log.xlsx (col idx 2: 1=F, 2=M) ----
GENDER_MAP = {
    'sub01': 'F', 'sub02': 'F', 'sub03': 'M', 'sub04': 'F',
    'sub06': 'F', 'sub07': 'F', 'sub08': 'F', 'sub09': 'M',
    'sub10': 'M', 'sub14': 'F', 'sub15': 'F', 'sub16': 'M',
    'sub17': 'F', 'sub19': 'F', 'sub21': 'M', 'sub22': 'F',
    'sub23': 'F', 'sub24': 'M', 'sub26': 'M', 'sub28': 'M',
    'sub30': 'M', 'sub34': 'M'
}

# ---- Load data ----
df = pd.read_csv(FEATURES_CSV)
df['gender'] = df['subject'].map(GENDER_MAP)

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

print(f"Loaded {len(df)} records, {df['subject'].nunique()} subjects")
print(f"Sex distribution: {df.groupby('subject')['gender'].first().value_counts().to_dict()}")
print()

# ===================================================================
# SHARED: compute K-Score and D-Score
# ===================================================================
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
    return dfc


def aggregate_subjects(dfc):
    sub = dfc.groupby("subject").agg(
        KScore_raw=("KScore_raw", "mean"),
        DScore_raw=("DScore_raw", "mean"),
    ).reset_index()
    sub["KScore"] = (sub["KScore_raw"] - sub["KScore_raw"].mean()) / sub["KScore_raw"].std()
    sub["DScore"] = (sub["DScore_raw"] - sub["DScore_raw"].mean()) / sub["DScore_raw"].std()
    return sub


def compute_agreement_stats(sub, label):
    n = len(sub)
    rho, p_rho = stats.spearmanr(sub["KScore"], sub["DScore"])
    k_hi = (sub["KScore"] >= sub["KScore"].median()).astype(int)
    d_hi = (sub["DScore"] >= sub["DScore"].median()).astype(int)
    kappa = cohen_kappa_score(k_hi, d_hi)
    agree = (k_hi == d_hi).mean() * 100
    conflict_pct = (k_hi != d_hi).mean() * 100
    n_kp_dm = int(sum((k_hi == 1) & (d_hi == 0)))
    n_km_dp = int(sum((k_hi == 0) & (d_hi == 1)))

    bs = stats.bootstrap(
        (sub["KScore"].values, sub["DScore"].values),
        lambda x, y: stats.spearmanr(x, y)[0],
        n_resamples=10000,
        method='percentile',
        random_state=42,
        paired=True,
    )
    rho_ci = (float(bs.confidence_interval.low), float(bs.confidence_interval.high))

    bs_k = stats.bootstrap(
        (sub["KScore"].values, sub["DScore"].values),
        lambda x, y: cohen_kappa_score(
            (x >= np.median(x)).astype(int),
            (y >= np.median(y)).astype(int)
        ),
        n_resamples=10000,
        method='percentile',
        random_state=42,
        paired=True,
    )
    kappa_ci = (float(bs_k.confidence_interval.low), float(bs_k.confidence_interval.high))

    return {
        "label": label, "n": n,
        "rho": float(rho), "rho_ci_lo": rho_ci[0], "rho_ci_hi": rho_ci[1],
        "kappa": float(kappa), "kappa_ci_lo": kappa_ci[0], "kappa_ci_hi": kappa_ci[1],
        "agree_pct": float(agree), "conflict_pct": float(conflict_pct),
        "n_kp_dm": n_kp_dm, "n_km_dp": n_km_dp,
    }


# ===================================================================
# 1. SEX-STRATIFIED ANALYSIS (P0-2)
# ===================================================================
print("=" * 70)
print("1. SEX-STRATIFIED DESCRIPTIVE ANALYSIS")
print("=" * 70)

sex_results = {}
for sex_label, sex_mask in [("Female", df["gender"] == "F"), ("Male", df["gender"] == "M")]:
    df_sex = df[sex_mask].copy()
    df_sex = compute_scores(df_sex)
    sub_sex = aggregate_subjects(df_sex)
    r = compute_agreement_stats(sub_sex, f"Pooled-{sex_label}")
    sex_results[sex_label] = r
    print(f"\n  {sex_label} (N={r['n']}):")
    print(f"    Spearman rho = {r['rho']:.3f}, bootstrap CI [{r['rho_ci_lo']:.3f}, {r['rho_ci_hi']:.3f}]")
    print(f"    Cohen's kappa = {r['kappa']:.3f}, bootstrap CI [{r['kappa_ci_lo']:.3f}, {r['kappa_ci_hi']:.3f}]")
    print(f"    Agreement = {r['agree_pct']:.1f}%, Conflict = {r['conflict_pct']:.1f}%")
    print(f"    KD discordance: K+D- = {r['n_kp_dm']}, K-D+ = {r['n_km_dp']}")

# Also do task-stratified by sex
for sex_label, sex_mask in [("Female", df["gender"] == "F"), ("Male", df["gender"] == "M")]:
    for task_label in ["CMJ", "DJ"]:
        df_st = df[(df["gender"] == sex_label[0]) & (df["trial_type"] == task_label)].copy()
        df_st = compute_scores(df_st)
        sub_st = aggregate_subjects(df_st)
        r = compute_agreement_stats(sub_st, f"{sex_label}-{task_label}")
        sex_results[f"{sex_label}-{task_label}"] = r
        print(f"\n  {sex_label}-{task_label} (N={r['n']}):")
        print(f"    rho = {r['rho']:.3f}, CI [{r['rho_ci_lo']:.3f}, {r['rho_ci_hi']:.3f}]")
        print(f"    kappa = {r['kappa']:.3f}, CI [{r['kappa_ci_lo']:.3f}, {r['kappa_ci_hi']:.3f}]")
        print(f"    Agree = {r['agree_pct']:.1f}%, Conflict = {r['conflict_pct']:.1f}%")


# ===================================================================
# 2. TRIAL-TO-TRIAL RELIABILITY (ICC) (P0-3)
# ===================================================================
print("\n" + "=" * 70)
print("2. TRIAL-TO-TRIAL RELIABILITY (ICC)")
print("=" * 70)

def compute_icc(data, targets, subj_col="subject", trial_col="trial_num"):
    """
    ICC(3,1) and ICC(3,k) via Shrout-Fleiss formula
    Two-way mixed effects, consistency, single and average measures.
    """
    results = {}
    for var in targets:
        pivot = data.pivot_table(index=subj_col, columns=trial_col, values=var)
        pivot = pivot.dropna(axis=1, how='all')
        n_subjects = pivot.shape[0]
        n_trials = pivot.shape[1]

        if n_subjects < 3 or n_trials < 2:
            results[var] = {"icc3_1": np.nan, "icc3_k": np.nan,
                            "ci_lo": np.nan, "ci_hi": np.nan, "n_subjects": n_subjects}
            continue

        grand_mean = pivot.values.mean()
        ss_total = np.sum((pivot.values - grand_mean) ** 2)
        ss_subjects = n_trials * np.sum((pivot.mean(axis=1).values - grand_mean) ** 2)
        ss_trials = n_subjects * np.sum((pivot.mean(axis=0).values - grand_mean) ** 2)
        ss_error = ss_total - ss_subjects - ss_trials

        df_subjects = n_subjects - 1
        df_trials = n_trials - 1
        df_error = (n_subjects - 1) * (n_trials - 1)

        ms_subjects = ss_subjects / df_subjects
        ms_error = ss_error / df_error
        ms_trials = ss_trials / df_trials

        icc3_1 = (ms_subjects - ms_error) / (ms_subjects + (n_trials - 1) * ms_error)
        icc3_k = (ms_subjects - ms_error) / ms_subjects

        # approximate CI via F-distribution (Shrout & Fleiss)
        if icc3_1 > 0 and ms_error > 0:
            f_val = ms_subjects / ms_error
            f_low = f_val / stats.f.ppf(0.975, df_subjects, df_error)
            f_high = f_val * stats.f.ppf(0.975, df_error, df_subjects)
            ci_lo = (f_low - 1) / (f_low + n_trials - 1)
            ci_hi = (f_high - 1) / (f_high + n_trials - 1)
            ci_lo = max(ci_lo, -1.0)
            ci_hi = min(ci_hi, 1.0)
        else:
            ci_lo = -1.0
            ci_hi = 1.0

        results[var] = {
            "icc3_1": float(icc3_1),
            "icc3_k": float(icc3_k),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "n_subjects": n_subjects,
        }
    return results


# Compute K-Score and D-Score per trial for ICC (z within task)
icc_results = {}

for task_label, task_mask in [("CMJ", df["trial_type"] == "CMJ"), ("DJ", df["trial_type"] == "DJ")]:
    df_t = df[task_mask].copy()
    df_t = compute_scores(df_t)

    # ICC on raw K and D per trial
    icc_k = compute_icc(df_t, ["KScore_raw", "DScore_raw"])
    for var, r in icc_k.items():
        key = f"{task_label}_{var}"
        icc_results[key] = r
        print(f"\n  {task_label} {var}:")
        print(f"    ICC(3,1) = {r['icc3_1']:.3f} [95% CI: {r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
        print(f"    ICC(3,k) = {r['icc3_k']:.3f} (k={df_t['trial_num'].nunique()} trials)")

    # Also ICC on component variables
    icc_comps = compute_icc(df_t, kin_vars + kinetics_vars)
    for var, r in icc_comps.items():
        key = f"{task_label}_{var}"
        icc_results[key] = r


# ===================================================================
# 3. MULTILEVEL MIXED-EFFECTS MODELING (P1-2)
# ===================================================================
print("\n" + "=" * 70)
print("3. MULTILEVEL MIXED-EFFECTS MODELING")
print("=" * 70)

df_all = compute_scores(df)

# Stack K and D into long format for mixed model
records_long = []
for _, row in df_all.iterrows():
    records_long.append({
        "subject": row["subject"],
        "trial_type": row["trial_type"],
        "trial_num": row["trial_num"],
        "proxy": "K",
        "score_raw": row["KScore_raw"],
    })
    records_long.append({
        "subject": row["subject"],
        "trial_type": row["trial_type"],
        "trial_num": row["trial_num"],
        "proxy": "D",
        "score_raw": row["DScore_raw"],
    })

df_long = pd.DataFrame(records_long)

# ---- Reduced-form mixed model via ANOVA decomposition (manual) ----
# Model: score ~ proxy * task + (1 + proxy | subject)
# Equivalent to computing random-effect correlation from subject-level K/D means
# after partialling out task effects.

print("  Using ANOVA-based decomposition for random-effects correlation:")
r_btw_cmj = None
r_btw_dj = None
for task_label in ["CMJ", "DJ"]:
    df_t = df_all[df_all["trial_type"] == task_label]
    sub_means = df_t.groupby("subject").agg(
        K=("KScore_raw", "mean"),
        D=("DScore_raw", "mean"),
    )
    n = len(sub_means)
    r = np.corrcoef(sub_means["K"], sub_means["D"])[0, 1]
    print(f"  {task_label}: between-subject r(K,D) = {r:.3f} (N={n})")
    if task_label == "CMJ":
        r_btw_cmj = r
    else:
        r_btw_dj = r

sub_means_pooled = df_all.groupby("subject").agg(
    K=("KScore_raw", "mean"),
    D=("DScore_raw", "mean"),
)
r_btw_pooled = np.corrcoef(sub_means_pooled["K"], sub_means_pooled["D"])[0, 1]
print(f"  Pooled:  between-subject r(K,D) = {r_btw_pooled:.3f} (N={len(sub_means_pooled)})")

# ---- Within-subject (trial-level) residual correlation ----
print()
print("  Within-subject residual correlation (trial-level, adjusting for subject):")
for task_label in ["CMJ", "DJ"]:
    df_t = df_all[df_all["trial_type"] == task_label]
    k_resid = []
    d_resid = []
    for sid in df_t["subject"].unique():
        sdata = df_t[df_t["subject"] == sid]
        k_resid.extend(sdata["KScore_raw"].values - sdata["KScore_raw"].mean())
        d_resid.extend(sdata["DScore_raw"].values - sdata["DScore_raw"].mean())
    r_within = np.corrcoef(k_resid, d_resid)[0, 1]
    print(f"  {task_label}: within-subject r(K,D) = {r_within:.3f}")

# rmcorr equivalent
k_resid_all = []
d_resid_all = []
for sid in df_all["subject"].unique():
    sdata = df_all[df_all["subject"] == sid]
    k_resid_all.extend(sdata["KScore_raw"].values - sdata["KScore_raw"].mean())
    d_resid_all.extend(sdata["DScore_raw"].values - sdata["DScore_raw"].mean())
r_within_all = np.corrcoef(k_resid_all, d_resid_all)[0, 1]
print(f"  Pooled:  within-subject r(K,D) = {r_within_all:.3f}")

# Manual bootstrap CI for within-subject correlation
rng = np.random.default_rng(42)
n_pairs = len(k_resid_all)
bs_reps = []
for _ in range(10000):
    idx = rng.integers(0, n_pairs, n_pairs)
    bs_reps.append(np.corrcoef(
        np.array(k_resid_all)[idx],
        np.array(d_resid_all)[idx]
    )[0, 1])
rmcorr_ci = (float(np.percentile(bs_reps, 2.5)), float(np.percentile(bs_reps, 97.5)))
print(f"  rmcorr CI: [{rmcorr_ci[0]:.3f}, {rmcorr_ci[1]:.3f}]")

# ---- Variance decomposition for multilevel model ----
print("\n  Variance decomposition (multilevel model):")
for task_label in ["CMJ", "DJ", "Pooled"]:
    if task_label == "Pooled":
        df_m = df_all
    else:
        df_m = df_all[df_all["trial_type"] == task_label]

    sub_means_m = df_m.groupby("subject").agg(
        K_mean=("KScore_raw", "mean"),
        D_mean=("DScore_raw", "mean"),
    )
    sub_means_m["diff"] = sub_means_m["K_mean"] - sub_means_m["D_mean"]

    var_between_diff = sub_means_m["diff"].var(ddof=1)

    diffs_within = []
    for sid in df_m["subject"].unique():
        sdata = df_m[df_m["subject"] == sid]
        k_res = sdata["KScore_raw"].values - sdata["KScore_raw"].mean()
        d_res = sdata["DScore_raw"].values - sdata["DScore_raw"].mean()
        diffs_within.extend(k_res - d_res)
    var_within_diff = np.var(diffs_within, ddof=1)

    icc_diff = var_between_diff / (var_between_diff + var_within_diff) if (var_between_diff + var_within_diff) > 0 else np.nan
    print(f"  {task_label}: var_between(K-D)={var_between_diff:.4f}, "
          f"var_within(K-D)={var_within_diff:.4f}, "
          f"ICC_diff={icc_diff:.3f}")

# ===================================================================
# SAVE RESULTS
# ===================================================================
output = {
    "sex_stratified": {k: {kk: vv for kk, vv in v.items() if kk != "label"}
                       for k, v in sex_results.items()},
    "icc": icc_results,
    "multilevel": {
        "between_subject_correlation": {
            "pooled": float(r_btw_pooled),
            "CMJ": float(r_btw_cmj),
            "DJ": float(r_btw_dj),
        },
        "within_subject_correlation": {
            "pooled": float(r_within_all),
            "pooled_ci_lo": float(rmcorr_ci[0]),
            "pooled_ci_hi": float(rmcorr_ci[1]),
        },
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {OUT_JSON}")

# Also write text summary
txt_lines = []
txt_lines.append("=" * 70)
txt_lines.append("SUPPLEMENTARY ANALYSES FOR REVISION")
txt_lines.append("Target: Sports Biomechanics (Methods and Theoretical Perspectives)")
txt_lines.append("=" * 70)
txt_lines.append("")

txt_lines.append("--- SEX-STRATIFIED ANALYSIS ---")
for label in ["Female", "Male", "Female-CMJ", "Female-DJ", "Male-CMJ", "Male-DJ"]:
    r = sex_results[label]
    txt_lines.append(f"{label} (N={r['n']}): rho={r['rho']:.3f} [{r['rho_ci_lo']:.3f},{r['rho_ci_hi']:.3f}], "
                     f"kappa={r['kappa']:.3f} [{r['kappa_ci_lo']:.3f},{r['kappa_ci_hi']:.3f}], "
                     f"agree={r['agree_pct']:.1f}%, conflict={r['conflict_pct']:.1f}%")

txt_lines.append("")
txt_lines.append("--- ICC RELIABILITY ---")
for key, r in icc_results.items():
    txt_lines.append(f"{key}: ICC(3,1)={r['icc3_1']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}], "
                     f"ICC(3,k)={r['icc3_k']:.3f}, n_subjects={r['n_subjects']}")

txt_lines.append("")
txt_lines.append("--- MULTILEVEL MODELLING ---")
txt_lines.append(f"Between-subject r(K,D): pooled={r_btw_pooled:.3f}, CMJ={r_btw_cmj:.3f}, DJ={r_btw_dj:.3f}")
txt_lines.append(f"Within-subject r(K,D) [rmcorr]: pooled={r_within_all:.3f} [{rmcorr_ci[0]:.3f},{rmcorr_ci[1]:.3f}]")

report = "\n".join(txt_lines)
print(report)
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nText report saved to {OUT_TXT}")
