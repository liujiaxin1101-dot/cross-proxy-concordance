# Core Computational Logic: Spearman ρ and Cohen's κ

**Date**: 2026-05-20  
**Corresponding script**: `04_compute_scores.py` (main analysis pipeline)  
**Diagnostic script**: `diag_rho_kappa.py` (data source for this document; can be run independently for verification)

---

## Overview: Computational Pipeline

```
features_raw.csv (132 trials x 6 features)
  │
  ├── Step 1: z-score standardization of 6 variables (132 records),
  │     directional sign convention reversal, equal-weight averaging
  │     → KScore_raw (trial-level), DScore_raw (trial-level)
  │
  ├── Step 2: aggregate by subject mean, yielding 22 independent observations
  │     → second z-score standardization on subject means
  │     → KScore, DScore (final subject-level scores)
  │
  ├── Step 3: Spearman ρ = 0.558, p = 0.007
  │     Fisher z → 95% CI = [0.179, 0.793]
  │
  ├── Step 4: median-split dichotomization (K ≥ 0.00684, D ≥ −0.38406)
  │     Confusion matrix: 9+2 / 2+9
  │     κ = 0.636, conflict = 18.2%
  │
  └── Step 5: McNemar χ² = 0.25 (p = 0.62), Bland-Altman LoA = [−2.00, 2.00]
```

---

## Step 0: Input Data Overview

**Source**: `features_raw.csv`  
**Trials**: 132 records (22 subjects × 3 CMJ + 3 DJ)

| Variable (column name) | Mean | SD | Min | Max |
|------------------------|------|-----|------|------|
| `hip_flex` | 6.8349 | 3.6758 | 0.6443 | 25.2438 |
| `knee_valg` | 12.0537 | 3.4028 | 0.2798 | 20.7870 |
| `trunk_lean` | 2.9290 | 2.6360 | 0.0172 | 16.8437 |
| `ankle_dorsi` | 153.4555 | 41.9501 | 6.3451 | 178.9141 |
| `peak_vgrf_bw` | 4.7903 | 1.6489 | 2.3248 | 11.7082 |
| `loading_rate_bw_s` | 132.5415 | 191.2872 | 28.2425 | 1254.0872 |

---

## Step 1: Trial-Level z-Score Standardization + Directional Convention

**Execution logic** (see `04_compute_scores.py` L70–93):

```python
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_dorsi": -1}

for col in kin_vars:
    z = (df[col] - df[col].mean()) / df[col].std()
    if kin_dir[col] == -1:
        z = -z
    df[f"z_{col}"] = z

df["KScore_raw"] = df[[f"z_{col}" for col in kin_vars]].mean(axis=1)
df["DScore_raw"] = df[[f"z_{col}" for col in kinetics_vars]].mean(axis=1)
```

**Actual z-score parameters for each variable**:

| Variable | Mean (μ) | SD (σ) | Direction (kin_dir) | Operation |
|----------|----------|--------|----------------------|-----------|
| `hip_flex` | 6.8349 | 3.6619 | −1 (protective) | z = −(x−μ)/σ |
| `knee_valg` | 12.0537 | 3.3899 | +1 (risk) | z = +(x−μ)/σ |
| `trunk_lean` | 2.9290 | 2.6260 | −1 (protective) | z = −(x−μ)/σ |
| `ankle_dorsi` | 153.4555 | 41.7908 | −1 (protective) | z = −(x−μ)/σ |
| `peak_vgrf_bw` | 4.7903 | 1.6427 | +1 (risk) | z = +(x−μ)/σ |
| `loading_rate_bw_s` | 132.5415 | 190.5613 | +1 (risk) | z = +(x−μ)/σ |

> **Note**: `df[col].std()` uses `ddof=1` (pandas default) in the script, so σ is slightly smaller than the population std of `features_raw.csv`. The σ values in the table above are the actual values used.

**Verification**: After standardization, each z-scored variable has mean ≈ 0 and SD = 1. Confirmed.

**Trial-level equal-weight means**:

| Score | Mean | SD |
|-------|------|-----|
| KScore_raw | 0.0000 | 0.4996 |
| DScore_raw | 0.0000 | 0.9482 |

> DScore_raw's SD (0.95) is approximately twice that of KScore_raw (0.50) — this is because D-Score comprises only 2 kinetic indicators (mean of 2 items → variance = 1/2 of a single indicator), while K-Score comprises 4 kinematic indicators (mean of 4 items → variance = 1/4 of a single indicator). However, the intercorrelation among kinetic indicators is lower than that among kinematic indicators.

---

## Step 2: Subject Means (22 subjects)

```python
sub_agg = df.groupby("subject").agg(
    KScore=("KScore_raw", "mean"),
    DScore=("DScore_raw", "mean"),
).reset_index()
```

**Raw means of 22 subjects** (before second z-score standardization):

| Subject | K_raw_mean | K_raw_std | D_raw_mean | D_raw_std | n |
|---------|-----------|-----------|-----------|-----------|----|
| sub01 | 0.0793 | 0.3376 | 0.5892 | 0.4048 | 6 |
| sub02 | −0.0516 | 0.3058 | −0.5702 | 0.1850 | 6 |
| sub03 | −0.2278 | 0.1504 | −0.4890 | 0.1425 | 6 |
| sub04 | −0.2356 | 0.3831 | −0.3750 | 0.2039 | 6 |
| sub06 | −0.3214 | 0.3204 | −0.5876 | 0.2448 | 6 |
| sub07 | 0.1697 | 0.2922 | 0.7688 | 0.4834 | 6 |
| sub08 | −0.4455 | 0.6464 | −0.3750 | 0.2332 | 6 |
| sub09 | 0.2831 | 0.1675 | −0.4645 | 0.3812 | 6 |
| sub10 | 0.5479 | 0.1319 | −0.3151 | 0.3003 | 6 |
| sub14 | 0.0985 | 0.1999 | −0.2901 | 0.4117 | 6 |
| sub15 | −0.1547 | 0.1069 | −0.0326 | 0.3730 | 6 |
| sub16 | 0.3225 | 0.5954 | −0.0239 | 0.5104 | 6 |
| sub17 | −0.0472 | 0.2142 | −0.4091 | 0.3773 | 6 |
| sub19 | −0.3231 | 0.3357 | −0.0126 | 0.2725 | 6 |
| sub21 | 0.0510 | 0.3881 | −0.2072 | 0.2980 | 6 |
| sub22 | −0.2405 | 0.1574 | −0.5414 | 0.2927 | 6 |
| sub23 | −0.2110 | 0.4433 | −0.3894 | 0.1749 | 6 |
| **sub24** | **0.3156** | 1.1964 | **2.7925** | 1.8980 | 6 |
| sub26 | 0.3518 | 0.5361 | 0.5624 | 0.4971 | 6 |
| sub28 | 0.1587 | 0.8578 | 1.1518 | 1.4615 | 6 |
| sub30 | −0.4201 | 0.2321 | −0.6177 | 0.1772 | 6 |
| sub34 | 0.3002 | 0.1910 | −0.1643 | 0.2490 | 6 |

**Subject-mean distribution parameters**:

| | Mean | SD |
|---|--------|--------|
| Subject-mean KScore_raw | −0.0000 | 0.2838 |
| Subject-mean DScore_raw | −0.0000 | 0.7878 |

> sub24's D_raw_mean = 2.7925 is 3.54 standard deviations above the sample mean. Its K_raw_std (1.20) and D_raw_std (1.90) are the largest among all subjects — suggesting sub24's landing strategy exhibits substantial intra-trial variability across the 6 trials.

---

## Step 3: Second z-Score Standardization (Final Subject Scores)

```python
sub_agg["KScore"] = (sub_agg["KScore"] - sub_agg["KScore"].mean()) / sub_agg["KScore"].std()
sub_agg["DScore"] = (sub_agg["DScore"] - sub_agg["DScore"].mean()) / sub_agg["DScore"].std()
```

**Final K-Score and D-Score for 22 subjects** (mean = 0, SD = 1):

| Subject | KScore | DScore | Diff (K−D) |
|---------|--------|--------|-------------|
| sub01 | 0.2795 | 0.7478 | −0.4683 |
| sub02 | −0.1818 | −0.7237 | +0.5419 |
| sub03 | −0.8026 | −0.6207 | −0.1818 |
| sub04 | −0.8303 | −0.4760 | −0.3543 |
| sub06 | −1.1324 | −0.7459 | −0.3865 |
| sub07 | 0.5979 | 0.9758 | −0.3779 |
| sub08 | −1.5696 | −0.4759 | −1.0936 |
| sub09 | 0.9973 | −0.5896 | +1.5870 |
| sub10 | 1.9306 | −0.3999 | +2.3305 |
| sub14 | 0.3471 | −0.3682 | +0.7153 |
| sub15 | −0.5452 | −0.0413 | −0.5038 |
| sub16 | 1.1362 | −0.0304 | +1.1665 |
| sub17 | −0.1662 | −0.5193 | +0.3532 |
| sub19 | −1.1383 | −0.0160 | −1.1223 |
| sub21 | 0.1798 | −0.2630 | +0.4428 |
| sub22 | −0.8475 | −0.6872 | −0.1603 |
| sub23 | −0.7433 | −0.4942 | −0.2491 |
| sub24 | 1.1121 | 3.5445 | −2.4324 |
| sub26 | 1.2395 | 0.7138 | +0.5256 |
| sub28 | 0.5592 | 1.4620 | −0.9028 |
| sub30 | −1.4801 | −0.7840 | −0.6961 |
| sub34 | 1.0578 | −0.2086 | +1.2664 |

---

## Step 4: Spearman Rank Correlation Coefficient

### 4.1 Principle

Spearman's ρ is Pearson's r applied to ranks. For data without ties, the simplified formula is `ρ = 1 − 6Σd² / (n(n²−1))`, where d = rank(x_i) − rank(y_i).

### 4.2 Ranks and Differences

| Subject | KScore | rank_K | DScore | rank_D | d | d² |
|---------|--------|--------|--------|--------|---|---|
| sub01 | 0.2795 | 13 | 0.7478 | 19 | −6 | 36 |
| sub02 | −0.1818 | 10 | −0.7237 | 3 | +7 | 49 |
| sub03 | −0.8026 | 7 | −0.6207 | 5 | +2 | 4 |
| sub04 | −0.8303 | 6 | −0.4760 | 9 | −3 | 9 |
| sub06 | −1.1324 | 4 | −0.7459 | 2 | +2 | 4 |
| sub07 | 0.5979 | 16 | 0.9758 | 20 | −4 | 16 |
| sub08 | −1.5696 | 1 | −0.4759 | 10 | −9 | 81 |
| sub09 | 0.9973 | 17 | −0.5896 | 6 | +11 | 121 |
| sub10 | 1.9306 | 22 | −0.3999 | 11 | +11 | 121 |
| sub14 | 0.3471 | 14 | −0.3682 | 12 | +2 | 4 |
| sub15 | −0.5452 | 9 | −0.0413 | 15 | −6 | 36 |
| sub16 | 1.1362 | 20 | −0.0304 | 16 | +4 | 16 |
| sub17 | −0.1662 | 11 | −0.5193 | 7 | +4 | 16 |
| sub19 | −1.1383 | 3 | −0.0160 | 17 | −14 | 196 |
| sub21 | 0.1798 | 12 | −0.2630 | 13 | −1 | 1 |
| sub22 | −0.8475 | 5 | −0.6872 | 4 | +1 | 1 |
| sub23 | −0.7433 | 8 | −0.4942 | 8 | 0 | 0 |
| sub24 | 1.1121 | 19 | 3.5445 | 22 | −3 | 9 |
| sub26 | 1.2395 | 21 | 0.7138 | 18 | +3 | 9 |
| sub28 | 0.5592 | 15 | 1.4620 | 21 | −6 | 36 |
| sub30 | −1.4801 | 2 | −0.7840 | 1 | +1 | 1 |
| sub34 | 1.0578 | 18 | −0.2086 | 14 | +4 | 16 |

### 4.3 Calculation

```
Σ d² = 782.0
n = 22

ρ (tie-uncorrected) = 1 − 6 × 782.0 / (22 × (22² − 1))
                     = 1 − 4692 / (22 × 483)
                     = 1 − 4692 / 10626
                     = 1 − 0.441558
                     = 0.558442
```

**scipy.stats.spearmanr** (with tie correction) returns `ρ = 0.558442, p = 0.006909`.  
Since there is only one tie in the ranks (sub23's K=8, D=8 have the same rank), the manual computation matches scipy exactly.

### 4.4 Fisher z-Transformation for 95% CI

```
Fisher z = arctanh(ρ) = arctanh(0.558442) = 0.630566

SE = 1 / sqrt(n − 3) = 1 / sqrt(19) = 0.229416

z_lo = 0.630566 − 1.96 × 0.229416 = 0.180911
z_hi = 0.630566 + 1.96 × 0.229416 = 1.080220

CI = [tanh(0.180911), tanh(1.080220)]
   = [0.178963, 0.793281]
```

**Final**: **Spearman ρ = 0.558, 95% CI [0.179, 0.793], p = 0.007**

> **Cohen (1988) benchmark**: ρ ≥ 0.50 is considered "large." 0.558 just crosses this threshold.

---

## Step 5: Median-Split Dichotomization

```
median K-Score = 0.006840
median D-Score = −0.384060
```

> Note: The medians of K-Score and D-Score differ. K-Score's median is close to 0 (its distribution is generally symmetric), while D-Score's median is −0.384, negatively skewed — because sub24's D-Score = 3.54 is extremely right-skewed, pulling D-Score's overall mean (0) upward while keeping the median in the lower half. Visually, this means the scatter plot's D-quantile line will lie below the K-quantile line.

**Classification results for each subject**:

| Subject | KScore | K≥0.00684? | DScore | D≥−0.38406? | Agree? |
|---------|--------|------------|--------|-------------|--------|
| sub01 | 0.2795 | HIGH | 0.7478 | HIGH | YES |
| sub02 | −0.1818 | low | −0.7237 | low | YES |
| sub03 | −0.8026 | low | −0.6207 | low | YES |
| sub04 | −0.8303 | low | −0.4760 | low | YES |
| sub06 | −1.1324 | low | −0.7459 | low | YES |
| sub07 | 0.5979 | HIGH | 0.9758 | HIGH | YES |
| sub08 | −1.5696 | low | −0.4759 | low | YES |
| **sub09** | **0.9973** | **HIGH** | **−0.5896** | **low** | **NO** |
| **sub10** | **1.9306** | **HIGH** | **−0.3999** | **low** | **NO** |
| sub14 | 0.3471 | HIGH | −0.3682 | HIGH | YES |
| **sub15** | **−0.5452** | **low** | **−0.0413** | **HIGH** | **NO** |
| sub16 | 1.1362 | HIGH | −0.0304 | HIGH | YES |
| sub17 | −0.1662 | low | −0.5193 | low | YES |
| **sub19** | **−1.1383** | **low** | **−0.0160** | **HIGH** | **NO** |
| sub21 | 0.1798 | HIGH | −0.2630 | HIGH | YES |
| sub22 | −0.8475 | low | −0.6872 | low | YES |
| sub23 | −0.7433 | low | −0.4942 | low | YES |
| sub24 | 1.1121 | HIGH | 3.5445 | HIGH | YES |
| sub26 | 1.2395 | HIGH | 0.7138 | HIGH | YES |
| sub28 | 0.5592 | HIGH | 1.4620 | HIGH | YES |
| sub30 | −1.4801 | low | −0.7840 | low | YES |
| sub34 | 1.0578 | HIGH | −0.2086 | HIGH | YES |

**4 conflicting subjects**:
- **sub09, sub10**: K=HIGH, D=low (K+D−) — classified as high-risk by kinematics but not by kinetics
- **sub15, sub19**: K=low, D=HIGH (K−D+) — classified as high-risk by kinetics but not by kinematics

Conflicts are symmetric (2+2), indicating no systematic directional bias.

---

## Step 6: Cohen's κ

### 6.1 Confusion Matrix

| | D=Low | D=High | Total |
|---|-------|--------|-------|
| **K=Low** | 9 (Agree) | 2 (K−D+) | 11 |
| **K=High** | 2 (K+D−) | 9 (Agree) | 11 |
| **Total** | 11 | 11 | 22 |

n_00 = 9, n_01 = 2, n_10 = 2, n_11 = 9

### 6.2 Calculation

```
Observed agreement  p_o = (9 + 9) / 22 = 18/22 = 0.818182
Expected agreement  p_e = 0.5  (under a median split, the chance-expected agreement probability)

κ (manual) = (p_o − p_e) / (1 − p_e)
           = (0.818182 − 0.500000) / (1 − 0.500000)
           = 0.318182 / 0.500000
           = 0.636364
```

**sklearn.metrics.cohen_kappa_score** returns `κ = 0.636364`, consistent with the manual calculation.

### 6.3 Confidence Interval

```
κ_SE = sqrt(p_o × (1−p_o) / (n × (1−p_e)²))
     = sqrt(0.818182 × 0.181818 / (22 × 0.250000))
     = sqrt(0.148760 / 5.500000)
     = sqrt(0.027047)
     = 0.164461

κ 95% CI = [0.636364 − 1.96 × 0.164461, 0.636364 + 1.96 × 0.164461]
         = [0.314021, 0.958707]
```

**Final**: **Cohen's κ = 0.636, 95% CI [0.314, 0.959]**

> **Landis & Koch (1977) benchmark**: 0.61 ≤ κ < 0.80 is "substantial."  
> 0.636 falls exactly within this range.

---

## Step 7: McNemar Test

```
McNemar χ² (Yates continuity correction) = (|n_01 − n_10| − 1)² / (n_01 + n_10)
                                          = (|2 − 2| − 1)² / (2 + 2)
                                          = (−1)² / 4
                                          = 1 / 4
                                          = 0.250

p = 1 − CDF_χ²(0.250, df=1) = 0.617
```

**p = 0.62** → Cannot reject the null hypothesis that the marginal distributions of the two classification methods are identical. That is, neither risk proxy is systematically more stringent than the other.

---

## Step 8: Bland-Altman

```
diffs = KScore − DScore  (22 difference values)
Bias  = mean(diffs) = −0.0000  (a degenerate identity, since both variables are z-score standardized to mean 0)
SD    = std(diffs, ddof=0) = 1.0225

LoA = Bias ± 1.96 × SD
    = [−0.0000 − 1.96 × 1.0225, −0.0000 + 1.96 × 1.0225]
    = [−2.0041, 2.0041]
```

**LoA = [−2.00, 2.00]**

> **Interpretation**: For z-score variables, ±2 SD covers approximately 95% of individual differences. This means that the discrepancy between K-Score and D-Score can span nearly the entire sample distribution — although κ reaches a substantial level, individual differences remain large.

---

## Step 9: Power Analysis

**Formula**: `N = ((z_α/2 + z_β) / arctanh(ρ))² + 3`  
where `z_α/2 = 1.96 (α = 0.05)`, `z_β = 0.8416 (β = 0.20, power = 80%)`

| ρ (assumed true effect) | N required for 80% power |
|--------------------------|---------------------------|
| 0.20 | 194 |
| 0.25 | 124 |
| 0.30 | 85 |
| 0.35 | 62 |
| 0.40 | 47 |
| 0.50 | 30 |

**At current N = 22**:

| | ρ |
|---|-----|
| 50% power to detect | ρ ≥ 0.422 |
| 80% power to detect | ρ ≥ 0.567 |

> The observed ρ = 0.558 falls just below the 80% power detection threshold (0.567), but exceeds the 50% power threshold (0.422). This aligns well with the positioning of a pilot study: the effect-size estimate is meaningful, but the confidence interval is relatively wide.

---

## Numerical Consistency Verification Checklist

| Quantity | Manual | scipy/sklearn | results_summary.json | tables.tex |
|----------|--------|---------------|----------------------|------------|
| ρ | 0.558442 | 0.558442 | 0.558 | 0.558 |
| ρ CI | [0.179, 0.793] | — | [0.179, 0.793] | [0.179, 0.793] |
| κ | 0.636364 | 0.636364 | 0.636 | 0.636 |
| κ CI | [0.314, 0.959] | — | [0.314, 0.959] | [0.314, 0.959] |
| Conflicts | 4/22 (18.2%) | — | 4/22 (18.2%) | 18.2% (4/22) |
| McNemar χ² | 0.250 (p = 0.617) | — | — | 0.250 (p = 0.617) |
| BA Bias | −0.000 | — | 0.000 | 0.000 |
| BA LoA | [−2.004, 2.004] | — | [−2.004, 2.004] | [−2.004, 2.004] |

All consistent. ✅
