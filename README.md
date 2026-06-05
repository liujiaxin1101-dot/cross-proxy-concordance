# Same Person, Different Risk: Mapping the Fragile Agreement Between Kinematic and Kinetic Landing-Risk Proxies

Analysis pipeline for evaluating agreement between kinematic (K-Score) and kinetic (D-Score) proxy measures of ACL injury risk during jump-landing tasks.

## Data

Motion capture data from Calisti, M., Mohr, M., & Federolf, P. (2025). *Motion capture data of six jump-landings, fatigued and non-fatigued, after anterior cruciate ligament injury.* Scientific Data, 12(1), 1645.

- **DOI**: [10.1038/s41597-025-05934-5](https://doi.org/10.1038/s41597-025-05934-5)
- **Figshare**: [10.6084/m9.figshare.28890545](https://doi.org/10.6084/m9.figshare.28890545)
- **License**: CC BY 4.0

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Quick Start — Two Reproduction Modes

### Mode A: Core reproduction (no raw data needed)

Runs the full statistical analysis and generates all figures using the **pre-extracted `data/features_raw.csv`** included in this repository. No external downloads required.

```bash
cd scripts
run_all.bat
```

This reproduces: Spearman ρ, Cohen's κ, bootstrap CIs, Bland-Altman, all figures (05–10), and the final results report.

### Mode B: Full reproduction (with raw C3D data)

Re-extracts all features from the original `.c3d` motion capture files, then runs the complete pipeline including IC-threshold robustness checks.

1. Download the dataset from [Figshare](https://doi.org/10.6084/m9.figshare.28890545)
2. Set the environment variable pointing to the extracted dataset root:

   ```bash
   set C3D_DATA_DIR=C:\path\to\extracted\dataset
   ```

3. Run the pipeline:

   ```bash
   cd scripts
   run_all.bat
   ```

The pipeline auto-detects whether C3D data is available and adapts accordingly.

## What Can Be Reproduced Without Raw Data

The following results are **fully reproducible from the CSV files in `data/`** without downloading any C3D files:

| Analysis | Script | Needs C3D? |
|----------|--------|:----------:|
| Primary Spearman ρ & Cohen's κ | `04_compute_scores.py` | No |
| Bootstrap confidence intervals | `bootstrap_ci.py` | No |
| Robustness: weights, rmcorr, cutpoints | `robustness_*.py` | No |
| Sensitivity: sub24 exclusion | `sensitivity_sub24.py` | No |
| PABAK & weighted kappa | `compute_pabak_weighted_kappa.py` | No |
| All figures (05–10) | `05_visualize.py`, etc. | No |
| Specification curve | `spec_curve.py`, `spec_curve_plot.py` | No |
| Combinatorial perturbations | `combinatorial_*.py` | No |
| Upgraded D-Score (joint moments) | `upgrade_dscore.py` | No |

The following require the original C3D data:

| Analysis | Script |
|----------|--------|
| Feature re-extraction | `03_extract_features.py` |
| IC threshold = 50 N | `robustness_IC50N.py` |
| Kinematic IC detection | `robustness_kinematic_IC.py` |
| Knee flexion extraction | `add_knee_flexion.py` |
| Joint moments extraction | `extract_joint_moments.py` |
| Sub24 time-series diagnostic | `sub24_diag.py` |

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `03_extract_features.py` | Extract features from raw C3D files (Mode B only) |
| 2 | `04_compute_scores.py` | Compute K-Score / D-Score, Spearman ρ, Cohen's κ |
| 3 | `bootstrap_ci.py` | Bootstrap confidence intervals (10,000 resamples) |
| 4 | `robustness_*.py` | Robustness analyses (weights, rmcorr, cutpoints, IC variants) |
| 5 | `sensitivity_*.py` | Sensitivity analyses (knee flexion, sub24 exclusion) |
| 6 | `compute_pabak_weighted_kappa.py` | PABAK and weighted kappa |
| 7 | `05_visualize.py`, `06_kappa_forest.py`, etc. | Generate all figures (PNG + PDF) |
| 8 | LaTeX compilation | Compile `paper/main.tex` (if paper/ directory present) |

## Key Results

- **Spearman ρ = 0.558** (95% CI [0.179, 0.793], p = 0.007)
- **Cohen's κ = 0.636** (95% CI [0.314, 0.959])
- 18.2% classification conflict rate (4/22 subjects)
- McNemar test: no systematic directional bias (p = 0.62)

## Documentation

- `ANALYSIS_REPORT.md` — Complete computational logic, results, and figure descriptions
- `core_logic_rho_kappa.md` — Detailed step-by-step derivation of ρ and κ
- `figures/FIGURE_DESCRIPTIONS.md` — Figure-by-figure descriptions

## Directory Structure

```
├── data/              # Derived features, result summaries, and intermediate CSVs
├── figures/           # All figures (PNG + PDF) + figure descriptions
├── scripts/           # All analysis scripts + run_all.bat
├── paper/             # LaTeX manuscript source (if included)
├── .gitignore
├── README.md
├── ANALYSIS_REPORT.md
├── core_logic_rho_kappa.md
└── requirements.txt
```

## License

Analysis code: MIT

Data: CC BY 4.0 — Calisti et al. (2025)
