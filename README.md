# Fragile Consistency: A Perturbation Framework for Cross-Proxy Consistency in Landing Assessment

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

## Quick Start

Place the downloaded C3D dataset in a directory and set the environment variable:

```bash
set C3D_DATA_DIR=E:\path\to\c3d_dataset
```

Then run the full pipeline:

```bash
cd scripts
run_all.bat
```

Or run individual scripts as needed.

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `03_extract_features.py` | Extract 8 features (4 kinematic + 2 kinetic + ankle + BW) from raw C3D files |
| 2 | `04_compute_scores.py` | Compute K-Score / D-Score, Spearman ρ, Cohen's κ |
| 3 | `bootstrap_ci.py` | Bootstrap confidence intervals (10,000 resamples) |
| 4 | `robustness_*.py` | Robustness analyses (IC threshold, weights, rmcorr, cutpoints) |
| 5 | `sensitivity_*.py` | Sensitivity analyses (knee flexion, sub24 exclusion) |
| 6 | `compute_pabak_weighted_kappa.py` | PABAK and weighted kappa |
| 7 | `05_visualize.py`, `06_kappa_forest.py`, etc. | Generate all figures |
| 8 | LaTeX compilation | Compile `paper/main.tex` |

## Key Results

- **Spearman ρ = 0.558** (95% CI [0.179, 0.793], p = 0.007)
- **Cohen's κ = 0.636** (95% CI [0.314, 0.959])
- 18.2% classification conflict rate (4/22 subjects)
- McNemar test: no systematic directional bias (p = 0.62)

## Directory Structure

```
├── data/           # Derived features and result summaries
├── figures/        # All figures (PNG + PDF)
├── scripts/        # Analysis scripts
├── paper/          # LaTeX manuscript source (if included)
├── requirements.txt
└── README.md
```

## License

Analysis code: MIT License (or specify your license)

Data: CC BY 4.0 — Calisti et al. (2025)
