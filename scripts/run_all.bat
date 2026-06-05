@echo off
REM ==============================================================================
REM run_all.bat — Reproduction pipeline
REM Paper: Same Person, Different Risk
REM Data: Calisti et al. (2025), CC BY 4.0
REM ==============================================================================
REM
REM TWO MODES:
REM   Mode A (core):  requires ONLY features_raw.csv (included in repo)
REM                    Runs Steps 2-7, reproducing all figures and statistics
REM   Mode B (full):  requires C3D raw data + C3D_DATA_DIR env var
REM                    Re-extracts features from .c3d files, then runs full pipeline
REM
REM Quick check: if C3D_DATA_DIR is NOT set, skips C3D-dependent steps.
REM ==============================================================================

echo.
echo ======================================================================
echo  CROSS-PROXY CONCORDANCE — Reproduction Pipeline
echo  Paper: Same Person, Different Risk
echo ======================================================================
echo.

REM --- Detect C3D data availability ---
set HAS_C3D=0
if defined C3D_DATA_DIR (
    if exist "%C3D_DATA_DIR%\Kinematic_data\Kinematic_data\Raw_c3d_files" (
        set HAS_C3D=1
        echo [DETECTED] C3D_DATA_DIR = %C3D_DATA_DIR%
        echo             Full reproduction mode (Mode B)
    ) else (
        echo [WARNING] C3D_DATA_DIR is set but Raw_c3d_files not found, falling back to core mode
    )
)
if %HAS_C3D%==0 (
    echo [INFO]    C3D_DATA_DIR not set — core reproduction mode (Mode A)
    echo             Using pre-extracted features_raw.csv from data/
)

echo.
echo ======================================================================
echo STEP 1/8 — Extract features from raw C3D files (requires C3D_DATA_DIR)
echo ======================================================================
if %HAS_C3D%==1 (
    python 03_extract_features.py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Feature extraction failed. Check C3D_DATA_DIR path.
        exit /b %ERRORLEVEL%
    )
) else (
    echo [SKIP]    C3D data not available — using existing data/features_raw.csv
)

echo.
echo ======================================================================
echo STEP 2/8 — Compute primary scores and task-stratified results
echo ======================================================================
python 04_compute_scores.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ======================================================================
echo STEP 3/8 — Run bootstrap CIs
echo ======================================================================
python bootstrap_ci.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ======================================================================
echo STEP 4/8 — Robustness: weights, rmcorr, cutpoints, sub24
echo ======================================================================
python robustness_weights.py
python robustness_rmcorr.py
python robustness_cutpoints.py
python sensitivity_sub24.py

echo.
echo ======================================================================
echo STEP 4b/8 — Robustness: IC50N & kinematic IC (requires C3D_DATA_DIR)
echo ======================================================================
if %HAS_C3D%==1 (
    python robustness_IC50N.py
    python robustness_kinematic_IC.py
) else (
    echo [SKIP]    C3D data not available — IC threshold variants not re-extracted
)

echo.
echo ======================================================================
echo STEP 5/8 — Knee flexion sensitivity (requires C3D_DATA_DIR)
echo ======================================================================
if %HAS_C3D%==1 (
    python add_knee_flexion.py
    python sensitivity_knee_flexion.py
) else (
    echo [SKIP]    C3D data not available — using existing data/features_with_knee.csv
    python sensitivity_knee_flexion.py
)

echo.
echo ======================================================================
echo STEP 6/8 — PABAK and weighted kappa
echo ======================================================================
python compute_pabak_weighted_kappa.py

echo.
echo ======================================================================
echo STEP 7/8 — Generate all figures
echo ======================================================================
python 05_visualize.py
python 06_kappa_forest.py
python plot_bland_altman.py

echo.
echo ======================================================================
echo STEP 8/8 — Compile LaTeX (if paper/main.tex exists)
echo ======================================================================
if exist ..\paper\main.tex (
    cd ..\paper
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
    cd ..\scripts
    echo DONE — main.pdf generated in paper/
) else (
    echo [SKIP]    paper/main.tex not found — LaTeX compilation skipped
)

echo.
echo ======================================================================
echo DONE — Pipeline complete
echo ======================================================================
echo.
echo Output files:
echo   data/results_summary.json   — primary statistics
echo   data/final_results.txt      — full results report
echo   figures/*.png, *.pdf        — all figures
echo.
