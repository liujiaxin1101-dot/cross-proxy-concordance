@echo off
REM ==============================================================================
REM run_all.bat — Reproduction pipeline for Cross-Proxy Concordance paper
REM Requires: Python 3.10+, dependencies from requirements.txt
REM Data: Calisti et al. (2025), CC BY 4.0
REM ==============================================================================
echo ======================================================================
echo STEP 1/8 — Extract features from raw c3d files
echo ======================================================================
python 03_extract_features.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

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
echo STEP 4/8 — Run robustness analyses (IC50N, kinematic IC, weights, rmcorr, cutpoints, sub24)
echo ======================================================================
python robustness_IC50N.py
python robustness_kinematic_IC.py
python robustness_weights.py
python robustness_rmcorr.py
python robustness_cutpoints.py
python sensitivity_sub24.py

echo.
echo ======================================================================
echo STEP 5/8 — Knee flexion sensitivity analysis
echo ======================================================================
python add_knee_flexion.py
python sensitivity_knee_flexion.py

echo.
echo ======================================================================
echo STEP 6/8 — PABAK and weighted kappa
echo ======================================================================
python compute_pabak_weighted_kappa.py

echo.
echo ======================================================================
echo STEP 7/8 — Generate figures
echo ======================================================================
python 05_visualize.py
python 06_kappa_forest.py
python plot_bland_altman.py

echo.
echo ======================================================================
echo STEP 8/8 — Compile LaTeX
echo ======================================================================
cd ..\paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..\scripts

echo.
echo ======================================================================
echo DONE — main.pdf generated in paper/
echo ======================================================================
