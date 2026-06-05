"""
Extract knee and hip joint moments from .c3d files (quasi-static approach).

Output: data/joint_moments.csv
"""

import ezc3d
import numpy as np
import pandas as pd
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inverse_dynamics import (
    compute_leg_joint_moments, FP1, FP2,
    GRF_FREQ, VIDEO_FREQ, ANALOG_RATIO, WINDOW_S,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
C3D_DIR = os.path.join(DATA_BASE, "Kinematic_data", "Kinematic_data", "Raw_c3d_files")
PARTICIPANT_LOG = os.path.join(DATA_BASE, "Participants", "participant_log.xlsx")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "joint_moments.csv")
OUTPUT_DIAG = os.path.join(PROJECT_ROOT, "data", "joint_moments_diagnostics.json")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

CONTROL_IDS = ["sub01", "sub02", "sub03", "sub04", "sub06", "sub07", "sub08",
               "sub09", "sub10", "sub14", "sub15", "sub16", "sub17", "sub19",
               "sub21", "sub22", "sub23", "sub24", "sub26", "sub28", "sub30", "sub34"]

TRIAL_TYPES = ["CMJ", "DJ"]
TRIAL_NUMS = [1, 2, 3]
IC_THRESHOLD_N = 20.0


def load_body_weights():
    df_raw = pd.read_excel(PARTICIPANT_LOG, header=None)
    col0 = df_raw.iloc[:, 0].astype(str)
    has_sub = col0.str.match(r'^sub\d+\s*$')
    data_rows = df_raw[has_sub]
    bw_map = {}
    for _, row in data_rows.iterrows():
        sub_id = str(row.iloc[0]).strip()
        weight = float(row.iloc[5])
        bw_map[sub_id] = weight  # kg
    return bw_map


bw_map = load_body_weights()
print(f"Loaded body masses for {len(bw_map)} subjects")


def detect_ic(analogs):
    """Find initial contact frame from vGRF (both plates summed)."""
    vgrf_raw = analogs[0, 2, :] + analogs[0, 8, :]  # Fz1 + Fz2
    vgrf_abs = np.abs(vgrf_raw)
    n_analog = len(vgrf_abs)
    peak_analog = int(np.argmax(vgrf_abs))
    ic_analog = None
    for a in range(peak_analog, 1, -1):
        if vgrf_abs[a] < IC_THRESHOLD_N and vgrf_abs[a - 1] < IC_THRESHOLD_N:
            ic_analog = a
            if vgrf_abs[ic_analog] < IC_THRESHOLD_N:
                ic_analog += 1
            break
    if ic_analog is None or ic_analog < 10:
        for a in range(n_analog):
            if vgrf_abs[a] > IC_THRESHOLD_N:
                ic_analog = a
                break
    return ic_analog


# ==============================================================================
# Main loop
# ==============================================================================
records = []
diagnostics = {}
n_ok = 0
n_err = 0

for sub_id in CONTROL_IDS:
    sub_dir = os.path.join(C3D_DIR, sub_id)
    if not os.path.isdir(sub_dir):
        continue

    body_mass = bw_map.get(sub_id, 70.0)
    file_prefix = sub_id.replace("sub", "s")

    for trial_type in TRIAL_TYPES:
        for tn in TRIAL_NUMS:
            fname = f"{file_prefix}_{trial_type}_t{tn}.c3d"
            fpath = os.path.join(sub_dir, fname)
            if not os.path.exists(fpath):
                continue

            tag = f"{sub_id}/{trial_type}_t{tn}"
            try:
                c3d = ezc3d.c3d(fpath)
            except Exception as e:
                n_err += 1
                continue

            markers_raw = c3d['data']['points']     # (4, n_markers, n_frames)
            analogs_raw = c3d['data']['analogs']     # (1, n_channels, n_analog)
            labels = list(c3d['parameters']['POINT']['LABELS']['value'])
            marker_idx = {name: i for i, name in enumerate(labels)}
            markers_3d = np.transpose(markers_raw[:3, :, :], (2, 1, 0))  # (n_video, n_markers, 3) mm
            analog_data = analogs_raw[0, :, :].T  # (n_analog, n_channels)

            # IC detection
            ic_analog = detect_ic(analogs_raw)
            if ic_analog is None:
                n_err += 1
                continue

            # Standard convention for bilateral landing tasks:
            # Left foot → Plate 2, Right foot → Plate 1
            # (both plates are used; each foot has its own plate)
            result_L = compute_leg_joint_moments(
                markers_3d, analog_data, marker_idx, FP2, body_mass, ic_analog, side="left"
            )
            result_R = compute_leg_joint_moments(
                markers_3d, analog_data, marker_idx, FP1, body_mass, ic_analog, side="right"
            )

            err_L = result_L.get("error")
            err_R = result_R.get("error")

            knee_L = result_L.get("knee_moment_peak_Nmkg", np.nan)
            hip_L = result_L.get("hip_moment_peak_Nmkg", np.nan)
            knee_R = result_R.get("knee_moment_peak_Nmkg", np.nan)
            hip_R = result_R.get("hip_moment_peak_Nmkg", np.nan)

            # Average left and right
            vals = []
            for v in [knee_L, knee_R]:
                if np.isfinite(v):
                    vals.append(v)
            knee_avg = float(np.mean(vals)) if vals else np.nan

            vals = []
            for v in [hip_L, hip_R]:
                if np.isfinite(v):
                    vals.append(v)
            hip_avg = float(np.mean(vals)) if vals else np.nan

            records.append({
                "subject": sub_id,
                "trial_type": trial_type,
                "trial_num": tn,
                "ic_analog_frame": ic_analog,
                "knee_moment_L_Nmkg": knee_L,
                "hip_moment_L_Nmkg": hip_L,
                "knee_moment_R_Nmkg": knee_R,
                "hip_moment_R_Nmkg": hip_R,
                "knee_moment_avg_Nmkg": knee_avg,
                "hip_moment_avg_Nmkg": hip_avg,
            })

            status = "OK" if np.isfinite(knee_avg) else "FAIL"
            if status == "OK":
                n_ok += 1
            else:
                n_err += 1

            print(f"  [{status}] {tag}: knee_avg={knee_avg:.3f}, hip_avg={hip_avg:.3f} Nm/kg")
            diagnostics[tag] = {
                "ic_analog": ic_analog,
                "knee_L": knee_L, "hip_L": hip_L,
                "knee_R": knee_R, "hip_R": hip_R,
                "knee_avg": knee_avg, "hip_avg": hip_avg,
                "err_L": err_L, "err_R": err_R,
            }

# Save
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} records to {OUTPUT_CSV}")
print(f"Success: {n_ok}, Errors/Skips: {n_err}")

with open(OUTPUT_DIAG, "w") as f:
    json.dump(diagnostics, f, indent=2, default=str)

valid = df.dropna(subset=["knee_moment_avg_Nmkg"])
if len(valid) > 0:
    print(f"\nSummary (Nm/kg):")
    for col in ["knee_moment_avg_Nmkg", "hip_moment_avg_Nmkg"]:
        s = valid[col]
        print(f"  {col}: mean={s.mean():.3f}, std={s.std():.3f}, "
              f"min={s.min():.3f}, max={s.max():.3f}")
