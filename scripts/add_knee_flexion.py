import ezc3d
import numpy as np
import pandas as pd
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", r"features_raw.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", r"features_with_knee.csv")
DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
C3D_DIR = os.path.join(DATA_BASE, "Kinematic_data", "Kinematic_data", "Raw_c3d_files")

df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} records from {FEATURES_CSV}")


def get_marker_pos(markers, name_idx, frame):
    return markers[:3, name_idx, frame]


def get_midpoint(markers, name_idx_a, name_idx_b, frame):
    pa = markers[:3, name_idx_a, frame]
    pb = markers[:3, name_idx_b, frame]
    return (pa + pb) / 2.0


def compute_knee_flex_sagittal(hip, knee, ankle):
    thigh_xz = np.array([knee[0] - hip[0], knee[2] - hip[2]])
    shank_xz = np.array([ankle[0] - knee[0], ankle[2] - knee[2]])
    norm_t = np.linalg.norm(thigh_xz)
    norm_s = np.linalg.norm(shank_xz)
    if norm_t < 1e-6 or norm_s < 1e-6:
        return np.nan
    cos_angle = np.dot(thigh_xz, shank_xz) / (norm_t * norm_s)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


name_to_idx = {}
knee_flex_vals = []

for _, row in df.iterrows():
    sub_id = row["subject"]
    trial_type = row["trial_type"]
    tn = int(row["trial_num"])
    ic_video = int(row["ic_video_frame"])

    sub_dir = os.path.join(C3D_DIR, sub_id)
    file_prefix = sub_id.replace("sub", "s")
    fname = f"{file_prefix}_{trial_type}_t{tn}.c3d"
    fpath = os.path.join(sub_dir, fname)

    if not os.path.exists(fpath):
        fname_cmj = f"{file_prefix}_{trial_type}_l_t{tn}.c3d"
        fpath = os.path.join(sub_dir, fname_cmj)
    if not os.path.exists(fpath):
        fname_cmj = f"{file_prefix}_{trial_type}_r_t{tn}.c3d"
        fpath = os.path.join(sub_dir, fname_cmj)

    if not os.path.exists(fpath):
        print(f"  MISSING: {fname} for {sub_id}")
        knee_flex_vals.append(np.nan)
        continue

    try:
        c3d = ezc3d.c3d(fpath)
    except Exception as e:
        print(f"  ERROR reading {fname}: {e}")
        knee_flex_vals.append(np.nan)
        continue

    markers = c3d['data']['points']

    if sub_id not in name_to_idx:
        labels = list(c3d['parameters']['POINT']['LABELS']['value'])
        name_to_idx[sub_id] = {name: i for i, name in enumerate(labels)}

    idx = name_to_idx[sub_id]

    required = ["LGT", "RGT", "LKNE", "LKNEM", "RKNE", "RKNEM",
                "LANK", "LANKM", "RANK", "RANKM"]
    missing = [m for m in required if m not in idx]
    if missing:
        print(f"  {fname}: missing markers {missing}")
        knee_flex_vals.append(np.nan)
        continue

    hip_L = get_marker_pos(markers, idx["LGT"], ic_video)
    hip_R = get_marker_pos(markers, idx["RGT"], ic_video)
    knee_L = get_midpoint(markers, idx["LKNE"], idx["LKNEM"], ic_video)
    knee_R = get_midpoint(markers, idx["RKNE"], idx["RKNEM"], ic_video)
    ankle_L = get_midpoint(markers, idx["LANK"], idx["LANKM"], ic_video)
    ankle_R = get_midpoint(markers, idx["RANK"], idx["RANKM"], ic_video)

    kf_L = compute_knee_flex_sagittal(hip_L, knee_L, ankle_L)
    kf_R = compute_knee_flex_sagittal(hip_R, knee_R, ankle_R)
    kf = np.mean([kf_L, kf_R])
    knee_flex_vals.append(kf)

df["knee_flex"] = knee_flex_vals
n_missing = df["knee_flex"].isna().sum()
print(f"Knee flexion extracted: {len(df) - n_missing}/{len(df)} valid, {n_missing} missing")

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")

print(f"\n=== Knee flexion descriptive stats ===")
print(df.groupby("trial_type")["knee_flex"].describe())
print(f"\nPooled: mean={df['knee_flex'].mean():.1f}, SD={df['knee_flex'].std():.1f}")
