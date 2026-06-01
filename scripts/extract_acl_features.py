import ezc3d
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
C3D_DIR = os.path.join(DATA_BASE, "Kinematic_data", "Kinematic_data", "Raw_c3d_files")
PARTICIPANT_LOG = os.path.join(DATA_BASE, "Participants", "participant_log.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTROL_IDS = ["sub01", "sub02", "sub03", "sub04", "sub06", "sub07", "sub08",
               "sub09", "sub10", "sub14", "sub15", "sub16", "sub17", "sub19",
               "sub21", "sub22", "sub23", "sub24", "sub26", "sub28", "sub30", "sub34"]

ACL_IDS = ["sub11", "sub12", "sub13", "sub18", "sub20", "sub25", "sub27",
           "sub29", "sub31", "sub32", "sub33", "sub35", "sub36", "sub37",
           "sub38", "sub39", "sub40", "sub41", "sub42", "sub43", "sub44"]

TRIAL_TYPES = ["CMJ", "DJ"]
TRIAL_NUMS = [1, 2, 3]

FZ1_CH = 2
FZ2_CH = 8
GRF_FREQ = 1000.0
VIDEO_FREQ = 250.0
ANALOG_RATIO = int(GRF_FREQ / VIDEO_FREQ)

IC_THRESHOLD_N = 20.0
FILTER_CUTOFF = 15.0


def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data, axis=0)


def get_marker_pos(markers, name_idx, frame):
    return markers[:3, name_idx, frame]


def get_midpoint(markers, name_idx_a, name_idx_b, frame):
    pa = markers[:3, name_idx_a, frame]
    pb = markers[:3, name_idx_b, frame]
    return (pa + pb) / 2.0


def compute_angle_sagittal(v_prox, v_dist):
    v = v_dist - v_prox
    horiz = np.abs(v[0])
    vert = np.abs(v[2])
    if vert < 1e-6:
        return 90.0
    return np.degrees(np.arctan2(horiz, vert))


def compute_knee_fppa(hip, knee, ankle):
    ha_yz = np.array([ankle[1] - hip[1], ankle[2] - hip[2]])
    ka_yz = np.array([ankle[1] - knee[1], ankle[2] - knee[2]])

    norm_ha = np.linalg.norm(ha_yz)
    norm_ka = np.linalg.norm(ka_yz)
    if norm_ha < 1e-6 or norm_ka < 1e-6:
        return np.nan

    cos_angle = np.dot(ha_yz, ka_yz) / (norm_ha * norm_ka)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    cross = ha_yz[0] * ka_yz[1] - ha_yz[1] * ka_yz[0]
    fppa = np.degrees(np.arctan2(cross, np.dot(ha_yz, ka_yz)))
    return fppa


def compute_ankle_angle_sagittal(knee, ankle, toe, heel):
    shank = np.array([knee[0] - ankle[0], knee[2] - ankle[2]])
    foot = np.array([toe[0] - heel[0], toe[2] - heel[2]])

    norm_s = np.linalg.norm(shank)
    norm_f = np.linalg.norm(foot)
    if norm_s < 1e-6 or norm_f < 1e-6:
        return np.nan

    cos_angle = np.dot(shank, foot) / (norm_s * norm_f)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def load_body_weights():
    df_raw = pd.read_excel(PARTICIPANT_LOG, header=None)
    col0 = df_raw.iloc[:, 0].astype(str)
    has_sub = col0.str.match(r'^sub\d+\s*$')
    data_rows = df_raw[has_sub]
    bw_map = {}
    for _, row in data_rows.iterrows():
        sub_id = str(row.iloc[0]).strip()
        weight = float(row.iloc[5])
        bw_map[sub_id] = weight * 9.81
    return bw_map


def extract_features(subject_ids, group_label):
    bw_map = load_body_weights()
    name_to_idx = {}
    records = []

    for sub_id in subject_ids:
        sub_dir = os.path.join(C3D_DIR, sub_id)
        if not os.path.isdir(sub_dir):
            print(f"  SKIP {sub_id}: directory not found")
            continue

        bw = bw_map.get(sub_id, 700.0)
        file_prefix = sub_id.replace("sub", "s")

        for trial_type in TRIAL_TYPES:
            for tn in TRIAL_NUMS:
                fname = f"{file_prefix}_{trial_type}_t{tn}.c3d"
                fpath = os.path.join(sub_dir, fname)
                if not os.path.exists(fpath):
                    continue

                try:
                    c3d = ezc3d.c3d(fpath)
                except Exception:
                    continue

                markers = c3d['data']['points']
                analogs = c3d['data']['analogs']

                if sub_id not in name_to_idx:
                    labels = list(c3d['parameters']['POINT']['LABELS']['value'])
                    name_to_idx[sub_id] = {name: i for i, name in enumerate(labels)}

                idx = name_to_idx[sub_id]

                n_marker_frames = markers.shape[2]
                n_analog_frames = analogs.shape[2]

                vgrf_raw = analogs[0, FZ1_CH, :] + analogs[0, FZ2_CH, :]
                vgrf_abs = np.abs(vgrf_raw)

                peak_analog = int(np.argmax(vgrf_abs))
                ic_analog = None
                for a in range(peak_analog, 1, -1):
                    if vgrf_abs[a] < IC_THRESHOLD_N and vgrf_abs[a - 1] < IC_THRESHOLD_N:
                        ic_analog = a + 1
                        break

                if ic_analog is None or ic_analog < 10:
                    for a in range(n_analog_frames):
                        if vgrf_abs[a] > IC_THRESHOLD_N:
                            ic_analog = a
                            break

                if ic_analog is None:
                    continue

                ic_video = min(ic_analog // ANALOG_RATIO, n_marker_frames - 1)

                required = ["LASI", "RASI", "LPSI", "RPSI",
                            "LKNE", "LKNEM", "RKNE", "RKNEM",
                            "LANK", "LANKM", "RANK", "RANKM",
                            "LGT", "RGT", "LHEE", "RHEE", "LTOE", "RTOE",
                            "C7", "T10", "CLAV", "STRN"]
                missing_markers = [m for m in required if m not in idx]
                if missing_markers:
                    continue

                hip_L = get_marker_pos(markers, idx["LGT"], ic_video)
                hip_R = get_marker_pos(markers, idx["RGT"], ic_video)
                knee_L = get_midpoint(markers, idx["LKNE"], idx["LKNEM"], ic_video)
                knee_R = get_midpoint(markers, idx["RKNE"], idx["RKNEM"], ic_video)
                ankle_L = get_midpoint(markers, idx["LANK"], idx["LANKM"], ic_video)
                ankle_R = get_midpoint(markers, idx["RANK"], idx["RANKM"], ic_video)

                hip_flex_L = compute_angle_sagittal(hip_L, knee_L)
                hip_flex_R = compute_angle_sagittal(hip_R, knee_R)
                hip_flex = np.mean([hip_flex_L, hip_flex_R])

                knee_valg_L = compute_knee_fppa(hip_L, knee_L, ankle_L)
                knee_valg_R = compute_knee_fppa(hip_R, knee_R, ankle_R)
                knee_valg = np.mean([knee_valg_L, knee_valg_R])

                c7 = get_marker_pos(markers, idx["C7"], ic_video)
                t10 = get_marker_pos(markers, idx["T10"], ic_video)
                lasi = get_marker_pos(markers, idx["LASI"], ic_video)
                rasi = get_marker_pos(markers, idx["RASI"], ic_video)
                trunk_mid = (c7 + t10) / 2.0
                pelvis_mid = (lasi + rasi) / 2.0
                trunk_lean = compute_angle_sagittal(pelvis_mid, trunk_mid)

                heel_L = markers[:3, idx["LHEE"], ic_video]
                toe_L = markers[:3, idx["LTOE"], ic_video]
                heel_R = markers[:3, idx["RHEE"], ic_video]
                toe_R = markers[:3, idx["RTOE"], ic_video]

                ankle_angle_sagittal_L = compute_ankle_angle_sagittal(knee_L, ankle_L, toe_L, heel_L)
                ankle_angle_sagittal_R = compute_ankle_angle_sagittal(knee_R, ankle_R, toe_R, heel_R)
                ankle_angle_sagittal = np.mean([ankle_angle_sagittal_L, ankle_angle_sagittal_R])

                ic_analog_end = min(ic_analog + int(0.3 * GRF_FREQ), n_analog_frames)
                vgrf_window = np.abs(vgrf_raw[ic_analog:ic_analog_end])
                peak_idx = np.argmax(vgrf_window)
                peak_vgrf = vgrf_window[peak_idx]
                time_to_peak_ms = peak_idx / GRF_FREQ * 1000.0

                peak_vgrf_bw = peak_vgrf / bw
                if time_to_peak_ms > 0:
                    loading_rate = peak_vgrf / (time_to_peak_ms / 1000.0) / bw
                else:
                    loading_rate = np.nan

                records.append({
                    "subject": sub_id,
                    "group": group_label,
                    "trial_type": trial_type,
                    "trial_num": tn,
                    "ic_video_frame": ic_video,
                    "hip_flex": hip_flex,
                    "knee_valg": knee_valg,
                    "trunk_lean": trunk_lean,
                    "ankle_angle_sagittal": ankle_angle_sagittal,
                    "peak_vgrf_bw": peak_vgrf_bw,
                    "loading_rate_bw_s": loading_rate,
                    "bw_N": bw,
                })

    return pd.DataFrame(records)


print("Extracting ACL group features...")
acl_df = extract_features(ACL_IDS, "ACL")
print(f"  ACL group: {len(acl_df)} records from {acl_df.subject.nunique()} subjects")

if len(acl_df) > 0:
    acl_out = os.path.join(OUTPUT_DIR, "features_acl_raw.csv")
    acl_df.to_csv(acl_out, index=False)
    print(f"  Saved to {acl_out}")

print("\nExtracting Control group features (for combined dataset)...")
ctl_df = extract_features(CONTROL_IDS, "Control")
print(f"  Control group: {len(ctl_df)} records from {ctl_df.subject.nunique()} subjects")

combined_df = pd.concat([ctl_df, acl_df], ignore_index=True)
combined_out = os.path.join(OUTPUT_DIR, "features_combined.csv")
combined_df.to_csv(combined_out, index=False)
print(f"\nCombined dataset: {len(combined_df)} records from {combined_df.subject.nunique()} subjects")
print(f"  Control: {combined_df[combined_df.group == 'Control'].subject.nunique()}")
print(f"  ACL:     {combined_df[combined_df.group == 'ACL'].subject.nunique()}")
print(f"  Saved to {combined_out}")
