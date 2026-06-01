import ezc3d
import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.metrics import cohen_kappa_score
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
C3D_DIR = os.path.join(DATA_BASE, "Kinematic_data", "Kinematic_data", "Raw_c3d_files")
PARTICIPANT_LOG = os.path.join(DATA_BASE, "Participants", "participant_log.xlsx")

CONTROL_IDS = ["sub01", "sub02", "sub03", "sub04", "sub06", "sub07", "sub08",
               "sub09", "sub10", "sub14", "sub15", "sub16", "sub17", "sub19",
               "sub21", "sub22", "sub23", "sub24", "sub26", "sub28", "sub30", "sub34"]

FZ1_CH, FZ2_CH = 2, 8
GRF_FREQ, VIDEO_FREQ = 1000.0, 250.0
RATIO = int(GRF_FREQ / VIDEO_FREQ)

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]


def get_marker_pos(markers, idx, name, frame):
    return markers[:3, idx[name], frame]


def get_midpoint(markers, idx, a, b, frame):
    return (markers[:3, idx[a], frame] + markers[:3, idx[b], frame]) / 2


def sag_angle(v_prox, v_dist):
    d = v_dist - v_prox
    v = np.abs(d[2])
    if v < 1e-6:
        return 90.0
    return np.degrees(np.arctan2(np.abs(d[0]), v))


def fppa(hip, knee, ankle):
    ha = np.array([ankle[1] - hip[1], ankle[2] - hip[2]])
    ka = np.array([ankle[1] - knee[1], ankle[2] - knee[2]])
    na, nk = np.linalg.norm(ha), np.linalg.norm(ka)
    if na < 1e-6 or nk < 1e-6:
        return np.nan
    c = np.clip(np.dot(ha, ka) / (na * nk), -1, 1)
    return np.degrees(np.arctan2(ha[0] * ka[1] - ha[1] * ka[0], np.dot(ha, ka)))


def ankle_ang(knee, ankle, toe, heel):
    s = np.array([knee[0] - ankle[0], knee[2] - ankle[2]])
    f = np.array([toe[0] - heel[0], toe[2] - heel[2]])
    ns, nf = np.linalg.norm(s), np.linalg.norm(f)
    if ns < 1e-6 or nf < 1e-6:
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(s, f) / (ns * nf), -1, 1)))


def load_bw():
    df = pd.read_excel(PARTICIPANT_LOG, header=None)
    mask = df.iloc[:, 0].astype(str).str.match(r'^sub\d+\s*$')
    return {str(r.iloc[0]).strip(): float(r.iloc[5]) * 9.81 for _, r in df[mask].iterrows()}


def extract_trial(sub_id, fname, fpath, bw, threshold):
    c3d = ezc3d.c3d(fpath)
    markers = c3d['data']['points']
    analogs = c3d['data']['analogs']
    labels = list(c3d['parameters']['POINT']['LABELS']['value'])
    idx = {n: i for i, n in enumerate(labels)}

    vgrf = np.abs(analogs[0, FZ1_CH, :] + analogs[0, FZ2_CH, :])
    n_a = len(vgrf)
    peak_a = int(np.argmax(vgrf))
    ic_a = peak_a
    for a in range(peak_a, 10, -1):
        if vgrf[a] < threshold and vgrf[a - 1] < threshold:
            ic_a = a + 1
            break
    if ic_a < 10:
        for a in range(n_a):
            if vgrf[a] > threshold:
                ic_a = a
                break
    ic_v = min(ic_a // RATIO, markers.shape[2] - 1)
    n_mf = markers.shape[2]
    n_af = analogs.shape[2]

    required = ["LGT", "RGT", "LKNE", "LKNEM", "RKNE", "RKNEM",
                "LANK", "LANKM", "RANK", "RANKM", "LHEE", "RHEE",
                "LTOE", "RTOE", "C7", "T10", "LASI", "RASI"]
    if any(m not in idx for m in required):
        return None

    hip_L = get_marker_pos(markers, idx, "LGT", ic_v)
    hip_R = get_marker_pos(markers, idx, "RGT", ic_v)
    knee_L = get_midpoint(markers, idx, "LKNE", "LKNEM", ic_v)
    knee_R = get_midpoint(markers, idx, "RKNE", "RKNEM", ic_v)
    ankle_L = get_midpoint(markers, idx, "LANK", "LANKM", ic_v)
    ankle_R = get_midpoint(markers, idx, "RANK", "RANKM", ic_v)

    hip_flex = np.mean([sag_angle(hip_L, knee_L), sag_angle(hip_R, knee_R)])
    knee_valg = np.mean([fppa(hip_L, knee_L, ankle_L), fppa(hip_R, knee_R, ankle_R)])
    trunk_lean = sag_angle(
        (get_marker_pos(markers, idx, "LASI", ic_v) + get_marker_pos(markers, idx, "RASI", ic_v)) / 2,
        (get_marker_pos(markers, idx, "C7", ic_v) + get_marker_pos(markers, idx, "T10", ic_v)) / 2)

    heel_L = get_marker_pos(markers, idx, "LHEE", ic_v)
    toe_L = get_marker_pos(markers, idx, "LTOE", ic_v)
    heel_R = get_marker_pos(markers, idx, "RHEE", ic_v)
    toe_R = get_marker_pos(markers, idx, "RTOE", ic_v)
    ankle_angle_sagittal = np.mean([ankle_ang(knee_L, ankle_L, toe_L, heel_L),
                            ankle_ang(knee_R, ankle_R, toe_R, heel_R)])

    win_end = min(ic_a + int(0.3 * GRF_FREQ), n_af)
    vgrf_w = np.abs(vgrf[ic_a:win_end])
    pk_idx = np.argmax(vgrf_w)
    peak = vgrf_w[pk_idx]
    ttp = pk_idx / GRF_FREQ
    lr = peak / ttp / bw if ttp > 0 else np.nan

    return {"subject": sub_id, "trial_type": fname.split("_")[1],
            "trial_num": int(fname.split("_t")[1].replace(".c3d", "")),
            "hip_flex": hip_flex, "knee_valg": knee_valg,
            "trunk_lean": trunk_lean, "ankle_angle_sagittal": ankle_angle_sagittal,
            "peak_vgrf_bw": peak / bw, "loading_rate_bw_s": lr, "bw_N": bw,
            "ic_threshold": threshold}


def score_and_compare(df, label):
    dfc = df.copy()
    for c in kin_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
        if kin_dir[c] == -1:
            dfc[f"z_{c}"] = -dfc[f"z_{c}"]
    dfc["K_raw"] = dfc[[f"z_{c}" for c in kin_vars]].mean(axis=1)
    for c in kinetics_vars:
        dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
    dfc["D_raw"] = dfc[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)

    sub = dfc.groupby("subject").agg(K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
    sub["K"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
    sub["D"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

    n = len(sub)
    rho, _ = stats.spearmanr(sub["K"], sub["D"])
    se = 1 / np.sqrt(n - 3)
    ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se)
    ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se)

    md = {"K": sub["K"].median(), "D": sub["D"].median()}
    kh = (sub["K"] >= md["K"]).astype(int)
    dh = (sub["D"] >= md["D"]).astype(int)
    kappa = cohen_kappa_score(kh, dh)
    agree = (kh == dh).mean()
    conflict = (kh != dh).mean()

    return {"label": label, "n": n, "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "kappa": kappa, "agree": agree, "conflict": conflict}


print("=" * 65)
print("ROBUSTNESS CHECK: IC threshold 20N vs 50N")
print("=" * 65)

bw_map = load_bw()
records_50 = []

for sub_id in CONTROL_IDS:
    sd = os.path.join(C3D_DIR, sub_id)
    if not os.path.isdir(sd):
        continue
    fp = sub_id.replace("sub", "s")
    for tt in ["CMJ", "DJ"]:
        for tn in [1, 2, 3]:
            fn = f"{fp}_{tt}_t{tn}.c3d"
            fp_path = os.path.join(sd, fn)
            if not os.path.exists(fp_path):
                continue
            r = extract_trial(sub_id, fn, fp_path, bw_map.get(sub_id, 700), 50)
            if r:
                records_50.append(r)

df_50 = pd.DataFrame(records_50)
print(f"  IC=50N: {len(df_50)} records from {df_50['subject'].nunique()} subjects")
r_50 = score_and_compare(df_50, "IC=50N")

df_20 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", r"features_raw.csv"))
r_20 = score_and_compare(df_20, "IC=20N")

print()
print(f"  {'Metric':30s} {'IC=20N':>12s} {'IC=50N':>12s} {'Delta':>10s}")
print(f"  {'-'*65}")
for k in ["rho", "ci_lo", "ci_hi", "kappa", "agree", "conflict"]:
    v20 = r_20[k]
    v50 = r_50[k]
    if isinstance(v20, float):
        print(f"  {k:30s} {v20:12.3f} {v50:12.3f} {v50-v20:+10.3f}")
    else:
        print(f"  {k:30s} {str(v20):>12s} {str(v50):>12s}")

delta_rho = r_50["rho"] - r_20["rho"]
delta_kappa = r_50["kappa"] - r_20["kappa"]
print(f"\n  Delta rho = {delta_rho:+.3f}, Delta kappa = {delta_kappa:+.3f}")
print(f"  VERDICT: IC threshold shift from 20N to 50N changes rho by "
      f"{delta_rho:+.3f} and kappa by {delta_kappa:+.3f}. "
      f"{'Substantial impact �?needs discussion.' if abs(delta_rho) > 0.10 or abs(delta_kappa) > 0.10 else 'Minor impact �?conclusions robust.'}")
