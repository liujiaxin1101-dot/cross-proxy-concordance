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

ACL_IDS = ["sub11", "sub12", "sub13", "sub18", "sub20", "sub25", "sub27",
           "sub29", "sub31", "sub32", "sub33", "sub35", "sub36", "sub37",
           "sub38", "sub39", "sub40", "sub41", "sub42", "sub43", "sub44"]

CONTROL_IDS = ["sub01", "sub02", "sub03", "sub04", "sub06", "sub07", "sub08",
               "sub09", "sub10", "sub14", "sub15", "sub16", "sub17", "sub19",
               "sub21", "sub22", "sub23", "sub24", "sub26", "sub28", "sub30", "sub34"]

FZ1_CH, FZ2_CH = 2, 8
GRF_FREQ, VIDEO_FREQ = 1000.0, 250.0
RATIO = int(GRF_FREQ / VIDEO_FREQ)

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]


def load_bw():
    df = pd.read_excel(PARTICIPANT_LOG, header=None)
    mask = df.iloc[:, 0].astype(str).str.match(r'^sub\d+\s*$')
    return {str(r.iloc[0]).strip(): float(r.iloc[5]) * 9.81 for _, r in df[mask].iterrows()}


def extract_all(subject_ids, label):
    bw_map = load_bw()
    records = []
    for sub_id in subject_ids:
        sd = os.path.join(C3D_DIR, sub_id)
        if not os.path.isdir(sd):
            continue
        fp = sub_id.replace("sub", "s")
        for tt in ["CMJ", "DJ"]:
            for tn in [1, 2, 3]:
                fn = f"{fp}_{tt}_t{tn}.c3d"
                fpath = os.path.join(sd, fn)
                if not os.path.exists(fpath):
                    continue
                try:
                    c3d = ezc3d.c3d(fpath)
                except:
                    continue
                markers = c3d['data']['points']
                analogs = c3d['data']['analogs']
                labels = list(c3d['parameters']['POINT']['LABELS']['value'])
                idx = {n: i for i, n in enumerate(labels)}
                required = ["LGT", "RGT", "LKNE", "LKNEM", "RKNE", "RKNEM",
                            "LANK", "LANKM", "RANK", "RANKM", "LHEE", "RHEE",
                            "LTOE", "RTOE", "C7", "T10", "LASI", "RASI"]
                if any(m not in idx for m in required):
                    continue

                vgrf = np.abs(analogs[0, FZ1_CH, :] + analogs[0, FZ2_CH, :])
                n_a = len(vgrf)
                peak_a = int(np.argmax(vgrf))
                ic_a = peak_a
                for a in range(peak_a, 10, -1):
                    if vgrf[a] < 20 and vgrf[a - 1] < 20:
                        ic_a = a + 1
                        break
                if ic_a < 10:
                    for a in range(n_a):
                        if vgrf[a] > 20: ic_a = a; break
                ic_v = min(ic_a // RATIO, markers.shape[2] - 1)
                n_mf = markers.shape[2]

                hip_L = markers[:3, idx["LGT"], ic_v]
                hip_R = markers[:3, idx["RGT"], ic_v]
                knee_L = (markers[:3, idx["LKNE"], ic_v] + markers[:3, idx["LKNEM"], ic_v]) / 2
                knee_R = (markers[:3, idx["RKNE"], ic_v] + markers[:3, idx["RKNEM"], ic_v]) / 2
                ankle_L = (markers[:3, idx["LANK"], ic_v] + markers[:3, idx["LANKM"], ic_v]) / 2
                ankle_R = (markers[:3, idx["RANK"], ic_v] + markers[:3, idx["RANKM"], ic_v]) / 2

                def sag_angle(p, d):
                    v = d - p
                    vz = np.abs(v[2])
                    if vz < 1e-6: return 90.0
                    return np.degrees(np.arctan2(np.abs(v[0]), vz))

                def fppa(hip, knee, ankle):
                    ha = np.array([ankle[1] - hip[1], ankle[2] - hip[2]])
                    ka = np.array([ankle[1] - knee[1], ankle[2] - knee[2]])
                    na, nk = np.linalg.norm(ha), np.linalg.norm(ka)
                    if na < 1e-6 or nk < 1e-6: return np.nan
                    c = np.clip(np.dot(ha, ka) / (na * nk), -1, 1)
                    return np.degrees(np.arctan2(ha[0] * ka[1] - ha[1] * ka[0], np.dot(ha, ka)))

                def ankle_ang(knee, ankle, toe, heel):
                    s = np.array([knee[0] - ankle[0], knee[2] - ankle[2]])
                    f = np.array([toe[0] - heel[0], toe[2] - heel[2]])
                    ns, nf = np.linalg.norm(s), np.linalg.norm(f)
                    if ns < 1e-6 or nf < 1e-6: return np.nan
                    return np.degrees(np.arccos(np.clip(np.dot(s, f) / (ns * nf), -1, 1)))

                hip_flex = np.mean([sag_angle(hip_L, knee_L), sag_angle(hip_R, knee_R)])
                knee_valg = np.mean([fppa(hip_L, knee_L, ankle_L), fppa(hip_R, knee_R, ankle_R)])
                trunk_lean = sag_angle(
                    (markers[:3, idx["LASI"], ic_v] + markers[:3, idx["RASI"], ic_v]) / 2,
                    (markers[:3, idx["C7"], ic_v] + markers[:3, idx["T10"], ic_v]) / 2)
                heel_L = markers[:3, idx["LHEE"], ic_v]
                toe_L = markers[:3, idx["LTOE"], ic_v]
                heel_R = markers[:3, idx["RHEE"], ic_v]
                toe_R = markers[:3, idx["RTOE"], ic_v]
                ankle_angle_sagittal = np.mean([ankle_ang(knee_L, ankle_L, toe_L, heel_L),
                                        ankle_ang(knee_R, ankle_R, toe_R, heel_R)])

                bw = bw_map.get(sub_id, 700.0)
                win_end = min(ic_a + int(0.3 * GRF_FREQ), len(vgrf))
                vgrf_w = np.abs(vgrf[ic_a:win_end])
                pk_idx = np.argmax(vgrf_w)
                peak = vgrf_w[pk_idx]
                ttp = pk_idx / GRF_FREQ
                lr = peak / ttp / bw if ttp > 0 else np.nan

                records.append({
                    "subject": sub_id, "group": label,
                    "trial_type": tt, "trial_num": tn,
                    "hip_flex": hip_flex, "knee_valg": knee_valg,
                    "trunk_lean": trunk_lean, "ankle_angle_sagittal": ankle_angle_sagittal,
                    "peak_vgrf_bw": peak / bw, "loading_rate_bw_s": lr, "bw_N": bw,
                })
    return pd.DataFrame(records)


def score_comparison(df_list, names):
    results = []
    for df, name in zip(df_list, names):
        if len(df) == 0: results.append(None); continue
        dfc = df.copy()
        for c in kin_vars:
            dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
            if kin_dir[c] == -1: dfc[f"z_{c}"] = -dfc[f"z_{c}"]
        dfc["K_raw"] = dfc[[f"z_{c}" for c in kin_vars]].mean(axis=1)
        for c in kinetics_vars:
            dfc[f"z_{c}"] = (dfc[c] - dfc[c].mean()) / dfc[c].std()
        dfc["D_raw"] = dfc[[f"z_{c}" for c in kinetics_vars]].mean(axis=1)
        sub = dfc.groupby("subject").agg(K=("K_raw", "mean"), D=("D_raw", "mean")).reset_index()
        sub["Kz"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
        sub["Dz"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()

        n = len(sub)
        rho, _ = stats.spearmanr(sub["Kz"], sub["Dz"])
        se = 1 / np.sqrt(n - 3)
        ci_lo = np.tanh(np.arctanh(rho) - 1.96 * se)
        ci_hi = np.tanh(np.arctanh(rho) + 1.96 * se)

        kh = (sub["Kz"] >= sub["Kz"].median()).astype(int)
        dh = (sub["Dz"] >= sub["Dz"].median()).astype(int)
        kappa = cohen_kappa_score(kh, dh)
        conflict = (kh != dh).mean()

        k_sd = sub["Kz"].std()
        d_sd = sub["Dz"].std()

        results.append({
            "name": name, "n": n, "n_trials": len(df),
            "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "kappa": kappa, "conflict": conflict,
            "k_sd": k_sd, "d_sd": d_sd,
            "sub": sub,
        })
    return results


print("=" * 70)
print("EXPLORATORY: Including ACL group")
print("=" * 70)

acl_df = extract_all(ACL_IDS, "ACL")
ctl_df = extract_all(CONTROL_IDS, "Control")

print(f"\n  ACL:     {len(acl_df)} records, {acl_df.subject.nunique()} subjects")
print(f"  Control: {len(ctl_df)} records, {ctl_df.subject.nunique()} subjects")

all_df = pd.concat([ctl_df, acl_df], ignore_index=True)
print(f"  Combined: {len(all_df)} records, {all_df.subject.nunique()} subjects")

results = score_comparison(
    [all_df, ctl_df, acl_df,
     all_df[all_df.trial_type == "CMJ"], all_df[all_df.trial_type == "DJ"],
     ctl_df[ctl_df.trial_type == "CMJ"], ctl_df[ctl_df.trial_type == "DJ"],
     acl_df[acl_df.trial_type == "CMJ"], acl_df[acl_df.trial_type == "DJ"]],
    ["All (N=43)", "Control (N=22)", "ACL (N=21)",
     "All-CMJ", "All-DJ",
     "CTL-CMJ", "CTL-DJ",
     "ACL-CMJ", "ACL-DJ"]
)

print(f"\n{'Group':25s} {'N':>4s} {'rho':>7s} {'95% CI':>22s} {'kappa':>7s} {'conflict':>9s}")
print("-" * 78)
for r in results:
    if r is None: continue
    print(f"  {r['name']:25s} {r['n']:4d} {r['rho']:7.3f} [{r['ci_lo']:7.3f},{r['ci_hi']:7.3f}] "
          f"{r['kappa']:7.3f} {r['conflict']*100:7.0f}%")

print(f"\n{'='*70}")
print("DESCRIPTIVES BY GROUP")
print("=" * 70)
for r in results:
    if r is None or 'sub' not in r: continue
    s = r['sub']
    print(f"\n  {r['name']}:")
    print(f"    K-score: range [{s['Kz'].min():.2f}, {s['Kz'].max():.2f}], SD={s['Kz'].std():.2f}")
    print(f"    D-score: range [{s['Dz'].min():.2f}, {s['Dz'].max():.2f}], SD={s['Dz'].std():.2f}")
