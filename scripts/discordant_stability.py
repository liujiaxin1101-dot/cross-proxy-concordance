"""
Stability of discordant subject sets across perturbations (S1-S8).

Question: When a perturbation changes κ, does it change WHO is discordant,
or just HOW MANY?

- Mechanism 1 (stable core): same individuals discordant across perturbations
- Mechanism 2 (specification-induced): discordant set membership changes with perturbation
"""

import numpy as np
import pandas as pd

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "features_raw.csv")

# ==============================================================================
# Load
# ==============================================================================
df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} records, {df.subject.nunique()} subjects")

# Filter to control group only
df = df[df.subject.isin([f"sub{s:02d}" for s in range(1, 44)])].copy()
df = df[df.subject.isin([f"sub{s:02d}" for s in [1,2,3,4,6,7,8,9,10,14,15,16,17,19,21,22,23,24,26,28,30,34]])].copy()
print(f"Control group: {df.subject.nunique()} subjects, {len(df)} trials")

kin_vars = ["hip_flex", "knee_valg", "trunk_lean", "ankle_angle_sagittal"]
kin_dir = {"hip_flex": -1, "knee_valg": 1, "trunk_lean": -1, "ankle_angle_sagittal": -1}
kinetics_vars = ["peak_vgrf_bw", "loading_rate_bw_s"]

# ==============================================================================
# Subject-level scoring helper
# ==============================================================================
def compute_subject_scores(df_in, kin_dir_override=None, k_components_override=None,
                           d_components_override=None, ic_threshold=None, exclude_sub24=False):
    """Compute K-Score and D-Score at subject level, return (scores_df, discordant_set)."""
    d = df_in.copy()
    
    dir_map = kin_dir_override if kin_dir_override else kin_dir
    k_comp = k_components_override if k_components_override else kin_vars
    d_comp = d_components_override if d_components_override else kinetics_vars
    
    if exclude_sub24:
        d = d[d.subject != "sub24"]
    
    # K-Score
    for c in k_comp:
        d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std()
        if dir_map.get(c, 1) == -1:
            d[f"z_{c}"] = -d[f"z_{c}"]
    d["K_raw"] = d[[f"z_{c}" for c in k_comp]].mean(axis=1)
    
    # D-Score
    for c in d_comp:
        d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std()
    d["D_raw"] = d[[f"z_{c}" for c in d_comp]].mean(axis=1)
    
    # Subject-level
    sub = d.groupby("subject").agg(
        K=("K_raw", "mean"), D=("D_raw", "mean")
    ).reset_index()
    sub["Kz"] = (sub["K"] - sub["K"].mean()) / sub["K"].std()
    sub["Dz"] = (sub["D"] - sub["D"].mean()) / sub["D"].std()
    
    # Median split
    sub["K_hi"] = (sub["Kz"] >= sub["Kz"].median()).astype(int)
    sub["D_hi"] = (sub["Dz"] >= sub["Dz"].median()).astype(int)
    sub["discordant"] = (sub["K_hi"] != sub["D_hi"]).astype(int)
    sub["quadrant"] = sub.apply(lambda r: f"{'K+' if r.K_hi else 'K-'}{'D+' if r.D_hi else 'D-'}", axis=1)
    
    discordant_set = set(sub[sub.discordant == 1]["subject"].values)
    return sub, discordant_set


# ==============================================================================
# S2: IC threshold 50N — re-extract features from c3d files
# ==============================================================================
def load_features_ic50n():
    import ezc3d
    import os as _os
    DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
    C3D_DIR = _os.path.join(DATA_BASE, "Kinematic_data", "Kinematic_data", "Raw_c3d_files")
    PARTICIPANT_LOG = _os.path.join(DATA_BASE, "Participants", "participant_log.xlsx")
    CONTROL_IDS = ["sub01", "sub02", "sub03", "sub04", "sub06", "sub07", "sub08",
                   "sub09", "sub10", "sub14", "sub15", "sub16", "sub17", "sub19",
                   "sub21", "sub22", "sub23", "sub24", "sub26", "sub28", "sub30", "sub34"]
    FZ1_CH, FZ2_CH = 2, 8
    GRF_FREQ, VIDEO_FREQ = 1000.0, 250.0
    RATIO = int(GRF_FREQ / VIDEO_FREQ)

    bw_df = pd.read_excel(PARTICIPANT_LOG, header=None)
    mask = bw_df.iloc[:, 0].astype(str).str.match(r'^sub\d+\s*$')
    bw_map = {str(r.iloc[0]).strip(): float(r.iloc[5]) * 9.81 for _, r in bw_df[mask].iterrows()}

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

    records = []
    for sub_id in CONTROL_IDS:
        sd = os.path.join(C3D_DIR, sub_id)
        if not os.path.isdir(sd):
            continue
        fp = sub_id.replace("sub", "s")
        bw = bw_map.get(sub_id, 700)
        for tt in ["CMJ", "DJ"]:
            for tn in [1, 2, 3]:
                fn = f"{fp}_{tt}_t{tn}.c3d"
                fpath = os.path.join(sd, fn)
                if not os.path.exists(fpath):
                    continue
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
                    if vgrf[a] < 50 and vgrf[a - 1] < 50:
                        ic_a = a + 1
                        break
                if ic_a < 10:
                    for a in range(n_a):
                        if vgrf[a] > 50:
                            ic_a = a
                            break
                ic_v = min(ic_a // RATIO, markers.shape[2] - 1)
                required = ["LGT", "RGT", "LKNE", "LKNEM", "RKNE", "RKNEM",
                            "LANK", "LANKM", "RANK", "RANKM", "LHEE", "RHEE",
                            "LTOE", "RTOE", "C7", "T10", "LASI", "RASI"]
                if any(m not in idx for m in required):
                    continue
                hip_L = markers[:3, idx["LGT"], ic_v]
                hip_R = markers[:3, idx["RGT"], ic_v]
                knee_L = (markers[:3, idx["LKNE"], ic_v] + markers[:3, idx["LKNEM"], ic_v]) / 2
                knee_R = (markers[:3, idx["RKNE"], ic_v] + markers[:3, idx["RKNEM"], ic_v]) / 2
                ankle_L = (markers[:3, idx["LANK"], ic_v] + markers[:3, idx["LANKM"], ic_v]) / 2
                ankle_R = (markers[:3, idx["RANK"], ic_v] + markers[:3, idx["RANKM"], ic_v]) / 2
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
                win_end = min(ic_a + int(0.3 * GRF_FREQ), n_a)
                vgrf_w = vgrf[ic_a:win_end]
                pk_idx = np.argmax(vgrf_w)
                peak = vgrf_w[pk_idx]
                ttp = pk_idx / GRF_FREQ
                lr = peak / ttp / bw if ttp > 0 else np.nan
                records.append({
                    "subject": sub_id, "trial_type": tt, "trial_num": tn,
                    "hip_flex": hip_flex, "knee_valg": knee_valg,
                    "trunk_lean": trunk_lean, "ankle_angle_sagittal": ankle_angle_sagittal,
                    "peak_vgrf_bw": peak / bw, "loading_rate_bw_s": lr, "bw_N": bw
                })
    return pd.DataFrame(records)


# ==============================================================================
# Compute for each perturbation
# ==============================================================================
perturbations = {}

# Primary (pooled, correct direction, full components)
_, disc_primary = compute_subject_scores(df)
perturbations["S0_Primary"] = disc_primary

# S1: Task stratification - CMJ only
df_cmj = df[df.trial_type == "CMJ"]
_, disc_s1_cmj = compute_subject_scores(df_cmj)
perturbations["S1_CMJ"] = disc_s1_cmj

# S1: Task stratification - DJ only
df_dj = df[df.trial_type == "DJ"]
_, disc_s1_dj = compute_subject_scores(df_dj)
perturbations["S1_DJ"] = disc_s1_dj

# S2: IC threshold 50N (re-extract from c3d files)
print("Extracting IC=50N features from c3d files...")
df_ic50 = load_features_ic50n()
print(f"  IC=50N: {len(df_ic50)} records from {df_ic50['subject'].nunique()} subjects")
_, disc_s2 = compute_subject_scores(df_ic50)
perturbations["S2_IC50N"] = disc_s2

# S4c: K-Score = knee valgus only
_, disc_s4c = compute_subject_scores(df, k_components_override=["knee_valg"])
perturbations["S4c_KVonly"] = disc_s4c

# S4d: D-Score = peak GRF only
_, disc_s4d = compute_subject_scores(df, d_components_override=["peak_vgrf_bw"])
perturbations["S4d_GRFonly"] = disc_s4d

# S5: Direction convention reversed
dir_rev = {"hip_flex": 1, "knee_valg": 1, "trunk_lean": 1, "ankle_angle_sagittal": 1}
_, disc_s5 = compute_subject_scores(df, kin_dir_override=dir_rev)
perturbations["S5_DirRev"] = disc_s5

# S7: Outlier exclusion (sub24 removed)
_, disc_s7 = compute_subject_scores(df, exclude_sub24=True)
perturbations["S7_NoSub24"] = disc_s7


# ==============================================================================
# Jaccard overlap analysis
# ==============================================================================
def jaccard(a, b):
    if len(a | b) == 0:
        return 0.0
    return len(a & b) / len(a | b)

print(f"\n{'='*70}")
print(f"DISCORDANT SETS")
print(f"{'='*70}")
for name, dset in perturbations.items():
    n_subjects = 22
    if name == "S7_NoSub24":
        n_subjects = 21
    sorted_set = sorted(dset)
    set_str = ", ".join(sorted_set) if sorted_set else "(none)"
    print(f"  {name:<20s}: n={len(dset):>2d}/{n_subjects} ({len(dset)/n_subjects*100:.1f}%) | {set_str}")

print(f"\n{'='*70}")
print(f"JACCARD OVERLAP MATRIX (discordant subject sets)")
print(f"{'='*70}")

names = list(perturbations.keys())
# Header
print(f"{'':20s}", end="")
for n in names:
    print(f"{n:>10s}", end="")
print()

for n1 in names:
    print(f"{n1:20s}", end="")
    for n2 in names:
        j = jaccard(perturbations[n1], perturbations[n2])
        print(f"{j:10.3f}", end="")
    print()

print(f"\n{'='*70}")
print(f"KEY CONTRASTS")
print(f"{'='*70}")

key_pairs = [
    ("Primary vs Direction Reversal", "S0_Primary", "S5_DirRev"),
    ("Primary vs K-Score=KV only", "S0_Primary", "S4c_KVonly"),
    ("Primary vs D-Score=GRF only", "S0_Primary", "S4d_GRFonly"),
    ("Primary vs No Sub24", "S0_Primary", "S7_NoSub24"),
    ("Primary vs IC 50N", "S0_Primary", "S2_IC50N"),
    ("Direction Reversal vs K-Score=KV only", "S5_DirRev", "S4c_KVonly"),
    ("CMJ vs DJ", "S1_CMJ", "S1_DJ"),
]

for label, k1, k2 in key_pairs:
    j = jaccard(perturbations[k1], perturbations[k2])
    shared = perturbations[k1] & perturbations[k2]
    only1 = perturbations[k1] - perturbations[k2]
    only2 = perturbations[k2] - perturbations[k1]
    print(f"\n  {label}: Jaccard = {j:.3f}")
    print(f"    |S(k1)|={len(perturbations[k1])}, |S(k2)|={len(perturbations[k2])}")
    print(f"    Shared: {sorted(shared) if shared else '(none)'}")
    print(f"    Only in {k1}: {sorted(only1) if only1 else '(none)'}")
    print(f"    Only in {k2}: {sorted(only2) if only2 else '(none)'}")

# ==============================================================================
# Quadrant-level analysis: do discordant subjects shift quadrants?
# ==============================================================================
print(f"\n{'='*70}")
print(f"QUADRANT-LEVEL ANALYSIS")
print(f"{'='*70}")
print(f"\n  K+=high K-Score, K-=low K-Score; D+=high D-Score, D-=low D-Score")
print(f"  Discordant = K+D- (good kinematics, high kinetics) or K-D+ (bad kinematics, low kinetics)")

for pname, (sub_df, _) in [
    ("S0_Primary", compute_subject_scores(df)),
    ("S5_DirRev", compute_subject_scores(df, kin_dir_override=dir_rev)),
    ("S4c_KVonly", compute_subject_scores(df, k_components_override=["knee_valg"])),
]:
    qcounts = sub_df.groupby("quadrant")["subject"].count()
    print(f"\n  {pname}:")
    for q in ["K+D+", "K+D-", "K-D+", "K-D-"]:
        n_q = qcounts.get(q, 0)
        subjects = sorted(sub_df[sub_df.quadrant == q]["subject"].values)
        print(f"    {q}: {n_q:>2d} subjects | {', '.join(subjects) if subjects else '(none)'}")
