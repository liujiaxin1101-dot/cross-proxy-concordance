import ezc3d
import numpy as np
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
C3D_DIR = os.path.join(DATA_BASE, "Kinematic_data", "Kinematic_data", "Raw_c3d_files")

cases = [
    ("sub01", "s01_CMJ_t1.c3d"),
    ("sub07", "s07_CMJ_t3.c3d"),
    ("sub24", "s24_CMJ_t1.c3d"),
    ("sub28", "s28_CMJ_t1.c3d"),
]

FZ1, FZ2 = 2, 8
GRF_FREQ = 1000.0
VIDEO_FREQ = 250.0
RATIO = int(GRF_FREQ / VIDEO_FREQ)

for sub_id, fname in cases:
    fpath = os.path.join(C3D_DIR, sub_id, fname)
    c3d = ezc3d.c3d(fpath)
    markers = c3d['data']['points']
    analogs = c3d['data']['analogs']
    labels = list(c3d['parameters']['POINT']['LABELS']['value'])
    idx = {n: i for i, n in enumerate(labels)}

    vgrf = np.abs(analogs[0, FZ1, :] + analogs[0, FZ2, :])
    peak_a = int(np.argmax(vgrf))
    ic_a = peak_a
    for a in range(peak_a, 10, -1):
        if vgrf[a] < 20 and vgrf[a-1] < 20:
            ic_a = a + 1
            break
    ic_v = min(ic_a // RATIO, markers.shape[2] - 1)

    knee_L = (markers[:3, idx["LKNE"], ic_v] + markers[:3, idx["LKNEM"], ic_v]) / 2
    ankle_L = (markers[:3, idx["LANK"], ic_v] + markers[:3, idx["LANKM"], ic_v]) / 2
    toe_L = markers[:3, idx["LTOE"], ic_v]
    heel_L = markers[:3, idx["LHEE"], ic_v]

    shank = knee_L - ankle_L
    foot = toe_L - heel_L

    shank_xz = np.array([shank[0], shank[2]])
    foot_xz = np.array([foot[0], foot[2]])

    norm_s = np.linalg.norm(shank_xz)
    norm_f = np.linalg.norm(foot_xz)
    cos_a = np.dot(shank_xz, foot_xz) / (norm_s * norm_f)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    angle_arccos = np.degrees(np.arccos(cos_a))

    shank_horiz_angle = np.degrees(np.arctan2(np.abs(shank_xz[0]), np.abs(shank_xz[1])))
    foot_horiz_angle = np.degrees(np.arctan2(np.abs(foot_xz[0]), np.abs(foot_xz[1])))

    print(f"\n{sub_id}/{fname}")
    print(f"  ic_video={ic_v}")
    print(f"  knee_L  = [{knee_L[0]:.1f}, {knee_L[1]:.1f}, {knee_L[2]:.1f}]")
    print(f"  ankle_L = [{ankle_L[0]:.1f}, {ankle_L[1]:.1f}, {ankle_L[2]:.1f}]")
    print(f"  toe_L   = [{toe_L[0]:.1f}, {toe_L[1]:.1f}, {toe_L[2]:.1f}]")
    print(f"  heel_L  = [{heel_L[0]:.1f}, {heel_L[1]:.1f}, {heel_L[2]:.1f}]")
    print(f"  shank_xz = {np.round(shank_xz, 1)}")
    print(f"  foot_xz  = {np.round(foot_xz, 1)}")
    print(f"  shank_angle_from_vertical = {shank_horiz_angle:.1f} deg")
    print(f"  foot_angle_from_horizontal = {foot_horiz_angle:.1f} deg")
    print(f"  ankle_angle (arccos)       = {angle_arccos:.1f} deg")
    print(f"  ankle_angle (foot-shank)   = {np.degrees(np.arctan2(foot_xz[0], foot_xz[1]) - np.arctan2(shank_xz[0], shank_xz[1])):.1f} deg")
