import ezc3d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
C3D_DIR = DATA_BASE + r"\Kinematic_data\Kinematic_data\Raw_c3d_files"

cases = [
    ("sub24", "s24_CMJ_t1.c3d"),
    ("sub24", "s24_CMJ_t2.c3d"),
    ("sub24", "s24_CMJ_t3.c3d"),
    ("sub07", "s07_CMJ_t3.c3d"),
]

FZ1, FZ2 = 2, 8
GRF_FREQ = 1000.0
VIDEO_FREQ = 250.0
RATIO = int(GRF_FREQ / VIDEO_FREQ)

fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

for case_idx, (sub_id, fname) in enumerate(cases):
    fpath = C3D_DIR + f"\\{sub_id}\\{fname}"
    c3d = ezc3d.c3d(fpath)
    markers = c3d['data']['points']
    analogs = c3d['data']['analogs']
    labels = list(c3d['parameters']['POINT']['LABELS']['value'])
    idx = {n: i for i, n in enumerate(labels)}

    n_mf = markers.shape[2]
    t_video = np.arange(n_mf) / VIDEO_FREQ

    vgrf = np.abs(analogs[0, FZ1, :] + analogs[0, FZ2, :])
    t_analog = np.arange(len(vgrf)) / GRF_FREQ
    peak_a = int(np.argmax(vgrf))
    ic_a = peak_a
    for a in range(peak_a, 10, -1):
        if vgrf[a] < 20 and vgrf[a-1] < 20:
            ic_a = a + 1
            break
    ic_v = min(ic_a // RATIO, n_mf - 1)
    ic_t = ic_v / VIDEO_FREQ

    ax_idx_base = case_idx * 4

    ax = axes[ax_idx_base + 0]
    knee_L = (markers[:3, idx["LKNE"], :] + markers[:3, idx["LKNEM"], :]) / 2
    ankle_L = (markers[:3, idx["LANK"], :] + markers[:3, idx["LANKM"], :]) / 2
    hip_L = markers[:3, idx["LGT"], :]
    toe_L = markers[:3, idx["LTOE"], :]
    heel_L = markers[:3, idx["LHEE"], :]

    hip_z = hip_L[2, :]
    knee_z = knee_L[2, :]
    ankle_z = ankle_L[2, :]
    toe_z = toe_L[2, :]
    heel_z = heel_L[2, :]

    ax.plot(t_video, hip_z, label='Hip Z', lw=1.2)
    ax.plot(t_video, knee_z, label='Knee Z', lw=1.2)
    ax.plot(t_video, ankle_z, label='Ankle Z', lw=1.2)
    ax.plot(t_video, toe_z, label='Toe Z', lw=1.2)
    ax.plot(t_video, heel_z, label='Heel Z', lw=1.2)
    ax.axvline(ic_t, color='red', ls='--', alpha=0.7, label='IC')
    ax.set_ylabel('Z (mm)')
    ax.set_title(f'{sub_id} {fname} 鈥?Segment Z')
    ax.legend(fontsize=6)

    c7 = markers[:3, idx["C7"], :]
    t10 = markers[:3, idx["T10"], :]
    lasi = markers[:3, idx["LASI"], :]
    rasi = markers[:3, idx["RASI"], :]

    trunk = (c7 + t10) / 2
    pelvis = (lasi + rasi) / 2
    trunk_vec = trunk - pelvis
    hip_flex_time = np.degrees(np.arctan2(np.abs(trunk_vec[0, :]), np.abs(trunk_vec[2, :])))

    shank_L = knee_L - ankle_L
    foot_L = toe_L - heel_L
    shank_xz = np.sqrt(shank_L[0, :]**2 + shank_L[2, :]**2)
    foot_xz = np.sqrt(foot_L[0, :]**2 + foot_L[2, :]**2)
    cos_ankle = np.sum(shank_L[[0, 2], :] * foot_L[[0, 2], :], axis=0) / (shank_xz * foot_xz + 1e-9)
    cos_ankle = np.clip(cos_ankle, -1, 1)
    ankle_angle_time = np.degrees(np.arccos(cos_ankle))

    ax = axes[ax_idx_base + 1]
    ax.plot(t_video, hip_flex_time, label='Trunk lean', lw=1.2)
    ax.plot(t_video, ankle_angle_time, label='Ankle angle', lw=1.2)
    ax.axvline(ic_t, color='red', ls='--', alpha=0.7, label='IC')
    ax.set_ylabel('Angle (deg)')
    ax.set_title(f'{sub_id} {fname} 鈥?Angles')
    ax.legend(fontsize=6)

    ax = axes[ax_idx_base + 2]
    ax.plot(t_analog, vgrf, lw=1.2)
    ax.axvline(ic_a / GRF_FREQ, color='red', ls='--', alpha=0.7, label='IC')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|vGRF| (N)')
    ax.set_title(f'{sub_id} {fname} 鈥?Total vGRF')
    ax.legend(fontsize=6)

    win_start = max(0, ic_a - 200)
    win_end = min(len(vgrf), ic_a + 1500)
    ax = axes[ax_idx_base + 3]
    ax.plot(t_analog[win_start:win_end], vgrf[win_start:win_end], lw=1.2)
    ax.axvline(ic_a / GRF_FREQ, color='red', ls='--', alpha=0.7, label='IC')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|vGRF| (N)')
    ax.set_title(f'{sub_id} {fname} 鈥?vGRF zoom')
    ax.legend(fontsize=6)

    print(f"{sub_id}/{fname}: ic_v={ic_v}, ic_t={ic_t:.3f}s")
    print(f"  trunk_lean@IC = {hip_flex_time[ic_v]:.1f} deg")
    print(f"  ankle_angle@IC = {ankle_angle_time[ic_v]:.1f} deg")
    print(f"  peak_vGRF = {vgrf[peak_a]:.0f} N at {peak_a/GRF_FREQ:.3f}s")

plt.tight_layout()
out_path = os.path.join(PROJECT_ROOT, "figures", "sub24_diagnostic.png")
plt.savefig(out_path, dpi=200)
print(f"\nSaved to {out_path}")
