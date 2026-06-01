import ezc3d
import numpy as np
import pandas as pd
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
C3D_DIR = os.path.join(DATA_BASE, "Kinematic_data", "Kinematic_data", "Raw_c3d_files")

CONTROL_IDS = ["sub01", "sub02", "sub03", "sub04", "sub06", "sub07", "sub08",
               "sub09", "sub10", "sub14", "sub15", "sub16", "sub17", "sub19",
               "sub21", "sub22", "sub23", "sub24", "sub26", "sub28", "sub30", "sub34"]

FZ1_CH, FZ2_CH = 2, 8
GRF_FREQ, VIDEO_FREQ = 1000.0, 250.0
RATIO = int(GRF_FREQ / VIDEO_FREQ)


def ic_by_grf(analogs, threshold=20.0):
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
    ic_v = min(int(ic_a // RATIO), 999999)
    return ic_a, ic_v


def ic_by_ankle_velocity(markers, idx):
    ank_L = (markers[:3, idx["LANK"], :] + markers[:3, idx["LANKM"], :]) / 2
    ank_R = (markers[:3, idx["RANK"], :] + markers[:3, idx["RANKM"], :]) / 2
    ank_z = (ank_L[2, :] + ank_R[2, :]) / 2
    vel_z = np.gradient(ank_z, 1.0 / VIDEO_FREQ)
    for f in range(1, len(vel_z)):
        if vel_z[f] > 0 and vel_z[f - 1] <= 0:
            return f
    return None


records = []
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
            try:
                c3d = ezc3d.c3d(fp_path)
            except Exception:
                continue
            markers = c3d['data']['points']
            analogs = c3d['data']['analogs']
            labels = list(c3d['parameters']['POINT']['LABELS']['value'])
            idx_map = {n: i for i, n in enumerate(labels)}
            required = ["LANK", "LANKM", "RANK", "RANKM"]
            if any(m not in idx_map for m in required):
                continue

            grf_ic_analog, grf_ic_video = ic_by_grf(analogs, 20.0)
            kin_ic_video = ic_by_ankle_velocity(markers, idx_map)

            if kin_ic_video is None:
                continue

            delta_frames = kin_ic_video - grf_ic_video
            delta_ms = delta_frames / VIDEO_FREQ * 1000.0
            records.append({
                "subject": sub_id,
                "trial_type": tt,
                "trial_num": tn,
                "grf_ic_video": grf_ic_video,
                "kin_ic_video": kin_ic_video,
                "delta_frames": delta_frames,
                "delta_ms": delta_ms,
            })

df = pd.DataFrame(records)
n = len(df)
mean_delta_ms = df["delta_ms"].mean()
sd_delta_ms = df["delta_ms"].std()
median_delta_ms = df["delta_ms"].median()
abs_delta_ms = df["delta_ms"].abs()
mean_abs_ms = abs_delta_ms.mean()
sd_abs_ms = abs_delta_ms.std()

print("=" * 65)
print("IC FRAME OFFSET: GRF-based vs Kinematic (ankle velocity)")
print("=" * 65)
print(f"\n  Valid trials: {n}")
print(f"  Delta (kinematic - GRF) in frames (250 Hz):")
print(f"    Mean = {df['delta_frames'].mean():.2f},  SD = {df['delta_frames'].std():.2f}")
print(f"    Median = {df['delta_frames'].median():.1f}")
print(f"  Delta in milliseconds:")
print(f"    Mean = {df['delta_ms'].mean():.2f},  SD = {df['delta_ms'].std():.2f}")
print(f"    Median = {df['delta_ms'].median():.1f}")
print(f"  Absolute delta (ms):")
print(f"    Mean = {mean_abs_ms:.2f},  SD = {sd_abs_ms:.2f}")
print(f"    Median = {abs_delta_ms.median():.1f}")
print(f"  Range: [{df['delta_ms'].min():.1f}, {df['delta_ms'].max():.1f}] ms")

bins = [-np.inf, -16, -12, -8, -4, 0, 4, 8, 12, 16, np.inf]
labels = ["< -16", "-16..-12", "-12..-8", "-8..-4", "-4..0", "0..4", "4..8", "8..12", "12..16", "> 16"]
df["delta_bin"] = pd.cut(df["delta_ms"], bins=bins, labels=labels)
print(f"\n  Distribution of delta (ms):")
print(df["delta_bin"].value_counts().sort_index().to_string())

print(f"\n  By task type:")
for tt_label in ["CMJ", "DJ"]:
    sub_df = df[df["trial_type"] == tt_label]
    print(f"    {tt_label}: n={len(sub_df)}, mean|delta| = {sub_df['delta_ms'].abs().mean():.2f} ms, "
          f"mean delta = {sub_df['delta_ms'].mean():.2f} ms")

pct_lt_4ms = (abs_delta_ms < 4.0).mean() * 100
pct_lt_8ms = (abs_delta_ms < 8.0).mean() * 100
pct_lt_12ms = (abs_delta_ms < 12.0).mean() * 100
pct_gt_16ms = (abs_delta_ms >= 16.0).mean() * 100
print(f"\n  % trials with |delta| < 4 ms:   {pct_lt_4ms:.1f}%")
print(f"  % trials with |delta| < 8 ms:   {pct_lt_8ms:.1f}%")
print(f"  % trials with |delta| < 12 ms:  {pct_lt_12ms:.1f}%")
print(f"  % trials with |delta| >= 16 ms: {pct_gt_16ms:.1f}%")

print(f"\n  INTERPRETATION:")
print(f"    Mean absolute IC offset = {mean_abs_ms:.1f} ms ({mean_abs_ms/4:.1f} frames @ 250Hz)")
print(f"    In the ~50ms impact absorption phase of landing, this represents")
print(f"    {mean_abs_ms/50*100:.0f}% of the total absorption window.")
print(f"    A {mean_abs_ms:.0f}ms shift in declared IC can alter extracted joint")
print(f"    angles by several degrees at typical landing angular velocities.")
