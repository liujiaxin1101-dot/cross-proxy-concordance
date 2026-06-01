import ezc3d
import numpy as np
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_BASE = os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data"))
C3D_DIR = os.path.join(DATA_BASE, "Kinematic_data", "Kinematic_data", "Raw_c3d_files")

sub_dirs = sorted([d for d in os.listdir(C3D_DIR) if d.startswith("sub")])
print(f"Found {len(sub_dirs)} subject folders: {sub_dirs}")
print()

trial_path = os.path.join(C3D_DIR, "sub01", "s01_CMJ_t1.c3d")
print(f"Reading: {trial_path}")
print()

c3d = ezc3d.c3d(trial_path)

print("=== HEADER ===")
print(f"Points (markers):    {c3d['header']['points']['size']}")
print(f"Frame rate:           {c3d['header']['points']['frame_rate']} Hz")
print(f"First frame:          {c3d['header']['points']['first_frame']}")
print(f"Last frame:           {c3d['header']['points']['last_frame']}")
print()

print("=== ANALOG (force plates) ===")
print(f"Analog channels:      {c3d['header']['analogs']['size']}")
print(f"Analog frame rate:    {c3d['header']['analogs']['frame_rate']} Hz")
print(f"Analog first frame:   {c3d['header']['analogs']['first_frame']}")
print(f"Analog last frame:    {c3d['header']['analogs']['last_frame']}")
print()

points = c3d['data']['points']
print(f"=== POINTS DATA ===")
print(f"Points shape: {points.shape}  (4, n_markers, n_frames)")
n_markers = points.shape[1]
n_frames = points.shape[2]
print(f"Number of markers: {n_markers}")
print(f"Number of frames:  {n_frames}")
print()

print("=== MARKER NAMES ===")
marker_names = list(c3d['parameters']['POINT']['LABELS']['value'])
for i, name in enumerate(marker_names):
    print(f"  [{i:3d}] {name}")
print()

analogs = c3d['data']['analogs']
print(f"=== ANALOG DATA ===")
print(f"Analogs shape: {analogs.shape}  (n_subframes, n_channels, n_frames)")
n_analog_subframes = analogs.shape[0]
n_analog_channels = analogs.shape[1]
n_analog_frames = analogs.shape[2]
print(f"Analog subframes: {n_analog_subframes}")
print(f"Analog channels:  {n_analog_channels}")
print(f"Analog frames:    {n_analog_frames}")
print()

print("=== ANALOG LABELS ===")
if 'ANALOG' in c3d['parameters'] and 'LABELS' in c3d['parameters']['ANALOG']:
    analog_labels = list(c3d['parameters']['ANALOG']['LABELS']['value'])
    for i, label in enumerate(analog_labels):
        print(f"  [{i:3d}] {label}")
else:
    print("  (no analog labels found)")
print()

print("=== FORCE PLATFORM INFO ===")
if 'FORCE_PLATFORM' in c3d['parameters']:
    fp = c3d['parameters']['FORCE_PLATFORM']
    print(f"  Number of plates: {fp['USED']['value'][0]}")
    for i in range(fp['USED']['value'][0]):
        ch_start = fp['CHANNEL']['value'][i, 0]
        ch_count = fp['CHANNEL']['value'][i, 1] if fp['CHANNEL']['value'].shape[1] > 1 else 6
        print(f"  Plate {i+1}: channels {int(ch_start)}-{int(ch_start+ch_count-1)}, type={fp['TYPE']['value'][i]}")
else:
    print("  (no FORCE_PLATFORM parameter)")

print()
print("=== ANALOG CHANNEL STATS ===")
for i in range(n_analog_channels):
    ch_data = analogs[0, i, :]
    print(f"  [{i:2d}] {analog_labels[i]:16s}: min={ch_data.min():+10.3f}, max={ch_data.max():+10.3f}, mean={ch_data.mean():+10.3f}")

print()
print("=== TIMING ===")
print(f"Video duration:  {n_frames / c3d['header']['points']['frame_rate']:.2f} s")
print(f"Analog duration: {n_analog_frames / c3d['header']['analogs']['frame_rate']:.2f} s")
print(f"Ratio (analog/video): {n_analog_frames / max(n_frames, 1):.1f}x")
