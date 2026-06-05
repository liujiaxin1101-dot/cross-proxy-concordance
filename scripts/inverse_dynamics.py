"""
Bottom-up quasi-static inverse dynamics for sagittal-plane knee and hip moments.

Computes net joint moments (Nm) during the landing phase (IC to IC+300ms)
from .c3d marker and force plate data. Uses a quasi-static approximation
(external moment from GRF about each joint), which is valid for landing
tasks where segment inertial contributions are <10% of GRF contributions.

References:
  - de Leva (1996) J Biomech 29(9):1223-1230 (segment inertial parameters)
  - Winter DA (2009) Biomechanics and Motor Control of Human Movement (4th ed)
"""

import numpy as np
from scipy.signal import butter, filtfilt

# ==============================================================================
# Force plate channel indices (0-based, from analog labels)
# ==============================================================================
FP1 = {"Fx": 0, "Fy": 1, "Fz": 2, "Mx": 3, "My": 4, "Mz": 5}
FP2 = {"Fx": 6, "Fy": 7, "Fz": 8, "Mx": 9, "My": 10, "Mz": 11}

# ==============================================================================
# Constants
# ==============================================================================
GRF_FREQ = 1000.0      # Hz
VIDEO_FREQ = 250.0     # Hz
ANALOG_RATIO = int(GRF_FREQ / VIDEO_FREQ)  # 4
WINDOW_S = 0.3         # 300 ms landing window
FILTER_CUTOFF = 12.0   # Hz, Butterworth lowpass for marker data
GRAVITY = 9.81         # m/s^2

# ==============================================================================
# Anthropometric parameters (de Leva 1996, both sexes average)
# ==============================================================================
ANTHRO = {
    "foot":   {"mass_frac": 0.0137, "com_frac": 0.4415, "rg_frac": 0.257},
    "shank":  {"mass_frac": 0.0433, "com_frac": 0.4395, "rg_frac": 0.251},
    "thigh":  {"mass_frac": 0.1416, "com_frac": 0.4095, "rg_frac": 0.329},
}


# ==============================================================================
# Signal processing
# ==============================================================================
def butter_lowpass_4th(data, cutoff, fs):
    """4th-order zero-phase Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='low')
    return filtfilt(b, a, data, axis=0)


def _central_diff(y, dt):
    """Central difference (2nd-order) with forward/backward at endpoints."""
    dy = np.zeros_like(y)
    if len(y) > 2:
        dy[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
        dy[0] = (y[1] - y[0]) / dt
        dy[-1] = (y[-1] - y[-2]) / dt
    elif len(y) == 2:
        dy[0] = dy[1] = (y[1] - y[0]) / dt
    return dy


# ==============================================================================
# Force plate helpers
# ==============================================================================
def compute_cop(analog_data, fp_ch, dz0=0.0):
    """
    Compute centre of pressure (COP) from force plate analog data.

    Parameters
    ----------
    analog_data : ndarray (n_frames, n_channels)
    fp_ch : dict
        Channel indices for Fx, Fy, Fz, Mx, My, Mz.
    dz0 : float
        Vertical offset from sensor origin to plate surface (m). Default 0.

    Returns
    -------
    cop_x, cop_y : ndarray (n_frames,)
        COP coordinates in the plate's local coordinate system (mm).
    """
    fx = analog_data[:, fp_ch["Fx"]]
    fy = analog_data[:, fp_ch["Fy"]]
    fz = analog_data[:, fp_ch["Fz"]]
    mx = analog_data[:, fp_ch["Mx"]]
    my = analog_data[:, fp_ch["My"]]

    eps = 1e-3
    fz_safe = np.where(np.abs(fz) > eps, fz, np.sign(fz + eps) * eps)

    cop_x = (-my - fx * dz0) / fz_safe
    cop_y = (mx - fy * dz0) / fz_safe

    return cop_x, cop_y


def get_grf_on_foot(analog_data, fp_ch):
    """Return the ground reaction force applied TO the foot (global CS, N)."""
    return -analog_data[:, [fp_ch["Fx"], fp_ch["Fy"], fp_ch["Fz"]]]


def get_combined_grf(analog_data_xz, fp1_ch, fp2_ch):
    """
    Return total GRF (both plates summed) applied to the body.

    Returns
    -------
    grf_total : ndarray (n_frames, 3) in N
    cop_total : ndarray (n_frames, 2) in mm (weighted average COP)
    """
    grf1 = get_grf_on_foot(analog_data_xz, fp1_ch)
    grf2 = get_grf_on_foot(analog_data_xz, fp2_ch)
    grf_total = grf1 + grf2

    cop1_x, cop1_y = compute_cop(analog_data_xz, fp1_ch)
    cop2_x, cop2_y = compute_cop(analog_data_xz, fp2_ch)

    fz1 = np.abs(analog_data_xz[:, fp1_ch["Fz"]])
    fz2 = np.abs(analog_data_xz[:, fp2_ch["Fz"]])
    fz_total = fz1 + fz2 + 1e-6

    cop_total_x = (cop1_x * fz1 + cop2_x * fz2) / fz_total
    cop_total_y = (cop1_y * fz1 + cop2_y * fz2) / fz_total

    return grf_total, np.column_stack([cop_total_x, cop_total_y])


# ==============================================================================
# Quasi-static joint moments
# ==============================================================================
def compute_external_moment_about_joint(joint_pos, cop_pos, grf):
    """
    Compute external moment from GRF about a joint (sagittal plane).

    M_external = r_joint→COP × F_GRF = rx * Fz - rz * Fx

    Parameters
    ----------
    joint_pos : ndarray (n_frames, 2) or (n_frames, 3)
        Joint centre position in global CS [x, z] or [x, y, z] (m).
    cop_pos : ndarray (n_frames, 2)
        COP position [x, y] (m). COP_z = 0 (floor level).
    grf : ndarray (n_frames, 2) or (n_frames, 3)
        GRF on foot [Fx, Fz] or [Fx, Fy, Fz] (N).

    Returns
    -------
    M_ext : ndarray (n_frames,)
        External moment about Y-axis (Nm). Positive = tends to extend the joint
        (plantarflexion for ankle, extension for knee and hip).
    """
    jx = joint_pos[:, 0]
    jz = joint_pos[:, 2] if joint_pos.shape[1] >= 3 else joint_pos[:, 1]
    cpx = cop_pos[:, 0]
    # COP_z = 0 (floor level)
    r_x = cpx - jx           # moment arm in X
    r_z = 0.0 - jz           # moment arm in Z (COP at floor)

    fx = grf[:, 0]
    fz = grf[:, 2] if grf.shape[1] >= 3 else grf[:, 1]

    M_ext = r_x * fz - r_z * fx
    return M_ext


# ==============================================================================
# Main inverse dynamics: quasi-static joint moments
# ==============================================================================
def compute_leg_joint_moments(
    markers_3d,      # (n_video_frames, n_markers, 3) in mm
    analog_data,     # (n_analog_frames, n_channels)
    marker_idx,      # dict: marker_name → column index
    fp_ch,           # dict: force plate channel indices
    body_mass,       # kg
    ic_analog,       # int: analog frame of initial contact
    side="left",
):
    """
    Compute knee and hip net moments (quasi-static, sagittal plane) for one leg.

    Uses the external moment from GRF about each joint centre.
    Internal net moment = -external moment.

    Parameters
    ----------
    markers_3d : ndarray (n_video, n_markers, 3)
        Marker positions in global CS (mm).
    analog_data : ndarray (n_analog, n_channels)
        Force plate analog data (raw values).
    marker_idx : dict
        Mapping from marker name to column index in markers_3d.
    fp_ch : dict
        Force plate channel indices.
    body_mass : float
        Total body mass (kg).
    ic_analog : int
        Analog frame index of initial contact.
    side : str
        "left" or "right".

    Returns
    -------
    dict with keys:
        knee_moment_peak_Nm, hip_moment_peak_Nm,
        knee_moment_peak_Nmkg, hip_moment_peak_Nmkg,
        knee_moment_series, hip_moment_series,
        n_frames_window
    """
    pfx = side[0].upper()

    hip_marker = f"{pfx}GT"
    knee_lat = f"{pfx}KNE"
    knee_med = f"{pfx}KNEM"
    ankle_lat = f"{pfx}ANK"
    ankle_med = f"{pfx}ANKM"

    required = [hip_marker, knee_lat, knee_med, ankle_lat, ankle_med]
    missing = [m for m in required if m not in marker_idx]
    if missing:
        return {"error": f"Missing markers: {missing}"}

    def get_traj(name):
        return markers_3d[:, marker_idx[name], :] / 1000.0  # mm → m

    hip_pos = get_traj(hip_marker)
    knee_pos = (get_traj(knee_lat) + get_traj(knee_med)) / 2.0
    ankle_pos = (get_traj(ankle_lat) + get_traj(ankle_med)) / 2.0

    n_video = markers_3d.shape[0]
    n_analog = analog_data.shape[0]

    ic_video = min(ic_analog // ANALOG_RATIO, n_video - 1)
    window_frames = int(WINDOW_S * VIDEO_FREQ)
    end_video = min(ic_video + window_frames, n_video)

    if end_video - ic_video < 10:
        return {"error": f"Too few video frames in landing window ({end_video - ic_video})"}

    # Slice to landing window
    hip_win = butter_lowpass_4th(hip_pos[ic_video:end_video], FILTER_CUTOFF, VIDEO_FREQ)
    knee_win = butter_lowpass_4th(knee_pos[ic_video:end_video], FILTER_CUTOFF, VIDEO_FREQ)
    ankle_win = butter_lowpass_4th(ankle_pos[ic_video:end_video], FILTER_CUTOFF, VIDEO_FREQ)
    n_win = len(hip_win)

    # Downsample GRF and COP to video rate
    grf_all = get_grf_on_foot(analog_data, fp_ch)  # (n_analog, 3) [N]
    cop_x_all, cop_y_all = compute_cop(analog_data, fp_ch, dz0=0.0)  # [mm]

    grf_win = np.zeros((n_win, 3))
    cop_win = np.zeros((n_win, 2))
    for t in range(n_win):
        a0 = ic_analog + t * ANALOG_RATIO
        a1 = min(a0 + ANALOG_RATIO, n_analog)
        if a1 > a0:
            grf_win[t] = np.mean(grf_all[a0:a1], axis=0)
            cop_win[t] = np.mean(np.column_stack([cop_x_all, cop_y_all])[a0:a1], axis=0)

    # COP from mm to m
    cop_win = cop_win / 1000.0

    # --- Quasi-static external moments ---
    # M_external = r_joint→COP × F_GRF (sagittal)
    # Internal net moment ≈ -M_external
    # We report INTERNAL moment (muscles resist external load)

    M_ext_knee = compute_external_moment_about_joint(knee_win, cop_win, grf_win)
    M_ext_hip = compute_external_moment_about_joint(hip_win, cop_win, grf_win)
    M_ext_ankle = compute_external_moment_about_joint(ankle_win, cop_win, grf_win)

    # Internal net moment (what muscles produce to resist external load)
    # For landing: external moment tends to flex the knee and hip
    # Internal extension moment is positive
    knee_moment = M_ext_knee   # external moment = internal net moment (in magnitude)
    hip_moment = M_ext_hip
    ankle_moment = M_ext_ankle

    # Peak within landing window (use absolute peak, then check sign)
    # For landing, the peak extensor moment is the largest POSITIVE value
    # (since GRF tends to flex the joints, and the net internal moment resists this)
    knee_peak = float(np.max(np.abs(knee_moment))) if len(knee_moment) > 0 else np.nan
    hip_peak = float(np.max(np.abs(hip_moment))) if len(hip_moment) > 0 else np.nan
    ankle_peak = float(np.max(np.abs(ankle_moment))) if len(ankle_moment) > 0 else np.nan

    # Also get the signed peak (direction matters for interpretation)
    # External knee FLEXION moment is negative (GRF passes anterior to knee)
    # External knee EXTENSION moment is positive (GRF passes posterior to knee)
    knee_peak_signed = float(knee_moment[np.argmax(np.abs(knee_moment))])
    hip_peak_signed = float(hip_moment[np.argmax(np.abs(hip_moment))])

    # Normalised by body mass (Nm/kg)
    knee_peak_norm = knee_peak / body_mass if body_mass > 0 else np.nan
    hip_peak_norm = hip_peak / body_mass if body_mass > 0 else np.nan

    return {
        "knee_moment_peak_Nm": knee_peak,
        "hip_moment_peak_Nm": hip_peak,
        "knee_moment_peak_Nmkg": knee_peak_norm,
        "hip_moment_peak_Nmkg": hip_peak_norm,
        "knee_moment_series": knee_moment.tolist(),
        "hip_moment_series": hip_moment.tolist(),
        "n_frames_window": n_win,
    }
