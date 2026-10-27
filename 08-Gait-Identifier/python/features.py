"""Turns one window of IMU samples into the numbers the two models use.

A "window" is a list of (ax, ay, az, gx, gy, gz) tuples, already converted to
physical units (g's for accel, deg/s for gyro) by main.py.
"""
import numpy as np

# Feature names Model 2 uses, in a fixed order - this list IS the model's
# input vector definition, so keep it in sync with extract_features() below.
FEATURE_KEYS = [
    "accel_mag_mean",
    "accel_mag_std",
    "accel_mag_ptp",
    "accel_var_x",
    "accel_var_y",
    "accel_var_z",
    "zero_crossing_rate",
    "gyro_mag_mean",
    "gyro_mag_std",
]


def _as_array(window):
    return np.array(window, dtype=float)  # shape (n_samples, 6)


def single_axis_peaks(window):
    """Peak-to-peak (max - min) of each accelerometer axis over the window.

    This is ALL Model 1 is allowed to see: one of these three numbers.
    """
    arr = _as_array(window)
    return {
        "peak_ax": float(np.ptp(arr[:, 0])),
        "peak_ay": float(np.ptp(arr[:, 1])),
        "peak_az": float(np.ptp(arr[:, 2])),
    }


def extract_features(window):
    """Features computed from all 6 IMU channels, for Model 2."""
    arr = _as_array(window)
    accel = arr[:, 0:3]
    gyro = arr[:, 3:6]

    accel_mag = np.sqrt((accel ** 2).sum(axis=1))
    gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))

    # The axis with the most movement is the best one to count "steps" on.
    dominant_axis = int(np.argmax(np.ptp(accel, axis=0)))
    dominant_signal = accel[:, dominant_axis] - accel[:, dominant_axis].mean()
    signs = dominant_signal >= 0
    zero_crossings = np.count_nonzero(signs[1:] != signs[:-1])
    zero_crossing_rate = zero_crossings / len(dominant_signal)

    return {
        "accel_mag_mean": float(accel_mag.mean()),
        "accel_mag_std": float(accel_mag.std()),
        "accel_mag_ptp": float(np.ptp(accel_mag)),
        "accel_var_x": float(accel[:, 0].var()),
        "accel_var_y": float(accel[:, 1].var()),
        "accel_var_z": float(accel[:, 2].var()),
        "zero_crossing_rate": float(zero_crossing_rate),
        "gyro_mag_mean": float(gyro_mag.mean()),
        "gyro_mag_std": float(gyro_mag.std()),
    }
