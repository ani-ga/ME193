"""Everything the board needs to count steps, using numpy only.

The net is trained with PyTorch on the laptop (train_steps.py), then exported
to step_net.npz. This file loads that file and runs the same forward pass in
plain numpy, so the Uno Q never needs PyTorch. It also holds the signal
preprocessing and peak picking shared by training, evaluation and the live app.
"""
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FS = 50                  # Hz: every recording is resampled to this rate
CHANNELS = 8             # ax ay az gx gy gz, |accel|, |gyro|
MIN_GAP_S = 0.35         # two steps are never closer than this (95% of real steps are >= 0.45 s apart)
THRESHOLD = 0.6          # step-probability needed to count as a step
MODEL_PATH = Path(__file__).resolve().parent / "step_net.npz"


def resample(t, raw, fs=FS):
    """Samples arrive at uneven times; interpolate onto an even fs grid.

    t: (N,) seconds, raw: (N, 6). Returns (t_grid, (M, 6)).
    """
    t = np.asarray(t, dtype=float)
    raw = np.asarray(raw, dtype=float)
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    out = np.stack([np.interp(grid, t, raw[:, i]) for i in range(raw.shape[1])], axis=1)
    return grid, out


def make_channels(raw):
    """(N, 6) -> (N, 8). The two magnitudes don't depend on how the sensor is
    rotated on the body, which helps when it is strapped on a little differently."""
    accel_mag = np.sqrt((raw[:, 0:3] ** 2).sum(axis=1, keepdims=True))
    gyro_mag = np.sqrt((raw[:, 3:6] ** 2).sum(axis=1, keepdims=True))
    return np.hstack([raw, accel_mag, gyro_mag])


def find_peaks(p, threshold=THRESHOLD, min_gap=int(MIN_GAP_S * FS)):
    """Indices of local maxima of p above threshold, strongest first wins
    when two are closer than min_gap samples. Returned in time order."""
    p = np.asarray(p)
    if len(p) < 3:
        return []
    is_max = (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > threshold)
    cand = np.nonzero(is_max)[0] + 1
    chosen = []
    for i in cand[np.argsort(-p[cand])]:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(int(i))
    return sorted(chosen)


def baseline_peaks(raw):
    """The no-neural-net comparison: detrend |accel|, smooth it, pick peaks
    above a fixed 0.15 g. raw is (N, 6) already resampled to FS."""
    mag = np.sqrt((raw[:, 0:3] ** 2).sum(axis=1))
    mag = mag - np.convolve(mag, np.ones(FS) / FS, mode="same")
    mag = np.convolve(mag, np.ones(5) / 5, mode="same")
    return find_peaks(mag, threshold=0.15, min_gap=int(0.3 * FS))


def _conv_same(x, w, b, dilation):
    """1D convolution with zero 'same' padding. x (C_in, T), w (C_out, C_in, K)."""
    k = w.shape[2]
    span = (k - 1) * dilation + 1
    pad = span // 2
    xp = np.pad(x, ((0, 0), (pad, span - 1 - pad)))
    win = sliding_window_view(xp, span, axis=1)[:, :, ::dilation]  # (C_in, T, K)
    return np.einsum("oik,itk->ot", w, win) + b[:, None]


class NumpyStepNet:
    """The trained StepNet's forward pass (BatchNorm already folded into the
    conv weights at export time). Input (N, 6) raw samples at FS, output
    (N,) step probabilities."""

    def __init__(self, path=MODEL_PATH):
        d = np.load(path)
        self.ch_mean, self.ch_std = d["ch_mean"], d["ch_std"]
        self.dilations = [int(v) for v in d["dilations"]]
        self.layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(len(self.dilations) + 1)]

    def predict(self, raw):
        x = ((make_channels(raw) - self.ch_mean) / self.ch_std).T  # (C, T)
        for (w, b), dil in zip(self.layers[:-1], self.dilations):
            x = np.maximum(_conv_same(x, w, b, dil), 0.0)
        w, b = self.layers[-1]
        logits = _conv_same(x, w, b, 1)[0]
        return 1.0 / (1.0 + np.exp(-logits))
