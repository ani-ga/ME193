"""Loads recorded trials (see recorder.py) and builds training targets."""
from pathlib import Path

import numpy as np

from step_infer import FS, resample

TAP_LAG_S = 0.15     # people tap a little after the foot lands; shift taps earlier by this
TARGET_SIGMA = 3.0   # samples (0.06 s): width of the bump drawn at each step


class Trial:
    def __init__(self, path, label, known_count, raw, tap_times, t0):
        self.path = Path(path)
        self.label = label
        self.raw = raw                  # (N, 6) at FS Hz
        self.tap_times = tap_times      # seconds from the start of raw
        self.known_count = known_count  # your hand count, or None
        self.t0 = t0

    @property
    def name(self):
        return self.path.name

    @property
    def true_count(self):
        """Hand count if you gave one, else the number of taps."""
        return self.known_count if self.known_count is not None else len(self.tap_times)

    def target(self, lag_s=TAP_LAG_S):
        """Per-sample step curve: a Gaussian bump of height 1 at each foot strike."""
        n = len(self.raw)
        x = np.arange(n)
        y = np.zeros(n)
        for tt in self.tap_times:
            y = np.maximum(y, np.exp(-0.5 * ((x - (tt - lag_s) * FS) / TARGET_SIGMA) ** 2))
        return y


def load_trial(path):
    label, known = "unknown", None
    with open(path) as f:
        first = f.readline()
    if first.startswith("#"):
        for part in first[1:].split():
            k, _, v = part.partition("=")
            if k == "label":
                label = v
            elif k == "known_count":
                known = int(v)
    data = np.genfromtxt(path, delimiter=",", skip_header=2)
    data = data[~np.isnan(data).any(axis=1)]  # a power loss can truncate the last row
    t, raw, taps = data[:, 0], data[:, 1:7], data[:, 7]
    grid, raw50 = resample(t, raw)
    tap_idx = np.nonzero(taps)[0]
    tap_times = []
    for i in tap_idx:
        tap_times += [t[i] - grid[0]] * int(taps[i])
    return Trial(path, label, known, raw50, np.array(tap_times), grid[0])


def load_trials(data_dir):
    files = sorted(Path(data_dir).glob("trial_*.csv"))
    return [load_trial(f) for f in files]
