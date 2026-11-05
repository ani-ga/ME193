"""Records labeled IMU trials (with foot-strike taps) to CSV for training.

One trial = one CSV in data/steps/. The first line is a comment holding the
activity and, optionally, the true step count you counted yourself:
    # label=walking known_count=20
Then one row per IMU sample: t, ax, ay, az, gx, gy, gz, tap
where tap is how many times the Tap button was hit since the previous sample.
"""
import threading
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "steps"
HEADER = "t,ax,ay,az,gx,gy,gz,tap\n"


class StepRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._file = None
        self.label = None
        self.rows = 0
        self.taps = 0
        self._pending_taps = 0

    @property
    def is_recording(self):
        return self._file is not None

    def start(self, label, known_count=None):
        with self._lock:
            if self._file is not None:
                return False
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            name = f"trial_{time.strftime('%Y%m%d_%H%M%S')}_{label}.csv"
            self._file = open(DATA_DIR / name, "w")
            meta = f"# label={label}" + (f" known_count={known_count}" if known_count is not None else "")
            self._file.write(meta + "\n" + HEADER)
            self.label, self.rows, self.taps, self._pending_taps = label, 0, 0, 0
            return True

    def tap(self):
        with self._lock:
            if self._file is None:
                return False
            self._pending_taps += 1
            self.taps += 1
            return True

    def on_sample(self, t, sample):
        """Called by the sampler for every IMU reading."""
        with self._lock:
            if self._file is None:
                return
            vals = ",".join(f"{v:.5f}" for v in sample)
            self._file.write(f"{t:.4f},{vals},{self._pending_taps}\n")
            self._pending_taps = 0
            self.rows += 1

    def stop(self):
        with self._lock:
            if self._file is None:
                return None
            self._file.close()
            self._file = None
            summary = {"label": self.label, "rows": self.rows, "taps": self.taps}
            self.label = None
            return summary

    @staticmethod
    def trial_files():
        return sorted(DATA_DIR.glob("trial_*.csv")) if DATA_DIR.exists() else []
