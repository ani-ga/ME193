"""Recording calibration data from the live IMU stream into a CSV, and
(re)training both models from it.
"""
import csv
import threading
import time
from pathlib import Path

from features import extract_features, single_axis_peaks, FEATURE_KEYS

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "calibration.csv"

PEAK_KEYS = ["peak_ax", "peak_ay", "peak_az"]
CSV_FIELDS = ["label"] + PEAK_KEYS + FEATURE_KEYS

RECORD_INTERVAL_SECONDS = 0.5  # one labeled window saved this often while recording


class CalibrationManager:
    """Owns the calibration CSV and the recording thread.

    recording_label is None when idle, or the activity name while a
    recording is in progress (one label at a time).
    """

    def __init__(self, sampler, model1, model2, window_size):
        self.sampler = sampler
        self.model1 = model1
        self.model2 = model2
        self.window_size = window_size
        self.recording_label = None
        self.rows_recorded_this_session = 0

        self._stop_event = threading.Event()
        self._thread = None

        DATA_DIR.mkdir(exist_ok=True)
        if CSV_PATH.exists():
            self.train()

    @property
    def is_recording(self):
        return self.recording_label is not None

    def start_recording(self, label):
        if self.is_recording:
            return False
        self.recording_label = label
        self.rows_recorded_this_session = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        return True

    def stop_recording(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.recording_label = None

    def _record_loop(self):
        file_is_new = not CSV_PATH.exists()
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if file_is_new:
                writer.writeheader()

            while not self._stop_event.is_set():
                window = self.sampler.window()
                if len(window) >= self.window_size:
                    row = {"label": self.recording_label}
                    row.update(single_axis_peaks(window))
                    row.update(extract_features(window))
                    writer.writerow(row)
                    f.flush()
                    self.rows_recorded_this_session += 1
                time.sleep(RECORD_INTERVAL_SECONDS)

    def load_rows(self):
        """Loads calibration rows, skipping any that are malformed.

        A row can get truncated or zero-filled if the board loses power or
        network mid-write (e.g. while recording, on battery, away from a
        desk) - that should cost at most one row, not crash the whole app
        on the next startup.
        """
        if not CSV_PATH.exists():
            return []
        rows = []
        skipped = 0
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    for key in PEAK_KEYS + FEATURE_KEYS:
                        r[key] = float(r[key])
                    rows.append(r)
                except (TypeError, ValueError):
                    skipped += 1
        if skipped:
            print(f"calibration.csv: skipped {skipped} malformed row(s)", flush=True)
        return rows

    def train(self):
        rows = self.load_rows()
        self.model1.train(rows)
        self.model2.train(rows)
        return len(rows)

    def counts_by_label(self):
        counts = {}
        for r in self.load_rows():
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        return counts
