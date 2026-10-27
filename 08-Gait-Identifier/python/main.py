"""Linux-side entry point for the Gait Identifier app.

Talks to the MCU over the Bridge to read the MPU6050 (see ../sketch/sketch.ino),
keeps a rolling window of samples, runs both classifiers, and serves a phone
web page for calibration + live predictions.
"""
import time
import threading
from collections import deque

from arduino.app_utils import App, Logger, brick, call
from arduino.app_bricks.web_ui import WebUI

from features import extract_features, single_axis_peaks
from models import Model1Threshold, Model2KNN
from calibration import CalibrationManager

logger = Logger("GaitIdentifier")

SAMPLE_HZ = 50
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLE_HZ * WINDOW_SECONDS)
HISTORY_SECONDS = 10
# "No motion" shortcut: if the last STILL_SECONDS of data barely vary, report
# standing right away instead of waiting for the 2 s window to clear.
STILL_SECONDS = 0.4
STILL_ACCEL_RANGE = 0.08  # g, max-min of accel magnitude
STILL_GYRO_MAX = 15.0     # deg/s, peak rotation rate
GRAPH_STRIDE = 5  # send every 5th sample (10 points/s) to the page

# MPU6050 default full-scale ranges: +-2g for accel, +-250 deg/s for gyro.
ACCEL_SCALE = 1.0 / 16384.0  # g per raw LSB
GYRO_SCALE = 1.0 / 131.0     # deg/s per raw LSB

LABELS = ["standing", "walking", "running", "biking", "hopping"]


# Each of these calls the matching function the sketch registered with
# Bridge.provide() (see sketch/sketch.ino) and blocks for its return value.
@call()
def read_ax() -> int: ...
@call()
def read_ay() -> int: ...
@call()
def read_az() -> int: ...
@call()
def read_gx() -> int: ...
@call()
def read_gy() -> int: ...
@call()
def read_gz() -> int: ...


@brick
class ImuSampler:
    """Polls the MCU at SAMPLE_HZ and keeps the most recent WINDOW_SIZE
    samples, converted to physical units."""

    def __init__(self, window_size):
        self.buffer = deque(maxlen=window_size)
        # Longer history (10 s) for the web page's graphs.
        self.history = deque(maxlen=int(SAMPLE_HZ * HISTORY_SECONDS))
        self._lock = threading.Lock()
        self.connected = True

    def loop(self):
        try:
            sample = (
                read_ax() * ACCEL_SCALE, read_ay() * ACCEL_SCALE, read_az() * ACCEL_SCALE,
                read_gx() * GYRO_SCALE, read_gy() * GYRO_SCALE, read_gz() * GYRO_SCALE,
            )
            self.connected = True
        except Exception as e:
            if self.connected:
                logger.error(f"Lost contact with the sketch over the Bridge: {e}")
            self.connected = False
            time.sleep(0.5)
            return
        with self._lock:
            self.buffer.append(sample)
            self.history.append(sample)
        time.sleep(1.0 / SAMPLE_HZ)

    def recent(self):
        with self._lock:
            return list(self.history)

    def window(self):
        with self._lock:
            return list(self.buffer)


sampler = ImuSampler(WINDOW_SIZE)
model1 = Model1Threshold()
model2 = Model2KNN(k=5)
calib = CalibrationManager(sampler, model1, model2, WINDOW_SIZE)

state = {
    "prediction1": "collecting data...",
    "prediction2": "collecting data...",
    "loo_accuracy": None,
}


def is_still(samples):
    """True when the most recent STILL_SECONDS show basically no motion."""
    n = int(SAMPLE_HZ * STILL_SECONDS)
    if len(samples) < n:
        return False
    recent = samples[-n:]
    mags = [(ax * ax + ay * ay + az * az) ** 0.5 for ax, ay, az, _, _, _ in recent]
    spin = max((gx * gx + gy * gy + gz * gz) ** 0.5 for _, _, _, gx, gy, gz in recent)
    return max(mags) - min(mags) < STILL_ACCEL_RANGE and spin < STILL_GYRO_MAX


@brick
class LiveClassifier:
    """Checks for stillness 10x a second (instant "standing"), and twice a
    second classifies the full window with both models - unless we're
    recording calibration data, in which case stay out of the way so the
    recording loop gets the CPU."""

    def __init__(self):
        self._last_full = 0.0

    def loop(self):
        time.sleep(0.1)
        if calib.is_recording:
            return
        window = sampler.window()
        if not sampler.connected:
            state["prediction1"] = state["prediction2"] = "sketch not responding"
            return
        if is_still(window):
            state["prediction1"] = state["prediction2"] = "standing"
            return
        if time.time() - self._last_full < 0.5:
            return
        self._last_full = time.time()
        if len(window) < WINDOW_SIZE:
            state["prediction1"] = state["prediction2"] = "collecting data..."
            return
        peaks = single_axis_peaks(window)
        feats = extract_features(window)
        state["prediction1"] = model1.predict(peaks)
        state["prediction2"] = model2.predict(feats)


live_classifier = LiveClassifier()

# The app's files land at /app/<app-name's local layout>, so our python/
# folder is /app/python, not /app - WebUI's /app/assets default doesn't
# match our layout, so point it at the folder's actual location.
web_ui = WebUI(assets_dir_path="/app/python/assets")


def get_status():
    return {
        "connected": sampler.connected,
        "prediction1": state["prediction1"],
        "prediction2": state["prediction2"],
        "recording": calib.is_recording,
        "recording_label": calib.recording_label,
        "rows_recorded_this_session": calib.rows_recorded_this_session,
        "labels": LABELS,
        "counts_by_label": calib.counts_by_label(),
        "loo_accuracy": state["loo_accuracy"],
    }


def get_telemetry():
    """Last ~10 s of motion for the graphs. The IMU can't measure true speed
    (integrating acceleration drifts within seconds), so 'intensity' is how
    hard the sensor is accelerating beyond gravity, in g."""
    samples = sampler.recent()[::GRAPH_STRIDE]
    intensity, rotation = [], []
    for ax, ay, az, gx, gy, gz in samples:
        intensity.append(round(abs((ax * ax + ay * ay + az * az) ** 0.5 - 1.0), 3))
        rotation.append(round((gx * gx + gy * gy + gz * gz) ** 0.5, 1))
    return {"hz": SAMPLE_HZ / GRAPH_STRIDE, "intensity": intensity, "rotation": rotation}


def start_recording(label: str):
    if label not in LABELS:
        return {"ok": False, "error": f"unknown label '{label}'"}
    started = calib.start_recording(label)
    return {"ok": started}


def stop_recording():
    calib.stop_recording()
    return {"ok": True}


def train():
    n_rows = calib.train()
    state["loo_accuracy"] = model2.leave_one_out_accuracy()
    return {"ok": True, "rows": n_rows, "loo_accuracy": state["loo_accuracy"]}


web_ui.expose_api("GET", "/status", get_status)
web_ui.expose_api("GET", "/telemetry", get_telemetry)
web_ui.expose_api("POST", "/start_recording", start_recording)
web_ui.expose_api("POST", "/stop_recording", stop_recording)
web_ui.expose_api("POST", "/train", train)

# WebUI is itself a brick (like ImuSampler and LiveClassifier above), so
# App.run() starts its web server automatically - no manual web_ui.start().
logger.info(f"Web UI will be available at {web_ui.url}")

App.run()
