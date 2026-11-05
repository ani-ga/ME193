"""Linux-side entry point for the Step Counter app.

Talks to the MCU over the Bridge to read the MPU6050 (see ../sketch/sketch.ino),
counts steps live with the trained neural net (step_net.npz, numpy forward
pass), and serves a phone page that shows the count and records labeled
trials (with a Tap button on every foot strike) for training.
"""
import time
import threading
from collections import deque

import numpy as np
from arduino.app_utils import App, Logger, brick, call
from arduino.app_bricks.web_ui import WebUI

from recorder import StepRecorder
from step_infer import FS, MODEL_PATH, MIN_GAP_S, NumpyStepNet, baseline_peaks, find_peaks, resample

logger = Logger("StepCounter")

SAMPLE_HZ = 50
LIVE_SECONDS = 6          # how much recent signal the net looks at each update
UPDATE_SECONDS = 0.5
CONFIRM_SAMPLES = 15      # a peak this close to "now" may still move, so wait
EDGE_SAMPLES = 28         # the net is unreliable near the window start (zero padding)
GRAPH_SECONDS = 6

# MPU6050 default full-scale ranges: +-2g for accel, +-250 deg/s for gyro.
ACCEL_SCALE = 1.0 / 16384.0
GYRO_SCALE = 1.0 / 131.0

# "step": you tap every foot strike. "not_step": any motion that is not a step
# (standing, biking, arm swinging, hopping...). Its true count is always 0.
LABELS = ["step", "not_step"]


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


recorder = StepRecorder()


@brick
class ImuSampler:
    """Polls the MCU at SAMPLE_HZ, timestamps each reading, keeps the last
    LIVE_SECONDS, and feeds the recorder while a trial is being recorded."""

    def __init__(self):
        self.buffer = deque(maxlen=int(SAMPLE_HZ * LIVE_SECONDS * 1.5))  # (t, ax..gz)
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
        t = time.time()
        with self._lock:
            self.buffer.append((t,) + sample)
        recorder.on_sample(t, sample)
        time.sleep(1.0 / SAMPLE_HZ)

    def recent(self, seconds):
        with self._lock:
            data = list(self.buffer)
        cutoff = time.time() - seconds
        return [row for row in data if row[0] >= cutoff]


sampler = ImuSampler()

try:
    net = NumpyStepNet()
    logger.info(f"Loaded {MODEL_PATH.name}: counting with the neural net")
except Exception:
    net = None
    logger.warning(f"{MODEL_PATH.name} not found: counting with the simple baseline until you train and copy it here")

state = {
    "steps": 0, "baseline_steps": 0, "since": time.time(),
    "last_t": 0.0, "last_base_t": 0.0,
    "curve": [], "intensity": [],
}


@brick
class LiveStepCounter:
    """Every UPDATE_SECONDS, runs the net over the last LIVE_SECONDS of signal
    and counts each new step peak exactly once. Windows overlap, so a step
    is only counted if it comes at least MIN_GAP_S after the last counted
    step, and only once it is CONFIRM_SAMPLES old."""

    def loop(self):
        time.sleep(UPDATE_SECONDS)
        rows = sampler.recent(LIVE_SECONDS)
        # The Bridge delivers fewer samples per second than SAMPLE_HZ, so test
        # for 2 seconds of data by time span, not by sample count.
        if len(rows) < 20 or rows[-1][0] - rows[0][0] < 2.0 or not sampler.connected:
            return
        arr = np.array(rows)
        grid, raw = resample(arr[:, 0], arr[:, 1:7])
        if len(raw) < 2 * EDGE_SAMPLES:
            return

        if net is not None:
            curve = net.predict(raw)
            peaks = find_peaks(curve)
        else:
            curve = np.zeros(len(raw))
            peaks = baseline_peaks(raw)
        for i in peaks:
            t = grid[i]
            if EDGE_SAMPLES <= i < len(raw) - CONFIRM_SAMPLES and t - state["last_t"] >= MIN_GAP_S:
                state["steps"] += 1
                state["last_t"] = t
        for i in baseline_peaks(raw):
            t = grid[i]
            if EDGE_SAMPLES <= i < len(raw) - CONFIRM_SAMPLES and t - state["last_base_t"] >= 0.3:
                state["baseline_steps"] += 1
                state["last_base_t"] = t

        mag = np.abs(np.sqrt((raw[:, 0:3] ** 2).sum(axis=1)) - 1.0)
        state["curve"] = [round(float(v), 3) for v in curve[::5]]
        state["intensity"] = [round(float(v), 3) for v in mag[::5]]


live_counter = LiveStepCounter()

# The app's files land at /app/python, so point WebUI's assets there.
web_ui = WebUI(assets_dir_path="/app/python/assets")


def get_status():
    return {
        "connected": sampler.connected,
        "using_net": net is not None,
        "steps": state["steps"],
        "baseline_steps": state["baseline_steps"],
        "seconds": round(time.time() - state["since"], 1),
        "recording": recorder.is_recording,
        "recording_label": recorder.label,
        "recording_rows": recorder.rows,
        "recording_taps": recorder.taps,
        "labels": LABELS,
        "trials_saved": len(recorder.trial_files()),
    }


def get_telemetry():
    return {"hz": SAMPLE_HZ / 5, "curve": state["curve"], "intensity": state["intensity"]}


def reset_count():
    state["steps"] = state["baseline_steps"] = 0
    state["since"] = time.time()
    return {"ok": True}


def start_recording(label: str, known_count: str = ""):
    if label not in LABELS:
        return {"ok": False, "error": f"unknown label '{label}'"}
    known = 0 if label == "not_step" else (int(known_count) if known_count.strip().isdigit() else None)
    return {"ok": recorder.start(label, known)}


def tap():
    return {"ok": recorder.tap()}


def stop_recording():
    return {"ok": True, "summary": recorder.stop()}


web_ui.expose_api("GET", "/status", get_status)
web_ui.expose_api("GET", "/telemetry", get_telemetry)
web_ui.expose_api("POST", "/reset", reset_count)
web_ui.expose_api("POST", "/start_recording", start_recording)
web_ui.expose_api("POST", "/tap", tap)
web_ui.expose_api("POST", "/stop_recording", stop_recording)

logger.info(f"Web UI will be available at {web_ui.url}")

App.run()
