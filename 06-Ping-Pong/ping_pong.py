"""
Virtual ping-pong with a LEGO Education Double Motor as the paddle.

The player holds the Double Motor in their hand like a paddle. A virtual ball
is drawn over the webcam view, flying toward the player. To return it:
    - Pose: the player's hand (MediaPipe Pose) must be on the ball when it
      reaches the hit zone, and
    - IMU:  the player must actually swing (gyro spike on the Double Motor)
      within a short window of that moment.
Hit -> the ball goes back to a virtual opponent, which returns it faster.
Miss -> the rally ends and the streak resets.

AprilTags (36h11) run the menu:
    tag 1 = Easy, tag 2 = Medium, tag 3 = Hard   (sets ball speed)
    tag 0 = start the game
Each tag must be held steady for TAG_HOLD_S seconds to count.

The record number of continuous hits is published in real time to MQTT as a
floating point number.

Haptic feedback (the "new" thing): the Double Motor's own motors are pulsed
back and forth so the paddle vibrates in the player's hand - a short tick on
a hit, a long rumble on a miss, a triple buzz on a new record.

Keys:  q = quit,  r = back to the level menu,  space = swing (testing
without the robot),  [ / ] = lower / raise the swing threshold.

Setup:
    pip install legoeducation mediapipe opencv-contrib-python paho-mqtt
    Run make_tags.py once to make the tag PNGs.
"""

import math
import os
import queue
import random
import threading
import time
import urllib.request
from collections import deque

import cv2
import legoeducation as le
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from mqttlib import MQTTClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "1129"

CAMERA_INDEX = 1          # built-in webcam (index 0 is the iPhone Continuity Camera)
MAX_CAMERA_INDEX = 3      # other indices are tried if it doesn't deliver frames
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
POSE_INPUT_WIDTH = 640    # pose and tags run on a downscaled copy (faster, same result)
TAG_INPUT_WIDTH = 800
MIRROR_PREVIEW = True     # flip the image so it behaves like a mirror

# Record number of continuous hits is published here as a float.
MQTT_TOPIC = "ME193/Rogers/Ani"

# Which hand holds the paddle (the person's own left/right). Press 'h' in game
# to swap if the paddle follows the wrong arm.
PADDLE_HAND = "right"

# The virtual paddle head sits just past the wrist, pointing along the forearm
# (elbow -> wrist), like a paddle held in the fist.
PADDLE_HANDLE = 0.3       # handle length, as a fraction of the head's half-length
PADDLE_ASPECT = 0.8       # head half-width / half-length (1.0 = round)
VIS_GOOD = 0.8            # landmark visibility at/above this = solid tracking
VIS_OK = 0.5              # below this the landmark counts as lost

# Pose smoothing (One Euro filter): heavy smoothing when the hand is still,
# almost none when it moves fast, so swings aren't delayed.
SMOOTH_MIN_CUTOFF = 1.0   # Hz - lower = smoother when still
SMOOTH_BETA = 8.0         # higher = less lag during fast movement
POSE_HOLD_S = 0.25        # keep the last pose this long if a frame misses the body

# AprilTag menu
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
START_TAG_ID = 0
TAG_HOLD_S = 1.0

# speed: fraction of the table the ball covers per second (1.0 = 1 s per trip)
# hit_radius: paddle head half-length, as a fraction of frame width
# aim_away: probability the opponent aims away from the player's hand
LEVELS = {
    1: {"name": "EASY", "speed": 0.45, "hit_radius": 0.16, "aim_away": 0.0},
    2: {"name": "MEDIUM", "speed": 0.70, "hit_radius": 0.12, "aim_away": 0.4},
    3: {"name": "HARD", "speed": 1.00, "hit_radius": 0.09, "aim_away": 0.8},
}
SPEEDUP_PER_HIT = 0.05    # rally gets 5% faster every hit...
MAX_SPEEDUP = 2.0         # ...up to 2x the level's base speed

HIT_ZONE_Z = 0.80         # ball is hittable once it is this close (0 = far, 1 = player)
MISS_Z = 1.15             # ball is past the player -> miss
SERVE_DELAY_S = 1.0
MISS_BANNER_S = 1.5
COUNTDOWN_S = 3.0         # also used to calibrate the IMU while the paddle is held still

# Swing detection: a swing is when the gyro magnitude rises this far above the
# still-hand baseline measured during the countdown. Units are the raw IMU
# units; watch the swing meter and tune with [ and ].
SWING_THRESHOLD = 250.0
SWING_WINDOW_S = 0.35     # swing must be this recent when the ball is in the hit zone
SWING_COOLDOWN_S = 0.30   # one physical swing only counts once

# Haptic patterns: (duty cycle, number of back-and-forth pulses, pulse length s)
HAPTIC_HIT = (80, 2, 0.04)
HAPTIC_MISS = (100, 12, 0.04)
HAPTIC_RECORD = (100, 3, 0.08)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# MediaPipe Pose landmark indices
POSE_SHOULDER = {"left": 11, "right": 12}
POSE_ELBOW = {"left": 13, "right": 14}
POSE_WRIST = {"left": 15, "right": 16}
POSE_INDEX = {"left": 19, "right": 20}

# Skeleton lines to draw (face points are left out to keep it readable).
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),          # shoulders + arms
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    (11, 23), (12, 24), (23, 24),                              # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                    # legs
]

WAIT_LEVEL, WAIT_START, COUNTDOWN, SERVE, PLAY, MISSED = (
    "WAIT_LEVEL", "WAIT_START", "COUNTDOWN", "SERVE", "PLAY", "MISSED")


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading pose landmark model (one-time download)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def open_camera():
    """Return the first camera that actually delivers frames, trying CAMERA_INDEX first.

    Some indices (e.g. an iPhone Continuity Camera) open fine but never return a frame.
    """
    indices = [CAMERA_INDEX] + [i for i in range(MAX_CAMERA_INDEX + 1) if i != CAMERA_INDEX]
    for index in indices:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            for _ in range(30):
                ok, _ = cap.read()
                if ok:
                    print(f"Using camera index {index}")
                    return CameraStream(cap)
                time.sleep(0.05)
        cap.release()
    return None


class CameraStream:
    """Reads the camera on a background thread and keeps only the newest frame.

    Without this, the game loop blocks on cap.read() waiting for the next frame,
    and when a loop takes slightly longer than one camera frame it falls a whole
    frame behind - which made the game stutter.
    """

    def __init__(self, cap):
        self._cap = cap
        self._frame = None
        self._frame_id = 0
        self._last_returned = 0
        self._running = True
        self._new_frame = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._new_frame:
                self._frame = frame
                self._frame_id += 1
                self._new_frame.notify_all()

    def isOpened(self):
        return self._running and self._cap.isOpened()

    def read(self, timeout=1.0):
        """Wait for a frame newer than the last one returned (each frame is processed once)."""
        with self._new_frame:
            if not self._new_frame.wait_for(lambda: self._frame_id > self._last_returned, timeout):
                return False, None
            self._last_returned = self._frame_id
            return True, self._frame

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self._cap.release()


class OneEuroFilter:
    """Low-pass filter whose cutoff rises with speed (Casiez et al., 2012)."""

    def __init__(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None or t <= self.t:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx += self._alpha(1.0, dt) * (dx - self.dx)
        cutoff = SMOOTH_MIN_CUTOFF + SMOOTH_BETA * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


class PoseSmoother:
    """Smooths every landmark and briefly holds the last pose through dropped frames."""

    def __init__(self):
        self._filters = None
        self._last = None
        self._last_t = -1e9

    def update(self, body, now):
        if body is None:
            if now - self._last_t <= POSE_HOLD_S:
                return self._last
            self._filters = None  # person gone: start fresh next time
            return None
        if self._filters is None:
            self._filters = [(OneEuroFilter(), OneEuroFilter()) for _ in body]
        self._last = [(fx(x, now), fy(y, now), vis)
                      for (x, y, vis), (fx, fy) in zip(body, self._filters)]
        self._last_t = now
        return self._last


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Robot: IMU reading + haptics (motors, light, beep) on a worker thread so
# the camera loop never blocks on Bluetooth.
# ---------------------------------------------------------------------------
class Paddle:
    def __init__(self):
        self.robot = le.DoubleMotor()
        print(f"Connecting to Double Motor (orange card, serial {CARD_SERIAL})...")
        self.robot.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL,
                           device_notification_delay=20)
        self.connected = self.robot.connected
        if not self.connected:
            print("Could not connect to the Double Motor. Playing in keyboard mode "
                  "(press space to swing, no haptics).")
            return
        self._commands = queue.Queue()
        self._worker = threading.Thread(target=self._run_commands, daemon=True)
        self._worker.start()

    def gyro_magnitude(self):
        """Magnitude of the angular velocity vector, or None before the first IMU update."""
        if not self.connected:
            return None
        imu = self.robot.imu_device
        gx, gy, gz = imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ
        if gx != gx:  # NaN: no IMU notification yet
            return None
        return math.sqrt(gx * gx + gy * gy + gz * gz)

    def feedback(self, haptic, color, beep_freq=None, beep_pattern=le.SOUND_PATTERN_BEEP_SINGLE):
        """Queue a vibration + light (+ optional beep). Returns how long the vibration lasts."""
        duty, pulses, pulse_s = haptic
        if self.connected:
            self._commands.put((haptic, color, beep_freq, beep_pattern))
        return 2 * pulses * pulse_s

    def light(self, color):
        if self.connected:
            self._commands.put((None, color, None, None))

    def _run_commands(self):
        while True:
            item = self._commands.get()
            if item is None:
                return
            haptic, color, beep_freq, beep_pattern = item
            try:
                self.robot.light_color(color, blocking=False)
                if beep_freq:
                    self.robot.beep(pattern=beep_pattern, frequency=beep_freq, blocking=False)
                if haptic:
                    self._vibrate(*haptic)
            except Exception as exc:  # a dropped BLE command shouldn't kill the game
                print(f"Robot command failed: {exc}")

    def _vibrate(self, duty, pulses, pulse_s):
        # Flipping the motor direction rapidly makes the hub shake instead of spin.
        for _ in range(pulses):
            self.robot.motor_set_duty_cycle(duty, motor=le.MOTOR_BOTH, blocking=False)
            time.sleep(pulse_s)
            self.robot.motor_set_duty_cycle(-duty, motor=le.MOTOR_BOTH, blocking=False)
            time.sleep(pulse_s)
        self.robot.motor_stop(motor=le.MOTOR_BOTH, blocking=False)

    def close(self):
        if not self.connected:
            return
        self._commands.put(None)
        self._worker.join(timeout=2)
        self.robot.motor_stop(motor=le.MOTOR_BOTH)
        self.robot.light_color(le.LEGO_COLOR_WHITE)
        self.robot.disconnect()


class SwingDetector:
    """Threshold on gyro magnitude above a calibrated still-hand baseline."""

    def __init__(self):
        self.baseline = 0.0
        self.threshold = SWING_THRESHOLD
        self._samples = []
        self.last_swing_t = -1e9
        self.ignore_until = 0.0  # ignore our own haptic vibration
        self.level = 0.0         # current gyro above baseline, for the on-screen meter

    def start_calibration(self):
        self._samples = []

    def add_calibration_sample(self, mag):
        if mag is not None:
            self._samples.append(mag)

    def finish_calibration(self):
        if self._samples:
            self.baseline = sum(self._samples) / len(self._samples)
        print(f"IMU baseline gyro magnitude: {self.baseline:.1f}")

    def update(self, mag, now):
        if mag is None:
            self.level = 0.0
            return
        self.level = max(0.0, mag - self.baseline)
        if (self.level > self.threshold and now >= self.ignore_until
                and now - self.last_swing_t > SWING_COOLDOWN_S):
            self.last_swing_t = now

    def force_swing(self, now):
        self.last_swing_t = now

    def swung_recently(self, now):
        return now - self.last_swing_t <= SWING_WINDOW_S


# ---------------------------------------------------------------------------
# Table geometry. The table is drawn in perspective: depth d = 0 is the far
# (opponent) edge, d = 1 the near (player) edge, all in normalized screen units.
# ---------------------------------------------------------------------------
TABLE_FAR_Y, TABLE_NEAR_Y = 0.32, 0.88
TABLE_FAR_HALF_W, TABLE_NEAR_HALF_W = 0.16, 0.40
CAMERA_DIST = 1.6         # bigger = stronger perspective


def perspective(d):
    """Map table depth (0 far .. 1 near) to screen progress (0 .. 1), compressing the far end."""
    far, near = 1.0 / (1.0 + CAMERA_DIST), 1.0
    return (1.0 / (1.0 + (1.0 - d) * CAMERA_DIST) - far) / (near - far)


def table_y(d):
    return lerp(TABLE_FAR_Y, TABLE_NEAR_Y, perspective(d))


def table_half_w(d):
    return lerp(TABLE_FAR_HALF_W, TABLE_NEAR_HALF_W, perspective(d))


class Ball:
    """Ball travelling between the opponent's paddle and the player's paddle.

    z = 0 is the opponent's hit point, z = 1 is the player's hit point (`near`,
    where the opponent aimed). On screen the ball arcs over the net and bounces
    once on the receiving side of the table, with a shadow on the table below it.
    """

    START_D, END_D = -0.06, 1.06   # table depth at z = 0 and z = 1
    BOUNCE_T = 0.72                # fraction of each trip where the ball bounces
    ARC = 0.06                     # extra height of the arc over the net (screen units)

    def __init__(self):
        self.far = (0.5, 0.18)
        self.near = (0.5, 0.55)
        self.z = 0.0
        self.direction = 1  # +1 toward the player, -1 toward the opponent
        self.speed = 0.5

    def depth(self):
        return lerp(self.START_D, self.END_D, self.z)

    def height(self):
        """Height above the table (screen units): starts at the hitter's paddle, arcs
        over the net down to the bounce, then rises to the receiver's paddle."""
        h_far = table_y(self.START_D) - self.far[1]
        h_near = table_y(self.END_D) - self.near[1]
        if self.direction == 1:
            t, h_start, h_end = self.z, h_far, h_near
        else:
            t, h_start, h_end = 1.0 - self.z, h_near, h_far
        tb = self.BOUNCE_T
        if t < tb:
            s = t / tb
            return (1 - s) * h_start + 4 * self.ARC * s * (1 - s)
        u = (t - tb) / (1 - tb)
        return h_end * u * (2 - u)

    def shadow(self):
        x = lerp(self.far[0], self.near[0], self.z)
        return x, table_y(self.depth())

    def pos(self):
        x, ground = self.shadow()
        return x, ground - self.height()

    def radius(self, frame_w):
        return lerp(0.008, 0.026, perspective(self.depth())) * frame_w

    def step(self, dt):
        self.z += self.direction * self.speed * dt


def opponent_choose_target(level, hand):
    """Opponent policy: aim at a random reachable spot, or away from the player's hand."""
    candidates = [(random.uniform(0.2, 0.8), random.uniform(0.35, 0.7)) for _ in range(6)]
    if hand is not None and random.random() < level["aim_away"]:
        return max(candidates, key=lambda c: math.dist(c, hand))
    return candidates[0]


def opponent_far_point(x=None):
    """Where the opponent's paddle is when it hits: just behind the far edge."""
    if x is None:
        x = random.uniform(0.4, 0.6)
    return x, random.uniform(0.17, 0.23)

# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------
def detect_pose(landmarker, frame, timestamp_ms, mirrored=False):
    """Return the body's landmarks as a list of (x, y, visibility) in `frame`'s
    normalized coordinates, or None if no person is found.

    Like the tags, pose runs on the un-mirrored image: the model decides left vs.
    right arm from normal (non-mirrored) photos, so a mirrored frame would swap them.
    """
    scale = POSE_INPUT_WIDTH / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    rgb = cv2.cvtColor(cv2.flip(small, 1) if mirrored else small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.pose_landmarks:
        return None
    return [((1.0 - lm.x) if mirrored else lm.x, lm.y, lm.visibility)
            for lm in result.pose_landmarks[0]]


class PaddlePose:
    """Virtual paddle placed along the paddle arm's forearm.

    All math is in pixels so the paddle isn't stretched by the frame's aspect ratio.
    """

    def __init__(self, body, hand, half_len, frame_w, frame_h):
        self.status = "NO BODY"
        self.center = None   # normalized (x, y), used by the game logic
        self.tilt = 0.0      # forearm angle from vertical, radians (+ = leaning right)
        self.wrist_px = self.center_px = None
        self.axis = (0.0, -1.0)  # unit vector along the paddle, pixels
        if body is None:
            return
        wrist, elbow, index = body[POSE_WRIST[hand]], body[POSE_ELBOW[hand]], body[POSE_INDEX[hand]]
        if wrist[2] < VIS_OK:
            self.status = "HAND LOST"
            return
        self.status = "HAND TRACKED" if wrist[2] >= VIS_GOOD else "HAND WEAK"

        self.wrist_px = np.array([wrist[0] * frame_w, wrist[1] * frame_h])
        # Forearm direction; fall back to wrist -> index knuckle if the elbow is hidden.
        base = elbow if elbow[2] >= VIS_OK else None
        if base is not None:
            vec = self.wrist_px - np.array([base[0] * frame_w, base[1] * frame_h])
        else:
            vec = np.array([index[0] * frame_w, index[1] * frame_h]) - self.wrist_px
        length = float(np.linalg.norm(vec))
        axis = vec / length if length > 1e-3 else np.array([0.0, -1.0])
        self.axis = tuple(axis)
        self.center_px = self.wrist_px + axis * half_len * (1 + PADDLE_HANDLE)
        self.center = (self.center_px[0] / frame_w, self.center_px[1] / frame_h)
        self.tilt = math.atan2(self.axis[0], -self.axis[1])

    def contains(self, point, half_len, frame_w, frame_h):
        """Is the normalized point inside the paddle head (an ellipse along the forearm)?"""
        if self.center_px is None:
            return False
        d = np.array([point[0] * frame_w, point[1] * frame_h]) - self.center_px
        along = d[0] * self.axis[0] + d[1] * self.axis[1]
        across = -d[0] * self.axis[1] + d[1] * self.axis[0]
        return (along / half_len) ** 2 + (across / (half_len * PADDLE_ASPECT)) ** 2 <= 1.0

    def offset(self, point, half_len, frame_w):
        """Sideways hit position on the paddle, -1 (left edge) .. 1 (right edge)."""
        d = point[0] * frame_w - self.center_px[0]
        return max(-1.0, min(1.0, d / (half_len * PADDLE_ASPECT)))


class TagHold:
    """Report a tag ID only after it has been seen continuously for TAG_HOLD_S."""

    def __init__(self):
        self.current = None
        self.since = 0.0

    def update(self, tag_id, now):
        if tag_id != self.current:
            self.current, self.since = tag_id, now
            return None, 0.0
        if tag_id is None:
            return None, 0.0
        held = now - self.since
        return (tag_id if held >= TAG_HOLD_S else None), min(1.0, held / TAG_HOLD_S)


def detect_tag(detector, frame, mirrored=False):
    """Find the largest AprilTag. Detection runs on the un-mirrored image, since a
    mirror-flipped tag no longer matches its code; corners come back in the
    coordinates of `frame`."""
    scale = min(1.0, TAG_INPUT_WIDTH / frame.shape[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if mirrored:
        gray = cv2.flip(gray, 1)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None
    w = gray.shape[1]
    corners = [c.copy() for c in corners]
    for c in corners:
        if mirrored:
            c[..., 0] = w - 1 - c[..., 0]
        c /= scale  # back to full-frame pixels
    # Largest tag in view wins (the one held closest to the camera).
    best = max(range(len(ids)), key=lambda i: cv2.contourArea(corners[i][0]))
    return int(ids.flatten()[best]), corners[best]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX

# Colors (BGR)
TABLE_BLUE = (140, 72, 12)
TABLE_SIDE = (85, 40, 5)
TABLE_LEG = (45, 45, 45)
LINE_WHITE = (245, 245, 245)
NET_MESH = (30, 30, 30)
BALL_DARK, BALL_MAIN, BALL_SHINE = (20, 105, 215), (55, 165, 255), (200, 230, 255)
RUBBER_RED, RUBBER_EDGE = (50, 45, 205), (25, 20, 110)
WOOD, WOOD_EDGE = (70, 135, 195), (30, 65, 100)
PANEL = (28, 22, 18)
WHITE, GREY = (245, 245, 245), (170, 170, 170)
GOLD, GREEN, RED = (40, 200, 255), (100, 220, 90), (70, 70, 235)
LEVEL_COLORS = {"EASY": GREEN, "MEDIUM": GOLD, "HARD": RED}


def px(point, w, h):
    return int(point[0] * w), int(point[1] * h)


def blend(frame, draw_fn, alpha):
    """Run draw_fn on a copy of the frame and blend it back in (for see-through shapes)."""
    overlay = frame.copy()
    draw_fn(overlay)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def panel(frame, x1, y1, x2, y2, alpha=0.7, radius=14, color=PANEL):
    """Rounded, see-through dark panel. Only the panel's region is blended."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    rw, rh = x2 - x1, y2 - y1
    r = min(radius, rw // 2, rh // 2)
    cv2.rectangle(overlay, (r, 0), (rw - r, rh), color, -1)
    cv2.rectangle(overlay, (0, r), (rw, rh - r), color, -1)
    for cx, cy in ((r, r), (rw - r - 1, r), (r, rh - r - 1), (rw - r - 1, rh - r - 1)):
        cv2.circle(overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
    mask = np.zeros((rh, rw), np.uint8)
    cv2.rectangle(mask, (r, 0), (rw - r, rh), 255, -1)
    cv2.rectangle(mask, (0, r), (rw, rh - r), 255, -1)
    for cx, cy in ((r, r), (rw - r - 1, r), (r, rh - r - 1), (rw - r - 1, rh - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    roi[mask > 0] = blended[mask > 0]


def text_size(text, scale, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return tw, th


def put_text(frame, text, org, scale=0.7, color=WHITE, thickness=1):
    x, y = org
    # Same thickness as the text: thicker Hershey strokes also widen the letter spacing.
    cv2.putText(frame, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def put_text_centered(frame, text, cx, y, scale=0.7, color=WHITE, thickness=1):
    tw, _ = text_size(text, scale, thickness)
    put_text(frame, text, (int(cx - tw / 2), int(y)), scale, color, thickness)


# --- scene -----------------------------------------------------------------
def draw_background(frame):
    """Dim the camera image so the table and ball stand out."""
    cv2.convertScaleAbs(frame, frame, alpha=0.5, beta=0)


def draw_table(frame):
    h, w = frame.shape[:2]
    far_l = (int((0.5 - TABLE_FAR_HALF_W) * w), int(TABLE_FAR_Y * h))
    far_r = (int((0.5 + TABLE_FAR_HALF_W) * w), int(TABLE_FAR_Y * h))
    near_r = (int((0.5 + TABLE_NEAR_HALF_W) * w), int(TABLE_NEAR_Y * h))
    near_l = (int((0.5 - TABLE_NEAR_HALF_W) * w), int(TABLE_NEAR_Y * h))
    thick = int(0.025 * h)

    # Legs, then the table's front edge, then the top.
    leg_w = max(6, int(0.014 * w))
    for x in (near_l[0] + int(0.05 * w), near_r[0] - int(0.05 * w)):
        cv2.rectangle(frame, (x - leg_w // 2, near_l[1] + thick), (x + leg_w // 2, h), TABLE_LEG, -1)
    for x in (far_l[0] + int(0.025 * w), far_r[0] - int(0.025 * w)):
        cv2.rectangle(frame, (x - leg_w // 4, far_l[1]), (x + leg_w // 4, far_l[1] + int(0.14 * h)), TABLE_LEG, -1)

    side = np.array([near_l, near_r, (near_r[0], near_r[1] + thick), (near_l[0], near_l[1] + thick)])
    cv2.fillConvexPoly(frame, side, TABLE_SIDE, cv2.LINE_AA)
    top = np.array([far_l, far_r, near_r, near_l])
    # Slightly see-through so the player can still see themselves behind it.
    blend(frame, lambda o: cv2.fillConvexPoly(o, top, TABLE_BLUE, cv2.LINE_AA), 0.82)

    # Soft light on the far half so the surface doesn't look flat.
    mid_y = int(table_y(0.5) * h)
    mid_hw = int(table_half_w(0.5) * w)
    sheen = np.array([far_l, far_r, (w // 2 + mid_hw, mid_y), (w // 2 - mid_hw, mid_y)])
    blend(frame, lambda o: cv2.fillConvexPoly(o, sheen, (190, 120, 50), cv2.LINE_AA), 0.18)

    cv2.polylines(frame, [top], True, LINE_WHITE, 3, cv2.LINE_AA)
    cv2.line(frame, (w // 2, far_l[1]), (w // 2, near_l[1]), LINE_WHITE, 2, cv2.LINE_AA)


def draw_net(frame):
    h, w = frame.shape[:2]
    y = int(table_y(0.5) * h)
    hw = int((table_half_w(0.5) + 0.02) * w)
    nh = int(0.05 * h)
    x1, x2, top = w // 2 - hw, w // 2 + hw, y - nh

    def mesh(o):
        cv2.rectangle(o, (x1, top), (x2, y), NET_MESH, -1)
        for x in range(x1, x2, 7):
            cv2.line(o, (x, top), (x, y), (90, 90, 90), 1)
        for yy in range(top, y, 7):
            cv2.line(o, (x1, yy), (x2, yy), (90, 90, 90), 1)

    blend(frame, mesh, 0.6)
    cv2.line(frame, (x1, top), (x2, top), LINE_WHITE, 3, cv2.LINE_AA)
    for x in (x1, x2):
        cv2.line(frame, (x, y + 6), (x, top - 3), (25, 25, 25), 6, cv2.LINE_AA)


def draw_shadow(frame, ball):
    d = ball.depth()
    if not 0.0 <= d <= 1.02:
        return  # only on the table
    h, w = frame.shape[:2]
    center = px(ball.shadow(), w, h)
    r = ball.radius(w)
    height = max(0.0, min(1.0, ball.height() / 0.3))
    axes = (max(2, int(r * (1.2 - 0.4 * height))), max(1, int(r * 0.4)))
    blend(frame, lambda o: cv2.ellipse(o, center, axes, 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA),
          0.55 - 0.3 * height)


def draw_ball(frame, ball, hittable, trail):
    h, w = frame.shape[:2]
    # Motion trail: a faint streak, older positions smaller.
    def streak(o):
        for i, (tx, ty, tr) in enumerate(trail):
            k = (i + 1) / (len(trail) + 1)
            cv2.circle(o, px((tx, ty), w, h), max(1, int(tr * (0.3 + 0.6 * k))), BALL_MAIN, -1, cv2.LINE_AA)
    if trail:
        blend(frame, streak, 0.3)

    cx, cy = px(ball.pos(), w, h)
    r = max(3, int(ball.radius(w)))
    if hittable:
        cv2.circle(frame, (cx, cy), r + 7, GREEN, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, BALL_DARK, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx - r // 8, cy - r // 8), int(r * 0.85), BALL_MAIN, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx - r // 3, cy - r // 3), max(1, r // 4), BALL_SHINE, -1, cv2.LINE_AA)


def draw_opponent_paddle(frame, far):
    h, w = frame.shape[:2]
    cx, cy = px(far, w, h)
    r = int(0.022 * w)
    cv2.line(frame, (cx, cy + r), (cx + r // 2, cy + int(r * 1.9)), WOOD_EDGE, 7, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + r), (cx + r // 2, cy + int(r * 1.9)), WOOD, 4, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (15, 15, 15), 2, cv2.LINE_AA)


def draw_paddle(frame, paddle_pose, half_len, can_hit):
    """The player's paddle: red rubber head along the forearm, wooden handle to the wrist."""
    if paddle_pose.center_px is None:
        return
    center = tuple(int(v) for v in paddle_pose.center_px)
    angle = math.degrees(math.atan2(paddle_pose.axis[1], paddle_pose.axis[0]))
    axes = (int(half_len), int(half_len * PADDLE_ASPECT))

    handle_end = paddle_pose.center_px - np.array(paddle_pose.axis) * half_len * 0.9
    wrist = tuple(int(v) for v in paddle_pose.wrist_px)
    handle_end = tuple(int(v) for v in handle_end)
    cv2.line(frame, wrist, handle_end, WOOD_EDGE, 14, cv2.LINE_AA)
    cv2.line(frame, wrist, handle_end, WOOD, 9, cv2.LINE_AA)

    blend(frame, lambda o: cv2.ellipse(o, center, axes, angle, 0, 360, RUBBER_RED, -1, cv2.LINE_AA), 0.5)
    cv2.ellipse(frame, center, axes, angle, 0, 360, RUBBER_EDGE, 4, cv2.LINE_AA)
    if can_hit:
        glow = (axes[0] + 8, axes[1] + 8)
        cv2.ellipse(frame, center, glow, angle, 0, 360, GREEN, 3, cv2.LINE_AA)


def vis_color(visibility):
    """Green = solid, yellow = weak, red = lost."""
    if visibility >= VIS_GOOD:
        return GREEN
    if visibility >= VIS_OK:
        return GOLD
    return RED


def draw_skeleton(frame, body, hand):
    """Draw what MediaPipe sees: every joint colored by confidence, paddle arm highlighted."""
    if body is None:
        return
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in body]
    arm = {(POSE_SHOULDER[hand], POSE_ELBOW[hand]), (POSE_ELBOW[hand], POSE_WRIST[hand])}
    for a, b in POSE_CONNECTIONS:
        if min(body[a][2], body[b][2]) < 0.3:
            continue
        if (a, b) in arm:
            cv2.line(frame, pts[a], pts[b], (255, 210, 60), 5, cv2.LINE_AA)
        else:
            cv2.line(frame, pts[a], pts[b], (150, 150, 150), 2, cv2.LINE_AA)
    for i in {i for c in POSE_CONNECTIONS for i in c}:
        if body[i][2] >= 0.3:
            cv2.circle(frame, pts[i], 5, vis_color(body[i][2]), -1, cv2.LINE_AA)
    wrist = POSE_WRIST[hand]
    cv2.circle(frame, pts[wrist], 11, vis_color(body[wrist][2]), 2, cv2.LINE_AA)


class Effects:
    """Short-lived animations: expanding rings, floating text, a screen-edge flash."""

    def __init__(self):
        self.items = []

    def ring(self, pos, color, now, duration=0.45):
        self.items.append(("ring", pos, color, now, duration, None))

    def text(self, pos, text, color, now, duration=0.9):
        self.items.append(("text", pos, color, now, duration, text))

    def flash(self, color, now, duration=0.6):
        self.items.append(("flash", None, color, now, duration, None))

    def draw(self, frame, now):
        h, w = frame.shape[:2]
        alive = []
        for item in self.items:
            kind, pos, color, start, duration, text = item
            k = (now - start) / duration
            if k >= 1.0:
                continue
            alive.append(item)
            if kind == "ring":
                cv2.circle(frame, px(pos, w, h), int(12 + 90 * k), color, max(1, int(6 * (1 - k))), cv2.LINE_AA)
            elif kind == "text":
                x, y = px(pos, w, h)
                put_text_centered(frame, text, x, y - 30 - int(50 * k), 1.1, color, 2)
            elif kind == "flash":
                border = int(0.04 * w)
                blend(frame, lambda o: cv2.rectangle(o, (0, 0), (w - 1, h - 1), color, border), 0.6 * (1 - k))
        self.items = alive


# --- HUD -------------------------------------------------------------------
def draw_scoreboard(frame, level, streak, record):
    h, w = frame.shape[:2]
    pw, ph = 300, 96
    x1, y1 = 14, 14
    panel(frame, x1, y1, x1 + pw, y1 + ph)
    for cx, label, value, color in ((x1 + pw // 4, "RALLY", streak, WHITE),
                                    (x1 + 3 * pw // 4, "BEST", record, GOLD)):
        put_text_centered(frame, label, cx, y1 + 26, 0.55, GREY)
        put_text_centered(frame, str(value), cx, y1 + 78, 1.6, color, 2)
    cv2.line(frame, (x1 + pw // 2, y1 + 16), (x1 + pw // 2, y1 + ph - 16), (90, 90, 90), 1, cv2.LINE_AA)

    if level:
        name = level["name"]
        tw, _ = text_size(name, 0.55)
        panel(frame, x1, y1 + ph + 8, x1 + tw + 28, y1 + ph + 38, radius=12)
        put_text(frame, name, (x1 + 14, y1 + ph + 30), 0.55, LEVEL_COLORS[name])


def draw_status_bar(frame, paddle_pose, hand, robot_ok, mqtt_ok):
    h, w = frame.shape[:2]
    hand_colors = {"HAND TRACKED": GREEN, "HAND WEAK": GOLD, "HAND LOST": RED, "NO BODY": RED}
    items = [("ROBOT" if robot_ok else "ROBOT OFF (space = swing)", GREEN if robot_ok else RED),
             ("MQTT", GREEN if mqtt_ok else RED),
             (f"{paddle_pose.status} ({hand})", hand_colors[paddle_pose.status])]
    widths = [text_size(t, 0.5)[0] + 34 for t, _ in items]
    x, y = 14, h - 46
    panel(frame, x, y, x + sum(widths) + 10, y + 34, radius=12)
    x += 12
    for (text, color), width in zip(items, widths):
        cv2.circle(frame, (x + 6, y + 17), 6, color, -1, cv2.LINE_AA)
        put_text(frame, text, (x + 18, y + 23), 0.5, WHITE)
        x += width

    keys = "q quit   r menu   h hand   [ ] swing threshold"
    tw, _ = text_size(keys, 0.45)
    put_text(frame, keys, (w - tw - 16, h - 22), 0.45, GREY)


def draw_swing_meter(frame, swing):
    h, w = frame.shape[:2]
    x1, y1, pw, ph = w - 70, 160, 56, 270
    panel(frame, x1, y1, x1 + pw, y1 + ph)
    put_text_centered(frame, "SWING", x1 + pw // 2, y1 + 22, 0.45, GREY)
    bx1, by1, bx2, by2 = x1 + 18, y1 + 34, x1 + pw - 18, y1 + ph - 34
    bh = by2 - by1
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 60), -1)
    fill = min(1.0, swing.level / (swing.threshold * 1.5))
    swinging = swing.level > swing.threshold
    if fill > 0:
        cv2.rectangle(frame, (bx1, by2 - int(fill * bh)), (bx2, by2), GREEN if swinging else GOLD, -1)
    thresh_y = by2 - int(bh / 1.5)
    cv2.line(frame, (bx1 - 6, thresh_y), (bx2 + 6, thresh_y), RED, 2, cv2.LINE_AA)
    put_text_centered(frame, str(int(swing.threshold)), x1 + pw // 2, y1 + ph - 12, 0.45, GREY)


def draw_banner(frame, title, subtitle=None, title_color=WHITE, progress=None, title_scale=1.4, y=None):
    """Centered card with a title, optional subtitle, and optional progress bar (0..1)."""
    h, w = frame.shape[:2]
    tw, th = text_size(title, title_scale, 2)
    sw = text_size(subtitle, 0.7)[0] if subtitle else 0
    pw = max(tw, sw, 260) + 60
    ph = th + 50 + (36 if subtitle else 0) + (26 if progress is not None else 0)
    y1 = int(h * 0.5 - ph / 2) if y is None else y
    panel(frame, w // 2 - pw // 2, y1, w // 2 + pw // 2, y1 + ph, alpha=0.78, radius=18)
    cy = y1 + 25 + th
    put_text_centered(frame, title, w // 2, cy, title_scale, title_color, 2)
    if subtitle:
        cy += 38
        put_text_centered(frame, subtitle, w // 2, cy, 0.7, GREY)
    if progress is not None:
        cy += 20
        bx1, bx2 = w // 2 - 110, w // 2 + 110
        cv2.rectangle(frame, (bx1, cy), (bx2, cy + 8), (70, 70, 70), -1)
        cv2.rectangle(frame, (bx1, cy), (bx1 + int(220 * progress), cy + 8), GREEN, -1)

# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class ScorePublisher:
    def __init__(self):
        self.client = None
        try:
            self.client = MQTTClient()
            self.client.connect()
            print(f"MQTT connected, publishing record to {MQTT_TOPIC}")
        except Exception as exc:
            print(f"MQTT unavailable ({exc}); scores will only be shown locally.")
            self.client = None

    @property
    def ok(self):
        return self.client is not None

    def publish(self, record):
        if self.client:
            self.client.publish(MQTT_TOPIC, f"{float(record)}", qos=1, retain=True)

    def close(self):
        if self.client:
            self.client.disconnect()


def main():
    ensure_model_downloaded()

    paddle = Paddle()
    swing = SwingDetector()
    scores = ScorePublisher()

    aruco_dict = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    tag_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    tag_hold = TagHold()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = open_camera()
    if cap is None:
        print("Error: could not open the camera. Check camera permission in "
              "System Settings > Privacy & Security > Camera.")
        paddle.close()
        scores.close()
        return

    hand_side = PADDLE_HAND
    state = WAIT_LEVEL
    state_t = time.monotonic()
    level = None
    streak = 0
    record = 0
    ball = Ball()
    scores.publish(record)
    paddle.light(le.LEGO_COLOR_BLUE)

    def set_state(new_state):
        nonlocal state, state_t
        state, state_t = new_state, time.monotonic()

    def serve():
        ball.far = opponent_far_point()
        ball.near = opponent_choose_target(level, paddle_pose.center)
        ball.z = 0.0
        ball.direction = 1
        ball.speed = level["speed"]
        set_state(SERVE)

    prev_t = time.monotonic()
    start_ms = int(prev_t * 1000)
    paddle_pose = PaddlePose(None, hand_side, 1, 1, 1)
    failed_reads = 0
    effects = Effects()
    pose_smoother = PoseSmoother()
    trail = deque(maxlen=8)

    try:
        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads > 5:  # each read waits up to 1 s
                        print("Error: the camera stopped delivering frames.")
                        break
                    continue
                failed_reads = 0
                if MIRROR_PREVIEW:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                now = time.monotonic()
                dt = min(0.1, now - prev_t)  # clamp so a stalled frame doesn't teleport the ball
                prev_t = now

                # --- sense ---
                body = pose_smoother.update(
                    detect_pose(landmarker, frame, int(now * 1000) - start_ms, MIRROR_PREVIEW), now)
                paddle_half_len = (level["hit_radius"] if level else 0.12) * w
                paddle_pose = PaddlePose(body, hand_side, paddle_half_len, w, h)
                gyro = paddle.gyro_magnitude()
                swing.update(gyro, now)

                # Tags are read from the raw camera image, before anything is drawn on it.
                tag_id = corners = None
                if state in (WAIT_LEVEL, WAIT_START):
                    tag_id, corners = detect_tag(tag_detector, frame, MIRROR_PREVIEW)

                # --- decide / act ---
                in_zone = hand_on_ball = False
                progress = 0.0
                if state in (WAIT_LEVEL, WAIT_START):
                    held_id, progress = tag_hold.update(tag_id, now)
                    if held_id in LEVELS and (level is None or LEVELS[held_id] is not level):
                        level = LEVELS[held_id]
                        print(f"Level set to {level['name']}")
                        paddle.feedback(HAPTIC_HIT, le.LEGO_COLOR_YELLOW, 660)
                        set_state(WAIT_START)
                    elif held_id == START_TAG_ID and state == WAIT_START:
                        swing.start_calibration()
                        paddle.light(le.LEGO_COLOR_WHITE)
                        set_state(COUNTDOWN)

                elif state == COUNTDOWN:
                    swing.add_calibration_sample(gyro)
                    if now - state_t >= COUNTDOWN_S:
                        swing.finish_calibration()
                        streak = 0
                        paddle.light(le.LEGO_COLOR_GREEN)
                        serve()

                elif state == SERVE:
                    if now - state_t >= SERVE_DELAY_S:
                        set_state(PLAY)

                elif state == PLAY:
                    ball.step(dt)
                    bx, by = ball.pos()
                    in_zone = ball.direction == 1 and ball.z >= HIT_ZONE_Z
                    hand_on_ball = paddle_pose.contains((bx, by), paddle_half_len, w, h)

                    if in_zone and hand_on_ball and swing.swung_recently(now):
                        # HIT: return the ball toward the opponent. Where it hit the
                        # paddle and how the forearm is tilted set the return angle.
                        streak += 1
                        offset = paddle_pose.offset((bx, by), paddle_half_len, w)
                        tilt = max(-1.0, min(1.0, paddle_pose.tilt / (math.pi / 3)))
                        far_x = 0.5 + 0.1 * offset + 0.1 * tilt
                        ball.near = (bx, by)
                        ball.far = opponent_far_point(min(0.65, max(0.35, far_x)))
                        ball.z = 1.0
                        ball.direction = -1
                        ball.speed = level["speed"] * min(MAX_SPEEDUP, 1 + SPEEDUP_PER_HIT * streak)
                        swing.last_swing_t = -1e9  # use the swing up
                        effects.ring((bx, by), GREEN, now)

                        if streak > record:
                            record = streak
                            scores.publish(record)
                            effects.text((bx, by), "NEW BEST!", GOLD, now)
                            duration = paddle.feedback(HAPTIC_RECORD, le.LEGO_COLOR_YELLOW, 1320,
                                                       le.SOUND_PATTERN_BEEP_TRIPLE)
                        else:
                            effects.text((bx, by), "+1", GREEN, now)
                            duration = paddle.feedback(HAPTIC_HIT, le.LEGO_COLOR_GREEN, 880)
                        swing.ignore_until = now + duration + 0.15

                    elif ball.direction == 1 and ball.z >= MISS_Z:
                        print(f"Miss! Rally ended at {streak} hits (record {record}).")
                        duration = paddle.feedback(HAPTIC_MISS, le.LEGO_COLOR_RED, 220)
                        swing.ignore_until = now + duration + 0.15
                        effects.flash(RED, now)
                        set_state(MISSED)

                    elif ball.direction == -1 and ball.z <= 0.0:
                        # Opponent returns the ball, aiming according to its policy.
                        effects.ring(ball.far, WHITE, now, duration=0.3)
                        ball.near = opponent_choose_target(level, paddle_pose.center)
                        ball.z = 0.0
                        ball.direction = 1

                elif state == MISSED:
                    ball.step(dt)  # let the ball fly past and drop off the table
                    if now - state_t >= MISS_BANNER_S:
                        streak = 0
                        paddle.light(le.LEGO_COLOR_GREEN)
                        serve()

                # --- draw ---
                show_ball = state in (SERVE, PLAY, MISSED)
                if state == PLAY:
                    bx, by = ball.pos()
                    trail.append((bx, by, ball.radius(w)))
                elif state != MISSED:
                    trail.clear()

                draw_background(frame)
                draw_table(frame)
                if show_ball:
                    draw_opponent_paddle(frame, ball.far)
                    if ball.depth() < 0.5:  # behind the net
                        draw_shadow(frame, ball)
                        draw_ball(frame, ball, False, trail)
                draw_net(frame)
                draw_skeleton(frame, body, hand_side)
                draw_paddle(frame, paddle_pose, paddle_half_len, in_zone and hand_on_ball)
                if show_ball and ball.depth() >= 0.5:
                    draw_shadow(frame, ball)
                    draw_ball(frame, ball, in_zone, trail)
                if corners is not None:
                    cv2.polylines(frame, [corners.reshape(-1, 2).astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
                effects.draw(frame, now)

                draw_scoreboard(frame, level, streak, record)
                draw_swing_meter(frame, swing)
                draw_status_bar(frame, paddle_pose, hand_side, paddle.connected, scores.ok)

                tag_progress = progress if tag_id is not None and progress < 1.0 else None
                if state == WAIT_LEVEL:
                    draw_banner(frame, "LEGO PING-PONG", "Show tag 1 / 2 / 3  for  Easy / Medium / Hard",
                                GOLD, tag_progress)
                elif state == WAIT_START:
                    draw_banner(frame, level["name"], "Show tag 0 to start", LEVEL_COLORS[level["name"]],
                                tag_progress)
                elif state == COUNTDOWN:
                    remaining = max(1, math.ceil(COUNTDOWN_S - (now - state_t)))
                    draw_banner(frame, str(remaining), "Hold the paddle still", GOLD, title_scale=3.0)
                elif state == SERVE:
                    draw_banner(frame, "Ready...", title_scale=1.0, y=int(0.62 * h))
                elif state == MISSED:
                    draw_banner(frame, "MISS", f"Rally {streak}   -   Best {record}", RED, title_scale=2.0)

                cv2.imshow("LEGO Ping-Pong", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    swing.force_swing(now)
                elif key == ord("r"):
                    level = None
                    tag_hold = TagHold()
                    paddle.light(le.LEGO_COLOR_BLUE)
                    set_state(WAIT_LEVEL)
                elif key == ord("h"):
                    hand_side = "left" if hand_side == "right" else "right"
                    print(f"Paddle hand: {hand_side}")
                elif key == ord("["):
                    swing.threshold = max(20.0, swing.threshold - 20)
                elif key == ord("]"):
                    swing.threshold += 20
    finally:
        print(f"Final record: {record} continuous hits")
        cap.release()
        cv2.destroyAllWindows()
        paddle.close()
        scores.close()


if __name__ == "__main__":
    main()
