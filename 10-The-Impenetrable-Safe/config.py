"""All settings for FingerLock in one place. Edit here, not in the other files."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
CROP_FILE = ROOT / "crop.json"          # written by scripts/find_crop.py

# ---- camera / input ----
CAMERA_INDEX = 0                         # USB webcam (icspring). 1 = iPhone Continuity cam, 2 = FaceTime - do NOT use
CAM_WIDTH, CAM_HEIGHT = 640, 480         # same resolution on laptop and UNO Q (faster, smaller)
IMG_SIZE = 64                            # model input is IMG_SIZE x IMG_SIZE, grayscale
NUM_CLASSES = 6                          # 0 = no hand / fist, 1..5 = fingers

def load_crop():
    """Red-square crop as (x, y, w, h) in camera pixels."""
    if CROP_FILE.exists():
        return tuple(json.loads(CROP_FILE.read_text())["crop"])
    return (220, 140, 200, 200)          # placeholder until find_crop.py is run

# ---- lock policy ----
LOCK_CODE = (2, 5, 3)                    # placeholder: three DIFFERENT digits from 1-5
BUTTON_CONFIRMS_DIGITS = True            # True: press the button to lock in each digit (no need to drop your hand)
BUTTON_MIN_FRAMES = 4                    # ...but the same number must have been seen this many frames in a row
STABLE_FRAMES = 12                       # (hold-to-accept mode only) same answer this many frames in a row
CONF_THRESHOLD = 0.85                    # softmax confidence needed
TIMEOUT_SECONDS = 30                     # give up after arming
MAX_ATTEMPTS = 3                         # wrong codes before lockout
LOCKOUT_SECONDS = 30
OPEN_SECONDS = 8                         # door stays open this long

# ---- training ----
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 0

# ---- UNO Q pins / servo (MG995: 180 deg positional) ----
PIN_SERVO, PIN_BUTTON = 9, 2
PIN_LEDS = (5, 6, 7)
SERVO_PULSE_US = (500, 2500)             # MG995 range; adjust if it doesn't reach 180
LOCKED_ANGLE, OPEN_ANGLE = 0, 180
