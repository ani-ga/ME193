"""Linux side of the safe. Runs ON the UNO Q: webcam -> crop -> CNN (numpy) -> lock policy.
The sketch does the physical work (button, servo) and is reached over the Bridge.

Flow: press the button to start -> show digit 1, press the button to lock it in -> repeat for
digits 2 and 3 -> right code = servo to 180 degrees."""
import time
from pathlib import Path

from arduino.app_utils import App, Bridge, Logger
from arduino.app_peripherals.camera import Camera
from arduino.app_bricks.web_ui import WebUI
from fastapi.responses import Response
import cv2

import safe_config as config
import preprocess
from np_net import NumpyFingerNet
from lock_policy import LockPolicy

logger = Logger("Safe")

AUTO_ARM = False   # True = re-arm automatically (use if the button isn't wired yet)

net = NumpyFingerNet(Path(__file__).resolve().parent / "fingernet.npz")


def bridge(name):
    """Call a sketch function; log (don't crash) if the MCU doesn't answer."""
    try:
        return Bridge.call(name)
    except Exception as e:
        logger.error(f"Bridge call {name} failed: {e}")
        return None


# What the web page shows (written by loop(), read by the web handlers below).
view = {"pred": None, "conf": 0.0, "msg": "Press the button (or Start on this page), then show digit 1.",
        "jpeg": b"", "fps": 0.0, "infer_ms": 0.0, "web_press": False}


def on_open():
    logger.info("CODE CORRECT -> UNLOCKED, servo opening")
    view["msg"] = "Code correct!"
    bridge("open_door")


def on_close():
    logger.info("locking, servo closing")
    view["msg"] = "Locked again."
    bridge("close_door")


def on_digit(d, n):
    logger.info(f"accepted digit {n}: {d}")
    view["msg"] = f"Locked in digit {n} of {len(config.LOCK_CODE)}: {d}. Show the next digit, then press the button."


policy = LockPolicy(on_open=on_open, on_close=on_close, on_digit=on_digit)

# Same webcam, same resolution as on the laptop.
camera = Camera(resolution=(config.CAM_WIDTH, config.CAM_HEIGHT), fps=15)
camera.start()
logger.info(f"Safe ready. Code length {len(config.LOCK_CODE)}. Press the button to start, then press it again to lock in each digit.")

# ---- web page (served from the board on port 7000) ----
web_ui = WebUI(assets_dir_path="/app/python/assets")


def get_state():
    return {"policy": policy.state, "digits": policy.digits, "code_len": len(config.LOCK_CODE),
            "pred": view["pred"], "conf": view["conf"], "msg": view["msg"],
            "fps": view["fps"], "infer_ms": view["infer_ms"]}


def get_frame():
    return Response(content=view["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def web_start():
    view["web_press"] = True            # loop() acts on it, so everything stays on one thread
    return {"ok": True}


def set_code(code: str):
    digits = [int(c) for c in code if c in "12345"]
    if len(digits) != 3 or len(set(digits)) != 3:
        return {"error": "Need three DIFFERENT digits from 1-5"}
    config.LOCK_CODE = tuple(digits)    # the policy reads this each time it checks
    view["msg"] = "Code set. Press Start."
    return {"ok": True}


web_ui.expose_api("GET", "/state", get_state)
web_ui.expose_api("GET", "/frame.jpg", get_frame)
web_ui.expose_api("POST", "/start", web_start)
web_ui.expose_api("POST", "/set_code", set_code)

stats = {"frames": 0, "infer_ms": 0.0, "t0": time.time(), "state": None}


def loop():
    # 1. button -> arm the lock
    if bridge("button_pressed") == 1 or view["web_press"] or (AUTO_ARM and policy.state == "IDLE"):
        view["web_press"] = False
        was_idle = policy.state == "IDLE"
        result = policy.press_button()      # idle: arm.  armed: lock in the digit on show.
        if was_idle and policy.state == "ARMED":
            logger.info("armed: show digit 1, then press the button")
            view["msg"] = "Armed: show digit 1, then press the button to lock it in."
        elif result is False:
            logger.info("button pressed but no clear digit")
            view["msg"] = "I can't see a clear number. Hold your fingers steady and press again."

    # 2. camera -> network -> policy
    frame = camera.capture()
    if frame is None:
        time.sleep(0.05)
        return
    t = time.perf_counter()
    digit, conf = net.predict(preprocess.frame_to_array(frame))
    stats["infer_ms"] += (time.perf_counter() - t) * 1000
    stats["frames"] += 1
    view["pred"], view["conf"] = digit, conf
    failed_before = policy.failed
    policy.update(digit, conf)
    if policy.failed > failed_before:
        view["msg"] = (f"Too many wrong codes: LOCKED OUT." if policy.state == "LOCKOUT"
                       else f"Wrong code. Still locked. Attempt {policy.failed}/{config.MAX_ATTEMPTS}.")

    # picture for the web page: the frame with the crop box drawn on it
    x, y, w, h = config.load_crop()
    shown = frame.copy()
    cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
    ok, jpg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        view["jpeg"] = jpg.tobytes()

    # 3. log state changes, and FPS / latency every 5 s (for the computer-vs-UNO Q comparison)
    if policy.state != stats["state"]:
        stats["state"] = policy.state
        logger.info(f"state: {policy.state}")
    if time.time() - stats["t0"] >= 5:
        n = max(stats["frames"], 1)
        logger.info(f"{stats['frames'] / 5:.1f} fps, inference {stats['infer_ms'] / n:.1f} ms, "
                    f"seeing {digit} ({conf:.2f}), digits so far {policy.digits}")
        view["fps"], view["infer_ms"] = stats["frames"] / 5, stats["infer_ms"] / n
        stats.update(frames=0, infer_ms=0.0, t0=time.time())


App.run(user_loop=loop)
