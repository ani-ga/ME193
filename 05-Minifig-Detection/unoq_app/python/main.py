"""UnoQ side (Linux/Python half): listens for where the laptop sees our
minifig and decides how to drive.

Policy, run 20 times a second:
  - error = x - TARGET_X   (both 0..1 across the laptop camera image;
    TARGET_X = 0.5 is the middle of the screen)
  - |error| <= DEAD_ZONE                -> stop, we've arrived
  - otherwise speed = KP * error, clamped to MIN_SPEED..MAX_SPEED in size
    (MIN_SPEED so the motors don't stall just short of the target),
    and the sign picks forward or backward
  - no minifig seen, or no message for LOST_TIMEOUT_S -> stop

The speed and the dot position are passed to the microcontroller half
(sketch/sketch.ino) over the Bridge; it drives the motors and draws a dot
on the LED matrix at the scaled position.
"""

import json
import threading
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

TEAM = "RARA"
BROKER = "test.mosquitto.org"
TOPIC = f"ME193/{TEAM}/minifig"

TARGET_X = 0.5         # stop in the middle of the laptop screen
DEAD_ZONE = 0.04       # stop when within 4% of the image width of the target
KP = 250.0             # speed per unit of error: 0.4 off -> full speed
MIN_SPEED = 35         # weakest command (percent) that still turns the wheels
MAX_SPEED = 100        # percent
DIRECTION_SIGN = 1     # flip to -1 if the car drives away from the target
LOST_TIMEOUT_S = 0.5   # stop if the laptop goes quiet this long

MATRIX_COLS = 13
MATRIX_ROWS = 8

latest = {"msg": {"seen": False}, "time": 0.0}
lock = threading.Lock()
last_sent = {"speed": None, "dot": None}


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(TOPIC)
    print(f"Subscribed to {TOPIC}")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
    except ValueError:
        print(f"Ignoring bad message: {msg.payload!r}")
        return
    with lock:
        latest["msg"] = data
        latest["time"] = time.time()


def decide_speed(msg, age):
    """The policy: turn the latest position into a motor speed (-100..100)."""
    if age > LOST_TIMEOUT_S or not msg.get("seen"):
        return 0
    error = msg["x"] - TARGET_X
    if abs(error) <= DEAD_ZONE:
        return 0
    magnitude = min(MAX_SPEED, max(MIN_SPEED, abs(KP * error)))
    return int(DIRECTION_SIGN * (magnitude if error > 0 else -magnitude))


def dot_position(msg, age):
    """Scale (x, y) to an LED (col, row), or (-1, -1) to clear the matrix."""
    if age > LOST_TIMEOUT_S or not msg.get("seen"):
        return (-1, -1)
    col = min(MATRIX_COLS - 1, max(0, round(msg["x"] * (MATRIX_COLS - 1))))
    row = min(MATRIX_ROWS - 1, max(0, round(msg["y"] * (MATRIX_ROWS - 1))))
    return (col, row)


def loop():
    with lock:
        msg, age = latest["msg"], time.time() - latest["time"]

    speed = decide_speed(msg, age)
    # Resend the speed even when unchanged: the sketch stops the motors if it
    # hears nothing for a second, in case this program crashes.
    Bridge.call("drive", speed)
    if speed != last_sent["speed"]:
        print(f"speed {speed}  ({msg})")
        last_sent["speed"] = speed

    dot = dot_position(msg, age)
    if dot != last_sent["dot"]:
        Bridge.call("dot", dot[0], dot[1])
        last_sent["dot"] = dot

    time.sleep(0.05)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_start()

App.run(user_loop=loop)
