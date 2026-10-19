"""UnoQ side (Linux/Python half): the robot's tape sensor.

Connects to the LEGO Color Sensor on the back of the robot over Bluetooth
and, 20 times a second, publishes {"edge": 1, "color": 3, "reflection": 12}
to TOPIC. edge = 1 when the sensor sees the blue tape. The laptop's
sumo_controller.py uses it to force the robot forward off the edge.

Runs in App Lab, or straight from a UnoQ shell:
    python3 main.py
If the sensor disconnects, nothing is published (the laptop warns and
ignores the edge) and this program reconnects.

Only one computer can be connected to the Color Sensor at a time, so close
test_color_sensor.py on the laptop first.
"""

import json
import math
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # show prints in App Lab's log right away
print("Sumo tape sensor starting")

import legoeducation as le  # noqa: E402
import paho.mqtt.client as mqtt

try:
    from arduino.app_utils import App
except ImportError:
    App = None  # running from a shell instead of App Lab

BROKER = "test.mosquitto.org"
TOPIC = "ME193/sumo/ani/edge"  # must match EDGE_TOPIC in sumo_controller.py

# Color Sensor's Connection Card
SENSOR_CARD_COLOR = le.LEGO_COLOR_RED
SENSOR_CARD_SERIAL = "1142"

# Raw color codes that count as tape (3 = blue). Add 4 (azure) if the tape
# reads as light blue.
TAPE_COLORS = {3}

state = {"sensor": None, "last_print": 0.0, "last_edge": None}


def connect_sensor():
    sensor = le.ColorSensor()
    print(f"Connecting to Color Sensor (card {SENSOR_CARD_SERIAL})...")
    sensor.connect(card_color=SENSOR_CARD_COLOR, card_serial=SENSOR_CARD_SERIAL)
    if not sensor.connected:
        print("Couldn't connect to the Color Sensor; retrying. Is it on, and not connected to the laptop?")
        time.sleep(2)
        return None
    print("Color Sensor connected")
    return sensor


def loop():
    sensor = state["sensor"]
    if sensor is None or not sensor.connected:
        state["sensor"] = connect_sensor()
        return

    reading = sensor.sensor
    if math.isnan(float(reading.color)):
        time.sleep(0.05)  # no reading yet
        return

    color = int(reading.color)
    edge = int(color in TAPE_COLORS)
    msg = {"edge": edge, "color": color, "reflection": int(reading.reflection)}
    client.publish(TOPIC, json.dumps(msg))

    now = time.time()
    if edge != state["last_edge"] or now - state["last_print"] > 1.0:
        print("TAPE " if edge else "floor", msg)
        state["last_edge"] = edge
        state["last_print"] = now

    time.sleep(0.05)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, 1883)
client.loop_start()

if App is not None:
    App.run(user_loop=loop)
else:
    try:
        while True:
            loop()
    except KeyboardInterrupt:
        pass
    finally:
        if state["sensor"] is not None:
            state["sensor"].disconnect()
