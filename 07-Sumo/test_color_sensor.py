"""Test the back Color Sensor against the blue tape from the laptop, on its
own (no motors, no MQTT). Uses the same card and TAPE_COLORS as the UnoQ's
unoq_app/python/main.py, so what you see here is what the robot will decide.

Slide the sensor over the tape and the floor, about 1 cm above them. Each
reading prints with TAPE or floor, and the sensor beeps when it first sees
the tape. Ctrl+C prints a summary of the colors and reflection seen.

Usage:
    python test_color_sensor.py
    python test_color_sensor.py --card-color blue --card-serial 1234
"""

import argparse
import math
import time
from collections import Counter

import legoeducation as le

# Same values as unoq_app/python/main.py
SENSOR_CARD_COLOR = "red"
SENSOR_CARD_SERIAL = "1142"
TAPE_COLORS = {3}

# Raw color codes the sensor reports
COLOR_NAMES = {-1: "none", 0: "black", 1: "magenta", 2: "purple", 3: "blue",
               4: "azure", 5: "turquoise", 6: "green", 7: "yellow",
               8: "orange", 9: "red", 10: "white"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card-color", default=SENSOR_CARD_COLOR)
    parser.add_argument("--card-serial", default=SENSOR_CARD_SERIAL)
    args = parser.parse_args()

    sensor = le.ColorSensor()
    print(f"Connecting to Color Sensor (card {args.card_color} {args.card_serial})...")
    sensor.connect(card_color=getattr(le, f"LEGO_COLOR_{args.card_color.upper()}"),
                   card_serial=args.card_serial)
    if not sensor.connected:
        raise SystemExit("Error connecting to Color Sensor. Check it's on and the card matches.")

    tape_names = ", ".join(COLOR_NAMES.get(c, str(c)) for c in sorted(TAPE_COLORS))
    print(f"Connected. Counting {tape_names} as tape. Ctrl+C to stop.\n")

    seen = {"TAPE": Counter(), "floor": Counter()}
    reflections = {"TAPE": [], "floor": []}
    was_tape = False
    try:
        while True:
            reading = sensor.sensor
            if math.isnan(float(reading.color)):
                time.sleep(0.1)  # no reading yet
                continue
            color = int(reading.color)
            tag = "TAPE" if color in TAPE_COLORS else "floor"
            print(f"{tag:5}  color={color:2d} {COLOR_NAMES.get(color, '?'):9}"
                  f"  reflection={reading.reflection:3d}%"
                  f"  hue={reading.hue:3d} sat={reading.saturation:3d} val={reading.value:3d}")

            seen[tag][COLOR_NAMES.get(color, str(color))] += 1
            reflections[tag].append(reading.reflection)
            if tag == "TAPE" and not was_tape:
                sensor.beep(frequency=880, blocking=False)
            was_tape = tag == "TAPE"
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.disconnect()

    print("\nSummary")
    for tag in ("TAPE", "floor"):
        values = reflections[tag]
        if values:
            print(f"  {tag:5}  {len(values)} readings, colors {dict(seen[tag])}, "
                  f"reflection {min(values)}-{max(values)}%")
        else:
            print(f"  {tag:5}  no readings")


if __name__ == "__main__":
    main()
