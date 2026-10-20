"""
Sumo wrestler controller (runs on the computer).

Inputs:
  - Xbox controller paired with this computer over Bluetooth (teleoperation)
  - Tape detection from the Arduino UnoQ over MQTT (autonomous): the UnoQ
    reads the LEGO Color Sensor on the back of the robot and publishes
    {"edge": 1, "color": 3, "reflection": 12} to EDGE_TOPIC
Output:
  - LEGO Double Motor speeds (left, right)
  - LEGO Single Motor on the front (spins while X or Y is held)

Controls:
  left stick          drive (x steers, y throttle)
  A or right trigger  CHARGE
  hold B LONG_PRESS s OUT
  hold X / Y          spin the front motor forward / backward

Decision tree (first match wins):
  1. OUT            B held >= LONG_PRESS          -> stop, sad song, end
  2. FAILSAFE_STOP  Xbox controller disconnected  -> stop
  3. ESCAPE_SHARED  UnoQ says the tape is behind us -> forced forward, driver steers
  4. CHARGE         A or right trigger            -> full forward, driver steers
  5. TELEOP         otherwise                     -> arcade drive

If the UnoQ goes quiet for EDGE_TIMEOUT, the robot keeps driving on the
controller alone and a warning is printed.

Usage:
    python sumo_controller.py                  # connect motor, fight
    python sumo_controller.py --dry-run        # no LEGO hardware, prints motor commands
    python sumo_controller.py --show-input     # print controller + UnoQ readings only
    python sumo_controller.py --send-test      # publish fake tape messages (no UnoQ needed)

Install:  pip install paho-mqtt legoeducation pygame-ce
"""

import argparse
import json
import os
import threading
import time

from mqttlib import MQTTClient

# ---------------- CONFIG ----------------
BROKER = "test.mosquitto.org"  # or your class broker's IP
EDGE_TOPIC = "ME193/sumo/ani/edge"  # must match what the UnoQ publishes
LOOP_HZ = 20                  # decision rate
DEADZONE = 0.15               # ignore tiny stick drift
EDGE_TIMEOUT = 1.0            # seconds without a UnoQ message -> warn, ignore the edge
LONG_PRESS = 1.5              # seconds holding B -> "we're out"
TRIGGER_ON = 0.5              # right trigger pressed past half = charge
ESCAPE_POWER = 100            # forward power when tape is seen behind us
STEER_GAIN = 50               # how much stick x bends ESCAPE / CHARGE
TELEOP_SPEED = 50             # top speed (%) when driving with the stick; CHARGE and ESCAPE stay at 100

# Connection Card for the Double Motor
MOTOR_CARD_COLOR = "orange"
MOTOR_CARD_SERIAL = "1142"

# Connection Card for the front Single Motor. It shares the Double Motor's
# card; the library also matches the device type, so they don't get mixed up.
FRONT_CARD_COLOR = "orange"
FRONT_CARD_SERIAL = "1142"
FRONT_SPEED = 100             # % while X / Y is held; flip the sign if X spins it backward

# The two motors are mounted mirrored on the chassis (same as
# track_apriltag.py). Swap a -1 if the robot spins when it should go straight,
# or flip GLOBAL_DIRECTION_SIGN if forward and backward are swapped.
LEFT_SIGN = 1
RIGHT_SIGN = -1
GLOBAL_DIRECTION_SIGN = 1
# True if the robot turns left when the stick goes right (the Double Motor's
# "left" port is on the robot's right side).
SWAP_SIDES = True

# (frequency Hz, seconds); wah wah wah waaah, as in world_cup.py
SAD_SONG = [(494, 0.45), (466, 0.45), (440, 0.45), (415, 1.3)]
SONG_VOLUME = 0.4             # laptop speaker volume, 0..1


def start_song_on_laptop(song):
    """Start playing (Hz, seconds) notes through the laptop speakers; returns
    a function that waits for the song to finish."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        print("(install numpy + sounddevice to hear the song on the laptop)")
        return lambda: None

    rate = 44100
    parts = []
    for freq, seconds in song:
        t = np.arange(int(seconds * rate)) / rate
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)  # 10 ms fades, no clicks
        parts.append(np.sin(2 * np.pi * freq * t) * fade)
    try:
        sd.play((SONG_VOLUME * np.concatenate(parts)).astype(np.float32), rate)
    except Exception as exc:  # no audio device, etc.
        print(f"Could not play the song on the laptop: {exc}")
        return lambda: None
    return sd.wait


# ---------------- ROBOT INTERFACE ----------------
class Robot:
    """LEGO Education Double Motor (drive) + Single Motor (front) over Bluetooth."""

    def __init__(self):
        import legoeducation as le

        self.le = le
        self.last_motors = None
        self.last_front = None
        self.motor = le.DoubleMotor()
        print(f"Connecting to Double Motor (card {MOTOR_CARD_SERIAL})...")
        self.motor.connect(card_color=getattr(le, f"LEGO_COLOR_{MOTOR_CARD_COLOR.upper()}"),
                           card_serial=MOTOR_CARD_SERIAL)
        if not self.motor.connected:
            raise SystemExit("Error connecting to Double Motor. Check it's on and the card matches.")

        # The front motor is extra: if it doesn't connect, drive without it.
        self.front = le.SingleMotor()
        print(f"Connecting to front Single Motor (card {FRONT_CARD_SERIAL})...")
        self.front.connect(card_color=getattr(le, f"LEGO_COLOR_{FRONT_CARD_COLOR.upper()}"),
                           card_serial=FRONT_CARD_SERIAL)
        if not self.front.connected:
            print("WARNING: front Single Motor didn't connect -- driving without it")
            self.front = None

    def set_motors(self, left, right):
        # left/right in -100..100; only send when they change to keep Bluetooth quiet
        left, right = int(left), int(right)
        if (left, right) == self.last_motors:
            return
        self.last_motors = (left, right)
        le = self.le
        if left == 0 and right == 0:
            self.motor.motor_stop(motor=le.MOTOR_BOTH, blocking=False)
            return
        if SWAP_SIDES:
            left, right = right, left
        self.motor.motor_run(motor=le.MOTOR_LEFT, speed=GLOBAL_DIRECTION_SIGN * LEFT_SIGN * left, blocking=False)
        self.motor.motor_run(motor=le.MOTOR_RIGHT, speed=GLOBAL_DIRECTION_SIGN * RIGHT_SIGN * right, blocking=False)

    def set_front(self, speed):
        # speed in -100..100; only send when it changes
        speed = int(speed)
        if speed == self.last_front or self.front is None:
            return
        self.last_front = speed
        if speed == 0:
            self.front.motor_stop(blocking=False)
        else:
            self.front.motor_run(speed=speed, blocking=False)

    def play_sad_song(self):
        """Song on the laptop speakers, with the robot beeping along."""
        print("sad trombone")
        wait = start_song_on_laptop(SAD_SONG)
        for freq, seconds in SAD_SONG:
            start = time.time()
            try:
                self.motor.beep(frequency=freq, blocking=True)
            except Exception as exc:
                print(f"Robot beep failed: {exc}")
            time.sleep(max(0.0, seconds - (time.time() - start)))
        wait()

    def close(self):
        self.motor.disconnect()
        if self.front is not None:
            self.front.disconnect()


class FakeRobot:
    """Stands in for the LEGO hardware with --dry-run."""

    def __init__(self):
        self.last_motors = None
        self.last_front = None

    def set_motors(self, left, right):
        if (int(left), int(right)) != self.last_motors:
            self.last_motors = (int(left), int(right))
            print(f"motors L={left:4.0f} R={right:4.0f}")

    def set_front(self, speed):
        if int(speed) != self.last_front:
            self.last_front = int(speed)
            print(f"front motor {speed:4.0f}")

    def play_sad_song(self):
        print("sad trombone")
        start_song_on_laptop(SAD_SONG)()

    def close(self):
        pass


# ---------------- XBOX CONTROLLER ----------------
class XboxController:
    """Reads the first game controller paired with this computer, using SDL's
    standard Xbox layout so the stick and button numbers don't need guessing."""

    def __init__(self):
        os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"  # read it without a window
        import pygame
        from pygame._sdl2 import controller

        self.pg = pygame
        self.sdl = controller
        pygame.init()
        controller.init()
        self.pad = None
        self.b_since = None
        self.last_search = 0.0

    def _connect(self):
        if time.time() - self.last_search < 1.0:
            return
        self.last_search = time.time()
        for i in range(self.sdl.get_count()):
            if self.sdl.is_controller(i):
                self.pad = self.sdl.Controller(i)
                print(f"Controller connected: {self.pad.name}")
                return

    def snapshot(self):
        pg = self.pg
        pg.event.pump()
        if self.pad is not None and not self.pad.attached():
            print("Controller disconnected")
            self.pad = None
        if self.pad is None:
            self._connect()
        if self.pad is None:
            self.b_since = None
            return {"x": 0.0, "y": 0.0, "button": False, "out": False, "front": 0, "stale": True}

        x = self._dz(self.pad.get_axis(pg.CONTROLLER_AXIS_LEFTX) / 32767)
        y = self._dz(-self.pad.get_axis(pg.CONTROLLER_AXIS_LEFTY) / 32767)  # stick up is negative
        trigger = self.pad.get_axis(pg.CONTROLLER_AXIS_TRIGGERRIGHT) / 32767
        charge = bool(self.pad.get_button(pg.CONTROLLER_BUTTON_A)) or trigger > TRIGGER_ON

        if self.pad.get_button(pg.CONTROLLER_BUTTON_B):
            self.b_since = self.b_since or time.time()
        else:
            self.b_since = None
        out = self.b_since is not None and time.time() - self.b_since >= LONG_PRESS

        # Front motor: X = forward, Y = backward, neither (or both) = stop
        front = (int(bool(self.pad.get_button(pg.CONTROLLER_BUTTON_X)))
                 - int(bool(self.pad.get_button(pg.CONTROLLER_BUTTON_Y))))

        return {"x": x, "y": y, "button": charge, "out": out, "front": front, "stale": False}

    @staticmethod
    def _dz(v):
        v = max(-1.0, min(1.0, v))
        return 0.0 if abs(v) < DEADZONE else v


# ---------------- TAPE SENSOR (UnoQ over MQTT) ----------------
class EdgeSensor:
    """Holds the latest tape reading from the UnoQ, updated from the MQTT thread."""

    def __init__(self):
        self.edge = False
        self.color = None
        self.reflection = None
        self.last_msg = 0.0
        self.lock = threading.Lock()

    def on_message(self, topic, payload):
        try:
            data = json.loads(payload)
            with self.lock:
                self.edge = bool(data.get("edge", 0))
                self.color = data.get("color")
                self.reflection = data.get("reflection")
                self.last_msg = time.time()
        except (ValueError, TypeError, AttributeError):
            pass  # ignore malformed messages

    def alive(self):
        with self.lock:
            return time.time() - self.last_msg <= EDGE_TIMEOUT

    def edge_seen(self):
        with self.lock:
            return self.edge and time.time() - self.last_msg <= EDGE_TIMEOUT


# ---------------- DECISION TREE ----------------
def decide(edge_seen, joy, is_out):
    """Returns (action_name, left, right)."""
    # Node 1: have we lost?
    if is_out or joy["out"]:
        return ("OUT", 0, 0)

    # Node 2: is the operator's controller connected?
    if joy["stale"]:
        return ("FAILSAFE_STOP", 0, 0)

    steer = joy["x"] * STEER_GAIN

    # Node 3: is the tape behind us? -> shared control
    if edge_seen:
        return ("ESCAPE_SHARED", ESCAPE_POWER + steer, ESCAPE_POWER - steer)

    # Node 4: is the operator asking to charge?
    if joy["button"]:
        return ("CHARGE", 100 + steer, 100 - steer)

    # Node 5: default -> follow the stick (arcade drive)
    fwd = joy["y"] * TELEOP_SPEED
    turn = joy["x"] * TELEOP_SPEED
    return ("TELEOP", fwd + turn, fwd - turn)


def clamp(v):
    return max(-100, min(100, v))


def connect_edge_sensor():
    edge = EdgeSensor()
    mqtt = MQTTClient(broker=BROKER)
    mqtt.connect()
    mqtt.subscribe(EDGE_TOPIC, edge.on_message)
    print(f"Listening for the UnoQ on {BROKER} {EDGE_TOPIC}")
    return edge, mqtt


# ---------------- MODES ----------------
def show_input():
    """Print what the controller and the UnoQ are sending, without driving."""
    pad = XboxController()
    edge, mqtt = connect_edge_sensor()
    print("Move the sticks and slide tape under the sensor. Ctrl+C to stop.")
    try:
        while True:
            joy = pad.snapshot()
            unoq = (f"edge={int(edge.edge)} color={edge.color} reflection={edge.reflection}"
                    if edge.alive() else "UnoQ: no messages")
            pad_text = ("controller: not connected" if joy["stale"] else
                        f"x={joy['x']:+.2f} y={joy['y']:+.2f} charge={int(joy['button'])} "
                        f"out={int(joy['out'])} front={joy['front']:+d}")
            print(f"{pad_text}   {unoq}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        mqtt.disconnect()


def send_test():
    """Publish fake tape readings (tape every few seconds) so the controller
    can be tested without the UnoQ."""
    with MQTTClient(broker=BROKER) as mqtt:
        print(f"Publishing fake tape readings to {EDGE_TOPIC}. Ctrl+C to stop.")
        t0 = time.time()
        try:
            while True:
                edge = int((time.time() - t0) % 4 > 3)  # 1 s of tape every 4 s
                mqtt.publish(EDGE_TOPIC, json.dumps({"edge": edge, "color": 3 if edge else 10, "reflection": 0}))
                time.sleep(1 / LOOP_HZ)
        except KeyboardInterrupt:
            pass


# ---------------- MAIN LOOP ----------------
def run(dry_run):
    robot = FakeRobot() if dry_run else Robot()
    pad = XboxController()
    edge, mqtt = connect_edge_sensor()
    is_out = False
    last_action = None
    unoq_alive = None
    start = time.time()

    try:
        while True:
            joy = pad.snapshot()
            action, left, right = decide(edge.edge_seen(), joy, is_out)

            # Give the UnoQ a moment to be heard before warning that it's missing.
            alive = edge.alive()
            if alive != unoq_alive and (alive or time.time() - start > 2 * EDGE_TIMEOUT):
                unoq_alive = alive
                print("UnoQ tape sensor online" if unoq_alive else
                      "WARNING: no tape readings from the UnoQ -- edge escape is off")

            if action != last_action:
                print("->", action)
                last_action = action

            if action == "OUT":
                robot.set_motors(0, 0)
                robot.set_front(0)
                robot.play_sad_song()
                break

            robot.set_motors(clamp(left), clamp(right))
            # The front motor is the driver's alone, outside the decision tree;
            # it stops with everything else if the controller drops.
            robot.set_front(0 if action == "FAILSAFE_STOP" else joy["front"] * FRONT_SPEED)
            time.sleep(1 / LOOP_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        robot.set_motors(0, 0)
        robot.set_front(0)
        mqtt.disconnect()
        robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="no LEGO hardware; print motor commands")
    parser.add_argument("--show-input", action="store_true", help="print controller + UnoQ readings only")
    parser.add_argument("--send-test", action="store_true", help="publish fake tape messages")
    args = parser.parse_args()

    if args.show_input:
        show_input()
    elif args.send_test:
        send_test()
    else:
        run(args.dry_run)


if __name__ == "__main__":
    main()
