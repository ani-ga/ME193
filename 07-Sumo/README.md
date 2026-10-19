# Sumo Wrestler

A LEGO sumo robot that fights inside a ring of blue tape. The driver steers with an
Xbox controller; a Color Sensor on the back, read by an Arduino UnoQ, spots the tape
and forces the robot forward off the edge while the driver keeps steering. If the
robot is pushed out, the driver holds B and it stops and plays a sad song.

| Part | Runs on | Role |
| --- | --- | --- |
| `sumo_controller.py` | Laptop | Reads the Xbox controller, gets tape readings over MQTT, runs the decision tree, drives the Double Motor (orange card, 1142) |
| `unoq_app/python/main.py` | UnoQ (Linux shell) | Connects to the Color Sensor (red card, 1142) over Bluetooth and publishes `{"edge", "color", "reflection"}` to `ME193/sumo/ani/edge` 20 times a second |
| `test_color_sensor.py` | Laptop | Tests the Color Sensor against the tape on its own |

## Decision tree

Checked 20 times a second; the first match wins.

| Node | Condition | Action | Motors (left, right) |
| --- | --- | --- | --- |
| 1 | B held 1.5 s | OUT: stop, sad song, end | 0, 0 |
| 2 | Xbox controller disconnected | FAILSAFE_STOP | 0, 0 |
| 3 | UnoQ sees the tape behind us | ESCAPE_SHARED: forced forward, driver steers | 100 + 50x, 100 - 50x |
| 4 | A or right trigger | CHARGE: full forward, driver steers | 100 + 50x, 100 - 50x |
| 5 | Otherwise | TELEOP: arcade drive | 100y + 100x, 100y - 100x |

If the UnoQ goes quiet, the robot keeps driving on the controller alone and prints a warning.

## Running it

1. Pair the Xbox controller with the laptop (System Settings > Bluetooth).
2. On the UnoQ, from a shell (not App Lab: App Lab's container can't reach Bluetooth):

   ```bash
   # once
   python3 -m venv --without-pip ~/sumo-env
   curl -fsSL https://bootstrap.pypa.io/get-pip.py | ~/sumo-env/bin/python
   ~/sumo-env/bin/python -m pip install paho-mqtt legoeducation==1.1.1
   # copy main.py over from the laptop: adb push unoq_app/python/main.py /home/arduino/sumo_main.py

   # every time
   ~/sumo-env/bin/python ~/sumo_main.py
   ```

3. On the laptop:

   ```bash
   pip install paho-mqtt legoeducation pygame-ce
   python sumo_controller.py               # fight
   python sumo_controller.py --show-input  # check the controller + UnoQ readings
   python sumo_controller.py --dry-run     # no robot, prints motor commands
   ```

| Control | Does |
| --- | --- |
| Left stick | Drive (x steers, y throttle) |
| A or right trigger | Charge |
| Hold B 1.5 s | Out |
