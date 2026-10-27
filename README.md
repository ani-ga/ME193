# ME193 - AI for Mobile Robotics (Tufts, Fall 2026)

My work for ME193, one folder per assignment, roughly in the order we did them.

| # | Assignment | What it does |
|---|---|---|
| 01 | [Image Thresholding](01-Image-Thresholding/) | Converts a photo to black and white with live sliders for threshold and morphology (erode, dilate, open, close, gradient). |
| 02 | [AprilTags](02-AprilTags/) | `apriltag.py` generates printable AprilTags at an exact physical size; `track_apriltag.py` drives a LEGO Double Motor robot to keep a tag centered in the webcam view. |
| 03 | [MQTT](03-MQTT/) | A small `paho-mqtt` wrapper (`mqttlib.py`) and a publish/subscribe test against `test.mosquitto.org`. |
| 04 | [Whistle Spectrogram](04-Whistle-Spectrogram/) | Records audio from the microphone and plots its waveform, spectrum, and spectrogram. |
| 05 | [Minifig Detection](05-Minifig-Detection/) | YOLOv8 trained on photos of a green LEGO minifig; the laptop tracks it and sends its position over MQTT to an Arduino UnoQ car. Includes the dataset and training runs. |
| 06 | [Ping-Pong](06-Ping-Pong/) | Virtual ping-pong with the Double Motor as the paddle: MediaPipe Pose, IMU swing detection, AprilTag menus, haptic feedback, and the record streak published over MQTT. |
| 07 | [Sumo Wrestler](07-Sumo/) | LEGO sumo robot with shared control: Xbox controller driving plus a UnoQ-read Color Sensor that forces the robot off the ring's edge, combined in a decision tree. |
| 08 | [Activity Tracker](08-Activity-Tracker/) | Arduino UnoQ + MPU6050 on the ankle classifies standing / walking / running / biking / hopping with a threshold model and a hand-written KNN, served as a phone web page with an animated stick-figure bunny and live motion graphs. |

## Setup

Each folder runs on its own. Most use some of:

```bash
pip install opencv-contrib-python numpy matplotlib paho-mqtt legoeducation mediapipe sounddevice ultralytics
```

Folders with extra steps have their own README. Notes:

- **01:** `pic.py` expects a photo named `IMG_5534 (1).jpg` next to it (not included) - change `IMAGE_PATH` to any image.
- **02 / 06:** set `CARD_COLOR` / `CARD_SERIAL` to match your Double Motor's connection card.
