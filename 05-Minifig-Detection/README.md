# Minifig Detection: YOLO + MQTT + UnoQ

A laptop webcam watches a car carrying a green LEGO minifig. A YOLOv8 model
we trained finds the minifig, and the laptop publishes its position over
MQTT. The UnoQ on the car shows the position as a blue dot on its LED matrix,
and drives forward or backward until the minifig reaches the middle of the
screen.

```
 webcam ──> minifig_tracker.py ──MQTT──> UnoQ: python/main.py ──Bridge──> sketch.ino ──> motors + LED dot
            (YOLO)
```

## Files

| File | What it does |
|---|---|
| `dataset/` | 140 photos of the green minifig (version 2), labeled in Roboflow and exported in YOLOv8 format (98 train / 28 valid / 14 test) |
| `train_minifig.py` | Fine-tunes the pretrained `yolov8n.pt` on our photos and saves the result as `minifig.pt` |
| `minifig.pt` | The trained model the tracker loads |
| `minifig_tracker.py` | Runs on the laptop: finds the minifig in the webcam image and publishes its position |
| `mqttlib.py` | Small wrapper around paho-mqtt used by the tracker |
| `confusion_test.py` | Measures the model on photos it never trained on, and on recolored copies of them (question 3) |
| `unoq_app/python/main.py` | Runs on the UnoQ's Linux side: receives the position, decides the speed (the policy), and sends speed + dot to the microcontroller |
| `unoq_app/sketch/sketch.ino` | Runs on the UnoQ's microcontroller: drives the DC motors and lights the LED |

## Running it

**1. Set up Python:**
```bash
git clone https://github.com/ani-ga/minifig-detection.git
cd minifig-detection
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train (only when the dataset changes):**
```bash
python train_minifig.py      # writes minifig.pt; uses an NVIDIA or Apple GPU if there is one
```

**3. Set up the UnoQ in Arduino App Lab:** create a new app, then replace
its `python/main.py`, `python/requirements.txt` and `sketch/sketch.ino` with
the files in `unoq_app/`. Before running:
- Wire the Cytron motor driver as listed at the top of `sketch.ino`
  (MDD3A / Maker Drive: M1A→D3, M1B→D5, M2A→D6, M2B→D9, GND→GND). For an
  MDD10A, set `CYTRON_MDD10A = true` and use PWM1→D3, DIR1→D4, PWM2→D6, DIR2→D7.
- Run the app and check the car drives *toward* the target. If it drives away,
  set `DIRECTION_SIGN = -1` in `main.py`.

**4. Start the tracker on the laptop:**
```bash
python minifig_tracker.py
python minifig_tracker.py --dry-run    # just watch the detection, don't send anything
```

## MQTT messages

Topic: `ME193/RARA/minifig` on `test.mosquitto.org`, 10 messages a second.

```json
{"seen": true, "x": 0.42, "y": 0.61, "conf": 0.87}
{"seen": false}
```

`x` and `y` are the center of the minifig's box as a fraction of the image
(0 = left/top, 1 = right/bottom), so they don't depend on the camera's
resolution.

## Questions

### 1. Describe the policy. How does it make decisions?

The decision is made on the UnoQ (`decide_speed` in `unoq_app/python/main.py`),
20 times a second, from the latest message:

1. **error = x − 0.5**: how far the minifig is from the middle of the screen,
   as a fraction of the screen width.
2. If |error| ≤ 0.04, the car **stops**: it has arrived. This dead zone keeps
   it from twitching back and forth around the target because of small jitter
   in YOLO's box.
3. Otherwise it drives with **speed = 250 × error**, so it goes fast when far
   away and slows down as it gets close (proportional control). The size of
   the speed is kept between 35% and 100%: below 35% the motors can't
   overcome friction and the car would stall just short of the target.
4. The sign of the error picks **forward or backward**.

The laptop only reports where the minifig is: if YOLO finds more than one,
it uses the most confident box. The car does all the deciding.

The LED dot uses the same message: x is scaled to one of the matrix's 13
columns and y to one of its 8 rows.

### 2. What does your code do if no minifig is detected?

The car stops, and the dot disappears from the LED matrix.

- The laptop publishes `{"seen": false}` whenever it doesn't see the minifig
  (or YOLO is less than 50% confident), and once more when the tracker quits.
- If the laptop goes quiet (crash, Wi-Fi drop), the UnoQ stops after 0.5 s
  without a message.
- If the UnoQ's Python program dies, the microcontroller stops the motors
  after 1 s without a drive command.

### 3. How good is your model? Can you confuse it?

*These results are for the model trained on version 1 of the dataset (109
photos). Rerun `train_minifig.py` and `confusion_test.py` to update them for
version 2.*

**On the 11 version-1 test images it never trained on** (Ultralytics' validation, run
by `train_minifig.py`):

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.91 | 0.77 | 0.81 | 0.77 |

So when it reports a minifig it is almost always right, but it misses roughly
1 in 4. With only 76 training photos, that's about what we expected.

**Yes, we can confuse it, and color is the easiest way.** `confusion_test.py`
runs the model on all 33 held-out version-1 images (validation + test) and on recolored
copies of them:

| Version of the images | Minifigs found | Avg confidence | False alarms |
|---|---|---|---|
| Original (green) | 29 / 32 | 0.92 | 3 |
| Turned blue | 0 / 32 | – | 6 |
| Turned red | 0 / 32 | – | 15 |
| Turned yellow | 26 / 32 | 0.87 | 8 |
| Grayscale | 0 / 32 | – | 0 |

The model didn't really learn "minifig shape". It learned "green-ish thing
the size of a minifig". In grayscale, where the shape is unchanged, it finds
nothing. A yellow minifig, whose color is next to green, is still mostly
detected. Blue and red ones vanish, and in red images it fires on other
objects instead. Every training photo shows the same green minifig, so color
was the easiest pattern for it to pick up.

Other ways we expect to confuse it: other green objects about the size of a
minifig, a minifig much farther away (smaller) than in our photos, and very
different lighting or backgrounds.
