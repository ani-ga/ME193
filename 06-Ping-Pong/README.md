# LEGO Ping-Pong

Virtual ping-pong where a LEGO Education Double Motor (orange card, 1129) is the paddle.
The webcam tracks where your hand is, the Double Motor's IMU detects whether you
actually swung, AprilTags choose the level and start the game, and the record number of
continuous hits is published live over MQTT.

## Running it

```bash
pip install legoeducation mediapipe opencv-contrib-python paho-mqtt
python make_tags.py        # once - makes tag0..tag3 PNGs to print or show on a phone
python ping_pong.py
```

1. Hold up **tag 1 / 2 / 3** (Easy / Medium / Hard) for one second to pick the level.
2. Hold up **tag 0** to start. Hold the paddle still during the 3-2-1 countdown (the IMU is calibrating).
3. Get your paddle hand onto the ball as it arrives and **swing**.

| Key | Action |
|---|---|
| `q` | quit |
| `r` | back to level menu |
| `space` | swing (for testing without the robot) |
| `[` / `]` | lower / raise the swing threshold |
| `h` | swap which hand is the paddle |

Score topic: `ME193/Rogers/Ani`
The payload is the record streak as a float, e.g. `7.0`.

## How it uses what we've done in class

| Topic | Use in the game |
|---|---|
| Double Motor / `legoeducation` | Bluetooth connection to the paddle; motors, light and beeper for feedback |
| IMU | Gyro magnitude above a calibrated baseline = a swing |
| Pose (MediaPipe) | Elbow -> wrist (forearm) places and tilts a virtual paddle; full skeleton drawn with per-joint confidence |
| AprilTags | Tags 1-3 pick level (ball speed, hit radius, opponent aim); tag 0 starts |
| MQTT | Record streak published in real time |
| Image processing / OpenCV | Overlay of table, ball, paddle, swing meter on the live video |

**New thing - haptic feedback:** the Double Motor's own motors are flipped back and forth
very quickly so the hub *vibrates* in your hand instead of spinning: a short tick on a hit,
a long rumble on a miss, and a triple buzz (plus triple beep) when you set a new record.
The hub light also shows game state (blue = menu, white = calibrating, green = playing,
red = miss, yellow = new record).

## Questions

### 1. Describe the policy - how does it make decisions?

The game is a rule-based **finite state machine**:
`WAIT_LEVEL -> WAIT_START -> COUNTDOWN -> SERVE -> PLAY -> (MISSED -> SERVE ...)`.

- **Menu:** an AprilTag only counts after it is seen continuously for 1 s, which filters
  out flickering detections. Tags 1-3 set Easy / Medium / Hard, tag 0 starts the game.
- **Calibration:** during the 3-2-1 countdown the game averages the IMU's gyro magnitude
  while the paddle is held still. That becomes the "not swinging" baseline.
- **Hit decision:** every frame while the ball is in the hit zone (the last 20% of its
  path), two things are checked:
  1. **Pose** - the ball is inside the virtual paddle head, an oval placed just past the
     wrist and pointed along the forearm (elbow -> wrist).
  2. **IMU** - a swing (gyro more than a threshold above the baseline) happened in the
     last 0.35 s.

  Both true = **hit**. If the ball gets past the player first = **miss**.
- **Return:** where the ball landed on the paddle and how the forearm is tilted set the
  return direction. Each hit makes the rally 5% faster (capped at 2x).
- **Opponent:** aims at a random reachable spot, but on Medium (40%) and Hard (80%) it
  picks the spot farthest from your hand so you have to move.
- **Feedback:** hit, miss and new record each trigger a different vibration, light color
  and beep, and a new record is published to MQTT immediately. Swings are ignored while
  the paddle is vibrating so the haptics can't fake a swing.

### 2. What are the potential limitations of the game?

- **No real depth.** One webcam gives a 2-D hand position, so the game only knows your
  hand is over the ball on screen, not whether it is actually forward at the table.
- **Swing direction and strength aren't used.** Any fast rotation counts, so shaking the
  paddle works as well as a real stroke, and a harder swing doesn't hit the ball harder.
- **Latency.** Bluetooth, the camera, and the pose smoothing each add a little delay, so
  fast Hard rallies can feel slightly late; the 0.35 s swing window covers some of it.
- **Pose tracking can fail** in poor lighting, with a busy background, when the arm leaves
  the frame, or when the paddle blocks the wrist. Holding the last pose for 0.25 s only
  hides short dropouts.
- **The swing threshold is in raw IMU units** and may need tuning per person (the `[` `]`
  keys and on-screen meter are there for that).
- **AprilTags need decent lighting** and to be held flat and fairly close to the camera.
- **Performance depends on the computer.** It runs at about 20-30 fps on a laptop; a slower
  machine makes ball motion and hit timing feel worse.
- **The haptics are fairly weak** and drain the battery.
- **The public MQTT broker** (`test.mosquitto.org`) is not guaranteed to be up.

### 3. What AI/ML algorithms did you use and how do they work?

- **MediaPipe Pose (neural-network pose estimation):** one neural network finds the person
  in the image and a second predicts the position and visibility of 33 body landmarks. I
  use the elbow and wrist to place and tilt the paddle.
- **AprilTag detection (OpenCV ArUco):** the image is thresholded, four-sided outlines are
  found and unwarped into a square grid, and the black/white bits are decoded and matched
  against the 36h11 tag family, which can correct a few wrong bits.
- **IMU swing detection (threshold classifier):** the average gyro magnitude while still is
  the baseline, and any reading far enough above it counts as a swing. A cooldown stops
  one swing from counting twice.
- **One Euro filter (adaptive smoothing):** a low-pass filter whose cutoff rises with
  speed, so it smooths heavily when the hand is still (removing jitter) but barely at all
  during fast moves (so swings aren't delayed).
- **Rule-based policy (finite state machine):** hand-written if-then rules combine the pose
  and IMU signals to decide game state and hit vs. miss. The opponent is a simple random
  policy biased toward aiming away from your hand.

## Reflection (template questions)

> Fill these in with your own experience after testing.

- **What I'm proud of:** _e.g. fusing two independent sensors (camera pose + IMU) into one
  hit decision, and turning the drive motors into haptic feedback._
- **What was hardest:** _e.g. tuning the swing threshold / getting the hit timing to feel fair._
- **What I'd do next:** _e.g. use swing speed to set return speed, use accelerometer
  direction to require a forward stroke, or let two Double Motors play each other over MQTT._
- **Who I worked with / resources used:** _..._
