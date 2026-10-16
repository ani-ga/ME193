"""Finds the green LEGO minifig in the webcam image with our YOLOv8 model and
publishes where it is over MQTT on ME193/<team>/minifig.

Each message is JSON. When the minifig is visible:
    {"seen": true, "x": 0.42, "y": 0.61, "conf": 0.87}
where x and y are the box center as a fraction of the image width/height
(0 = left/top, 1 = right/bottom). When it isn't visible:
    {"seen": false}

The laptop only reports the position. The UnoQ decides for itself whether
to drive forward, backward or stop (see unoq_app/python/main.py).

Usage:
    python minifig_tracker.py
    python minifig_tracker.py --camera 1 --dry-run  # camera 1, don't publish
Press q or Esc in the video window to quit.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from mqttlib import MQTTClient

DEFAULT_TEAM = "RARA"
DEFAULT_MODEL = Path(__file__).with_name("minifig.pt")
PUBLISH_PERIOD_S = 0.1  # 10 messages per second is plenty for the car
GREEN = (0, 200, 0)


def open_camera(index):
    """Open camera `index` if it actually sends frames, else None.

    Some cameras (like an iPhone's Continuity Camera) open fine but never
    send a picture, so wait up to 2 s for a real frame.
    """
    cap = cv2.VideoCapture(index)
    deadline = time.time() + 2
    while cap.isOpened() and time.time() < deadline:
        if cap.read()[0]:
            return cap
    cap.release()
    return None


def find_minifig(model, frame, conf):
    """Return (box, confidence) of the most confident minifig, or (None, 0.0)."""
    result = model(frame, conf=conf, verbose=False)[0]
    if len(result.boxes) == 0:
        return None, 0.0
    best = int(result.boxes.conf.argmax())
    box = tuple(int(round(c)) for c in result.boxes.xyxy[best].tolist())
    return box, float(result.boxes.conf[best])


def make_message(box, score, width, height):
    if box is None:
        return {"seen": False}
    x1, y1, x2, y2 = box
    return {
        "seen": True,
        "x": round((x1 + x2) / 2 / width, 3),
        "y": round((y1 + y2) / 2 / height, 3),
        "conf": round(score, 2),
    }


def draw_overlay(frame, box, score):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), GREEN, 1)  # where the car stops
    if box is None:
        status = "not seen"
    else:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        cv2.rectangle(frame, box[:2], box[2:], GREEN, 2)
        cv2.circle(frame, (cx, cy), 5, GREEN, -1)
        cv2.putText(frame, f"minifig {score:.2f}", (box[0], box[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        status = f"x={cx / w:.2f} y={cy / h:.2f}"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (default: the first one that sends frames).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO weights (default minifig.pt).")
    parser.add_argument("--conf", type=float, default=0.5, help="Minimum YOLO confidence (default 0.5).")
    parser.add_argument("--team", default=DEFAULT_TEAM, help="Topic: ME193/<team>/minifig.")
    parser.add_argument("--dry-run", action="store_true", help="Detect and display, but don't publish.")
    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"No model at {args.model}. Run train_minifig.py first.")
    # Relative path on purpose: Ultralytics drops the ' from absolute paths
    # like ".../Fall '26/..." and then can't find the file.
    model = YOLO(os.path.relpath(args.model))
    topic = f"ME193/{args.team}/minifig"

    for index in [args.camera] if args.camera is not None else range(4):
        cap = open_camera(index)
        if cap:
            print(f"Using camera {index}")
            break
    else:
        sys.exit("No working camera found. Check System Settings > Privacy & Security > Camera.")

    client = None if args.dry_run else MQTTClient().__enter__()
    if client:
        print(f"Publishing to {topic}")
    last_publish = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera")
                break
            h, w = frame.shape[:2]
            box, score = find_minifig(model, frame, args.conf)

            now = time.time()
            if client and now - last_publish >= PUBLISH_PERIOD_S:
                last_publish = now
                client.publish(topic, json.dumps(make_message(box, score, w, h)))

            draw_overlay(frame, box, score)
            cv2.imshow("Minifig tracker (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        if client:
            client.publish(topic, json.dumps({"seen": False}))  # stop the car before we go quiet
            time.sleep(0.3)
            client.__exit__(None, None, None)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
