"""How good is the model, and can we confuse it?

Runs minifig.pt on every labeled image in the validation and test splits
(images it never trained on), first as-is and then with the colors changed:
the whole image's hue shifted so the green minifig turns blue, red or
yellow, and a grayscale version with no color at all. For each version it
reports how many of the minifigs were found, the average confidence, and
how many detections were false alarms (boxes that don't overlap a real
minifig).

Usage:
    python confusion_test.py
"""

import glob
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).parent
CONF = 0.5  # same threshold minifig_tracker.py uses


def hue_shift(degrees):
    def apply(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = ((hsv[..., 0].astype(int) + degrees // 2) % 180).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return apply


def grayscale(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


VARIANTS = {
    "original (green)": lambda img: img,
    "turned blue": hue_shift(100),
    "turned red": hue_shift(-120),
    "turned yellow": hue_shift(-60),
    "grayscale": grayscale,
}


def load_boxes(label_path, width, height):
    """Read YOLO labels (boxes or polygons) as pixel (x1, y1, x2, y2) boxes."""
    boxes = []
    for line in open(label_path):
        values = [float(v) for v in line.split()[1:]]
        if not values:
            continue
        if len(values) == 4:
            cx, cy, w, h = values
            xs, ys = [cx - w / 2, cx + w / 2], [cy - h / 2, cy + h / 2]
        else:
            xs, ys = values[0::2], values[1::2]
        boxes.append((min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main():
    # Relative path: Ultralytics drops the ' from "Fall '26" in absolute paths.
    model = YOLO(os.path.relpath(HERE / "minifig.pt"))
    images = sorted(glob.glob(str(HERE / "dataset" / "valid" / "images" / "*.jpg")) +
                    glob.glob(str(HERE / "dataset" / "test" / "images" / "*.jpg")))

    print(f"{len(images)} held-out images, confidence threshold {CONF}\n")
    print(f"{'variant':<18}{'found':>10}{'avg conf':>10}{'false alarms':>14}")
    for name, change in VARIANTS.items():
        found = total = false_alarms = 0
        confidences = []
        for image_path in images:
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            label = image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            truth = load_boxes(label, w, h)
            total += len(truth)

            result = model(change(img), conf=CONF, verbose=False)[0]
            boxes = result.boxes.xyxy.tolist()
            scores = result.boxes.conf.tolist()
            for t in truth:
                matches = [s for b, s in zip(boxes, scores) if iou(b, t) > 0.3]
                if matches:
                    found += 1
                    confidences.append(max(matches))
            false_alarms += sum(1 for b in boxes if all(iou(b, t) <= 0.3 for t in truth))

        avg = f"{np.mean(confidences):.2f}" if confidences else "-"
        print(f"{name:<18}{f'{found}/{total}':>10}{avg:>10}{false_alarms:>14}")


if __name__ == "__main__":
    main()
