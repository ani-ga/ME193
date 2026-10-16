"""Make AprilTag PNGs for the ping-pong menu.

tag 0 = start, tag 1 = Easy, tag 2 = Medium, tag 3 = Hard.
"""

import os

import cv2

LABELS = {0: "start", 1: "easy", 2: "medium", 3: "hard"}
TAG_SIZE_PX = 600

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

for tag_id, label in LABELS.items():
    tag = cv2.aruco.generateImageMarker(aruco_dict, tag_id, TAG_SIZE_PX, borderBits=1)
    # White quiet-zone border around the tag (needed for reliable detection).
    margin = TAG_SIZE_PX // 4
    tag = cv2.copyMakeBorder(tag, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    path = os.path.join(os.path.dirname(__file__), f"tag{tag_id}_{label}.png")
    cv2.imwrite(path, tag)
    print(f"Saved tag36h11 id={tag_id} to {path}")
