"""Show the live camera, drag a box around the red square, press S to save, Q to quit."""
import sys, json, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config, preprocess

cap = preprocess.open_camera()
box = {"pt1": None, "pt2": None, "drag": False}

def mouse(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN:
        box.update(pt1=(x, y), pt2=(x, y), drag=True)
    elif event == cv2.EVENT_MOUSEMOVE and box["drag"]:
        box["pt2"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        box["pt2"] = (x, y); box["drag"] = False

cv2.namedWindow("find_crop"); cv2.setMouseCallback("find_crop", mouse)
print("Drag a box inside the red square. S = save, Q = quit.")
while True:
    ok, frame = cap.read()
    if not ok: break
    if box["pt1"]:
        cv2.rectangle(frame, box["pt1"], box["pt2"], (0, 255, 0), 2)
    cv2.imshow("find_crop", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s") and box["pt1"]:
        (x1, y1), (x2, y2) = box["pt1"], box["pt2"]
        side = max(abs(x2 - x1), abs(y2 - y1))        # force a square
        x, y = min(x1, x2), min(y1, y2)
        config.CROP_FILE.write_text(json.dumps({"crop": [x, y, side, side]}))
        print("Saved crop", (x, y, side, side))
    elif key == ord("q"): break
cap.release(); cv2.destroyAllWindows()
