"""Collect training images. Usage: python -m scripts.capture <person_name>
Press 0-5 to save the current crop as that class, Q to quit."""
import sys, time, cv2
import config, preprocess

person = sys.argv[1] if len(sys.argv) > 1 else "anon"
cap = preprocess.open_camera()
x, y, w, h = config.load_crop()

def counts():
    return [len(list((config.DATA_DIR / str(k)).glob("*.png"))) for k in range(config.NUM_CLASSES)]

while True:
    ok, frame = cap.read()
    if not ok: break
    small = preprocess.to_gray_small(preprocess.crop_frame(frame))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, "counts " + str(counts()), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("capture", frame)
    cv2.imshow("what the net sees", cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    if ord("0") <= key <= ord("5"):
        k = key - ord("0")
        name = f"{person}_{int(time.time() * 1000)}.png"     # person name lets us split by person later
        cv2.imwrite(str(config.DATA_DIR / str(k) / name), small)
cap.release(); cv2.destroyAllWindows()
