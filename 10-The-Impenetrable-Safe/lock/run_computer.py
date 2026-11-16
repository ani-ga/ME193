"""Live laptop demo with simulated hardware. SPACE = button, Q = quit.
Usage: python -m lock.run_computer"""
import time, cv2, torch
import config, preprocess
from model.net import FingerNet
from lock.policy import LockPolicy

net = FingerNet(); net.load_state_dict(torch.load(config.OUT_DIR / "fingernet.pt")); net.eval()
leds = {"on": False}
policy = LockPolicy(on_leds=lambda on: leds.update(on=on),
                    on_open=lambda: print("DOOR OPEN"), on_close=lambda: print("DOOR LOCKED"),
                    on_digit=lambda d, n: print(f"digit {n}: {d}"))
cap = preprocess.open_camera()
x, y, w, h = config.load_crop()
lat, t_prev = [], time.time()
log = open(config.OUT_DIR / "computer_perf.log", "w")

while True:
    ok, frame = cap.read()
    if not ok: break
    arr = torch.from_numpy(preprocess.frame_to_array(frame))[None]
    t = time.perf_counter()
    with torch.no_grad(): probs = torch.softmax(net(arr), 1)[0]
    ms = (time.perf_counter() - t) * 1000
    conf, pred = probs.max(0); pred, conf = pred.item(), conf.item()
    policy.update(pred, conf)
    fps = 1 / (time.time() - t_prev); t_prev = time.time()
    log.write(f"{fps:.1f},{ms:.2f}\n")
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    lines = [f"state {policy.state}  LEDs {'ON' if leds['on'] else 'off'}",
             f"pred {pred}  conf {conf:.2f}", f"digits {policy.digits}",
             f"{fps:.0f} fps  {ms:.1f} ms"]
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("FingerLock", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "): policy.press_button()
    elif key == ord("q"): break
cap.release(); cv2.destroyAllWindows()
