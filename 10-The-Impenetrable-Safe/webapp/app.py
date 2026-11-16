"""FingerLock web app: collect labeled images with the USB webcam and train the CNN.
Run from the project folder:  python -m webapp.app   then open http://127.0.0.1:5001
Training runs on THIS computer (never on the UNO Q)."""
import json, subprocess, sys, threading, time
from pathlib import Path
# Make sure OUR config.py (in the project folder) is found, even if this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from flask import Flask, Response, jsonify, request, send_file
import config, preprocess

app = Flask(__name__)
cam_lock = threading.Lock()
cap = preprocess.open_camera()
latest = {"frame": None}
train = {"proc": None, "log": []}

def grab_loop():
    """Keep reading the camera in the background so the newest frame is always ready."""
    while True:
        ok, frame = cap.read()
        if ok:
            with cam_lock: latest["frame"] = frame
        else: time.sleep(0.1)
threading.Thread(target=grab_loop, daemon=True).start()

def counts():
    return [len(list((config.DATA_DIR / str(k)).glob("*.png"))) for k in range(config.NUM_CLASSES)]

@app.route("/")
def index(): return PAGE

@app.route("/video")
def video():
    def gen():
        while True:
            with cam_lock: f = None if latest["frame"] is None else latest["frame"].copy()
            if f is not None:
                x, y, w, h = config.load_crop()
                cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 2)   # the red-square crop
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", f)[1].tobytes() + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

@app.route("/preview")
def preview():
    """What the network will actually see (cropped, gray, 64x64, enlarged)."""
    with cam_lock: f = latest["frame"]
    if f is None: return "", 204
    small = preprocess.to_gray_small(preprocess.crop_frame(f))
    big = cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST)
    return Response(cv2.imencode(".png", big)[1].tobytes(), mimetype="image/png")

@app.route("/crop", methods=["GET", "POST"])
def crop():
    if request.method == "POST":
        x, y, s = (int(request.json[k]) for k in ("x", "y", "s"))
        config.CROP_FILE.write_text(json.dumps({"crop": [x, y, s, s]}))
    x, y, w, h = config.load_crop()
    fh, fw = (latest["frame"].shape[:2] if latest["frame"] is not None else (480, 640))
    return jsonify(x=x, y=y, s=w, fw=fw, fh=fh)

@app.route("/save", methods=["POST"])
def save():
    k = int(request.json["label"]); person = "".join(c for c in request.json["person"] if c.isalnum()) or "anon"
    with cam_lock: f = latest["frame"]
    if f is None: return jsonify(error="no camera frame"), 500
    small = preprocess.to_gray_small(preprocess.crop_frame(f))
    cv2.imwrite(str(config.DATA_DIR / str(k) / f"{person}_{int(time.time()*1000)}.png"), small)
    return jsonify(counts=counts())

@app.route("/counts")
def get_counts(): return jsonify(counts=counts())

@app.route("/train", methods=["POST"])
def start_train():
    if train["proc"] and train["proc"].poll() is None: return jsonify(error="already training"), 400
    cmd = [sys.executable, "-u", "-m", "model.train"]
    for flag in ("val", "test"):
        names = request.json.get(flag, "").replace(",", " ").split()
        if names: cmd += [f"--{flag}", *names]
    train["log"] = []
    train["proc"] = subprocess.Popen(cmd, cwd=config.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    threading.Thread(target=lambda: [train["log"].append(l.rstrip()) for l in train["proc"].stdout], daemon=True).start()
    return jsonify(ok=True)

@app.route("/train_status")
def train_status():
    running = bool(train["proc"] and train["proc"].poll() is None)
    return jsonify(running=running, log=train["log"][-40:])

@app.route("/curves")
def curves():
    p = config.OUT_DIR / "training_curves.png"
    return send_file(p) if p.exists() else ("", 204)

PAGE = """<!doctype html><meta charset=utf-8><title>FingerLock trainer</title>
<style>body{font-family:system-ui;margin:20px;max-width:1000px}img{border:1px solid #999}
button{font-size:18px;padding:10px 18px;margin:3px}.row{display:flex;gap:20px;flex-wrap:wrap}
pre{background:#111;color:#8f8;padding:8px;height:200px;overflow:auto}</style>
<h2>FingerLock trainer</h2><p><a href=/unlock>Go to the unlock page</a></p>
<div class=row><div><b>Camera (green = crop)</b><br><img src=/video width=480></div>
<div><b>What the net sees</b><br><img id=pv width=256></div></div>
<p>Person: <input id=person value=ani> (use a different name for each person!)</p>
<p>Click a number (or press 0-5 on the keyboard) to save the current crop with that label:<br>
<span id=btns></span></p><p id=cnt></p>
<details><summary>Crop position (move the green box onto the red square)</summary>
x <input type=range id=cx> y <input type=range id=cy> size <input type=range id=cs min=40 max=480></details>
<h3>Train (on this computer)</h3>
<p>Validation people: <input id=val> Test people: <input id=test> (names, space-separated; leave empty for a first try)</p>
<button onclick=train()>Train</button><pre id=log></pre><img id=curves width=700>
<script>
const $=id=>document.getElementById(id);
for(let k=0;k<6;k++)$('btns').innerHTML+=`<button onclick=save(${k})>${k}</button>`;
async function save(k){const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:k,person:$('person').value})});
 show((await r.json()).counts)}
function show(c){if(c)$('cnt').textContent='images per class (0..5): '+c.join(', ')}
document.addEventListener('keydown',e=>{if(e.target.tagName!='INPUT'&&e.key>='0'&&e.key<='5')save(+e.key)});
setInterval(()=>$('pv').src='/preview?'+Date.now(),300);
fetch('/counts').then(r=>r.json()).then(d=>show(d.counts));
async function loadCrop(){const c=await (await fetch('/crop')).json();
 $('cx').max=c.fw;$('cy').max=c.fh;$('cx').value=c.x;$('cy').value=c.y;$('cs').value=c.s}
async function setCrop(){await fetch('/crop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x:$('cx').value,y:$('cy').value,s:$('cs').value})})}
['cx','cy','cs'].forEach(i=>$(i).onchange=setCrop);loadCrop();
async function train(){await fetch('/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({val:$('val').value,test:$('test').value})})}
setInterval(async()=>{const s=await (await fetch('/train_status')).json();$('log').textContent=s.log.join('\\n');
 if(!s.running&&s.log.length)$('curves').src='/curves?'+Date.now()},1500);
</script>"""

import webapp.unlock as unlock
unlock.register(app, latest, cam_lock)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, threaded=True)
