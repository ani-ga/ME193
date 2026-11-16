"""Unlock page: set a 3-digit code, press Start, show the digits with your fingers.
The trained CNN reads each digit and the SAME LockPolicy used on the UNO Q decides
whether the (simulated) safe unlocks. Open http://127.0.0.1:5001/unlock"""
import threading, time
import torch
from flask import jsonify, request
import config, preprocess
from model.net import FingerNet
from lock.policy import LockPolicy

state = {"pred": None, "conf": 0.0, "msg": "Press Start, then show your code.",
         "opened_at": None, "digits_log": []}
net = {"model": None}
started = {"flag": False}

def load_model():
    m = FingerNet()
    m.load_state_dict(torch.load(config.OUT_DIR / "fingernet.pt"))
    m.eval()
    net["model"] = m

def on_digit(d, n):
    state["digits_log"].append(d)
    state["msg"] = f"Locked in digit {n} of {len(config.LOCK_CODE)}: {d}. Show the next digit, then press the button."

def on_open():
    state["msg"] = "UNLOCKED"

policy = LockPolicy(on_open=on_open, on_digit=on_digit,
                    on_close=lambda: state.update(msg="Locked again."))
_last = {"state": "IDLE", "failed": 0}

def loop(latest, cam_lock):
    """Run the model on the newest camera frame ~20x per second and feed the policy."""
    while True:
        time.sleep(0.05)
        if net["model"] is None: continue
        with cam_lock: f = None if latest["frame"] is None else latest["frame"].copy()
        if f is None: continue
        x = torch.from_numpy(preprocess.frame_to_array(f))[None]
        with torch.no_grad(): probs = torch.softmax(net["model"](x), 1)[0]
        conf, pred = probs.max(0)
        state["pred"], state["conf"] = pred.item(), conf.item()
        before_failed = policy.failed
        policy.update(state["pred"], state["conf"])
        if policy.failed > before_failed and policy.state != "LOCKOUT":
            state["msg"] = f"Wrong code ({policy.digits}). Still locked. Attempt {policy.failed}/{config.MAX_ATTEMPTS}."
        elif policy.state == "LOCKOUT": state["msg"] = "Too many wrong codes: LOCKED OUT."
        elif policy.state == "IDLE" and _last["state"] == "ARMED" and policy.failed == before_failed and len(policy.digits) < len(config.LOCK_CODE):
            state["msg"] = "Timed out. Press Start to try again."
        _last["state"] = policy.state

def register(app, latest, cam_lock):
    @app.route("/unlock")
    def unlock_page():
        if not started["flag"]:
            started["flag"] = True
            try: load_model()
            except Exception as e: state["msg"] = f"No trained model yet ({e}). Train first."
            threading.Thread(target=loop, args=(latest, cam_lock), daemon=True).start()
        return UNLOCK_PAGE

    @app.route("/unlock/state")
    def unlock_state():
        return jsonify(policy=policy.state, digits=policy.digits, code_len=len(config.LOCK_CODE),
                       **{k: state[k] for k in ("pred", "conf", "msg")})

    @app.route("/unlock/set_code", methods=["POST"])
    def set_code():
        code = [int(c) for c in request.json["code"] if c in "12345"]
        if len(code) != 3 or len(set(code)) != 3:
            return jsonify(error="Need three DIFFERENT digits from 1-5"), 400
        config.LOCK_CODE = tuple(code)              # policy reads this every time it checks
        state["msg"] = "Code set. Press Start."
        return jsonify(ok=True)

    @app.route("/unlock/start", methods=["POST"])
    def start():
        if policy.state == "ARMED":                  # already listening: this press locks in the digit
            if policy.press_button() is False:
                state["msg"] = "I can't see a clear number. Hold steady and press again."
            return jsonify(ok=True)
        load_model()                                 # pick up a freshly trained model
        policy.state, policy.failed = "IDLE", 0
        policy.press_button()
        state["msg"] = "Armed: show digit 1, then press the button to lock it in."
        return jsonify(ok=True)

UNLOCK_PAGE = """<!doctype html><meta charset=utf-8><title>FingerLock unlock</title>
<style>body{font-family:system-ui;margin:20px;max-width:900px}img{border:1px solid #999}
#big{font-size:64px;font-weight:700;padding:20px;margin:10px 0;border-radius:12px;background:#ddd;text-align:center}
.open{background:#2ecc71!important;color:#fff}.bad{background:#e74c3c!important;color:#fff}
button{font-size:18px;padding:10px 18px}</style>
<h2>FingerLock</h2>
<p>Secret code (3 different digits, 1-5): <input id=code size=5 value=253>
<button onclick=setCode()>Set code</button> <button onclick=start()>Button: start / lock in digit</button> <a href=/>back to trainer</a></p>
<div id=big>LOCKED</div>
<p id=msg></p><p>Seeing: <b id=pred>-</b> (confidence <span id=conf>-</span>) &nbsp; Digits so far: <b id=digits></b></p>
<img src=/video width=480>
<script>
const $=id=>document.getElementById(id);
async function setCode(){const r=await fetch('/unlock/set_code',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:$('code').value})});
 if(!r.ok)alert((await r.json()).error)}
async function start(){await fetch('/unlock/start',{method:'POST'})}
setInterval(async()=>{const s=await (await fetch('/unlock/state')).json();
 $('pred').textContent=s.pred;$('conf').textContent=(s.conf||0).toFixed(2);$('msg').textContent=s.msg;
 $('digits').textContent=s.digits.join(' ')+' '+'_ '.repeat(Math.max(0,s.code_len-s.digits.length));
 const b=$('big');b.className='';
 if(s.policy=='UNLOCKED'){b.textContent='UNLOCKED';b.className='open'}
 else if(s.policy=='LOCKOUT'){b.textContent='LOCKED OUT';b.className='bad'}
 else if(s.policy=='ARMED'){b.textContent='LISTENING...'}
 else b.textContent='LOCKED'},200);
</script>"""
