"""Assemble the UNO Q app in ~/ArduinoApps/safe from this project (one source of truth).
Usage (from the project folder):  python unoq/build_app.py
Then deploy:  cd ~/ArduinoApps && python3 tools/deploy.py safe
Run `python -m model.export` first so outputs/fingernet.npz is up to date."""
import re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = Path.home() / "ArduinoApps" / "safe"
(APP / "python").mkdir(parents=True, exist_ok=True)
(APP / "sketch").mkdir(parents=True, exist_ok=True)

def copy_rewriting_config(src, dst):
    """The board uses safe_config.py (our own name, so nothing else called 'config' can clash)."""
    text = Path(src).read_text()
    text = re.sub(r"^import config, preprocess$", "import safe_config as config\nimport preprocess", text, flags=re.M)
    text = re.sub(r"^import config$", "import safe_config as config", text, flags=re.M)
    Path(dst).write_text(text)

npz = ROOT / "outputs" / "fingernet.npz"
if not npz.exists(): sys.exit("Run `python -m model.export` first.")
shutil.copy(npz, APP / "python" / "fingernet.npz")
shutil.copy(ROOT / "config.py", APP / "python" / "safe_config.py")
shutil.copy(ROOT / "crop.json", APP / "python" / "crop.json")
copy_rewriting_config(ROOT / "preprocess.py", APP / "python" / "preprocess.py")
copy_rewriting_config(ROOT / "lock" / "policy.py", APP / "python" / "lock_policy.py")
shutil.copy(ROOT / "model" / "np_net.py", APP / "python" / "np_net.py")
shutil.copy(ROOT / "unoq" / "safe_app" / "python" / "main.py", APP / "python" / "main.py")
shutil.copy(ROOT / "unoq" / "safe_app" / "sketch" / "sketch.ino", APP / "sketch" / "sketch.ino")
shutil.copy(ROOT / "unoq" / "safe_app" / "sketch" / "sketch.yaml", APP / "sketch" / "sketch.yaml")
(APP / "python" / "assets").mkdir(exist_ok=True)
shutil.copy(ROOT / "unoq" / "safe_app" / "python" / "assets" / "index.html", APP / "python" / "assets" / "index.html")
(APP / "app.yaml").write_text("name: safe\n\nicon: 🔐\n\nports: [7000]\n\nbricks:\n  - arduino:web_ui\n")
print("Built", APP)
