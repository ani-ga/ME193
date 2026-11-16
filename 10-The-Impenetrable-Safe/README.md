# 10 - The Impenetrable Safe (FingerLock)

A safe that opens when you show a 3-digit code (digits 1-5) with your fingers under a webcam.
CNN in PyTorch (3 conv layers + pooling + linear -> 6 classes), lock policy state machine, then deployed to the Arduino UNO Q.

Run everything from this folder, in this order:

```bash
pip install -r requirements.txt
python -m webapp.app                       # web app: crop, capture images, train (http://127.0.0.1:5001)
python -m scripts.find_crop                 # drag a box on the red square, press S
python -m scripts.capture <person>          # press 0-5 to save images (get several people!)
python -m model.net                         # print layer shapes
python -m model.train --val Bob --test Carol
python -m model.evaluate --test Carol       # confusion matrix
python -m unittest lock.test_policy         # policy unit tests
python -m lock.run_computer                 # live demo, SPACE = button
```

Settings (lock code, thresholds, pins) are in `config.py`. Lock code is a placeholder for now.
