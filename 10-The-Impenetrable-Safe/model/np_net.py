"""The same CNN as net.py, but written in plain numpy so the UNO Q can run it without PyTorch.
Weights come from outputs/fingernet.npz (made by model/export.py)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def conv2d(x, w, b):
    """x: (C,H,W)   w: (OUT,C,k,k)   b: (OUT,)  ->  (OUT,H,W)   ('same' padding, stride 1)"""
    k = w.shape[-1]
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))                    # zero padding
    windows = sliding_window_view(xp, (k, k), axis=(1, 2))      # every k x k patch: (C,H,W,k,k)
    out = np.tensordot(w, windows, axes=([1, 2, 3], [0, 3, 4])) # multiply patch by filter and add up
    return out + b[:, None, None]

def relu(x):
    return np.maximum(x, 0)

def maxpool2(x):
    """Keep the biggest value in each 2x2 block -> half the width and height."""
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))

class NumpyFingerNet:
    def __init__(self, path):
        self.p = np.load(path)

    def logits(self, img):
        """img: (1,H,W) float32 in 0..1 -> 6 raw scores."""
        p = self.p
        x = maxpool2(relu(conv2d(img, p["conv1.weight"], p["conv1.bias"])))
        x = maxpool2(relu(conv2d(x, p["conv2.weight"], p["conv2.bias"])))
        x = maxpool2(relu(conv2d(x, p["conv3.weight"], p["conv3.bias"])))
        return p["fc.weight"] @ x.reshape(-1) + p["fc.bias"]

    def predict(self, img):
        """Returns (digit, confidence) using softmax."""
        z = self.logits(img)
        e = np.exp(z - z.max())
        probs = e / e.sum()
        return int(probs.argmax()), float(probs.max())
