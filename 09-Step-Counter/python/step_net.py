"""The PyTorch step-detection network (laptop side, used for training)."""
import numpy as np
import torch
import torch.nn as nn

from step_infer import CHANNELS

KERNEL = 9
DILATIONS = [1, 2, 4]   # receptive field = 1 + 8*(1+2+4) = 57 samples, about 1.1 s
HIDDEN = 16


class StepNet(nn.Module):
    """Three dilated 1D conv layers (each followed by BatchNorm + ReLU) and a
    1x1 conv that turns the features at each moment into one logit: "is a
    foot strike happening right now?". It is fully convolutional, so it
    accepts any length of signal and returns one logit per sample."""

    def __init__(self, in_channels=CHANNELS):
        super().__init__()
        layers, c = [], in_channels
        for d in DILATIONS:
            layers += [
                nn.Conv1d(c, HIDDEN, KERNEL, dilation=d, padding=(KERNEL - 1) // 2 * d),
                nn.BatchNorm1d(HIDDEN),
                nn.ReLU(),
            ]
            c = HIDDEN
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(c, 1, 1)

    def forward(self, x):          # x: (batch, channels, time)
        return self.head(self.body(x)).squeeze(1)   # (batch, time) logits


def export_npz(model, ch_mean, ch_std, path):
    """Folds each BatchNorm into the conv before it and saves plain arrays,
    so step_infer.NumpyStepNet can reproduce the forward pass."""
    model.eval()
    out = {"ch_mean": np.asarray(ch_mean), "ch_std": np.asarray(ch_std),
           "dilations": np.array(DILATIONS)}
    for i in range(len(DILATIONS)):
        conv, bn = model.body[3 * i], model.body[3 * i + 1]
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        out[f"W{i}"] = (conv.weight * scale[:, None, None]).detach().numpy()
        out[f"b{i}"] = ((conv.bias - bn.running_mean) * scale + bn.bias).detach().numpy()
    out[f"W{len(DILATIONS)}"] = model.head.weight.detach().numpy()
    out[f"b{len(DILATIONS)}"] = model.head.bias.detach().numpy()
    np.savez(path, **out)
