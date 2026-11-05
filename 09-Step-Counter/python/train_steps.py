"""Train the step-detection net on recorded trials (run on the laptop).

    python train_steps.py                # uses data/steps/
    python train_steps.py --data DIR --epochs 40

Writes step_net.pt (PyTorch), step_net.npz (what the board loads),
split.json (which whole trials are held out for evaluate.py) and
results/loss_history.json (the loss curve, used in the animation).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from step_infer import CHANNELS, FS, make_channels
from step_net import StepNet, export_npz
from steps_data import TAP_LAG_S, load_trials

HERE = Path(__file__).resolve().parent
WINDOW = 128  # samples per training crop (2.56 s)


def split_trials(trials):
    """Hold out every 4th trial of each activity. Whole trials, never random
    windows: neighboring windows of one trial look almost identical, so
    splitting windows would leak the answers into the test set."""
    train, test = [], []
    by_label = {}
    for t in trials:
        by_label.setdefault(t.label, []).append(t)
    for group in by_label.values():
        for i, t in enumerate(group):
            (test if i % 4 == 3 else train).append(t)
    return train, test


def random_crop(chans, target, rng):
    """One augmented training example: random speed change, random crop,
    random per-channel gain, a little noise."""
    scale = rng.uniform(0.85, 1.15)  # >1 plays the trial faster
    idx = np.arange(0, len(chans) - 1, scale)
    chans = np.stack([np.interp(idx, np.arange(len(chans)), chans[:, c]) for c in range(chans.shape[1])], axis=1)
    target = np.interp(idx, np.arange(len(target)), target)
    start = rng.integers(0, len(chans) - WINDOW)
    x = chans[start:start + WINDOW] * rng.uniform(0.9, 1.1, size=chans.shape[1])
    x = x + rng.normal(0, 0.03, size=x.shape)
    return x.T.astype(np.float32), target[start:start + WINDOW].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "steps"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lag", type=float, default=TAP_LAG_S, help="seconds taps trail the real foot strike")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="train on every trial (no hold-out); test with evaluate.py --test-dir on fresh trials")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    trials = [t for t in load_trials(args.data) if len(t.raw) > WINDOW + 10]
    if len(trials) < (2 if args.all else 4):
        raise SystemExit(f"Not enough trials in {args.data}, found {len(trials)}.")
    train, test = (trials, []) if args.all else split_trials(trials)
    print(f"{len(train)} training trials, {len(test)} held-out trials")
    (HERE / "split.json").write_text(json.dumps({"data": str(args.data), "test": [t.name for t in test]}, indent=1))

    # Normalize each channel with training-set statistics.
    all_ch = np.vstack([make_channels(t.raw) for t in train])
    ch_mean, ch_std = all_ch.mean(axis=0), all_ch.std(axis=0) + 1e-6
    data = [((make_channels(t.raw) - ch_mean) / ch_std, t.target(args.lag)) for t in train]

    model = StepNet(CHANNELS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Steps are rare (about 1 sample in 25 is near a foot strike), so up-weight them.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
    history = []
    batches_per_epoch, batch_size = 30, 64

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for _ in range(batches_per_epoch):
            xs, ys = zip(*[random_crop(*data[rng.integers(len(data))], rng) for _ in range(batch_size)])
            x, y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
            loss = loss_fn(model(x), y)   # forward pass + loss
            opt.zero_grad()
            loss.backward()               # back-propagation: d(loss)/d(every weight)
            opt.step()                    # Adam nudges each weight against its gradient
            total += loss.item()
        history.append(total / batches_per_epoch)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {history[-1]:.4f}")

    torch.save(model.state_dict(), HERE / "step_net.pt")
    export_npz(model, ch_mean, ch_std, HERE / "step_net.npz")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "loss_history.json").write_text(json.dumps(history))
    print("saved step_net.pt, step_net.npz. Now run: python evaluate.py")


if __name__ == "__main__":
    main()
