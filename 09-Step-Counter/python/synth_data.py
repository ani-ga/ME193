"""Makes FAKE trials in the recorder's CSV format, to test the pipeline
(train -> export -> evaluate) before real data exists.

    python synth_data.py --out /tmp/fake_steps

Never report numbers from this data: it is simple on purpose, so the net
will look much better than it does on a real ankle.
"""
import argparse
from pathlib import Path

import numpy as np

HZ = 50


def make_trial(kind, rng, seconds=25):
    n = int(seconds * HZ)
    t = np.arange(n) / HZ + rng.uniform(0, 100)
    raw = np.zeros((n, 6))
    raw[:, 2] = 1.0                                   # gravity on z
    raw += rng.normal(0, 0.02, raw.shape) * np.array([1, 1, 1, 30, 30, 30])
    strikes = []
    if kind in ("walking", "running"):
        cadence = rng.uniform(1.6, 2.1) if kind == "walking" else rng.uniform(2.6, 3.2)  # steps/s
        amp = 0.5 if kind == "walking" else 1.4
        x = rng.uniform(0.5, 1.0)
        while x < seconds - 1:
            strikes.append(x)
            i = int(x * HZ)
            k = np.arange(-5, 15)
            pulse = np.exp(-0.5 * (k / 3.0) ** 2) * amp * rng.uniform(0.8, 1.2)
            sl = slice(max(i - 5, 0), min(i + 15, n))
            raw[sl, 2] += pulse[: sl.stop - sl.start]
            raw[sl, 0] += 0.4 * pulse[: sl.stop - sl.start]
            raw[sl, 3:6] += 60 * pulse[: sl.stop - sl.start, None] * rng.uniform(-1, 1, 3)
            x += 1.0 / cadence * rng.uniform(0.93, 1.07)
    elif kind == "biking":
        raw[:, 1] += 0.15 * np.sin(2 * np.pi * 1.0 * t)
        raw[:, 3] += 40 * np.sin(2 * np.pi * 1.0 * t)
    elif kind == "arm_swing":
        raw[:, 3] += 90 * np.sin(2 * np.pi * 1.5 * t)
        raw[:, 0] += 0.1 * np.sin(2 * np.pi * 1.5 * t)
    return t, raw, strikes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    kinds = ["walking"] * 12 + ["running"] * 8 + ["biking"] * 4 + ["arm_swing"] * 4 + ["standing"] * 4
    for j, kind in enumerate(kinds):
        t, raw, strikes = make_trial(kind, rng)
        taps = np.zeros(len(t), dtype=int)
        for s in strikes:  # taps land ~0.15 s after the strike, like a person's would
            idx = min(int((s + 0.15 + rng.normal(0, 0.03)) * HZ), len(t) - 1)
            taps[idx] += 1
        with open(out / f"trial_synth{j:02d}_{kind}.csv", "w") as f:
            f.write(f"# label={kind} known_count={len(strikes)}\nt,ax,ay,az,gx,gy,gz,tap\n")
            for ti, row, tp in zip(t, raw, taps):
                f.write(f"{ti:.4f}," + ",".join(f"{v:.5f}" for v in row) + f",{tp}\n")
    print(f"wrote {len(kinds)} fake trials to {out}")


if __name__ == "__main__":
    main()
