"""Uncertainty analysis: how good is the step counter on trials it never trained on?

    python evaluate.py

Runs the exported net (step_net.npz) and the no-net baseline over the
held-out trials listed in split.json, compares each count to the true count,
and writes results/metrics.json plus plots.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step_infer import NumpyStepNet, baseline_peaks, find_peaks
from steps_data import load_trial, load_trials

HERE = Path(__file__).resolve().parent


def summarize(name, truth, counted):
    err = counted - truth
    nonzero = truth > 0
    pct = 100.0 * err[nonzero] / truth[nonzero]
    sd = err.std(ddof=1) if len(err) > 1 else float("nan")
    return {
        "method": name,
        "trials": int(len(truth)),
        "mean_error_steps": float(err.mean()),
        "mean_abs_error_steps": float(np.abs(err).mean()),
        "error_sd_steps": float(sd),
        "interval95_steps": [float(err.mean() - 1.96 * sd), float(err.mean() + 1.96 * sd)],
        "mean_abs_pct_error": float(np.abs(pct).mean()) if len(pct) else None,
        "mean_pct_error": float(pct.mean()) if len(pct) else None,
        "pct_error_sd": float(pct.std(ddof=1)) if len(pct) > 1 else None,
        "false_steps_when_truth_is_0": int(counted[~nonzero].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", help="evaluate on every trial in this folder (fresh trials the net never trained on)")
    args = ap.parse_args()

    net = NumpyStepNet()
    if args.test_dir:
        trials = load_trials(args.test_dir)
    else:
        split = json.loads((HERE / "split.json").read_text())
        data_dir = Path(split["data"])
        if not data_dir.exists():          # split.json was written on another machine
            data_dir = HERE / "data" / "steps"
        trials = [load_trial(data_dir / name) for name in split["test"]]
    if not trials:
        raise SystemExit("No test trials found. Use --test-dir with a folder of fresh trials, or train without --all.")
    rows = []
    for trial in trials:
        net_count = len(find_peaks(net.predict(trial.raw)))
        base_count = len(baseline_peaks(trial.raw))
        rows.append((trial.name, trial.label, trial.true_count, net_count, base_count))

    truth = np.array([r[2] for r in rows], dtype=float)
    net_c = np.array([r[3] for r in rows], dtype=float)
    base_c = np.array([r[4] for r in rows], dtype=float)

    print(f"{'trial':44s} {'true':>5s} {'net':>5s} {'base':>5s}")
    for n, lab, tr, nc, bc in rows:
        print(f"{n:44s} {tr:5d} {nc:5d} {bc:5d}")
    metrics = [summarize("neural net", truth, net_c), summarize("baseline (no net)", truth, base_c)]
    for m in metrics:
        print(f"\n{m['method']}: mean error {m['mean_error_steps']:+.2f} steps, "
              f"MAE {m['mean_abs_error_steps']:.2f}, sd {m['error_sd_steps']:.2f}, "
              f"95% interval {m['interval95_steps'][0]:+.1f} to {m['interval95_steps'][1]:+.1f} steps, "
              f"mean |error| {m['mean_abs_pct_error']}% (walking trials), "
              f"false steps on zero-step trials: {m['false_steps_when_truth_is_0']}")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"metrics": metrics, "trials": [dict(zip(("trial", "label", "true", "net", "baseline"), r)) for r in rows]},
        indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    top = max(truth.max(), net_c.max(), base_c.max()) * 1.05
    ax[0].plot([0, top], [0, top], "k--", lw=1, label="perfect")
    ax[0].scatter(truth, base_c, c="tab:gray", label="baseline")
    ax[0].scatter(truth, net_c, c="tab:green", label="neural net")
    ax[0].set(xlabel="true steps", ylabel="counted steps", title="Held-out trials")
    ax[0].legend()
    bins = np.arange(min((net_c - truth).min(), (base_c - truth).min()) - 1,
                     max((net_c - truth).max(), (base_c - truth).max()) + 2)
    ax[1].hist([base_c - truth, net_c - truth], bins=bins, color=["tab:gray", "tab:green"],
               label=["baseline", "neural net"])
    ax[1].set(xlabel="counted - true (steps)", ylabel="trials", title="Count error")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=150)
    print(f"\nwrote {out / 'metrics.json'} and {out / 'accuracy.png'}")


if __name__ == "__main__":
    main()
