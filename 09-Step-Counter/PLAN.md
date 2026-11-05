# Assignment 9 plan: from gait identifier to step counter

**Goal:** count steps with a neural net in PyTorch, using every IMU channel, and report how accurate it is.
**Deliverables (from the assignment):** (1) GitHub link, reflection answers, clean layout; (2) video of the counter working plus an uncertainty analysis; (3) a Claude-made animation of the whole system, from the net and PyTorch through back-propagation; (4) at most three paragraphs on how the net works.
**Due:** Nov 3 by 11:59pm.

## Starting point (copied from 08-Gait-Identifier)

| Keep as is | Change | Add |
|---|---|---|
| `sketch/sketch.ino` (MPU6050 over I2C, Bridge reads), 50 Hz polling and unit scaling in `main.py`, web-page plumbing, `calibration.py` CSV recording | `features.py` and `models.py` (hand-built thresholds and KNN) get replaced by a PyTorch model | `train_steps.py` (laptop training), `step_net.py` (model definition), `evaluate.py` (uncertainty), step counter UI on the phone page |

The old classifier labels a 2 s window as standing/walking/running. A step counter needs *when* each foot strike happens, so the output changes from one label per window to one number per time sample.

## Design

**Sensors.** All 6 IMU channels (ax, ay, az, gx, gy, gz), plus two derived channels, accel magnitude and gyro magnitude, which don't depend on how the sensor is rotated on the ankle. Normalize each channel with training-set mean and std. If a second sensor is available (for example a Modulino Movement worn on the other leg or the waist), add its channels as extra inputs. More channels is a one-line change to `in_channels`.

**Model.** A small 1D convolutional net (about 3 conv layers with BatchNorm and ReLU, then a 1x1 conv head) that reads a sliding window of 128 samples (2.5 s at 50 Hz) and outputs a step-probability value for every time sample. Peak picking on that curve (minimum height, minimum spacing of about 0.25 s) gives step events, and the count is the number of peaks. This is better than regressing "steps in this window" directly because windows overlap and the counts can't simply be summed.

**Labels.** Add a "Tap" button to the web page. While recording, the user taps at every foot strike. Each tap becomes a Gaussian bump (sigma about 0.06 s) on a target curve, so small timing error in tapping is forgiven. Loss is `BCEWithLogitsLoss` with a positive weight, since steps are rare compared to non-steps. Also record trials with a *known* count (walk exactly 20 steps, then stop), used only for testing the final count.

**Training (laptop, PyTorch).** `calibration.py` already logs CSV rows. Extend it to log `timestamp, 6 channels, tap`. `train_steps.py` loads the CSVs, cuts windows with stride 16, trains with Adam (lr 1e-3, about 30 epochs), and saves `step_net.pt`. Printing loss per epoch also gives the data for the back-propagation animation.

**Inference (on the board).** Two options. Preferred: export weights to a `.npz` and run the forward pass in numpy in `main.py`, because PyTorch on the Uno Q is heavy and the net is tiny. Fallback: `torch.jit.trace` and run it with torch if the board has it installed. Each second: take the last 128 samples, run the net, find peaks, and increment a running total only for peaks past the last counted one, so overlapping windows never double count.

**Activities to include.** Walking slow, normal and fast, running, standing, hopping, biking (should count 0), and arm swinging while standing (should count 0). Non-step motion in the data is what teaches the net not to count.

## Steps

1. Create the folder (done): `09-Step-Counter/` copied from `08-Gait-Identifier/`.
2. Extend `calibration.py` and the web page with the Tap button and a timestamp column. Verify the CSV looks right.
3. Collect data: about 10 minutes total, several people or several sensor placements if possible. Hold out whole trials, not random windows, for testing.
4. Write `step_net.py` and `train_steps.py`. First goal: loss goes down and peaks line up with taps on a plot.
5. Write `evaluate.py` (below) and iterate: window size, channels, thresholds.
6. Port inference to `main.py`, add the live step count to the page, test on the board.
7. Record the demo video: walk a known count (for example 10, 20, 50 steps) with the counter visible.
8. Write the README: layout, three-paragraph explanation, reflection answers, GitHub link.
9. Make the Claude animation: data window → conv layers → step curve → loss → back-propagation → weight update, ending on the peak-counted result.
10. Commit as `09-Step-Counter` in ani-ga/ME193 and add a row to the top-level README table.

## Uncertainty analysis

- Run N held-out trials of known true count (target 15 or more, mixed speeds). For each, record true steps and counted steps.
- Report mean error, mean absolute error, and percent error per trial, plus mean and standard deviation of percent error, and a 95% interval (about mean ± 2 sd).
- Plot counted vs. true with a y = x line, and error vs. walking speed.
- Compare against a baseline (simple peak-picking on accel magnitude, no net) so the net's improvement is measurable.
- Note sources of error: tap timing noise in the labels, sensor placement shift, very slow shuffling steps, double peaks while running, and the one-step edge effect at the start and end of each trial.

## Risks

- **Too little data.** The net will memorize. Mitigate with augmentation (small time scaling, noise, random channel gain) and a small model.
- **Tap labels are late.** People tap a bit after foot strike. Estimate the average lag from the data and shift the labels.
- **Torch on the board.** Plan for the numpy export from the start.
- **Time.** The assignment was due Nov 3 and today is Nov 4, so check with the instructor about a late submission.
