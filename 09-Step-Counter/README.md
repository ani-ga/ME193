# How Many Steps? 👣

A wearable step counter for the Uno Q + MPU6050 (worn on the ankle). A small PyTorch neural net reads all six IMU channels and finds each foot strike; counting its peaks gives the step count. It grew out of [08-Gait-Identifier](../08-Gait-Identifier/) (same sensor, sketch and phone page).

**Video demo:** _add link_ · **Animation:** open [animation.html](animation.html) in a browser · **Plan:** [PLAN.md](PLAN.md)

## How it works

```
MPU6050 --I2C--> sketch.ino --Bridge--> main.py (50 Hz, timestamped)
                                          |
        phone page <--- live count <--- step_infer.py (numpy forward pass of the trained net)
        (tap to label, record trials)
                                          |
                      data/steps/*.csv --> train_steps.py (PyTorch, laptop) --> step_net.npz
                                          evaluate.py: held-out trials, uncertainty
```

| File | Job |
|---|---|
| `sketch/sketch.ino` | Reads the MPU6050 over I2C, serves the 6 raw values over the Bridge. |
| `python/main.py` | Samples at 50 Hz, runs the live counter, serves the phone page and recording API. |
| `python/step_infer.py` | Numpy-only: resampling, the 8 input channels, the net's forward pass, peak picking, and the no-net baseline. Runs on the board. |
| `python/step_net.py` | The PyTorch model (`StepNet`) and the export that folds BatchNorm into the conv weights. |
| `python/train_steps.py` | Trains on recorded trials, writes `step_net.pt`, `step_net.npz`, `split.json`, loss history. |
| `python/evaluate.py` | Counts steps on held-out trials, compares with true counts and the baseline, writes `results/`. |
| `python/recorder.py` | Saves a labeled trial (IMU samples plus your foot-strike taps) to `data/steps/`. |
| `python/synth_data.py` | Makes **fake** trials so the pipeline can be tested without hardware. Never report numbers from it. |

## Using it

1. Wire the MPU6050 (VCC 3.3V, GND, SDA to A4, SCL to A5) and deploy the app to the Uno Q as in 08. Open `http://<board-ip>:7000` on a phone.
2. **Record data.** There are two categories. *Steps*: walk or run (vary the speed) and tap the big button at every foot strike, left and right, optionally typing the true step count. *Not steps*: standing, biking, arm swinging, hopping, fidgeting; the true count is automatically 0. Aim for 12 or more *steps* trials at mixed speeds and 8 or more *not steps* trials (every 4th trial of a category is held out for testing).
3. **Copy the CSVs** from the board's `python/data/steps/` to the same folder on your laptop.
4. **Train and test** on the laptop:
   ```bash
   pip install torch numpy matplotlib
   cd python
   python train_steps.py        # ~1 minute on CPU
   python evaluate.py           # uncertainty analysis, writes results/
   ```
5. **Deploy the model:** copy `python/step_net.npz` to the board next to `main.py` and restart the app. The page then says "steps (neural net)". Without it the app falls back to the baseline counter.

To test the pipeline with no hardware: `python synth_data.py --out /tmp/fake`, then `python train_steps.py --data /tmp/fake` and `python evaluate.py`.

## The neural net, in three paragraphs

_Draft written by Claude. Rewrite parts in your own words before submitting; the assignment asks for yours and Claude's._

**What it sees and what it says.** The MPU6050 gives six numbers 50 times a second (three accelerations, three rotation rates); we add the overall size of each, giving eight channels, each normalized to a similar scale. The network is a stack of three 1D convolution layers followed by a 1x1 layer. A convolution slides a small set of learned filters along the time axis, so the same pattern detector works wherever in the window a step lands; the later layers use dilation (gaps of 1, 2 and 4 samples) so each moment is judged using about 1.1 seconds of context. For every sample it outputs one number between 0 and 1: how likely a foot strike is right now. Peaks of that curve above 0.6, at least 0.35 s apart, are the steps, and the count is the number of peaks, so overlapping windows never double-count.

**How it learns.** While recording, you tap a button at each foot strike; each tap becomes a smooth bump, which is the target curve the network should reproduce. In PyTorch, each training step feeds a batch of random crops of recorded signal through the network (the forward pass), compares its curve with the target using binary cross-entropy (steps are rare, so they are up-weighted), and calls `loss.backward()`. Back-propagation applies the chain rule layer by layer from the loss back to the input, which tells us, for every weight, how much a small increase would raise or lower the loss. The Adam optimizer then moves each weight a small step against its gradient. Repeating that a thousand times drives the loss down. Random speed changes, gains and noise on each crop keep the net from memorizing the training trials.

**Why this design.** Predicting a count directly would be hard to train and would double-count across overlapping windows; predicting "where are the steps" turns the problem into one the convolution is good at, and the counting step is a plain peak finder we can inspect. The net is tiny (a few thousand weights), so after training we fold BatchNorm into the convolutions and run it in plain numpy on the Uno Q, with no PyTorch needed there.

## Results and uncertainty

**Test set:** 6 fresh trials of exactly 10 steps each (true count typed in, mixed slow/normal/fast), recorded after training and never used for training or for choosing any setting (`python/test_trials/`, `python evaluate.py --test-dir test_trials`). Training used 13 trials (about 6 minutes, one person, sensor on the ankle).

| Trial | True steps | Neural net | Baseline (no net) |
|---|---|---|---|
| 1 | 10 | 12 | 13 |
| 2 | 10 | 11 | 14 |
| 3 | 10 | 11 | 12 |
| 4 | 10 | 14 | 13 |
| 5 | 10 | 12 | 13 |
| 6 | 10 | 11 | 11 |

| | Neural net | Baseline (peak-pick on \|accel\|) |
|---|---|---|
| Mean error | **+1.83 steps** (18%) | +2.67 steps (27%) |
| Std of error | 1.17 steps | 1.03 steps |
| 95% CI of the mean error (Student's t, n = 6) | +0.6 to +3.1 steps | +1.6 to +3.8 steps |
| 95% range for one new 10-step walk | -1.4 to +5.1 steps | -0.2 to +5.5 steps |
| False steps on the 6 not-step trials | 0 | not tested on this set |

In words: on a 10-step walk the net typically reports 11 to 13 steps. It overcounts every time, so the error is mostly a consistent bias (the interval does not include zero), not random noise. The net is a bit better than the no-net baseline, and the lower mean error is clear, but with 6 trials and overlapping intervals the two are not cleanly separated.

**Why it overcounts.** Checking where the extra peaks fall: not at the start or end of a trial, but in the middle, where a normal 0.7 s step is occasionally split into two peaks 0.35-0.5 s apart. The net's minimum peak gap (0.35 s) was chosen on training trials, where steps were slightly faster, and these short trials are slower and more varied. Percent errors look large (18%) partly because the walks are short: one extra step is 10%. On the earlier 20-step live demo the page showed exactly 20.

**Earlier check on 3 held-out training-set trials** (35, 42 and 0 true steps): net 40, 44, 0; baseline 39, 46, 2. The net also counted 0 false steps on all 6 not-step training trials (standing, biking, arm swinging), where the baseline invented 2.

**Limits.** One person, one sensor placement, and a small test set: accuracy on someone else, with the sensor strapped differently, or at running speed is untested. Known error sources: double peaks within a step (above), tap timing in the training labels (people tap about 0.15 s late; `--lag` corrects the average), very slow or shuffling steps, and the sensor sampling only about 15 times a second (each reading takes six separate Bridge calls), which blurs the foot-strike impact; sending all six values in one call would likely help.

**Idea for improving it:** choose the peak gap and threshold by leave-one-trial-out testing across all the trials, or scale the minimum gap to the walker's current cadence, then re-test on a new fresh set.

## Reflection questions

_Add the questions from the assignment page and your answers here._
