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

Trained on 13 recorded trials (about 6 minutes of data from 1 person, sensor on the ankle), tested on 3 trials held out of training (`python/results/metrics.json`, `accuracy.png`). Peak settings (threshold 0.6, minimum gap 0.35 s) were chosen on the training trials only. With the first settings (0.5 and 0.25 s) the net overcounted by about 30% because one step produced two peaks.

| Held-out trial | True steps | Neural net | Baseline (no net) |
|---|---|---|---|
| walking | 35 | 40 | 39 |
| walking | 42 | 44 | 46 |
| not stepping | 0 | 0 | 2 |

| | Neural net | Baseline (peak-pick on \|accel\|) |
|---|---|---|
| Held-out trials | 3 | 3 |
| Mean error (steps) | +2.33 | +3.33 |
| Mean absolute error (steps) | 2.33 | 3.33 |
| Std of error (steps) | 2.52 | 1.15 |
| Mean absolute % error (walking trials) | 9.5% | 10.5% |
| False steps on the not-step trial | 0 | 2 |

**Live test on the board:** walking 20 steps counted by hand, the page showed 20.

**How much to trust this.** Only 3 trials were held out, so these averages are rough: one trial can move them by a couple of steps, and the 95% interval that `evaluate.py` prints (about -2.6 to +7.3 steps for the net) rests on 3 points. Both methods overcount slightly on walking; the net's clear advantage so far is rejecting non-step motion (0 false steps on every one of the 6 not-step trials, held-out or not). All trials come from one person and one sensor placement, so accuracy on someone else, or with the sensor strapped differently, is untested. For a stronger analysis, record fresh test trials with known counts and run `python evaluate.py --test-dir <folder>`.

Known error sources: tap timing in the labels (people tap about 0.15 s late; `--lag` corrects the average), sensor placement shifts between trials, very slow or shuffling steps, double peaks at fast cadence, and edge effects at the start and end of a trial. The Bridge delivers only about 15 samples per second (each reading takes six separate calls), so the sharp impact at each foot strike is blurred; sending all six values in one call would likely help.

## Reflection questions

_Add the questions from the assignment page and your answers here._
