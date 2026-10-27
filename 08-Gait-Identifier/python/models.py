"""The two required classifiers.

Model1Threshold: sees only the peak-to-peak acceleration of ONE axis.
Model2KNN: sees features from all 6 IMU channels, classified with a
hand-rolled k-nearest-neighbors - no ML library, just numpy distance math,
so every line can be explained.
"""
import numpy as np

from features import FEATURE_KEYS

ACCEL_PEAK_KEYS = ["peak_ax", "peak_ay", "peak_az"]


class Model1Threshold:
    """Picks whichever accelerometer axis best separates the calibration
    classes, then classifies new windows by comparing that one axis's
    peak-to-peak value against thresholds set halfway between neighboring
    classes' average peaks.
    """

    def __init__(self):
        self.axis_key = "peak_az"
        self.class_order = []   # labels, sorted from least to most "active"
        self.thresholds = []    # len(class_order) - 1 cut points

    def train(self, rows):
        if not rows:
            return

        # Try each accelerometer axis in turn; keep the one whose class means
        # are most spread out (= easiest to tell apart with simple thresholds).
        best_axis, best_spread = self.axis_key, -1.0
        for axis in ACCEL_PEAK_KEYS:
            means_by_label = {}
            for r in rows:
                means_by_label.setdefault(r["label"], []).append(r[axis])
            means = [np.mean(v) for v in means_by_label.values()]
            spread = max(means) - min(means) if len(means) > 1 else 0.0
            if spread > best_spread:
                best_axis, best_spread = axis, spread
        self.axis_key = best_axis

        means_by_label = {}
        for r in rows:
            means_by_label.setdefault(r["label"], []).append(r[self.axis_key])
        self.class_order = sorted(means_by_label, key=lambda label: np.mean(means_by_label[label]))

        means = [np.mean(means_by_label[label]) for label in self.class_order]
        self.thresholds = [(means[i] + means[i + 1]) / 2 for i in range(len(means) - 1)]

    def predict(self, peaks: dict):
        if not self.class_order:
            return "not trained yet"
        value = peaks[self.axis_key]
        idx = sum(1 for t in self.thresholds if value > t)
        return self.class_order[idx]


class Model2KNN:
    """k-nearest-neighbors over all 6 IMU channels' features.

    Training just stores the calibration examples (normalized so no one
    feature dominates by having bigger raw units). Predicting measures the
    Euclidean distance from a new window to every stored example and takes
    a majority vote among the k closest.
    """

    def __init__(self, k=5):
        self.k = k
        self.X = None       # (n_examples, n_features), normalized
        self.y = None       # (n_examples,) labels
        self.mean = None
        self.std = None

    def train(self, rows):
        if not rows:
            return
        X = np.array([[r[key] for key in FEATURE_KEYS] for r in rows])
        self.y = np.array([r["label"] for r in rows])

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid divide-by-zero for a constant feature
        self.X = (X - self.mean) / self.std

    def _distances_to(self, x_normalized):
        return np.sqrt(((self.X - x_normalized) ** 2).sum(axis=1))

    def _vote(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, features: dict):
        if self.X is None:
            return "not trained yet"
        x = np.array([features[key] for key in FEATURE_KEYS])
        x = (x - self.mean) / self.std
        nearest = np.argsort(self._distances_to(x))[: self.k]
        return self._vote(self.y[nearest])

    def leave_one_out_accuracy(self):
        """Classify each training example using all the OTHER examples, and
        see how often that matches its real label. A quick, honest check of
        how well-separated the calibration data is, without needing a
        separate test set.
        """
        if self.X is None or len(self.y) < 2:
            return None
        correct = 0
        for i in range(len(self.y)):
            others = np.arange(len(self.y)) != i
            dists = np.sqrt(((self.X[others] - self.X[i]) ** 2).sum(axis=1))
            k = min(self.k, others.sum())
            nearest = np.argsort(dists)[:k]
            prediction = self._vote(self.y[others][nearest])
            if prediction == self.y[i]:
                correct += 1
        return correct / len(self.y)
