"""Load images from data/raw/<class>/<person>_<time>.png and augment them for training."""
import cv2, numpy as np, torch
from torch.utils.data import Dataset
import config, preprocess

def list_samples():
    """Returns [(path, label, person)]."""
    out = []
    for k in range(config.NUM_CLASSES):
        for p in sorted((config.DATA_DIR / str(k)).glob("*.png")):
            out.append((p, k, p.stem.split("_")[0]))
    return out

def split_by_person(samples, val_people, test_people):
    """Whole people go to val/test so the test accuracy is honest (new hands)."""
    tr = [s for s in samples if s[2] not in val_people + test_people]
    va = [s for s in samples if s[2] in val_people]
    te = [s for s in samples if s[2] in test_people]
    return tr, va, te

def augment(img, rng):
    """Random rotation, shift, brightness/contrast on a uint8 gray image."""
    s = config.IMG_SIZE
    angle = rng.uniform(-15, 15)
    tx, ty = rng.uniform(-0.08, 0.08, 2) * s
    M = cv2.getRotationMatrix2D((s / 2, s / 2), angle, 1.0)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (s, s), borderMode=cv2.BORDER_REPLICATE)
    img = img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-25, 25)
    return np.clip(img, 0, 255).astype(np.uint8)

class FingerDataset(Dataset):
    def __init__(self, samples, train=False):
        self.samples, self.train = samples, train
        self.rng = np.random.default_rng(config.SEED)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label, _ = self.samples[i]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if self.train: img = augment(img, self.rng)
        return torch.from_numpy(preprocess.to_tensor_array(img)), label
