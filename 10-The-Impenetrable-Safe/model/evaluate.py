"""Test accuracy + confusion matrix. Usage: python -m model.evaluate --test Carol"""
import argparse, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import config
from model.net import FingerNet
from model.dataset import FingerDataset, list_samples, split_by_person

ap = argparse.ArgumentParser(); ap.add_argument("--test", nargs="+", required=True)
args = ap.parse_args()
_, _, te = split_by_person(list_samples(), [], args.test)
net = FingerNet(); net.load_state_dict(torch.load(config.OUT_DIR / "fingernet.pt")); net.eval()
ds = FingerDataset(te); y_true, y_pred = [], []
with torch.no_grad():
    for x, y in ds:
        y_pred.append(net(x[None]).argmax(1).item()); y_true.append(y)
print("test accuracy:", np.mean(np.array(y_true) == np.array(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
fig, ax = plt.subplots(); ax.imshow(cm, cmap="Blues")
for i in range(6):
    for j in range(6): ax.text(j, i, cm[i, j], ha="center", va="center")
ax.set(xlabel="predicted", ylabel="true", title="Confusion matrix (watch the 3!)")
fig.savefig(config.OUT_DIR / "confusion_matrix.png", dpi=150)
