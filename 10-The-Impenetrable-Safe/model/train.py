"""Train the CNN. Usage: python -m model.train --val Bob --test Carol
(--val / --test are person names; leave empty to hold out nobody - only OK for a first try)."""
import argparse, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import config
from model.net import FingerNet
from model.dataset import FingerDataset, list_samples, split_by_person

ap = argparse.ArgumentParser()
ap.add_argument("--val", nargs="*", default=[]); ap.add_argument("--test", nargs="*", default=[])
args = ap.parse_args()

torch.manual_seed(config.SEED)
config.OUT_DIR.mkdir(exist_ok=True)
tr, va, te = split_by_person(list_samples(), args.val, args.test)
print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
train_dl = DataLoader(FingerDataset(tr, train=True), config.BATCH_SIZE, shuffle=True)
val_dl = DataLoader(FingerDataset(va), config.BATCH_SIZE) if va else None

net = FingerNet()
loss_fn = nn.CrossEntropyLoss()                       # classification loss
opt = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
hist = {"loss": [], "train_acc": [], "val_acc": []}

def accuracy(dl):
    net.eval(); right = n = 0
    with torch.no_grad():
        for x, y in dl:
            right += (net(x).argmax(1) == y).sum().item(); n += len(y)
    return right / max(n, 1)

for epoch in range(config.EPOCHS):
    net.train(); total = right = n = 0
    for x, y in train_dl:
        opt.zero_grad()
        out = net(x)                    # forward pass
        loss = loss_fn(out, y)
        loss.backward()                 # backpropagation: gradient of loss for every weight
        opt.step()                      # gradient descent step
        total += loss.item() * len(y); right += (out.argmax(1) == y).sum().item(); n += len(y)
    hist["loss"].append(total / n); hist["train_acc"].append(right / n)
    hist["val_acc"].append(accuracy(val_dl) if val_dl else float("nan"))
    print(f"epoch {epoch+1:2d}  loss {hist['loss'][-1]:.4f}  train {hist['train_acc'][-1]:.3f}  val {hist['val_acc'][-1]:.3f}")

torch.save(net.state_dict(), config.OUT_DIR / "fingernet.pt")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist["loss"]); ax[0].set(title="Loss vs epoch", xlabel="epoch")
ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
ax[1].set(title="Accuracy vs epoch", xlabel="epoch"); ax[1].legend()
fig.savefig(config.OUT_DIR / "training_curves.png", dpi=150)
