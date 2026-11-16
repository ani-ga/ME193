"""Export the trained PyTorch model to numpy weights for the UNO Q, and check they agree.
Usage: python -m model.export"""
import time, numpy as np, torch
import config
from model.net import FingerNet
from model.np_net import NumpyFingerNet
from model.dataset import list_samples, FingerDataset

net = FingerNet()
net.load_state_dict(torch.load(config.OUT_DIR / "fingernet.pt"))
net.eval()
out = config.OUT_DIR / "fingernet.npz"
np.savez(out, **{k: v.numpy() for k, v in net.state_dict().items()})
print("saved", out)

# Check: numpy and PyTorch must give the same answers on real images.
np_net = NumpyFingerNet(out)
ds = FingerDataset(list_samples()[::max(1, len(list_samples()) // 200)])
worst, agree, t = 0.0, 0, 0.0
for x, _ in ds:
    with torch.no_grad(): ref = net(x[None])[0].numpy()
    t0 = time.perf_counter(); got = np_net.logits(x.numpy()); t += time.perf_counter() - t0
    worst = max(worst, float(np.abs(ref - got).max()))
    agree += int(ref.argmax() == got.argmax())
print(f"{len(ds)} images: biggest score difference {worst:.2e}, same digit {agree}/{len(ds)}, "
      f"numpy {1000 * t / len(ds):.1f} ms/image on this laptop")
