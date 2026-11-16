"""The CNN: 3 conv layers (different kernel sizes), pooling after each, then a linear layer -> 6."""
import torch
import torch.nn as nn
import config

class FingerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layer 1: 1 channel (gray) -> 16 feature maps, big 7x7 window sees coarse shapes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, padding=3)
        # Conv layer 2: 16 -> 32 maps, 5x5 window combines the first features
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        # Conv layer 3: 32 -> 64 maps, small 3x3 window picks out fine detail (fingertips)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()                    # keeps positives, zeroes negatives
        self.pool = nn.MaxPool2d(2)              # keeps the max of each 2x2 block -> halves size
        side = config.IMG_SIZE // 8              # three poolings: /2 /2 /2
        self.fc = nn.Linear(64 * side * side, config.NUM_CLASSES)   # -> 6 scores

    def forward(self, x, verbose=False):
        def step(name, t):
            if verbose: print(f"{name:<8} -> {tuple(t.shape)}")
            return t
        x = step("input", x)
        x = step("conv1", self.pool(self.relu(self.conv1(x))))
        x = step("conv2", self.pool(self.relu(self.conv2(x))))
        x = step("conv3", self.pool(self.relu(self.conv3(x))))
        x = step("flatten", x.flatten(1))
        return step("linear", self.fc(x))        # raw scores (logits); softmax is applied later

if __name__ == "__main__":
    net = FingerNet()
    net(torch.zeros(1, 1, config.IMG_SIZE, config.IMG_SIZE), verbose=True)
    print("parameters:", sum(p.numel() for p in net.parameters()))
