# Fine-tunes a YOLOv8 detector to find the green LEGO minifigure.
#
# Dataset: dataset/ (Roboflow YOLOv8 export "Lego Minific v2",
# class "Lego-Minifig", 98 train / 28 valid / 14 test images).
# Training progress and the final weights land under
# runs/detect/green_minifig/weights/best.pt, and best.pt is copied to
# minifig.pt next to this script for minifig_tracker.py to load.

import os
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# Ultralytics deletes every ' from weight-file paths, so absolute paths
# like ".../Fall '26/..." break. Load weights by paths relative to here.
os.chdir(Path(__file__).resolve().parent)

DATASET_DIR = Path("dataset")
EPOCHS = 100  # small dataset (~100 images) benefits from more epochs than the usual 50
IMAGE_SIZE = 640
if torch.cuda.is_available():
    DEVICE = 0  # NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon GPU
else:
    DEVICE = "cpu"

# Roboflow's data.yaml uses paths like "../train/images", which Ultralytics
# resolves to the wrong place. Write a copy with absolute paths instead.
with open(DATASET_DIR / "data.yaml") as f:
    data = yaml.safe_load(f)
data["path"] = str(DATASET_DIR.resolve())
data["train"] = "train/images"
data["val"] = "valid/images"
data["test"] = "test/images"
DATA_YAML = DATASET_DIR / "data_fixed.yaml"
with open(DATA_YAML, "w") as f:
    yaml.safe_dump(data, f)


# Start from COCO-pretrained weights and fine-tune -- much faster than
# training from scratch, and works well with a few dozen images.
model = YOLO("yolov8n.pt")
try:
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig",
        project=str(Path("runs/detect").resolve()),
        device=DEVICE,
    )
except FileNotFoundError:
    # After the last epoch Ultralytics reloads best.pt by its absolute path,
    # which fails because of the '. The weights are already saved, so only
    # carry on if they're there.
    if not (Path(model.trainer.save_dir) / "weights" / "best.pt").exists():
        raise
    print("Skipped Ultralytics' final check (the ' in the folder name breaks it).")

best = Path(os.path.relpath(model.trainer.save_dir)) / "weights" / "best.pt"
shutil.copy(best, "minifig.pt")

# Score the trained model on the held-out test split.
YOLO("minifig.pt").val(data=str(DATA_YAML), split="test", project=str(Path("runs/detect").resolve()))

print(f"Done. Weights saved to {best} and copied to minifig.pt")
