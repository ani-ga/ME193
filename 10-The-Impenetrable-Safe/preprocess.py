"""Shared preprocessing. Used for capture, training AND live inference (computer and UNO Q),
so the network always sees images prepared in exactly the same way."""
import cv2
import numpy as np
import config

def open_camera():
    """Open the USB webcam at the shared resolution (used by every script)."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    return cap

def crop_frame(frame_bgr):
    """Cut out the red-square region from a full camera frame."""
    x, y, w, h = config.load_crop()
    return frame_bgr[y:y + h, x:x + w]

def to_gray_small(crop_bgr):
    """Crop -> grayscale -> IMG_SIZE x IMG_SIZE uint8 image. This is what we save to disk."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)

def to_tensor_array(gray_small):
    """uint8 image -> float32 array of shape (1, H, W) scaled to 0..1."""
    return (gray_small.astype(np.float32) / 255.0)[None, :, :]

def frame_to_array(frame_bgr):
    """Full camera frame -> network input array (1, H, W)."""
    return to_tensor_array(to_gray_small(crop_frame(frame_bgr)))
