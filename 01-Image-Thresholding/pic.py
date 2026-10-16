from pathlib import Path

import cv2

IMAGE_PATH = Path(__file__).parent / "IMG_5534 (1).jpg"

MORPH_OPS = {
    0: None,
    1: cv2.MORPH_ERODE,
    2: cv2.MORPH_DILATE,
    3: cv2.MORPH_OPEN,
    4: cv2.MORPH_CLOSE,
    5: cv2.MORPH_GRADIENT,
}
MORPH_OP_NAMES = {
    0: "None",
    1: "Erode",
    2: "Dilate",
    3: "Open",
    4: "Close",
    5: "Gradient",
}
KERNEL_SHAPES = {0: cv2.MORPH_RECT, 1: cv2.MORPH_ELLIPSE, 2: cv2.MORPH_CROSS}


def main() -> None:
    gray = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    window = "Grayscale (drag sliders, s to save, q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", window, 127, 255, lambda _: None)
    cv2.createTrackbar("Morph Op", window, 0, max(MORPH_OPS), lambda _: None)
    cv2.createTrackbar("Kernel Shape", window, 0, max(KERNEL_SHAPES), lambda _: None)
    cv2.createTrackbar("Kernel Size", window, 1, 10, lambda _: None)
    cv2.createTrackbar("Iterations", window, 1, 10, lambda _: None)

    while True:
        thresh_val = cv2.getTrackbarPos("Threshold", window)
        _, bw = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        morph_op = MORPH_OPS[cv2.getTrackbarPos("Morph Op", window)]
        if morph_op is not None:
            kernel_size = cv2.getTrackbarPos("Kernel Size", window) * 2 + 1
            kernel_shape = KERNEL_SHAPES[cv2.getTrackbarPos("Kernel Shape", window)]
            kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
            iterations = max(cv2.getTrackbarPos("Iterations", window), 1)
            bw = cv2.morphologyEx(bw, morph_op, kernel, iterations=iterations)

        cv2.imshow(window, bw)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("s"):
            output = IMAGE_PATH.with_stem(IMAGE_PATH.stem + "_bw")
            cv2.imwrite(str(output), bw)
            op_name = MORPH_OP_NAMES[cv2.getTrackbarPos("Morph Op", window)]
            print(f"Saved black & white image to {output} (threshold={thresh_val}, morph={op_name})")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
