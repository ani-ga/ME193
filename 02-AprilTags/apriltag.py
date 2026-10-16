"""Generate an AprilTag image using OpenCV's built-in AprilTag dictionaries."""

import cv2
import matplotlib.pyplot as plt

PAGE_SIZES_IN = {
    "letter": (8.5, 11.0),
    "a4": (8.27, 11.69),
}

# Common AprilTag families available in cv2.aruco
FAMILIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def generate_apriltag(tag_id, family="36h11", tag_size_px=200, border_bits=1, output_path=None):
    """Generate an AprilTag image.

    Args:
        tag_id: ID of the tag within the family.
        family: One of "16h5", "25h9", "36h10", "36h11".
        tag_size_px: Size (in pixels) of the tag image, excluding white margin.
        border_bits: Width (in bits) of the black border around the tag payload.
        output_path: If given, saves the image to this path.

    Returns:
        The generated tag as a numpy array (grayscale image).
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Choose from {list(FAMILIES)}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(FAMILIES[family])

    tag_image = cv2.aruco.generateImageMarker(
        aruco_dict, tag_id, tag_size_px, borderBits=border_bits
    )

    # Add a white quiet-zone border around the tag (recommended for reliable detection)
    margin = tag_size_px // 4
    tag_with_margin = cv2.copyMakeBorder(
        tag_image, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=255
    )

    if output_path:
        cv2.imwrite(output_path, tag_with_margin)
        print(f"Saved tag {family} id={tag_id} to {output_path}")

    return tag_with_margin


def export_apriltag_pdf(
    tag_id,
    family="36h11",
    tag_size_cm=5.0,
    border_bits=1,
    page_size="letter",
    output_path="tag.pdf",
    render_px=800,
):
    """Export an AprilTag to a PDF sized so it prints at an exact physical size.

    The tag is placed at the top of the page (not scaled to fill it) so that
    printing at 100% / "actual size" (no "fit to page") yields a tag whose
    black square measures exactly `tag_size_cm` centimeters on a side.

    Args:
        tag_id: ID of the tag within the family.
        family: One of "16h5", "25h9", "36h10", "36h11".
        tag_size_cm: Desired physical side length of the tag's black square, in cm.
        border_bits: Width (in bits) of the black border around the tag payload.
        page_size: "letter" or "a4".
        output_path: Path to write the PDF to.
        render_px: Pixel resolution to render the tag at before embedding (higher = crisper).
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Choose from {list(FAMILIES)}")
    if page_size not in PAGE_SIZES_IN:
        raise ValueError(f"Unknown page_size '{page_size}'. Choose from {list(PAGE_SIZES_IN)}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(FAMILIES[family])
    tag_image = cv2.aruco.generateImageMarker(aruco_dict, tag_id, render_px, borderBits=border_bits)

    page_w_in, page_h_in = PAGE_SIZES_IN[page_size]
    tag_size_in = tag_size_cm / 2.54

    fig = plt.figure(figsize=(page_w_in, page_h_in))

    # Position the tag near the top-center of the page, sized in exact inches.
    left = (page_w_in - tag_size_in) / 2 / page_w_in
    top_margin_in = 1.0
    bottom = (page_h_in - top_margin_in - tag_size_in) / page_h_in
    width = tag_size_in / page_w_in
    height = tag_size_in / page_h_in

    ax = fig.add_axes((left, bottom, width, height))
    ax.imshow(tag_image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")

    fig.text(
        0.5, bottom - 0.02,
        f"tag{family} id={tag_id} — black square = {tag_size_cm:.2f} cm\n"
        "Print at 100% / Actual Size (NOT 'Fit to page')",
        ha="center", va="top", fontsize=10,
    )

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved {tag_size_cm} cm tag {family} id={tag_id} to {output_path}")


if __name__ == "__main__":
    generate_apriltag(tag_id=0, family="36h11", tag_size_px=200, output_path="tag36h11_0.png")
    export_apriltag_pdf(tag_id=0, family="36h11", tag_size_cm=5.0, page_size="letter", output_path="tag36h11_0.pdf")
