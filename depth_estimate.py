import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# ─────────────────────────────────────────────
# 1. LOAD THE MiDaS MODEL FROM PYTORCH HUB
# ─────────────────────────────────────────────
# MiDaS is a pretrained monocular depth estimation model developed by Intel.
# "MiDaS_small" is the fastest version — good for real-time use.
# Other options: "DPT_Large" (most accurate), "DPT_Hybrid" (balanced)
print("Loading MiDaS model...")
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# ─────────────────────────────────────────────
# 2. SET UP DEVICE (GPU if available, else CPU)
# ─────────────────────────────────────────────
# Using a GPU speeds up inference significantly.
# If no GPU is found, it falls back to CPU automatically.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
midas.to(device)
midas.eval()  # Set model to evaluation mode (disables dropout/batchnorm training behavior)

# ─────────────────────────────────────────────
# 3. LOAD THE MIDAS TRANSFORMS
# ─────────────────────────────────────────────
# MiDaS requires images to be resized and normalized in a specific way.
# The transforms handle this automatically — different model types use different transforms.
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # matches "MiDaS_small" model


def estimate_depth(frame):
    """
    Takes a BGR image (as read by OpenCV) and returns a colorized depth map.

    Steps:
      1. Convert BGR to RGB (MiDaS expects RGB)
      2. Apply MiDaS transforms (resize + normalize)
      3. Run the model to get raw depth values
      4. Normalize depth values to 0-255 range
      5. Apply a colormap to visualize near/far distances
    """

    # ── Step 1: Convert BGR → RGB ──
    # OpenCV reads images in BGR format, but MiDaS expects RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Step 2: Apply MiDaS input transforms ──
    # Resizes and normalizes the image to what the model expects
    input_batch = transform(img_rgb).to(device)

    # ── Step 3: Run the depth estimation model ──
    with torch.no_grad():  # Disable gradient tracking (not training, just predicting)
        prediction = midas(input_batch)

        # Resize the output depth map back to the original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),               # add channel dimension
            size=frame.shape[:2],                  # target size: original image height x width
            mode="bicubic",                        # smooth interpolation
            align_corners=False,
        ).squeeze()                                # remove extra dimensions

    # ── Step 4: Convert to NumPy and normalize to 0–255 ──
    # Raw depth values are floating point — normalize so we can visualize them
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # ── Step 5: Apply colormap ──
    # COLORMAP_MAGMA: bright/white = close, dark/black = far
    # This makes depth visually intuitive at a glance
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    return depth_colormap


def run_on_image(image_path):
    """Load a static image, estimate depth, and show both side by side."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        return

    print("Estimating depth...")
    depth_colormap = estimate_depth(frame)

    # ── Display original image and depth map side by side using Matplotlib ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original image (convert BGR → RGB for correct colors in Matplotlib)
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # Depth map (convert BGR → RGB for Matplotlib display)
    axes[1].imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Depth Map  |  Bright = Close   Dark = Far", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("depth_output.png", dpi=150)  # save result to file
    print("Saved output to depth_output.png")
    plt.show()


def run_on_webcam():
    """Capture live webcam feed and show real-time depth estimation."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Running live depth estimation — press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Estimate depth for the current frame
        depth_colormap = estimate_depth(frame)

        # Stack original and depth map horizontally for side-by-side view
        combined = np.hstack((frame, depth_colormap))
        cv2.imshow("Original  |  Depth Map (Bright=Close, Dark=Far)", combined)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
# Pass an image path as an argument to run on a static image.
# Run with no arguments to use the live webcam feed.
if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_on_image(sys.argv[1])
    else:
        run_on_webcam()
