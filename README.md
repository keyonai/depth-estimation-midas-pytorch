# Depth Estimation â€” MiDaS + PyTorch

Estimates per-pixel depth from a single RGB image using the MiDaS model and PyTorch. Outputs a colorized depth map where bright/warm colors represent objects close to the camera and dark colors represent objects far away. No depth sensor or stereo camera required.

## What it does

- Loads a pretrained MiDaS model from PyTorch Hub
- Runs depth estimation on a static image or live webcam feed
- Outputs a heatmap depth map side by side with the original image
- Saves the result to `depth_output.png` when run on a static image

## Stack

- Python
- PyTorch
- MiDaS (Intel pretrained model via PyTorch Hub)
- OpenCV
- Matplotlib
- NumPy

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Webcam (live feed):**
```bash
python depth_estimate.py
```

**Static image:**
```bash
python depth_estimate.py path/to/image.jpg
```

## Depth map color guide

| Color | Meaning |
|-------|---------|
| Bright white/yellow | Very close to camera |
| Orange/red | Moderately close |
| Purple | Mid-range distance |
| Dark/black | Far from camera |

## Robotics connection

Depth estimation is a core component of robotic perception. This project demonstrates how a robot can approximate object distance using only a standard RGB camera â€” without expensive LiDAR or depth sensors. The depth values produced can be combined with object detection (e.g. YOLO bounding boxes) to estimate real-world distance to detected objects, which is critical for robotic navigation and manipulation.


## Docker

```bash
docker build -t depth-estimation-midas-pytorch .
docker run --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix depth-estimation-midas-pytorch
```

> **Note:** Webcam and display passthrough requires Linux. On Windows/Mac, run natively instead.