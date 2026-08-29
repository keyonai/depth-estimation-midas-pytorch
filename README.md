# Depth Estimation with MiDaS

Generates a depth map from a single RGB image using Intel's MiDaS model. No depth camera needed — bright colors are close, dark is far.

## Run it

```bash
pip install -r requirements.txt

# webcam
python depth_estimate.py

# static image
python depth_estimate.py path/to/image.jpg
```

## Docker

```bash
docker build -t depth-estimation-midas-pytorch .
docker run --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix depth-estimation-midas-pytorch
```

> Webcam passthrough requires Linux. On Windows, just run it natively.
