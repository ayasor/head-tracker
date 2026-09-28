# Head Tracker — low-latency head tracking

Computer Vision project (UPF). Detects people with a webcam and draws a crosshair on their
head in real time, with the goal of **minimizing end-to-end latency** (capture → display).

## How it works

```
[capture]    V4L2 camera ──► latest frame + driver timestamp
[inference]  GPU pre-processing + YOLO-pose (FP16, CUDA Graph) ──► NMS / end-to-end
             ──► IoU tracker ──► One Euro filter + prediction ──► crosshair + HUD
[display]    imshow / waitKey / video (main thread, never blocks inference)
```

- **Capture in its own thread** that only keeps the most recent frame: stale images from the
  driver buffer are never processed.
- **Inference with a CUDA Graph**: a *nano* network on GPU is bound by Python launching
  hundreds of kernels, not by compute. Recording pre-processing + network into a graph brings
  inference down from ~14 ms to ~3 ms.
- **Lightweight IoU tracker** instead of ByteTrack (which in ultralytics 8.4 adds ~11 ms of
  optical flow per frame).
- **Head center** from the nose/eyes/ears; if they aren't visible, it is estimated from the
  shoulders or the bounding box.
- **One Euro filter** per person (no jitter at rest, little lag in motion) and a
  **constant-velocity prediction** that moves the crosshair ahead by the time it takes to
  reach the screen.

## Performance

Measured on an RTX 5060 Laptop GPU with a 640x480 @ 30 fps webcam:

| | Original | Current |
|---|---|---|
| FPS | 15 | ~27–29 (limited by the camera) |
| Inference | 24 ms | ~3 ms |
| Capture → display (same clock) | ~26 ms | ~10 ms |

The HUD measures from the V4L2 driver timestamp, so it also includes the USB transfer
(~30 ms), which was not counted before.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and, for full speed, an NVIDIA GPU with CUDA. The weights
(`yolo26n-pose.pt`) are downloaded automatically on first run.

## Usage

```bash
python head_tracker.py                              # webcam 0, yolo26n-pose
python head_tracker.py --exposure 150 --brighten 3  # manual exposure (avoids dropping to 15 fps)
python head_tracker.py --model yolo11n-pose.pt      # a different model
python head_tracker.py --source video.mp4 --save out.mp4
python head_tracker.py --no-show --max-frames 300   # headless benchmark
```

Keys: `q` / `ESC` to quit, `p` to toggle prediction.

Useful options:

| Option | Description |
|---|---|
| `--exposure N` | Manual exposure (V4L2, units of 100 µs). Auto-exposure in low light drops the camera to 15 fps. |
| `--brighten X` | Digital gain if the image is too dark. |
| `--imgsz N` | Long side of the network input. |
| `--no-predict` | Disables latency compensation. |
| `--extra-lead-ms N` | Extra latency (exposure, monitor) to compensate for. |
| `--min-cutoff`, `--beta`, `--d-cutoff` | One Euro filter parameters. |
| `--no-graph` | Disables the CUDA Graph. |

`python head_tracker.py --help` lists all of them.

## Citation

See [`CITATION.cff`](CITATION.cff) (GitHub shows a *Cite this repository* button).

## License

[MIT](LICENSE). Note: this project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is distributed under AGPL-3.0.
