# YOLO26 Course

A hands-on, script-first course that demonstrates how to use Ultralytics **YOLO26** for five core computer vision tasks:

1. Image classification
2. Object detection
3. Oriented object detection (OBB)
4. Instance segmentation
5. Human pose estimation

Each lesson lives in a focused Python script under `scripts/` so students can read, run, and modify the workflow in one place.

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── scripts
│   ├── classify.py
│   ├── detect.py
│   ├── oriented_detect.py
│   ├── pose.py
│   └── segment.py
└── .gitignore
```

## Prerequisites

- Python 3.9+
- macOS, Linux, or Windows

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quickstart

### 1) Image Classification

```bash
python scripts/classify.py \
  --source https://ultralytics.com/images/bus.jpg
```

### 2) Object Detection

```bash
python scripts/detect.py \
  --source https://ultralytics.com/images/bus.jpg
```

### 3) Oriented Object Detection (OBB)

```bash
python scripts/oriented_detect.py \
  --source https://ultralytics.com/images/bus.jpg
```

### 4) Instance Segmentation

```bash
python scripts/segment.py \
  --source https://ultralytics.com/images/bus.jpg
```

### 5) Human Pose Estimation

```bash
python scripts/pose.py \
  --source https://ultralytics.com/images/zidane.jpg
```

By default, outputs are saved under `runs/` with task-specific subfolders. You can customize the model, confidence threshold, device, and output location using CLI flags in each script.

## Notes on YOLO26 Models

The scripts default to YOLO26 nano checkpoints for fast execution:

- Classification: `yolo26n-cls.pt`
- Detection: `yolo26n.pt`
- Oriented detection: `yolo26n-obb.pt`
- Segmentation: `yolo26n-seg.pt`
- Pose estimation: `yolo26n-pose.pt`

You can swap in `s`, `m`, `l`, or `x` versions for higher accuracy.

## Next Steps for the Course

- Add dataset-driven training notebooks for each task.
- Include evaluation scripts and metrics visualizations.
- Add export demos (ONNX, CoreML, TensorRT) for deployment.

---

Built with the Ultralytics YOLO26 models: https://docs.ultralytics.com/models/yolo26/
