# Vision-Based UAV Detection

A YOLOv11-based object detection system for identifying Unmanned Aerial Vehicles (UAVs/Drones) in real-time, with optimized training to minimize false positives from birds.

## 🎯 Project Overview

This project trains a deep learning model to accurately detect drones in images and video streams. The system is specifically designed to distinguish UAVs from visually similar objects like birds, making it suitable for security, airspace monitoring, and counter-drone applications.

## 📁 Project Structure

```
vision-based-uav-detection/
├── Dataset/                    # YOLO-formatted dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── runs/                       # Training outputs
│   └── uav_detection/
│       └── weights/
│           ├── best.pt        # Best model checkpoint
│           └── last.pt        # Latest model checkpoint
├── data.yaml                   # Dataset configuration
├── train.py                    # Training script
├── prepare_dataset.py          # Dataset preprocessing
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)

### Installation

```bash
pip install ultralytics
```

### Training

1. **Prepare Dataset** (First time only):
   ```bash
   python prepare_dataset.py
   ```
   This processes the dataset to treat bird images as negative samples, reducing false positives.

2. **Train Model**:
   ```bash
   python train.py
   ```

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/uav_detection/weights/best.pt")

# Predict on image
results = model.predict("path/to/image.jpg", conf=0.5)
results[0].show()
```

## ⚙️ Configuration

Training parameters in `train.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | YOLOv11s | Small variant, balanced speed/accuracy |
| Epochs | 15 | Training iterations |
| Image Size | 640×640 | Input resolution |

## 📊 Dataset

- **Source**: Roboflow - Drone,Birds dataset
- **Images**: ~20,000 annotated images
- **Classes**: 1 (UAV)
- **Format**: YOLO v7 PyTorch
- **Resolution**: 640×640 pixels

### Key Preprocessing

Bird images are included as **negative samples** (empty labels) rather than a separate class. This teaches the model to explicitly *not* detect birds, significantly reducing false positive rates.

## 📈 Model Performance (Test Set)

The model was evaluated on an unseen Test Set, demonstrating exceptional detection capabilities.

| Metric | Value | Interpretation |
|:---|:---|:---|
| **mAP50** | **90.72%** | Excellent accuracy in detecting small drones. |
| **mAP50-95** | **46.96%** | High precision in bounding box placement. |
| **Precision** | **90.96%** | Extremely low false positive rate (birds are not detected). |
| **Recall** | **80.38%** | Reliable detection of present threats. |

### Visual Results

#### Training Metrics
![Training Results](final_uav_results/content/runs/uav_colab_drive/results.png)

#### Confusion Matrix
![Confusion Matrix](final_uav_results/content/runs/uav_colab_drive/confusion_matrix.png)

#### Precision-Recall Curve
![PR Curve](final_uav_results/content/runs/uav_colab_drive/BoxPR_curve.png)

## 🛡️ Strategic Application & Future Work

This project works as the **"Sense"** layer in a wider autonomous defense pipeline:
1.  **Sense (YOLOv11)**: Passively detects potential threats without emitting radar signals.
2.  **Confirm (TCN)**: Verifies targets by analyzing flight trajectory (distinguishing rigid drone motion from flapping birds).
3.  **Neutralize (SAC)**: Uses Reinforcement Learning to continuously track and intercept the confirmed threat.

## 📝 License

Dataset: CC BY 4.0

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) for dataset hosting
