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
- ~4GB+ VRAM

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
| Batch Size | 4 | Optimized for 4GB VRAM |

## 📊 Dataset

- **Source**: Roboflow - Drone,Birds dataset
- **Images**: ~20,000 annotated images
- **Classes**: 1 (UAV)
- **Format**: YOLO v7 PyTorch
- **Resolution**: 640×640 pixels

### Key Preprocessing

Bird images are included as **negative samples** (empty labels) rather than a separate class. This teaches the model to explicitly *not* detect birds, significantly reducing false positive rates.

## 📈 Expected Results

After training, evaluation metrics will be saved to `runs/uav_detection/`:
- Confusion matrix
- Precision-Recall curves
- Training loss graphs
- Sample predictions

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4GB | 8GB+ |
| RAM | 8GB | 16GB |
| Storage | 5GB | 10GB |

## 📝 License

Dataset: CC BY 4.0

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) for dataset hosting
