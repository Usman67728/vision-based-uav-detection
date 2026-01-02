from ultralytics import YOLO
import os

def main():
    # Load model
    model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=r"e:\Computer_Vision\vision-based-uav-detection\data.yaml", 
        epochs=30, 
        imgsz=640, 
        project=r"e:\Computer_Vision\vision-based-uav-detection\runs", 
        name="uav_detection",
        exist_ok=True
    )

    # Validate
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # Export
    model.export(format="onnx")

if __name__ == "__main__":
    main()
