from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Test training with minimal epochs
results = model.train(data="/path/to/xView.yaml", epochs=1, imgsz=640, verbose=True)