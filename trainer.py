import torch
from ultralytics import YOLO
import os

dataset_root = 'datas'

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

dataset_yaml = os.path.join(dataset_root, 'coffee_dataset.yaml')

model = YOLO('yolov8n.pt')

results = model.train(
    data=dataset_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    patience=10,
    save=True,
    verbose=True,
    project='coffee_detection',
    name='baseline_run'
)