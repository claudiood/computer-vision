import os
from ultralytics import YOLO

DATASET_PATH = os.path.abspath("dataset/class")
modelo = YOLO("yolo26n-cls.pt")

modelo.train(
    data=DATASET_PATH,
    epochs=20,
    imgsz=512,
    batch=8,
    patience=10,
    seed=42,
    project="results",
    name="train1",
    exist_ok=True
)