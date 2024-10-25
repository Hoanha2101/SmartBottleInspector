from ultralytics import YOLO
model = YOLO("model_set/detect/best.pt")
model.export(format="engine", half=True)