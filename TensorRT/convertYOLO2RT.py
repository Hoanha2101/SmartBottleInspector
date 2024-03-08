from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('model_set/yolov8m.pt')

# Export the model to TensorRT format
model.export(format='engine', device="cuda")  # creates 'yolov8n.engine'

print("Success")