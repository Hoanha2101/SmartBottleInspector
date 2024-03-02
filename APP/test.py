from library import *
from ultralytics import YOLO
import cv2
from utils import *
yolo = YOLO("model_set/water_level.pt")

img = cv2.imread("label_361.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

results = yolo(img, classes = 0, verbose = False)

for result in results[0]:
        boxes = result.boxes.cpu().numpy() 

        for i in range(len(boxes)):
            box = boxes[i]
            
            if box.cls[0] == 0:
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                region_bottle = img[y_min_bottle:y_max_bottle,x_min_bottle:x_max_bottle]
                region_label  = region_bottle[int(region_bottle.shape[0]*0.42) : int(region_bottle.shape[0]*0.74),]
                
                rt = predict_cnn_label(region_label)
                
                print(rt)