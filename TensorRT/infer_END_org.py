import torch
import json
import numpy as np
import time
from __init__ import TensorrtBase
import cv2
import os

import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.preprocessing import image


model_cls_bottle = load_model("model_set/classification/org/model_cls_label.h5")
classes_cls_bottle = { 0:'GOOD',
                      1:'ERROR',}

def preprocess_image(image_path):
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (90, 90))
    img_tensor = image.img_to_array(img)  
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255   
    return img_tensor

def predict_cnn_bottle(img):
    pred_bottle = model_cls_bottle.predict(img,verbose = False)
    value = classes_cls_bottle[np.argmax(pred_bottle)]
    return value


path = "TensorRT/images/label_pepsi/test"

folder = os.listdir(path)
label_real = []
label_predict = []

for i in folder:
    if i[0] == "e":
        label_real.append(1)
    else:
        label_real.append(0)


def INFER():
    for i in range(len(folder)):
        PATH = os.path.join(path,folder[i])
        input_image = preprocess_image(PATH)
        result = predict_cnn_bottle(input_image)
        if result == "ERROR":
            label_predict.append(1)
        else:
            label_predict.append(0)
start = time.time()
INFER()
end = time.time()

time_infer = end - start

print()
print("---bottle-Origin CNN----")
print()
print("Acc:", accuracy_score(label_real, label_predict))
print()
print("Number of images infer:", len(folder))
print("With time:", round(time_infer, 4), " s")
print()
