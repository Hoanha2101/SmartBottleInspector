import torch
import json
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
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


model_cls_bottle = load_model("model_set/model_cls_bottle_ResNet18.h5")
classes_cls_bottle = { 0:'GOOD',
                      1:'ERROR',}

def preprocess_image(image_path):
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (90, 270))
    img_tensor = image.img_to_array(img)  
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255   
    return img_tensor

def predict_cnn_bottle(img):
    pred_bottle = model_cls_bottle.predict(img,verbose = False)
    value = classes_cls_bottle[np.argmax(pred_bottle)]
    return value

input_names = ['x']
output_names = ['dense_22']
batch = 1

path_error = "TensorRT/images/bottle_pepsi/error"
path_good = "TensorRT/images/bottle_pepsi/good"

net = TensorrtBase("TensorRT/model_rt/model_cls_bottle.trt",
                   input_names=input_names,
                   output_names=output_names,
                   max_batch_size=batch,
                   )

images = np.random.rand(1, 270, 90, 3).astype(np.float32)
images = np.ascontiguousarray(images)

binding_shape_map = {
    "x": images.shape,
    }

classes = ['GOOD','ERROR']

results_e = []
results_g = []

def INFER(total_status,path_status,results_status):
    for i in range(total_status):
        PATH = os.path.join(path_status,folder[i])
        input_image = preprocess_image(PATH)
        result = predict_cnn_bottle(input_image)
        results_status.append(result)

folder = os.listdir(path_error)
total_e = len(folder)
start_1 = time.time()
INFER(total_e,path_error,results_e)
end_1 = time.time()

folder = os.listdir(path_good)
total_g = len(folder)
start_2 = time.time()
INFER(total_g,path_good,results_g)
end_2 = time.time()

time_infer =  (end_1-start_1) + (end_2-start_2)
num_image_infer = total_e + total_g

num_good = 0
for i in results_g:
    if i == "GOOD":
        num_good += 1

num_error = 0
for i in results_e:
    if i == "ERROR":
        num_error += 1

print()
print("---bottle-Origin CNN----")
print()
print("Good acc:", num_good/total_g)
print("Error acc:", num_error/total_e)
print()
print("Number of images infer:", num_image_infer)
print("With time:", time_infer)
print()