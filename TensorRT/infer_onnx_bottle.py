import onnxruntime
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import time
import os


def load_image_cls_bottle(img):
    img = cv2.resize(img, (90, 270))
    img_tensor = image.img_to_array(img)  
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255   
    return img_tensor

classes_cls_bottle = {0: 'GOOD',
                      1: 'ERROR'}

ort_session = onnxruntime.InferenceSession("TensorRT/model_rt/model_cls_bottle.onnx")


results_e = []
results_g = []

path_error = "TensorRT/images/bottle_pepsi/error"

path = path_error
folder = os.listdir(path)
total_e = len(folder)
for i in range(total_e):
    
    img = cv2.imread(os.path.join(path,folder[i]))
    start = time.time()
    img = load_image_cls_bottle(img)
    outputs = ort_session.run(None, {'x': img})
    end = time.time()

    predicted_class_index = np.argmax(outputs[0], axis=1)[0]
    predicted_class = classes_cls_bottle[predicted_class_index]
    results_e.append(predicted_class)
num_error = 0
for i in results_e:
    if i == "ERROR":
        num_error += 1

path_good = "TensorRT/images/bottle_pepsi/good"
path = path_good
folder = os.listdir(path)
total_g = len(folder)
for i in range(total_g):
    
    img = cv2.imread(os.path.join(path,folder[i]))
    start = time.time()
    img = load_image_cls_bottle(img)
    outputs = ort_session.run(None, {'x': img})
    end = time.time()

    predicted_class_index = np.argmax(outputs[0], axis=1)[0]
    predicted_class = classes_cls_bottle[predicted_class_index]
    results_g.append(predicted_class)
    
num_good = 0
for i in results_g:
    if i == "GOOD":
        num_good += 1
        
# print("Result:", predicted_class, "----", round(outputs[0][0][np.argmax(outputs[0][0])],6))
# print("Time:", end-start ,"(s)")

print()
print("---bottle-ONNX----")
print()
print("good", num_good/total_g)
print("error", num_error/total_e)