import torch
import json
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
from __init__ import TensorrtBase
import cv2
import os

from tensorflow.keras.preprocessing import image

def preprocess_image(image_path):
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (90, 90))
    img_tensor = image.img_to_array(img)  
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255  
    return img_tensor

input_names = ['x']
output_names = ['dense_1']
batch = 1

path_error = "TensorRT/images/label_pepsi/error"
path_good = "TensorRT/images/label_pepsi/good"

net = TensorrtBase("model_set/model_cls_label_ResNet34_region.trt",
                   input_names=input_names,
                   output_names=output_names,
                   max_batch_size=batch,
                   )

images = np.random.rand(1, 90, 90, 3).astype(np.float32)
images = np.ascontiguousarray(images)

binding_shape_map = {
    "x": images.shape,
    }

classes = ['GOOD','ERROR']

results_e = []
results_g = []

def INFER(total_status,path_status,results_status):
    net.cuda_ctx.push()
    for i in range(total_status):
        PATH = os.path.join(path_status,folder[i])
        input_image = preprocess_image(PATH)
        # input_image = input_image.transpose((0, 3, 1, 2))
        images = np.ascontiguousarray(input_image).astype(np.float32)
        inf_in_list = [images]
        inputs, outputs, bindings, stream = net.buffers
        if binding_shape_map:
            net.context.set_optimization_profile_async(0, stream.handle)
            for binding_name, shape in binding_shape_map.items():
                net.context.set_input_shape(binding_name, shape)
            for i in range(len(inputs)):
                inputs[i].host = inf_in_list[i]
                cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
            stream.synchronize()
            net.context.execute_async_v2(
                                bindings=bindings,
                                stream_handle=stream.handle)  
            for i in range(len(outputs)):
                cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)
            stream.synchronize()
            trt_outputs = [out.host.copy() for out in outputs]
        out = trt_outputs[0].reshape(batch,-1)
        pred = np.argmax(out[0])
        results_status.append(classes[pred])
    net.cuda_ctx.pop()

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
print("---bottle-TensorRT----")
print()
print("Good acc:", num_good/total_g)
print("Error acc:", num_error/total_e)
print()
print("Number of images infer:", num_image_infer)
print("With time:", time_infer)
print()
