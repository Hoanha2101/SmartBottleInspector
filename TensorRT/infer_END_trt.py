import torch
import json
import numpy as np
import time
from __init__ import TensorrtBase
import cv2
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.preprocessing import image


input_names = ['x']
output_names = ['dense_22']
batch = 1

path_error = "TensorRT/images/bottle_pepsi/error"
path_good = "TensorRT/images/bottle_pepsi/good"

net = TensorrtBase("model_set/classification/tensorRT/model_cls_bottle.trt",
                   input_names=input_names,
                   output_names=output_names,
                   max_batch_size=batch,
                   )

images = np.random.rand(1, 270, 90, 3).astype(np.float32)
images = np.ascontiguousarray(images)

binding_shape_map = {
    "x": images.shape,
    }

def preprocess_image(image_path):
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (90, 270))
    img_tensor = image.img_to_array(img)  
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255   
    return img_tensor



path = "TensorRT/images/bottle_pepsi/test"

folder = os.listdir(path)
label_real = []
label_predict = []

for i in folder:
    if i[0] == "e":
        label_real.append(1)
    else:
        label_real.append(0)


def INFER():
    net.cuda_ctx.push()
    for i in range(len(folder)):
        PATH = os.path.join(path,folder[i])
        input_image = preprocess_image(PATH)
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
        label_predict.append(pred)
    net.cuda_ctx.pop()
        
start = time.time()
INFER()
end = time.time()

time_infer = end - start

print()
print("---bottle-TensorRT CNN----")
print()
print("Acc:", accuracy_score(label_real, label_predict))
print()
print("Number of images infer:", len(folder))
print("With time:", round(time_infer, 4), " s")
print()




# import torch
# import json
# import numpy as np
# import time
# from __init__ import TensorrtBase
# import cv2
# import os
# import pycuda.driver as cuda
# import pycuda.autoinit
# import tensorflow as tf
# from tensorflow import keras
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import accuracy_score
# from keras.models import load_model
# from keras.preprocessing import image


# input_names = ['x']
# output_names = ['dense_44']
# batch = 1

# net = TensorrtBase("model_set/classification/tensorRT/model_cls_label.trt",
#                    input_names=input_names,
#                    output_names=output_names,
#                    max_batch_size=batch,
#                    )

# images = np.random.rand(1, 90, 90, 3).astype(np.float32)
# images = np.ascontiguousarray(images)

# binding_shape_map = {
#     "x": images.shape,
#     }

# def preprocess_image(image_path):
#     # Đọc ảnh bằng OpenCV
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (90, 90))
#     img_tensor = image.img_to_array(img)  
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor /= 255   
#     return img_tensor


# path = "TensorRT/images/label_pepsi/test"

# folder = os.listdir(path)
# label_real = []
# label_predict = []

# for i in folder:
#     if i[0] == "e":
#         label_real.append(1)
#     else:
#         label_real.append(0)


# def INFER():
#     net.cuda_ctx.push()
#     for i in range(len(folder)):
#         PATH = os.path.join(path,folder[i])
#         input_image = preprocess_image(PATH)
#         images = np.ascontiguousarray(input_image).astype(np.float32)
#         inf_in_list = [images]
#         inputs, outputs, bindings, stream = net.buffers
#         if binding_shape_map:
#             net.context.set_optimization_profile_async(0, stream.handle)
#             for binding_name, shape in binding_shape_map.items():
#                 net.context.set_input_shape(binding_name, shape)
#             for i in range(len(inputs)):
#                 inputs[i].host = inf_in_list[i]
#                 cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
#             stream.synchronize()
#             net.context.execute_async_v2(
#                                 bindings=bindings,
#                                 stream_handle=stream.handle)  
#             for i in range(len(outputs)):
#                 cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)
#             stream.synchronize()
#             trt_outputs = [out.host.copy() for out in outputs]
#         out = trt_outputs[0].reshape(batch,-1)
#         pred = np.argmax(out[0])
#         label_predict.append(pred)
#     net.cuda_ctx.pop()
        
# start = time.time()
# INFER()
# end = time.time()

# time_infer = end - start

# print()
# print("---label-TensorRT CNN----")
# print()
# print("Acc:", accuracy_score(label_real, label_predict))
# print()
# print("Number of images infer:", len(folder))
# print("With time:", round(time_infer, 4), " s")
# print()