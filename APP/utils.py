import cv2
import os
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import torch
from ultralytics import YOLO
from __init__ import TensorrtBase
import pycuda.autoinit
import pycuda.driver as cuda
import csv
from statistics import mode
from deepSort.deep_sort.deep_sort import DeepSort
from deepSort.deep_sort.sort.tracker import Tracker
from PIL import Image
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> UI
>>>>>>> develop

decide_var = 15

"""decide_var: This variable is understood as the number of correct decisions in a list of the first 30 values of data_mask, 
GOOD or ERROR status. If it is greater than this variable, the final conclusion will be given for that bottle of water."""

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

"""------------------------------Yolo Model for modules-----------------------------------"""

# Path YOLOv8 model 
path_DetectMODEL = "model_set/detect/YOLOv8_combine.pt"

# Org model
path_cls_bottle_org = "model_set/classification/org/model_cls_bottle.h5"
path_cls_label_org = "model_set/classification/org/model_cls_label.h5"
path_cls_combine_org = "model_set/classification/org/model_combine.h5"

# TensorRT model
path_cls_bottle_RT = "model_set/classification/tensorRT/model_cls_bottle.trt"
path_cls_label_RT = "model_set/classification/tensorRT/model_cls_label.trt"
path_org_combine_RT = "model_set/classification/tensorRT/model_combine.trt"

<<<<<<< HEAD
=======
=======
from config import *
>>>>>>> UI-Hoan
<<<<<<< HEAD
=======
=======
from config import *
>>>>>>> UI-Hoan
>>>>>>> UI
>>>>>>> develop

MODEL_BOTTLE_AI = YOLO(path_DetectMODEL)
MODEL_WATER_LEVEL_AI = YOLO(path_DetectMODEL)
MODEL_LABEL_AI = YOLO(path_DetectMODEL)

<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
# MODEL_LABEL_AI = YOLO("yolov8m.pt")

>>>>>>> UI-Hoan
>>>>>>> UI
=======
# MODEL_LABEL_AI = YOLO("yolov8m.pt")

>>>>>>> UI-Hoan
>>>>>>> develop

# Initialize DeepSORT tracker   
deep_sort_weights = 'APP/deepSort/deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

"""-------------------------------------------------------------------"""
"""------------------------------Cls BOTTLE-----------------------------------"""

classes_cls_bottle = { 0:'GOOD', 1:'ERROR'}
input_names_bottle = ['x']
<<<<<<< HEAD
output_names_bottle = ['dense_22']
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> UI
output_names_bottle = ['dense_22']
=======
output_names_bottle = [output_names_bottle_str]
>>>>>>> UI-Hoan
<<<<<<< HEAD
=======
=======
output_names_bottle = [output_names_bottle_str]
>>>>>>> UI-Hoan
>>>>>>> UI
>>>>>>> develop
batch_bottle = 1
net_bottle_RT = TensorrtBase(path_cls_bottle_RT,
                              input_names=input_names_bottle,
                              output_names=output_names_bottle,
                              max_batch_size=batch_bottle,
                              gpu_id=0)
binding_shape_map = {"x": (1, 270, 90, 3)}

def predict_cnn_bottle_RT(img):
    net_bottle_RT.cuda_ctx.push()
    
    input_image = load_image_cls_bottle(img)
    images = np.ascontiguousarray(input_image).astype(np.float32)
    inf_in_list = [images]
    inputs, outputs, bindings, stream = net_bottle_RT.buffers
    if binding_shape_map:
        net_bottle_RT.context.set_optimization_profile_async(0, stream.handle)
        for binding_name, shape in binding_shape_map.items():
            net_bottle_RT.context.set_input_shape(binding_name, shape)
        for i in range(len(inputs)):
            inputs[i].host = inf_in_list[i]
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        stream.synchronize()
        net_bottle_RT.context.execute_async_v2(
                            bindings=bindings,
                            stream_handle=stream.handle)
        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)
        stream.synchronize()
        trt_outputs = [out.host.copy() for out in outputs]

    net_bottle_RT.cuda_ctx.pop()
    
    out = trt_outputs[0].reshape(batch_bottle,-1)
    pred = np.argmax(out[0])
    value = classes_cls_bottle[pred]
    return value

def load_image_cls_bottle(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(90,270))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255   
    return img_tensor

model_cls_bottle = load_model(path_cls_bottle_org)

def predict_cnn_bottle(img):
    img = load_image_cls_bottle(img)
    pred_bottle = model_cls_bottle.predict(img,verbose = False)
    value = classes_cls_bottle[np.argmax(pred_bottle)]
    return value



"""-------------------------------------------------------------------"""
"""------------------------------Cls LABEL-----------------------------------"""

classes_cls_label = { 0:'GOOD', 1:'ERROR'}
input_names_label = ['x']
<<<<<<< HEAD
output_names_label = ['dense_44']
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> UI
output_names_label = ['dense_44']
=======
output_names_label = [output_names_label_str]
>>>>>>> UI-Hoan
<<<<<<< HEAD
=======
=======
output_names_label = [output_names_label_str]
>>>>>>> UI-Hoan
>>>>>>> UI
>>>>>>> develop
batch_label = 1
net_label_RT = TensorrtBase(path_cls_label_RT,
                              input_names=input_names_label,
                              output_names=output_names_label,
                              max_batch_size=batch_label,
                              gpu_id=0)
binding_shape_map_label = {"x": (1, 90, 90, 3)}

def predict_cnn_label_RT(img):
    net_label_RT.cuda_ctx.push()
    
    input_image = load_image_cls_label(img)
    images = np.ascontiguousarray(input_image).astype(np.float32)
    inf_in_list = [images]
    inputs, outputs, bindings, stream = net_label_RT.buffers
    if binding_shape_map_label:
        net_label_RT.context.set_optimization_profile_async(0, stream.handle)
        for binding_name, shape in binding_shape_map_label.items():
            net_label_RT.context.set_input_shape(binding_name, shape)
        for i in range(len(inputs)):
            inputs[i].host = inf_in_list[i]
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        stream.synchronize()
        net_label_RT.context.execute_async_v2(
                            bindings=bindings,
                            stream_handle=stream.handle)  
        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)
        stream.synchronize()
        trt_outputs = [out.host.copy() for out in outputs] 
    net_label_RT.cuda_ctx.pop()
    
    out = trt_outputs[0].reshape(batch_bottle,-1)
    pred = np.argmax(out[0])
    value = classes_cls_label[pred]
    return value

classes_cls_label = { 0:'GOOD', 1:'ERROR'}

def load_image_cls_label(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(90,90))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255
    # imshow expects values in the range [0, 1]
    return img_tensor

model_cls_label = load_model(path_cls_label_org)

def predict_cnn_label(img):
    img = load_image_cls_label(img)
    pred_label = model_cls_label.predict(img,verbose = False)
    value = classes_cls_label[np.argmax(pred_label)]
    return value


"""-------------------------------------------------------------------"""
"""------------------------------Cls combine-----------------------------------"""

classes_cls_combine = { 0:'GOOD', 1:'ERROR'}
input_names_combine = ['x']
<<<<<<< HEAD
output_names_combine = ['dense_11']
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> UI
output_names_combine = ['dense_11']
=======
output_names_combine = [output_names_combine_str]
>>>>>>> UI-Hoan
<<<<<<< HEAD
=======
=======
output_names_combine = [output_names_combine_str]
>>>>>>> UI-Hoan
>>>>>>> UI
>>>>>>> develop
batch_combine = 1
net_combine_RT = TensorrtBase(path_org_combine_RT,
                              input_names=input_names_combine,
                              output_names=output_names_combine,
                              max_batch_size=batch_combine,
                              gpu_id=0)
binding_shape_map_combine = {"x": (1, 270, 90, 3)}

def predict_cnn_combine_RT(img):
    net_combine_RT.cuda_ctx.push()
    
    input_image = load_image_cls_combine(img)
    images = np.ascontiguousarray(input_image).astype(np.float32)
    inf_in_list = [images]
    inputs, outputs, bindings, stream = net_combine_RT.buffers
    if binding_shape_map_combine:
        net_combine_RT.context.set_optimization_profile_async(0, stream.handle)
        for binding_name, shape in binding_shape_map_combine.items():
            net_combine_RT.context.set_input_shape(binding_name, shape)
        for i in range(len(inputs)):
            inputs[i].host = inf_in_list[i]
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        stream.synchronize()
        net_combine_RT.context.execute_async_v2(
                            bindings=bindings,
                            stream_handle=stream.handle)
        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)
        stream.synchronize()
        trt_outputs = [out.host.copy() for out in outputs]

    net_combine_RT.cuda_ctx.pop()
    
    out = trt_outputs[0].reshape(batch_combine,-1)
    pred = np.argmax(out[0])
    value = classes_cls_combine[pred]
    return value

def load_image_cls_combine(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(90,270))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255   
    return img_tensor

model_cls_combine = load_model(path_cls_combine_org)

def predict_cnn_combine(img):
    img = load_image_cls_combine(img)
    pred_combine = model_cls_combine.predict(img,verbose = False)
    value = classes_cls_combine[np.argmax(pred_combine)]
    return value


#------------------------------------- CSV --------------------------------------------
def CLEAN_CSV_BOTTLE():
    # Clear "data_bottle.csv" before run AI module
    file_start = pd.read_csv("data\data_bottle.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data\data_bottle.csv', index=False)

    # Clear "data_mask_bottle.csv" before run AI module
    file_start = pd.read_csv("data_mask\data_mask_bottle.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data_mask\data_mask_bottle.csv', index=False)

def CLEAN_CSV_WATER_LEVEL():
    # Clear "data_water_level.csv" before run AI module
    file_start = pd.read_csv("data\data_water_level.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data\data_water_level.csv', index=False)

    # Clear "data_mask_water_level.csv" before run AI module
    file_start = pd.read_csv("data_mask\data_mask_water_level.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data_mask\data_mask_water_level.csv', index=False)

def CLEAN_CSV_LABEL():
    # Clear "data_label.csv" before run AI module
    file_start = pd.read_csv("data\data_label.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data\data_label.csv', index=False)

    # Clear "data_mask_label.csv" before run AI module
    file_start = pd.read_csv("data_mask\data_mask_label.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data_mask\data_mask_label.csv', index=False)
    
def CLEAN_CSV_COMBINE():
    # Clear "data_combine.csv" before run AI module
    file_start = pd.read_csv("data\data_combine.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data\data_combine.csv', index=False)

    # Clear "data_mask_combine.csv" before run AI module
    file_start = pd.read_csv("data_mask\data_mask_combine.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data_mask\data_mask_combine.csv', index=False)

def get_percent_area(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb2_area = abs((bb2['x1'] - bb2['x2']) * (bb2['y1'] - bb2['y2']))
    
    percent_area = intersection_area / bb2_area
    assert percent_area >= 0.0
    assert percent_area <= 1.0
    return percent_area


def ADD_DATA_CSV_BOTTLE(dict_info, limit_point_1, limit_point_2):
    for item in dict_info.items():
        data_csv_ = pd.read_csv("data\data_bottle.csv")
        unique_ids = data_csv_['id'].unique()
        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 100 < item[1][1][2] < limit_point_1:
                with open('data_mask\data_mask_bottle.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(list([item[1][1][4], item[1][0]]))

        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 10 < item[1][1][2] < limit_point_2:
                data_csv_ = pd.read_csv("data_mask\data_mask_bottle.csv")
                id_list_to_mode = list(data_csv_['id'][:30])
                if len(id_list_to_mode) > 0:
                    id_mode = mode(id_list_to_mode)
                    count_id_in_list_mode = 0
                    for i in id_list_to_mode:
                        if i == id_mode:
                            count_id_in_list_mode += 1
                    STATUS = data_csv_.loc[data_csv_['id'] == id_mode, 'status'].iloc[0]
                    if count_id_in_list_mode > decide_var:
                        with open('data\data_bottle.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(list([id_mode, STATUS]))
                        data_csv_ = pd.read_csv("data_mask\data_mask_bottle.csv", nrows=0)
                        clean_df = pd.DataFrame(columns=data_csv_.columns)
                        clean_df.to_csv('data_mask\data_mask_bottle.csv', index=False)

def ADD_DATA_CSV_WATER_LEVEL(dict_info, limit_point_1, limit_point_2):
    for item in dict_info.items():
        data_csv_ = pd.read_csv("data\data_water_level.csv")
        unique_ids = data_csv_['id'].unique()
        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 100 < item[1][1][2] < limit_point_1:
                with open('data_mask\data_mask_water_level.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(list([item[1][1][4], item[1][0]]))

        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 10 < item[1][1][2] < limit_point_2:
                data_csv_ = pd.read_csv("data_mask\data_mask_water_level.csv")
                id_list_to_mode = list(data_csv_['id'][:30])
                if len(id_list_to_mode) > 0:
                    id_mode = mode(id_list_to_mode)
                    count_id_in_list_mode = 0
                    for i in id_list_to_mode:
                        if i == id_mode:
                            count_id_in_list_mode += 1
                    STATUS = data_csv_.loc[data_csv_['id'] == id_mode, 'status'].iloc[0]
                    if count_id_in_list_mode > decide_var:
                        with open('data\data_water_level.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(list([id_mode, STATUS]))
                        data_csv_ = pd.read_csv("data_mask\data_mask_water_level.csv", nrows=0)
                        clean_df = pd.DataFrame(columns=data_csv_.columns)
                        clean_df.to_csv('data_mask\data_mask_water_level.csv', index=False)


def ADD_DATA_CSV_LABEL(dict_info, limit_point_1, limit_point_2):
    for item in dict_info.items():
        data_csv_ = pd.read_csv("data\data_label.csv")
        unique_ids = data_csv_['id'].unique()
        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 100 < item[1][1][2] < limit_point_1:
                with open('data_mask\data_mask_label.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(list([item[1][1][4], item[1][0]]))

        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 10 < item[1][1][2] < limit_point_2:
                data_csv_ = pd.read_csv("data_mask\data_mask_label.csv")
                id_list_to_mode = list(data_csv_['id'][:30])
                if len(id_list_to_mode) > 0:
                    id_mode = mode(id_list_to_mode)
                    count_id_in_list_mode = 0
                    for i in id_list_to_mode:
                        if i == id_mode:
                            count_id_in_list_mode += 1
                    STATUS = data_csv_.loc[data_csv_['id'] == id_mode, 'status'].iloc[0]
                    if count_id_in_list_mode > decide_var:
                        with open('data\data_label.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(list([id_mode, STATUS]))
                        data_csv_ = pd.read_csv("data_mask\data_mask_label.csv", nrows=0)
                        clean_df = pd.DataFrame(columns=data_csv_.columns)
                        clean_df.to_csv('data_mask\data_mask_label.csv', index=False)
                        
def ADD_DATA_CSV_COMBINE(dict_info, limit_point_1, limit_point_2):
    for item in dict_info.items():
        data_csv_ = pd.read_csv("data\data_combine.csv")
        unique_ids = data_csv_['id'].unique()
        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 100 < item[1][1][2] < limit_point_1:
                with open('data_mask\data_mask_combine.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(list([item[1][1][4], item[1][0]]))

        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 10 < item[1][1][2] < limit_point_2:
                data_csv_ = pd.read_csv("data_mask\data_mask_combine.csv")
                id_list_to_mode = list(data_csv_['id'][:30])
                if len(id_list_to_mode) > 0:
                    id_mode = mode(id_list_to_mode)
                    count_id_in_list_mode = 0
                    for i in id_list_to_mode:
                        if i == id_mode:
                            count_id_in_list_mode += 1
                    STATUS = data_csv_.loc[data_csv_['id'] == id_mode, 'status'].iloc[0]
                    if count_id_in_list_mode > decide_var:
                        with open('data\data_combine.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(list([id_mode, STATUS]))
                        data_csv_ = pd.read_csv("data_mask\data_mask_combine.csv", nrows=0)
                        clean_df = pd.DataFrame(columns=data_csv_.columns)
                        clean_df.to_csv('data_mask\data_mask_combine.csv', index=False)


def delete_files_in_folder_image_show():
    folder_path = "APP/image_show"
    # Lặp qua tất cả các tệp trong thư mục
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                # Xóa tệp nếu là một tệp
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # Đệ quy để xóa tất cả các tệp trong thư mục con nếu là một thư mục
                delete_files_in_folder_image_show(file_path)
        except Exception as e:
            print(f"Không thể xóa {file_path}: {e}")