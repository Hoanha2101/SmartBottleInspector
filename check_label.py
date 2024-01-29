from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import csv
from statistics import mode

import time



# model = YOLO("yolov8l.pt")
model = YOLO("model_set/label.pt")

model_cls = YOLO("model_set/best.pt")

cap = cv2.VideoCapture("video_label.mp4")

file_start = pd.read_csv("data_label.csv", nrows=0)
file_start_new = pd.DataFrame(columns=file_start.columns)
file_start_new.to_csv('data_label.csv', index=False)

file_start = pd.read_csv("data_label.csv", nrows=0)
file_start_new = pd.DataFrame(columns=file_start.columns)
file_start_new.to_csv('data_label.csv', index=False)


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


# Lấy kích thước video
width = int(cap.get(3))
height = int(cap.get(4))

# Định cấu hình đường thẳng màu đỏ
red_line_color = (0, 0, 255)  # BGR format
line_thickness = 2
line_position = width // 2  # Vị trí giữa màn hình

ret = True
min_id = 1
id_count = 1
ID_DEFAULT = ""
ERROR_DEFAULT = ""

while True:
    coordinates_bottle = []
    coordinates_label = []
    
    list_label_good = []
    list_label_error = []
    
    id_real = []
    ret, frame = cap.read()
    start_time = time.time()
    results = model.track(frame, persist = True, conf = 0.9, iou = 0.9)

    for result in results[0]:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            
            if (int(box.cls[0]) == 0):
                if box.id is not None and box.id:
                    id_real.append(int(box.id[0]))
                else:
                    pass
    id_real = list(set(id_real))
    
    if min_id < min(id_real, default=0):
        min_id = min(id_real, default=0)
        id_count += 1
    id_mask = [i for i in range(id_count,id_count + len(id_real) + 1)]
    
    
    for result in results[0]:
        
        boxes = result.boxes.cpu().numpy() 
        
        for i in range(len(boxes)):  
            box = boxes[i]
            
            if box.cls[0] == 0:
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                if (line_position + 50 < x_max_bottle < line_position+400) and ( box.id is not None):
                    index_id = id_real.index(int(box.id[0]))
                    id_show = str(id_mask[index_id])
                    coor_ = (x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, id_show)
                    coordinates_bottle.append(coor_)
                else:
                    continue
            if box.cls[0] == 1:
                x_min_label, y_min_label, x_max_label, y_max_label = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                coordinates_label.append([x_min_label, y_min_label, x_max_label, y_max_label])
            
            # Kiểm tra tỷ lệ phần giao nhau
        for coor_bottle in coordinates_bottle:
            bottle_box = {'x1': coor_bottle[0], 'y1': coor_bottle[1], 'x2': coor_bottle[2], 'y2': coor_bottle[3]}
            list_label_error.append(coor_bottle)
            for coor_label in coordinates_label:
                label_box = {'x1': coor_label[0], 'y1': coor_label[1], 'x2': coor_label[2], 'y2': coor_label[3]}
                overlap_ratio = get_percent_area(bottle_box, label_box)
                
                intermediate = ((coor_label[0],coor_label[1], coor_label[2],coor_label[3]), coor_bottle)
                
                if overlap_ratio > 0.9:
                    list_label_good.append(intermediate)
                    list_label_error = [item for item in list_label_error if item != intermediate[1]]   
                    
    list_label_good =  list(set(list_label_good))   
    ist_label_error = list(set(list_label_error))
    
    dict_info = {}
      
    if len(list_label_good) != 0:
        for i in list_label_good:
            region_label = frame[i[0][1]:i[0][3], i[0][0]: i[0][2]]
            results_cls = model_cls(region_label)
            if results_cls[0].names[results_cls[0].probs.top1] == "e":
                cv2.rectangle(frame, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (0, 0, 255), thickness=2) # bottle
                cv2.rectangle(frame, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), thickness=2) #label
                cv2.putText(frame, "ERROR", (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                dict_info[i[1][4]] = ("ERROR",(i[1][0], i[1][1], i[1][2], i[1][3],i[1][4]))
                cv2.putText(frame, str(i[1][4]), (i[1][0], i[1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 255, 0), thickness=2) #label
                cv2.rectangle(frame, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (0, 255, 0), thickness=2) # bottle
                cv2.putText(frame, "GOOD", (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, str(i[1][4]), (i[1][0], i[1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
                #
                dict_info[i[1][4]] = ("GOOD",(i[1][0], i[1][1], i[1][2], i[1][3],i[1][4]))
            
    if len(list_label_error) != 0:
        for i in list_label_error:
            cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), thickness=2)  
            cv2.putText(frame, "ERROR", (i[0], i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(i[4]), (i[0], i[1] + 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            #
            dict_info[i[4]] = ("ERROR",(i[0], i[1], i[2], i[3],i[4]))
    
    for item in dict_info.items():
        data_csv_ = pd.read_csv("data_label.csv")
        unique_ids = data_csv_['id'].unique()
        if int(item[1][1][4]) not in list(unique_ids):
            if line_position+300 < item[1][1][2] < line_position+400:
                with open('data_mask_label.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(list([item[1][1][4], item[1][0]]))

        if int(item[1][1][4]) not in list(unique_ids):
            if line_position< item[1][1][2] < line_position+320:
                data_csv_ = pd.read_csv("data_mask_label.csv")
                id_list_to_mode = list(data_csv_['id'][:30])
                if len(id_list_to_mode) > 0:
                    id_mode = mode(id_list_to_mode)
                    count_id_in_list_mode = 0
                    for i in id_list_to_mode:
                        if i == id_mode:
                            count_id_in_list_mode += 1
                    STATUS = data_csv_.loc[data_csv_['id'] == id_mode, 'status'].iloc[0]
                    if count_id_in_list_mode > 20:
                        with open('data_label.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(list([id_mode, STATUS]))
                        data_csv_ = pd.read_csv("data_mask_label.csv", nrows=0)
                        clean_df = pd.DataFrame(columns=data_csv_.columns)
                        clean_df.to_csv('data_mask_label.csv', index=False)
    
    data_csv_ = pd.read_csv("data_label.csv")
    if data_csv_.shape[0] > 0:
        ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
        ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]
    
    cv2.putText(frame, "ID: {}".format(ID_DEFAULT), (100,100), cv2.FONT_HERSHEY_SIMPLEX , 1, (200, 0, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, "ERROR: {}".format(ERROR_DEFAULT), (100,160), cv2.FONT_HERSHEY_SIMPLEX , 1, (200, 0, 100), 2, cv2.LINE_AA)
    
    # Vẽ đường thẳng đứng màu đỏ
    cv2.line(frame, (line_position, 0), (line_position, height), red_line_color, line_thickness)
    cv2.line(frame, (line_position+300, 0), (line_position+300, height), red_line_color, line_thickness)
    cv2.line(frame, (line_position+400, 0), (line_position+400, height), (255,123,0), line_thickness)
    # cv2.line(frame, (width-200, 0), (width-200, height), (255,255,0), line_thickness)
    # cv2.line(frame, (200, 0), (200, height), (255,255,0), line_thickness)
    
    # Kết thúc thời gian đo FPS
    end_time = time.time()
    
    # Tính FPS
    fps = 1.0 / (end_time - start_time)

    # Hiển thị FPS lên khung hình
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # frame_ = results[0].plot()
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break