from utils import *
import time


#--------------------------------------------------------------        AI MODULE        ---------------------------------------------------------
def CHECK_BOTTLE_AI(FRAME, start_time, is_ANNO, activate_optimize_RT):
    
    # *******************Introduction**************
    '''
    FRAME: Frame received from camera 1
    start_time: Timeline from the start of reading frame 1, to calculator FPS
    '''
    # *******************Action********************
    
    # (%%%%%) declare variable
    HEIGHT_FRAME_1 = FRAME.shape[0]
    WIDTH_FRAME_1 = FRAME.shape[1]
    
    limit_point_1_frame_1 = int(WIDTH_FRAME_1/2) + 20 
    limit_point_2_frame_1 = int(WIDTH_FRAME_1/2) + 180
    
    coordinates_bottle_f1 = []

    list_good_f1 = []
    list_error_f1 = []

    id_real_f1 = []
    
    min_id = 1
    id_count = 1
    ID_DEFAULT = ""
    ERROR_DEFAULT = ""
    dict_info = {}

    results = MODEL_BOTTLE_AI.track(FRAME, persist = True, classes = 0, verbose = False)

    for result in results[0]:
        boxes = result.boxes.cpu().numpy() 

        for i in range(len(boxes)):
            box = boxes[i]
            
            if box.cls[0] == 0:
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                
                #cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (255, 255, 0), thickness=2)
                
                if (limit_point_1_frame_1 - 150) < x_max_bottle < (limit_point_2_frame_1 + 20) and ( box.id is not None):
                    # index_id = id_real_f2.index(int(box.id[0]))
                    # id_show = str(id_mask[index_id])
                    coordinates_bottle_f1.append((x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                    region_bottle = FRAME[y_min_bottle:y_max_bottle,x_min_bottle:x_max_bottle]
                    
                        # +++ TensorRT +++
                    if activate_optimize_RT == True:
                        result_cls = predict_cnn_bottle_RT(region_bottle)
                    else:
                        # +++ Origin +++
                        result_cls = predict_cnn_bottle(region_bottle)
                    
                    data_csv_exist = pd.read_csv("data\data_bottle.csv")
                    unique_ids_exist = data_csv_exist['id'].unique()
                    
                    if box.id[0] in unique_ids_exist:
                        result_cls = str(data_csv_exist.loc[data_csv_exist.id == box.id[0]]["status"].values[0])
                        
                    if result_cls == "GOOD":
                        dict_info[box.id[0]] = ("GOOD",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                        cv2.putText(FRAME, str(box.id[0]), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        if (limit_point_1_frame_1 <= x_max_bottle <= (limit_point_2_frame_1 + 20)) and (box.id[0] in unique_ids_exist):
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 255, 0), thickness=2)
                            cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 0, 0), thickness=2)
                            if is_ANNO:
                                cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                    elif result_cls == "ERROR":
                        dict_info[box.id[0]] = ("ERROR",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                        cv2.putText(FRAME, str(box.id[0]), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        if (limit_point_1_frame_1 <= x_max_bottle <= (limit_point_2_frame_1 + 20)) and (box.id[0] in unique_ids_exist):
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (255, 0, 0), thickness=2)
                            cv2.putText(FRAME, "ERROR", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 0, 0), thickness=2)
                            if is_ANNO:
                                cv2.putText(FRAME, "ERROR", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                else:
                    continue
                
    ADD_DATA_CSV_BOTTLE(dict_info, limit_point_1_frame_1, limit_point_2_frame_1)
    
    cv2.line(FRAME, (limit_point_1_frame_1, 0), (limit_point_1_frame_1, HEIGHT_FRAME_1), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_2_frame_1, 0), (limit_point_2_frame_1, HEIGHT_FRAME_1), (255, 0, 0), thickness = 2)
    # cv2.line(FRAME, (limit_point_1_frame_1 - 150, 0), (limit_point_1_frame_1 - 150, HEIGHT_FRAME_1), (255,123,0), thickness = 2)
                
    data_csv_ = pd.read_csv("data\data_bottle.csv")
    
    if data_csv_.shape[0] > 0:
        ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
        ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]  
    
    # Kết thúc thời gian đo FPS
    end_time = time.time()
    
    # Tính FPS
    fps = 1.0 / (end_time - start_time)

    # Hiển thị FPS lên khung hình
    cv2.putText(FRAME, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return FRAME, ID_DEFAULT, ERROR_DEFAULT


# """Tracking by Deep Sort - Bottle"""
# def CHECK_BOTTLE_AI(FRAME, start_time, activate_optimize_RT):
    
#     HEIGHT_FRAME_1 = FRAME.shape[0]
#     WIDTH_FRAME_1 = FRAME.shape[1]
    
#     limit_point_1_frame_1 = int(WIDTH_FRAME_1/2) - 80
#     limit_point_2_frame_1 = int(WIDTH_FRAME_1/2) + 80
    
#     coordinates_bottle_f1 = []

#     ID_DEFAULT = ""
#     ERROR_DEFAULT = ""
#     dict_info = {}

#     results = MODEL_BOTTLE_AI(FRAME, classes = 0, verbose = False)
    
#     for result in results:
#         boxes = result.boxes
#         probs = result.probs
#         cls = boxes.cls.tolist()
#         xyxy = boxes.xyxy
#         conf = boxes.conf
#         xywh = boxes.xywh
    
#     pred_cls = np.array(cls)
#     conf = conf.detach().cpu().numpy()
#     xyxy = xyxy.detach().cpu().numpy()
#     bboxes_xywh = np.array(xywh.cpu().numpy(), dtype=float)
    
#     # Update tracker with detections
#     tracks = tracker.update(bboxes_xywh, conf, FRAME)
    
#     for track in tracker.tracker.tracks:
#         track_id = track.track_id
#         hits = track.hits
#         x1, y1, x2, y2 = track.to_tlbr()
#         x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = int(x1), int(y1), int(x2), int(y2)
#         coordinates_bottle_f1.append((x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, track_id))
#         region_bottle = FRAME[y_min_bottle:y_max_bottle,x_min_bottle:x_max_bottle]
        
#         if activate_optimize_RT == True:
#             result_cls = predict_cnn_bottle_RT(region_bottle)
#         else:
#             # +++ Origin +++
#             result_cls = predict_cnn_bottle(region_bottle)
        
#         cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (255, 255, 0), thickness=2)
#         if result_cls == "GOOD":
#             dict_info[track_id] = ("GOOD",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, track_id))
#             cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(FRAME, str(track_id), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
#         elif result_cls == "ERROR":
#             dict_info[track_id] = ("ERROR",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, track_id))
#             cv2.putText(FRAME, "ERROR", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 0, 0), 2, cv2.LINE_AA)
#             cv2.putText(FRAME, str(track_id), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
#     ADD_DATA_CSV_BOTTLE(dict_info, limit_point_1_frame_1, limit_point_2_frame_1)
    
#     cv2.line(FRAME, (limit_point_1_frame_1, 0), (limit_point_1_frame_1, HEIGHT_FRAME_1), (255, 0, 0), thickness = 2)
#     cv2.line(FRAME, (limit_point_2_frame_1, 0), (limit_point_2_frame_1, HEIGHT_FRAME_1), (255, 0, 0), thickness = 2)
#     # cv2.line(FRAME, (limit_point_1_frame_1 - 100, 0), (limit_point_1_frame_1 - 100, HEIGHT_FRAME_1), (255,123,0), thickness = 2)
                
#     data_csv_ = pd.read_csv("data\data_bottle.csv")
    
#     if data_csv_.shape[0] > 0:
#         ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
#         ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]  
    
#     # Kết thúc thời gian đo FPS
#     end_time = time.time()
    
#     # Tính FPS
#     fps = 1.0 / (end_time - start_time)

#     # Hiển thị FPS lên khung hình
#     cv2.putText(FRAME, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     return FRAME, ID_DEFAULT, ERROR_DEFAULT


def CHECK_WATER_LEVEL_AI(FRAME, start_time, is_ANNO):   
    # *******************Introduction**************
    '''
    FRAME: Frame received from camera 2
    start_time: Timeline from the start of reading frame 1, to calculator FPS
    '''
    # *******************Action********************
    
    # (%%%%%) declare variable
    
    HEIGHT_FRAME_2 = FRAME.shape[0]
    WIDTH_FRAME_2 = FRAME.shape[1]
    
    limit_point_1_frame_2 = int(WIDTH_FRAME_2/2) + 20
    limit_point_2_frame_2 = int(WIDTH_FRAME_2/2) + 180
    
    coordinates_bottle_f2 = []
    coordinates_water_f2 = []

    list_water_good_f2 = []
    list_water_error_f2 = []

    id_real_f2 = []
    
    min_id = 1
    id_count = 1
    ID_DEFAULT = ""
    ERROR_DEFAULT = ""
    
    
    # (%%%%%%) Get results tracking in each frame
    results = MODEL_WATER_LEVEL_AI.track(FRAME, persist = True, conf = 0.6, iou = 0.6, verbose = False)
    
    # (%%%%%%) During the tracking process, the ids generated from the YOLO model traking on the frame will have different ids for different objects, 
    # but these ids are arranged in an unordered manner, and are susceptible to noise.
    # ---> Rearrange the ids according to a corresponding order with each water bottle appearing in the first-to-last order.
    
        # Extract id
    # for result in results[0]:
    #     boxes = result.boxes.cpu().numpy()
    #     for box in boxes:
    #         if (int(box.cls[0]) == 0):
    #             if box.id is not None and box.id:
    #                 id_real_f2.append(int(box.id[0]))
    #             else:
    #                 pass
    # id_real_f2 = list(set(id_real_f2))
    
    #     # Reassign ids from smallest to largest in order
    # if min_id < min(id_real_f2, default=0):
    #     min_id = min(id_real_f2, default=0)
    #     id_count += 1
    # id_mask = [i for i in range(id_count,id_count + len(id_real_f2) + 1)]
    
    # (%%%%%) Classify the ids, positions, of detected objects in the frame for processing
    for result in results[0]:
        
        boxes = result.boxes.cpu().numpy() 

        for i in range(len(boxes)):  
            box = boxes[i]
            
            if box.cls[0] == 0:
                # Get the coordinate values of the 2 square corners, convert them to int type
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                
                # Limit the area where all processing and decisions on the object will take place, display good or error messages on the display above the UI
                if (limit_point_1_frame_2 - 150) < x_max_bottle < (limit_point_2_frame_2 + 20) and ( box.id is not None):
                    # index_id = id_real_f2.index(int(box.id[0]))
                    # id_show = str(id_mask[index_id])
                    coordinates_bottle_f2.append((x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                else:
                    continue
            if box.cls[0] == 1:
                x_min_water, y_min_water, x_max_water, y_max_water = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                coordinates_water_f2.append([x_min_water, y_min_water, x_max_water, y_max_water])
        
        
        # (%%%%%) Check the intersection: If the percentage of area of a certain no-water on a water bottle is greater than a threshold, 
        # then that is the no-water of that water bottle, meaning that water bottle has a no-water and is considered to have a no-water, otherwise . 
        for coor_bottle in coordinates_bottle_f2:
            bottle_box = {'x1': coor_bottle[0], 'y1': coor_bottle[1], 'x2': coor_bottle[2], 'y2': coor_bottle[3]}

            list_water_error_f2.append(((-1,-1,-1,-1), coor_bottle))
            
            for coor_water in coordinates_water_f2:
                water_box = {'x1': coor_water[0], 'y1': coor_water[1], 'x2': coor_water[2], 'y2': coor_water[3]}
                overlap_ratio = get_percent_area(bottle_box, water_box)
                intermediate = ((coor_water[0],coor_water[1], coor_water[2],coor_water[3]), coor_bottle)
                if overlap_ratio > 0.9:
                    list_water_error_f2.append(intermediate)
                    numerator = (coor_bottle[3] - coor_bottle[1])
                    denominator = (coor_water[3] - coor_water[1])
                
                    if 3.8 < (numerator/denominator) < 4.5:
                        list_water_good_f2.append(intermediate)
                        list_water_error_f2 = [item for item in list_water_error_f2 if item[1] != intermediate[1]]
                            
    list_water_good_f2 =  list(set(list_water_good_f2))
    
    list_water_error_f2 = list(set(list_water_error_f2))
    
    dict_info = {}
    
    data_csv_exist = pd.read_csv("data\data_water_level.csv")
    unique_ids_exist = data_csv_exist['id'].unique()
    
    if len(list_water_good_f2) != 0:
        for i in list_water_good_f2:
            result_status = "GOOD"
            STATUS_df = data_csv_exist.loc[data_csv_exist.id == i[1][3]]["status"]
            if i[1][4] in unique_ids_exist and (STATUS_df.shape[0] != 0):
                result_status = str(STATUS_df.values[0])
            #
            dict_info[i[1][4]] = (result_status,(i[1][0], i[1][1], i[1][2], i[1][3], i[1][4]))
            
            if (limit_point_1_frame_2 <= i[1][3] <= (limit_point_2_frame_2 + 20)) and (i[1][4] in unique_ids_exist):
                cv2.line(FRAME, (i[0][0], i[0][3]- 5), (i[0][2], i[0][3] - 5), (0, 255, 0), thickness=2) #water level
                cv2.rectangle(FRAME, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (0, 255, 0), thickness=2)
                cv2.putText(FRAME, result_status, (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(FRAME, str(i[1][4]), (i[1][0], i[1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            else: 
                cv2.line(FRAME, (i[0][0], i[0][3]- 5), (i[0][2], i[0][3] - 5), (0, 255, 0), thickness=2) #water level
                cv2.rectangle(FRAME, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (0, 0, 0), thickness=2)
                if is_ANNO:
                    cv2.putText(FRAME, result_status, (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(FRAME, str(i[1][4]), (i[1][0], i[1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
    if len(list_water_error_f2) != 0:
        for i in list_water_error_f2:
            result_status = "ERROR"
            STATUS_df = data_csv_exist.loc[data_csv_exist.id == i[1][3]]["status"]
            if i[1][4] in unique_ids_exist and (STATUS_df.shape[0] != 0):
                result_status = str(STATUS_df.values[0])
            #
            dict_info[i[1][4]] = (result_status,(i[1][0], i[1][1], i[1][2], i[1][3], i[1][4]))
            
            if (limit_point_1_frame_2 <= i[1][3] <= (limit_point_2_frame_2 + 20)) and (i[1][4] in unique_ids_exist):
                cv2.line(FRAME, (i[0][0], i[0][3]-  5), (i[0][2], i[0][3] - 5), (255, 0, 0), thickness=2) #water level
                cv2.rectangle(FRAME, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (255, 0, 0), thickness=2)
                cv2.putText(FRAME, result_status, (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(FRAME, str(i[1][4]), (i[1][0], i[1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.line(FRAME, (i[0][0], i[0][3]-  5), (i[0][2], i[0][3] - 5), (255, 0, 0), thickness=2) #water level
                cv2.rectangle(FRAME, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (0, 0, 0), thickness=2)
                if is_ANNO:
                    cv2.putText(FRAME, result_status, (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(FRAME, str(i[1][4]), (i[1][0], i[1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
    
    cv2.line(FRAME, (limit_point_1_frame_2, 0), (limit_point_1_frame_2, HEIGHT_FRAME_2), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_2_frame_2, 0), (limit_point_2_frame_2, HEIGHT_FRAME_2), (255, 0, 0), thickness = 2)
    # cv2.line(FRAME, (limit_point_1_frame_2 - 150, 0), (limit_point_1_frame_2 - 150, HEIGHT_FRAME_2), (255,123,0), thickness = 2)
# Save information into file csv and 
    ADD_DATA_CSV_WATER_LEVEL(dict_info, limit_point_1_frame_2, limit_point_2_frame_2)
    
    data_csv_ = pd.read_csv("data\data_water_level.csv")
    
    if data_csv_.shape[0] > 0:
        ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
        ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]
    
    # Kết thúc thời gian đo FPS
    end_time = time.time()
    
    # Tính FPS
    fps = 1.0 / (end_time - start_time)

    # Hiển thị FPS lên khung hình
    cv2.putText(FRAME, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return FRAME, ID_DEFAULT, ERROR_DEFAULT



"""Label - version 1"""
# def CHECK_LABEL_AI(FRAME, start_time, activate_optimize_RT):
     
#     # *******************Introduction**************
#     '''
#     FRAME: Frame received from camera 3
#     start_time: Timeline from the start of reading frame 1, to calculator FPS
    
#     '''
#     # *******************Action********************
    
#     # (%%%%%) declare variable
    
#     HEIGHT_FRAME_3 = FRAME.shape[0]
#     WIDTH_FRAME_3 = FRAME.shape[1]
    
#     limit_point_1_frame_3 = int(WIDTH_FRAME_3/2) - 80
#     limit_point_2_frame_3 = int(WIDTH_FRAME_3/2) + 80
    
#     coordinates_bottle_f3 = []
#     coordinates_label_f3 = []

#     list_label_good_f3 = []
#     list_label_error_f3 = []

#     id_real_f3 = []
    
#     min_id = 1
#     id_count = 1
#     ID_DEFAULT = ""
#     ERROR_DEFAULT = ""
    
    
#     # (%%%%%%) Get results tracking in each frame
#     results = MODEL_LABEL_AI.track(FRAME, persist = True, conf = 0.6, iou = 0.6,  verbose = False)
    
#     # (%%%%%%) During the tracking process, the ids generated from the YOLO model traking on the frame will have different ids for different objects, 
#     # but these ids are arranged in an unordered manner, and are susceptible to noise.
#     # ---> Rearrange the ids according to a corresponding order with each water bottle appearing in the first-to-last order.
    
#         # Extract id
#     # for result in results[0]:
#     #     boxes = result.boxes.cpu().numpy()
#     #     for box in boxes:
#     #         if (int(box.cls[0]) == 0):
#     #             if box.id is not None and box.id:
#     #                 id_real_f3.append(int(box.id[0]))
#     #             else:
#     #                 pass
#     # id_real_f3 = list(set(id_real_f3))
    
#     #     # Reassign ids from smallest to largest in order
#     # if min_id < min(id_real_f3, default=0):
#     #     min_id = min(id_real_f3, default=0)
#     #     id_count += 1
#     # id_mask = [i for i in range(id_count,id_count + len(id_real_f3) + 1)]
    
#     # (%%%%%) Classify the ids, positions, of detected objects in the frame for processing
#     for result in results[0]:
#         boxes = result.boxes.cpu().numpy() 

#         for i in range(len(boxes)):  
#             box = boxes[i]
            
#             if box.cls[0] == 0:
#                 # Get the coordinate values of the 2 square corners, convert them to int type
#                 x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                
#                 # Limit the area where all processing and decisions on the object will take place, display good or error messages on the display above the UI
#                 if (limit_point_1_frame_3 - 100) < x_max_bottle < (limit_point_2_frame_3 + 20) and ( box.id is not None):
#                     # index_id = id_real_f3.index(int(box.id[0]))
#                     # id_show = str(id_mask[index_id])
#                     coor_ = (x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0])
#                     coordinates_bottle_f3.append(coor_)
#                 else:
#                     continue
#             if box.cls[0] == 1:
#                 x_min_label, y_min_label, x_max_label, y_max_label = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
#                 coordinates_label_f3.append([x_min_label, y_min_label, x_max_label, y_max_label])
        
        
#         # (%%%%%) Check the intersection: If the percentage of area of a certain label on a water bottle is greater than a threshold, 
#         # then that is the label of that water bottle, meaning that water bottle has a label and is considered to have a label, otherwise . 
#         for coor_bottle in coordinates_bottle_f3:
#             bottle_box = {'x1': coor_bottle[0], 'y1': coor_bottle[1], 'x2': coor_bottle[2], 'y2': coor_bottle[3]}
#             list_label_error_f3.append(((-1,-1,-1,-1), coor_bottle))
#             for coor_label in coordinates_label_f3:
#                 label_box = {'x1': coor_label[0], 'y1': coor_label[1], 'x2': coor_label[2], 'y2': coor_label[3]}
#                 overlap_ratio = get_percent_area(bottle_box, label_box) 
#                 intermediate = ((coor_label[0],coor_label[1], coor_label[2],coor_label[3]), coor_bottle)
#                 if overlap_ratio > 0.9:
#                     list_label_good_f3.append(intermediate)
#                     list_label_error_f3 = [item for item in list_label_error_f3 if item[1] != intermediate[1]]   
    
#     list_label_good_f3 =  list(set(list_label_good_f3))
    
#     list_label_error_f3 = list(set(list_label_error_f3))
    
#     list_to_remove = []
    
#     for i in list_label_good_f3:
#         local_label = FRAME[i[0][1]:i[0][3], i[0][0]:i[0][2]]
#         if activate_optimize_RT == True:
#             '''+++ TensorRT +++'''
#             result_cnn = predict_cnn_label_RT(local_label)
#         else:
#             result_cnn = predict_cnn_label(local_label)
        
#         if result_cnn == "ERROR":
#             list_to_remove.append(i)
#             list_label_error_f3.append(i)
    
#     if len(list_to_remove) > 0:
#         list_label_good_f3 = [local for local in list_label_good_f3 if local not in list_to_remove]
            
#     dict_info = {}
    
#     if len(list_label_good_f3) != 0:
#         for i in list_label_good_f3:
#             cv2.rectangle(FRAME, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 255, 0), thickness=2) #label
#             cv2.rectangle(FRAME, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (255, 255, 0), thickness=2) # bottle
#             # cv2.putText(FRAME, "GOOD", (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(FRAME, str(i[1][4]), (i[1][0], i[1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
#             #
#             dict_info[i[1][4]] = ("GOOD",(i[1][0], i[1][1], i[1][2], i[1][3],i[1][4]))
            
#     if len(list_label_error_f3) != 0:
#         for i in list_label_error_f3:
#             cv2.rectangle(FRAME, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (255, 0, 0), thickness=2)
#             cv2.rectangle(FRAME, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (255, 255, 0), thickness=2)  
#             # cv2.putText(FRAME, "ERROR", (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 0, 0), 2, cv2.LINE_AA)
#             cv2.putText(FRAME, str(i[1][4]), (i[1][0], i[1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
#             #
#             dict_info[i[1][4]] = ("ERROR",(i[1][0], i[1][1], i[1][2], i[1][3],i[1][4]))
    
#     cv2.line(FRAME, (limit_point_1_frame_3, 0), (limit_point_1_frame_3, HEIGHT_FRAME_3), (255, 0, 0), thickness = 2)
#     cv2.line(FRAME, (limit_point_2_frame_3, 0), (limit_point_2_frame_3, HEIGHT_FRAME_3), (255, 0, 0), thickness = 2)
#     # cv2.line(FRAME, (limit_point_1_frame_3 - 100, 0), (limit_point_1_frame_3 - 100, HEIGHT_FRAME_3), (255,123,0), thickness = 2)
# # Save information into file csv and 
#     ADD_DATA_CSV_LABEL(dict_info, limit_point_1_frame_3, limit_point_2_frame_3)
    
#     data_csv_ = pd.read_csv("data\data_label.csv")
    
#     if data_csv_.shape[0] > 0:   
#         ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
#         ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]
    
#     # Kết thúc thời gian đo FPS
#     end_time = time.time()
    
#     # Tính FPS
#     fps = 1.0 / (end_time - start_time)

#     # Hiển thị FPS lên khung hình
#     cv2.putText(FRAME, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     return FRAME, ID_DEFAULT, ERROR_DEFAULT


"""Label - version 2"""
def CHECK_LABEL_AI(FRAME, start_time,is_ANNO ,activate_optimize_RT):
    
    # *******************Introduction**************
    '''
    FRAME: Frame received from camera 3
    start_time: Timeline from the start of reading frame 1, to calculator FPS
    '''
    # *******************Action********************
    
    # (%%%%%) declare variable
    HEIGHT_FRAME_3 = FRAME.shape[0]
    WIDTH_FRAME_3 = FRAME.shape[1]
    
    limit_point_1_frame_3 = int(WIDTH_FRAME_3/2) + 20
    limit_point_2_frame_3 = int(WIDTH_FRAME_3/2) + 180
    
    coordinates_bottle_f3 = []

    ID_DEFAULT = ""
    ERROR_DEFAULT = ""
    dict_info = {}

    results = MODEL_LABEL_AI.track(FRAME, persist = True, classes = 0, verbose = False)

    for result in results[0]:
        boxes = result.boxes.cpu().numpy() 

        for i in range(len(boxes)):
            box = boxes[i]
            
            if box.cls[0] == 0:
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                if (limit_point_1_frame_3 - 150) < x_max_bottle < (limit_point_2_frame_3 + 20) and ( box.id is not None):
                    # index_id = id_real_f2.index(int(box.id[0]))
                    # id_show = str(id_mask[index_id])
                    coordinates_bottle_f3.append((x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                    region_bottle = FRAME[y_min_bottle:y_max_bottle,x_min_bottle:x_max_bottle]
                    region_label  = region_bottle[int(region_bottle.shape[0]*0.42) : int(region_bottle.shape[0]*0.74),]
                        # +++ TensorRT +++
                    if activate_optimize_RT == True:
                        result_cls = predict_cnn_label_RT(region_label)
                    else:
                        # +++ Origin +++
                        result_cls = predict_cnn_label(region_label)
                    
                    data_csv_exist = pd.read_csv("data\data_label.csv")
                    unique_ids_exist = data_csv_exist['id'].unique()
                    
                    if box.id[0] in unique_ids_exist:
                        result_cls = str(data_csv_exist.loc[data_csv_exist.id == box.id[0]]["status"].values[0])
                    
                    if result_cls == "GOOD":
                        dict_info[box.id[0]] = ("GOOD",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                        cv2.putText(FRAME, str(box.id[0]), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        if (limit_point_1_frame_3 <= x_max_bottle <= (limit_point_2_frame_3 + 20)) and (box.id[0] in unique_ids_exist):
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 255, 0), thickness=2)
                            cv2.rectangle(FRAME,(x_min_bottle ,y_min_bottle + int(region_bottle.shape[0]*0.38)), (x_max_bottle,y_min_bottle + int(region_bottle.shape[0]*0.74)), (0, 180, 0), thickness=3)
                            cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 0, 0), thickness=2)
                            cv2.rectangle(FRAME,(x_min_bottle ,y_min_bottle + int(region_bottle.shape[0]*0.38)), (x_max_bottle,y_min_bottle + int(region_bottle.shape[0]*0.73)), (100, 0, 100), thickness=3)
                            if is_ANNO:
                                cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                    elif result_cls == "ERROR":
                        dict_info[box.id[0]] = ("ERROR",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                        cv2.putText(FRAME, str(box.id[0]), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        if (limit_point_1_frame_3 <= x_max_bottle <= (limit_point_2_frame_3 + 20)) and (box.id[0] in unique_ids_exist):
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (255, 0, 0), thickness=2)
                            cv2.rectangle(FRAME,(x_min_bottle ,y_min_bottle + int(region_bottle.shape[0]*0.38)), (x_max_bottle,y_min_bottle + int(region_bottle.shape[0]*0.73)), (180, 0, 0), thickness=3)
                            cv2.putText(FRAME, "ERROR", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 0, 0), thickness=2)
                            cv2.rectangle(FRAME,(x_min_bottle ,y_min_bottle + int(region_bottle.shape[0]*0.38)), (x_max_bottle,y_min_bottle + int(region_bottle.shape[0]*0.73)), (100, 0, 100), thickness=3)
                            if is_ANNO:
                                cv2.putText(FRAME, "ERROR", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                else:
                    continue
                
    ADD_DATA_CSV_LABEL(dict_info, limit_point_1_frame_3, limit_point_2_frame_3)
    
    cv2.line(FRAME, (limit_point_1_frame_3, 0), (limit_point_1_frame_3, HEIGHT_FRAME_3), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_2_frame_3, 0), (limit_point_2_frame_3, HEIGHT_FRAME_3), (255, 0, 0), thickness = 2)
    # cv2.line(FRAME, (limit_point_1_frame_3 - 150, 0), (limit_point_1_frame_3 - 150, HEIGHT_FRAME_3), (255,123,0), thickness = 2)
                
    data_csv_ = pd.read_csv("data\data_label.csv")
    
    if data_csv_.shape[0] > 0:
        ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
        ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]  
    
    # Kết thúc thời gian đo FPS
    end_time = time.time()
    
    # Tính FPS
    fps = 1.0 / (end_time - start_time)

    # Hiển thị FPS lên khung hình
    cv2.putText(FRAME, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return FRAME, ID_DEFAULT, ERROR_DEFAULT
# #-----------------------------------------------------------------------------------------------------------------------------------------------


"""AI COMBINE"""

def AI_COMBINE(FRAME, start_time, activate_optimize_RT):
    # *******************Introduction**************
    '''
    FRAME: Frame received from camera 1
    start_time: Timeline from the start of reading frame 1, to calculator FPS
    '''
    # *******************Action********************
    
    # (%%%%%) declare variable
    HEIGHT_FRAME_COMBINE = FRAME.shape[0]
    WIDTH_FRAME_COMBINE = FRAME.shape[1]
    
    limit_point_1_frame_1 = int(WIDTH_FRAME_COMBINE/2) + 20
    limit_point_2_frame_1 = int(WIDTH_FRAME_COMBINE/2) + 180
    
    coordinates_bottle_f1 = []

    list_good_f1 = []
    list_error_f1 = []

    id_real_f1 = []
    
    min_id = 1
    id_count = 1
    ID_DEFAULT = ""
    ERROR_DEFAULT = ""
    dict_info = {}

    results = MODEL_BOTTLE_AI.track(FRAME, persist = True, classes = 0, verbose = False)

    for result in results[0]:
        boxes = result.boxes.cpu().numpy() 

        for i in range(len(boxes)):
            box = boxes[i]
            
            if box.cls[0] == 0:
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                if (limit_point_1_frame_1 - 150) < x_max_bottle < (limit_point_2_frame_1 + 20) and ( box.id is not None):
                    # index_id = id_real_f2.index(int(box.id[0]))
                    # id_show = str(id_mask[index_id])
                    coordinates_bottle_f1.append((x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                    region_bottle = FRAME[y_min_bottle:y_max_bottle,x_min_bottle:x_max_bottle]
                    
                        # +++ TensorRT +++
                    if activate_optimize_RT == True:
                        result_cls = predict_cnn_combine_RT(region_bottle)
                    else:
                        # +++ Origin +++
                        result_cls = predict_cnn_combine(region_bottle)
                    
                    data_csv_exist = pd.read_csv("data\data_combine.csv")
                    unique_ids_exist = data_csv_exist['id'].unique()
                    
                    if box.id[0] in unique_ids_exist:
                        result_cls = str(data_csv_exist.loc[data_csv_exist.id == box.id[0]]["status"].values[0])
                    
                    if  x_max_bottle == limit_point_1_frame_1 - 10:
                        name_image_write = "APP/image_show/" + str(box.id[0]) + ".jpg"
                        if os.path.exists(name_image_write):
                            pass
                        else:
                            region_bottle_RGB = cv2.cvtColor(region_bottle, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(name_image_write,region_bottle_RGB)
                    
                    if result_cls == "GOOD":
                        dict_info[box.id[0]] = ("GOOD",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                        cv2.putText(FRAME, str(box.id[0]), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        if (limit_point_1_frame_1 <= x_max_bottle <= (limit_point_2_frame_1 + 20)) and (box.id[0] in unique_ids_exist):
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 255, 0), thickness=2)
                            cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 0, 0), thickness=2)
                            cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                    elif result_cls == "ERROR":
                        dict_info[box.id[0]] = ("ERROR",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                        cv2.putText(FRAME, str(box.id[0]), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        if (limit_point_1_frame_1 <= x_max_bottle <= (limit_point_2_frame_1 + 20)) and (box.id[0] in unique_ids_exist):
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (255, 0, 0), thickness=2)
                            cv2.putText(FRAME, "ERROR", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (0, 0, 0), thickness=2)
                            cv2.putText(FRAME, "ERROR", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                else:
                    continue
                
    ADD_DATA_CSV_COMBINE(dict_info, limit_point_1_frame_1, limit_point_2_frame_1)
    
    cv2.line(FRAME, (limit_point_1_frame_1, 0), (limit_point_1_frame_1, HEIGHT_FRAME_COMBINE), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_2_frame_1, 0), (limit_point_2_frame_1, HEIGHT_FRAME_COMBINE), (255, 0, 0), thickness = 2)
    # cv2.line(FRAME, (limit_point_1_frame_1 - 150, 0), (limit_point_1_frame_1 - 150, HEIGHT_FRAME_COMBINE), (255,123,0), thickness = 2)
                
    data_csv_ = pd.read_csv("data\data_combine.csv")
    
    if data_csv_.shape[0] > 0:
        ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
        ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]  
    
    # Kết thúc thời gian đo FPS
    end_time = time.time()
    
    # Tính FPS
    fps = 1.0 / (end_time - start_time)

    # Hiển thị FPS lên khung hình
    cv2.putText(FRAME, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return FRAME, ID_DEFAULT, ERROR_DEFAULT
