from library import *
from utils import *

# -------------------------------------------------------------         DECLARE         ---------------------------------------------------------

# MODEL_BOTTLE_AI = YOLO("model_bottle.pt")
MODEL_WATER_LEVEL_AI = YOLO("water2.pt")
MODEL_LABEL_AI = YOLO("model_label.pt")

model_test = YOLO("yolov8m.pt")

#------------------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------   SECONDARY FUNCTIONS   ---------------------------------------------------------




#--------------------------------------------------------------        AI MODULE        ---------------------------------------------------------
def CHECK_BOTTLE_AI(FRAME):
    results = model_test.track(FRAME, persist = True, conf = 0.9, iou = 0.9)
    FRAME = results[0].plot()
    return FRAME, 1, 1
    
def CHECK_WATER_LEVEL_AI(FRAME):
    # *******************Introduction**************
    '''
    FRAME: 

    
    
    '''
    
    
    # *******************Action********************
    
    # (%%%%%) declare variable
    
    HEIGHT_FRAME_2 = FRAME.shape[0]
    WIDTH_FRAME_2 = FRAME.shape[1]
    
    limit_point_1_frame_2 = int(WIDTH_FRAME_2/2) - 80
    limit_point_2_frame_2 = int(WIDTH_FRAME_2/2) + 80
    
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
    results = MODEL_WATER_LEVEL_AI.track(FRAME, persist = True, conf = 0.9, iou = 0.9)
    
    # (%%%%%%) During the tracking process, the ids generated from the YOLO model traking on the frame will have different ids for different objects, 
    # but these ids are arranged in an unordered manner, and are susceptible to noise.
    # ---> Rearrange the ids according to a corresponding order with each water bottle appearing in the first-to-last order.
    
        # Extract id
    for result in results[0]:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            if (int(box.cls[0]) == 0):
                if box.id is not None and box.id:
                    id_real_f2.append(int(box.id[0]))
                else:
                    pass
    id_real_f2 = list(set(id_real_f2))
    
        # Reassign ids from smallest to largest in order
    if min_id < min(id_real_f2, default=0):
        min_id = min(id_real_f2, default=0)
        id_count += 1
    id_mask = [i for i in range(id_count,id_count + len(id_real_f2) + 1)]
    
    # (%%%%%) Classify the ids, positions, of detected objects in the frame for processing
    for result in results[0]:
        
        boxes = result.boxes.cpu().numpy() 

        for i in range(len(boxes)):  
            box = boxes[i]
            
            if box.cls[0] == 0:
                # Get the coordinate values of the 2 square corners, convert them to int type
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                
                # Limit the area where all processing and decisions on the object will take place, display good or error messages on the display above the UI
                if (limit_point_1_frame_2 - 60) < x_max_bottle < (limit_point_2_frame_2 - 10) and ( box.id is not None):
                    index_id = id_real_f2.index(int(box.id[0]))
                    id_show = str(id_mask[index_id])
                    coordinates_bottle_f2.append((x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, id_show))
                else:
                    continue
            if box.cls[0] == 1:
                x_min_water, y_min_water, x_max_water, y_max_water = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                coordinates_water_f2.append([x_min_water, y_min_water, x_max_water, y_max_water])
        
        
        # (%%%%%) Check the intersection: If the percentage of area of a certain no-water on a water bottle is greater than a threshold, 
        # then that is the no-water of that water bottle, meaning that water bottle has a no-water and is considered to have a no-water, otherwise . 
        for coor_bottle in coordinates_bottle_f2:
            bottle_box = {'x1': coor_bottle[0], 'y1': coor_bottle[1], 'x2': coor_bottle[2], 'y2': coor_bottle[3]}
            
            for coor_water in coordinates_water_f2:
                water_box = {'x1': coor_water[0], 'y1': coor_water[1], 'x2': coor_water[2], 'y2': coor_water[3]}
                overlap_ratio = get_percent_area(bottle_box, water_box)
                
                intermediate = ((coor_water[0],coor_water[1], coor_water[2],coor_water[3]), coor_bottle)
                
                list_water_error_f2.append(intermediate)
                
                if overlap_ratio > 0.9:
                    
                    numerator = (coor_bottle[3] - coor_bottle[1])
                    denominator = (coor_water[3] - coor_water[1])
                    
                    if 3.8 < (numerator/denominator) < 4.1:
                        list_water_good_f2.append(intermediate)
                        list_water_error_f2 = [item for item in list_water_error_f2 if item[1] != intermediate[1]]   
    
    list_water_good_f2 =  list(set(list_water_good_f2))   
    list_water_error_f2 = list(set(list_water_error_f2))
    
    dict_info = {}
    
    if len(list_water_good_f2) != 0:
        for i in list_water_good_f2:
            cv2.line(frame, (i[0][0], i[0][3]+4), (i[0][2], i[0][3]+4), (0, 0, 255), thickness=2) #water level
            cv2.rectangle(frame, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (255, 255, 0), thickness=2)
            cv2.putText(frame, "ERROR", (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(i[1][4]), (i[1][0], i[1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            #
            dict_info[i[1][4]] = ("GOOD",(i[1][0], i[1][1], i[1][2], i[1][3], i[1][4]))
            
    if len(list_water_error_f2) != 0:
        for i in list_water_error_f2:
            cv2.line(FRAME, (i[0][0], i[0][3]+4), (i[0][2], i[0][3]+4), (0, 0, 255), thickness=2) #water level
            cv2.rectangle(FRAME, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (255, 255, 0), thickness=2)
            cv2.putText(FRAME, "ERROR", (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(FRAME, str(i[1][4]), (i[1][0], i[1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            #
            dict_info[i[1][4]] = ("ERROR",(i[1][0], i[1][1], i[1][2], i[1][3], i[1][4]))
    
    cv2.line(FRAME, (limit_point_1_frame_2, 0), (limit_point_1_frame_2, HEIGHT_FRAME_2), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_2_frame_2, 0), (limit_point_2_frame_2, HEIGHT_FRAME_2), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_1_frame_2 - 60, 0), (limit_point_1_frame_2 - 60, HEIGHT_FRAME_2), (255,123,0), thickness = 2)
# Save information into file csv and 
    ADD_DATA_CSV_WATER_LEVEL(dict_info, limit_point_1_frame_2, limit_point_2_frame_2)
    
    data_csv_ = pd.read_csv("data\data_water_level.csv")
    
    if data_csv_.shape[0] > 0:   
        ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
        ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]
    
    return FRAME, ID_DEFAULT, ERROR_DEFAULT





def CHECK_LABEL_AI(FRAME):
    # *******************Introduction**************
    '''
    FRAME: 

    
    
    '''
    
    
    # *******************Action********************
    
    # (%%%%%) declare variable
    
    HEIGHT_FRAME_3 = FRAME.shape[0]
    WIDTH_FRAME_3 = FRAME.shape[1]
    
    limit_point_1_frame_3 = int(WIDTH_FRAME_3/2) - 80
    limit_point_2_frame_3 = int(WIDTH_FRAME_3/2) + 80
    
    coordinates_bottle_f3 = []
    coordinates_label_f3 = []

    list_label_good_f3 = []
    list_label_error_f3 = []

    id_real_f3 = []
    
    min_id = 1
    id_count = 1
    ID_DEFAULT = ""
    ERROR_DEFAULT = ""
    
    
    # (%%%%%%) Get results tracking in each frame
    results = MODEL_LABEL_AI.track(FRAME, persist = True, conf = 0.9, iou = 0.9)
    
    # (%%%%%%) During the tracking process, the ids generated from the YOLO model traking on the frame will have different ids for different objects, 
    # but these ids are arranged in an unordered manner, and are susceptible to noise.
    # ---> Rearrange the ids according to a corresponding order with each water bottle appearing in the first-to-last order.
    
        # Extract id
    for result in results[0]:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            if (int(box.cls[0]) == 0):
                if box.id is not None and box.id:
                    id_real_f3.append(int(box.id[0]))
                else:
                    pass
    id_real_f3 = list(set(id_real_f3))
    
        # Reassign ids from smallest to largest in order
    if min_id < min(id_real_f3, default=0):
        min_id = min(id_real_f3, default=0)
        id_count += 1
    id_mask = [i for i in range(id_count,id_count + len(id_real_f3) + 1)]
    
    # (%%%%%) Classify the ids, positions, of detected objects in the frame for processing
    for result in results[0]:
        boxes = result.boxes.cpu().numpy() 

        for i in range(len(boxes)):  
            box = boxes[i]
            
            if box.cls[0] == 0:
                # Get the coordinate values of the 2 square corners, convert them to int type
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                
                # Limit the area where all processing and decisions on the object will take place, display good or error messages on the display above the UI
                if (limit_point_1_frame_3 - 60) < x_max_bottle < (limit_point_2_frame_3 - 10) and ( box.id is not None):
                    index_id = id_real_f3.index(int(box.id[0]))
                    id_show = str(id_mask[index_id])
                    coor_ = (x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, id_show)
                    coordinates_bottle_f3.append(coor_)
                else:
                    continue
            if box.cls[0] == 1:
                x_min_label, y_min_label, x_max_label, y_max_label = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                coordinates_label_f3.append([x_min_label, y_min_label, x_max_label, y_max_label])
        
        
        # (%%%%%) Check the intersection: If the percentage of area of a certain label on a water bottle is greater than a threshold, 
        # then that is the label of that water bottle, meaning that water bottle has a label and is considered to have a label, otherwise . 
        for coor_bottle in coordinates_bottle_f3:
            bottle_box = {'x1': coor_bottle[0], 'y1': coor_bottle[1], 'x2': coor_bottle[2], 'y2': coor_bottle[3]}
            list_label_error_f3.append(coor_bottle)
            for coor_label in coordinates_label_f3:
                label_box = {'x1': coor_label[0], 'y1': coor_label[1], 'x2': coor_label[2], 'y2': coor_label[3]}
                overlap_ratio = get_percent_area(bottle_box, label_box)
                
                intermediate = ((coor_label[0],coor_label[1], coor_label[2],coor_label[3]), coor_bottle)
                
                if overlap_ratio > 0.9:
                    list_label_good_f3.append(intermediate)
                    list_label_error_f3 = [item for item in list_label_error_f3 if item != intermediate[1]]   
    
    list_label_good_f3 =  list(set(list_label_good_f3))   
    list_label_error_f3 = list(set(list_label_error_f3))
    
    dict_info = {}
    
    if len(list_label_good_f3) != 0:
        for i in list_label_good_f3:
            cv2.rectangle(FRAME, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 255, 0), thickness=2) #label
            cv2.rectangle(FRAME, (i[1][0], i[1][1]), (i[1][2], i[1][3]), (0, 255, 0), thickness=2) # bottle
            cv2.putText(FRAME, "GOOD", (i[1][0], i[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(FRAME, str(i[1][4]), (i[1][0], i[1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            #
            dict_info[i[1][4]] = ("GOOD",(i[1][0], i[1][1], i[1][2], i[1][3],i[1][4]))
            
    if len(list_label_error_f3) != 0:
        for i in list_label_error_f3:
            cv2.rectangle(FRAME, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), thickness=2)  
            cv2.putText(FRAME, "ERROR", (i[0], i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(FRAME, str(i[4]), (i[0], i[1] + 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            #
            dict_info[i[4]] = ("ERROR",(i[0], i[1], i[2], i[3],i[4]))
    
    cv2.line(FRAME, (limit_point_1_frame_3, 0), (limit_point_1_frame_3, HEIGHT_FRAME_3), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_2_frame_3, 0), (limit_point_2_frame_3, HEIGHT_FRAME_3), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_1_frame_3 - 60, 0), (limit_point_1_frame_3 - 60, HEIGHT_FRAME_3), (255,123,0), thickness = 2)
# Save information into file csv and 
    ADD_DATA_CSV_LABEL(dict_info, limit_point_1_frame_3, limit_point_2_frame_3)
    
    data_csv_ = pd.read_csv("data\data_label.csv")
    
    if data_csv_.shape[0] > 0:   
        ID_DEFAULT = data_csv_['id'][(data_csv_.shape[0]-1)]
        ERROR_DEFAULT = data_csv_['status'][(data_csv_.shape[0]-1)]
    
    return FRAME, ID_DEFAULT, ERROR_DEFAULT
# #-----------------------------------------------------------------------------------------------------------------------------------------------