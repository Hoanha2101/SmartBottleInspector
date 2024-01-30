def CHECK_BOTTLE_AI(FRAME, start_time):
    
    # *******************Introduction**************
    '''
    FRAME: 

    
    
    '''
    
    
    # *******************Action********************
    
    # (%%%%%) declare variable
    HEIGHT_FRAME_1 = FRAME.shape[0]
    WIDTH_FRAME_1 = FRAME.shape[1]
    
    limit_point_1_frame_1 = int(WIDTH_FRAME_1/2) - 80
    limit_point_2_frame_1 = int(WIDTH_FRAME_1/2) + 80
    
    coordinates_bottle_f1 = []

    list_good_f1 = []
    list_error_f1 = []

    id_real_f1 = []
    
    min_id = 1
    id_count = 1
    ID_DEFAULT = ""
    ERROR_DEFAULT = ""
    dict_info = {}
    
    results = MODEL_BOTTLE_AI.track(FRAME, persist = True, classes = 39)
    
    for result in results[0]:
        boxes = result.boxes.cpu().numpy() 

        for i in range(len(boxes)):  
            box = boxes[i]
            
            if box.cls[0] == 39:
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = map(int,(box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
                if (limit_point_1_frame_1 - 100) < x_max_bottle < (limit_point_2_frame_1 + 20) and ( box.id is not None):
                        # index_id = id_real_f2.index(int(box.id[0]))
                        # id_show = str(id_mask[index_id])
                        coordinates_bottle_f1.append((x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                        region_bottle = FRAME[y_min_bottle:y_max_bottle,x_min_bottle:x_max_bottle]
                        
                        result_cls = predict_cnn_bottle(region_bottle)
                        
                        cv2.rectangle(FRAME,(x_min_bottle,y_min_bottle), (x_max_bottle,y_max_bottle), (255, 255, 0), thickness=2)
                        
                        if result_cls == "GOOD":
                            dict_info[box.id[0]] = ("GOOD",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                            cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(FRAME, str(box.id[0]), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        else:
                            dict_info[box.id[0]] = ("ERROR",(x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle, box.id[0]))
                            cv2.putText(FRAME, "GOOD", (x_min_bottle,y_min_bottle - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(FRAME, str(box.id[0]), (x_min_bottle,y_min_bottle + 15), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    continue
                
    ADD_DATA_CSV_BOTTLE(dict_info, limit_point_1_frame_1, limit_point_2_frame_1)
    
    cv2.line(FRAME, (limit_point_1_frame_1, 0), (limit_point_1_frame_1, HEIGHT_FRAME_1), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_2_frame_1, 0), (limit_point_2_frame_1, HEIGHT_FRAME_1), (255, 0, 0), thickness = 2)
    cv2.line(FRAME, (limit_point_1_frame_1 - 100, 0), (limit_point_1_frame_1 - 100, HEIGHT_FRAME_1), (255,123,0), thickness = 2)
                
    
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
    
    return FRAME, 1, 1