from library import *

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

def CLEAN_CSV_LEVEL():
    # Clear "data_label.csv" before run AI module
    file_start = pd.read_csv("data\data_label.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data\data_label.csv', index=False)

    # Clear "data_mask_label.csv" before run AI module
    file_start = pd.read_csv("data_mask\data_mask_label.csv", nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv('data_mask\data_mask_label.csv', index=False)
    
    
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


def ADD_DATA_CSV_WATER_LEVEL(dict_info, limit_point_1, limit_point_2):
    for item in dict_info.items():
        data_csv_ = pd.read_csv("data\data_water_level.csv")
        unique_ids = data_csv_['id'].unique()
        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 60 < item[1][1][2] < limit_point_1:
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
                    if count_id_in_list_mode > 20:
                        with open('data\data_water_level.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(list([id_mode, STATUS]))
                        data_csv_ = pd.read_csv("data_mask\data_mask_water_level.csv", nrows=0)
                        clean_df = pd.DataFrame(columns=data_csv_.columns)
                        clean_df.to_csv('data_mask\data_mask_water_level.csv', index=False)


def ADD_DATA_CSV_LABEL(dict_info, limit_point_1, limit_point_2):
    for item in dict_info.items():
        data_csv_ = pd.read_csv("data\data_water_level.csv")
        unique_ids = data_csv_['id'].unique()
        if int(item[1][1][4]) not in list(unique_ids):
            if limit_point_1 - 60 < item[1][1][2] < limit_point_1:
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
                    if count_id_in_list_mode > 20:
                        with open('data\data_water_level.csv', mode='a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(list([id_mode, STATUS]))
                        data_csv_ = pd.read_csv("data_mask\data_mask_water_level.csv", nrows=0)
                        clean_df = pd.DataFrame(columns=data_csv_.columns)
                        clean_df.to_csv('data_mask\data_mask_water_level.csv', index=False)
                        
