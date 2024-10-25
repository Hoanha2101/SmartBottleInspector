

decide_var = 10
"""decide_var: This variable is understood as the number of correct decisions in a list of the first 30 values of data_mask, 
GOOD or ERROR status. If it is greater than this variable, the final conclusion will be given for that bottle of water."""

output_names_bottle_str = 'dense_51'
output_names_label_str = 'dense_44'
output_names_combine_str = 'dense_11'

# Class names use in Deep sort
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
# path_DetectMODEL = "model_set/detect/YOLOv8_combine.pt"
path_DetectMODEL = "model_set/detect/best.engine"

# Org model
path_cls_bottle_org = "model_set/classification/org/model_cls_bottle_v2.h5"
path_cls_label_org = "model_set/classification/org/model_cls_label.h5"
path_cls_combine_org = "model_set/classification/org/model_combine.h5"

# TensorRT model
path_cls_bottle_RT = "model_set/classification/tensorRT/model_cls_bottle_v2.trt"
path_cls_label_RT = "model_set/classification/tensorRT/model_cls_label.trt"
path_org_combine_RT = "model_set/classification/tensorRT/model_combine.trt"