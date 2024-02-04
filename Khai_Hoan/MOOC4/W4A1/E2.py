from utils import *

distance, door_open_flag = verify("MOOC4/W4A1/images/camera_0.jpg", "younes", database, FRmodel)
assert np.isclose(distance, 0.5992949), "Distance not as expected"
assert isinstance(door_open_flag, bool), "Door open flag should be a boolean"
print("(", distance, ",", door_open_flag, ")")


verify("MOOC4/W4A1/images/camera_2.jpg", "kian", database, FRmodel)