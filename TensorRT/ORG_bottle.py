from library import *
from utils import *


img = cv2.imread("TensorRT/images/e_b (31).jpg")

start = time.time()
result_cls, con = predict_cnn_bottle(img)
end = time.time()

time_ = end - start

# print("Cuda:", torch.cuda.is_available())
# print()
print("Result",result_cls, "----", round(con,6))
print("Time:", time_, "(s)")

print()
