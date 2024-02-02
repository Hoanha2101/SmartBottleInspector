from utils import *

print("basic_sigmoid(1) = " + str(basic_sigmoid(1)))

x = [1, 2, 3] # x becomes a python list object
basic_sigmoid(np.array(x)) # you will see this give an error when you run it, because x is a vector.

# example of np.exp
t_x = np.array([1, 2, 3])
print(np.exp(t_x)) # result is (exp(1), exp(2), exp(3))

t_x = np.array([1, 2, 3])
print (t_x + 3)