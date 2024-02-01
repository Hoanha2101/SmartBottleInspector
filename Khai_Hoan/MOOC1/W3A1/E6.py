from utils import *

parameters, cache, t_X, t_Y = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, t_X, t_Y)
print("dW1 = "+ str(grads["dW1"]))
print("db1 = "+ str(grads["db1"]))
print("dW2 = "+ str(grads["dW2"]))
print("db2 = "+ str(grads["db2"]))

