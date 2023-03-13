
import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target = [1, 0, 0]

loss = -(target[0] * math.log(softmax_output[0], math.e) +
         target[1] * math.log(softmax_output[1], math.e) +
         target[2] * math.log(softmax_output[2], math.e))

# It is just simply the negative log of the target classes confidence

print(loss)
print("-----")

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

# 0: Dog
# 1: Cat
# 2: Human
class_targets = [0, 1, 1]  # this represent what value we except to be 100%

loss = -np.log(softmax_outputs[[0, 1, 2], class_targets])
average_loss = np.mean(-np.log(softmax_outputs[[0, 1, 2], class_targets]))
# np.array[which rows we are interested in, which column of these rows are
# we interested respectively]

print(loss)
print(average_loss)  # When the value is 0 np.log gives an error
