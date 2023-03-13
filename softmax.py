
import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

'''
raw python
e = math.e

exp_outputs = []
for i in layer_outputs:
    exp_outputs.append(e ** i)

norm_outputs = []
for i in exp_outputs:
    norm_outputs.append(i / np.sum(exp_outputs))

print(norm_outputs)
'''

exp_outputs = np.exp(layer_outputs)  # This tends to overflow i.e. e^1000 overflows
# We should  first subtract all numbers from the largest number then max will be e^0 which is 1
norm_outputs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
# axis 1 means rows sum and keep dims returns in same shape

print(norm_outputs)
