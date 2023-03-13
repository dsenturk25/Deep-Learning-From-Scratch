
# For example we want to predict system failure with sensor data and
# the three input values represent a sensor data.

import numpy as np

# giving the inputs as batches is just like getting the average of data
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# row number of weights contains weights of paths going to each output there are 3
biases = [2.0, 3.0, 0.5]


# second layer gets 3 input
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1.0, 2.0, 0.5]

#   # weights 1 represents links to this node
#   #
#   #
#

# transposing means swapping rows and columns we do this to multiply the matrices
# first we should convert weights to numpy array
layer_1_outputs = np.dot(inputs, np.array(weights).T) + biases  # switching of weights and inputs gives a shape error
layer_2_outputs = np.dot(layer_1_outputs, np.array(weights2).T) + biases2
# matrices are multiplied as first row * second col
'''
* np.dot fixes this loop issue
output = [0.0, 0.0, 0.0]
for i in range(len(weights)):
    output[i] += biases[i]
    for j in range(len(weights[i])):
        output[i] += inputs[j]*weights[i][j]

'''

print(layer_2_outputs)
