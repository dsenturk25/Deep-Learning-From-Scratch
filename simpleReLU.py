import numpy as np

X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)

inputs = [-5, 3, -3, 1, 0.2]
output = []

for i in inputs:
    output.append(max(0, i))

print(output)
