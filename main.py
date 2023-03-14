
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()
'''
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
'''
# These are all different test samples

# The reason we are getting are weights (randomized at first) between -1 and 1 because we dont want the data to get
# bigger and bigger

X, Y = spiral_data(samples=100, classes=3)  # y returns 300 size array identifying data belonging to different classes

np.random.seed(0)  # returns the same random numbers on each run


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)  # we dont have to do a transpose now
        self.biases = np.random.randn(1, n_neurons)  # vector with size of neurons
        self.output = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        reduced_inputs = inputs - np.max(inputs, axis=1, keepdims=True)  # each row max and maintain the shape
        exp_outputs = np.exp(reduced_inputs)
        probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def forward(self, y_pred, y_true):
        pass

    def calculate(self, output, y):  # y represents the targeted indexes
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):  # inheriting loss
    @staticmethod
    def forward(y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # makes the lower boundary non zero

        # expected data can be passed as two type
        # [1, 0, 1, 1]
        # [[0, 1], [1, 0], [0, 1]]

        correct_confidences = []
        if len(y_true.shape) == 1:  # scalar values
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # one-hot* encoded vector
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            '''
                [0.7, 0.2, 0.1] --> [1, 0, 0]
                [0.05, 0.9, 0.05] --> [0, 1, 0]
                multiplied and row sum
                --> [0.7, 0.9]
            '''
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


class Accuracy:
    @staticmethod
    def calculate(y_pred, y_true):
        accuracy_array = np.array(np.argmax(y_pred, axis=1) == y_true, dtype=int)
        return np.sum(accuracy_array) / len(accuracy_array)

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))  # Y is 0, 1, 2 so it creates 3 elem array
    one_hot_y[np.arange(y.size), y] = 1 # for each row and set the intended column to 1
    return one_hot_y


def forward_propagate(dense1, dense2):
    # dense1 = LayerDense(2, 3)  # in spiral data two inputs for each sample 2 unique values of x and y coordinates
    activation1 = ActivationReLU()
    # dense2 = LayerDense(3, 3)  # output layer --> we have 3 classes so end layer have 3 neurons (probabilities)
    activation2 = ActivationSoftmax()  # because it is an end layer

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss_function = LossCategoricalCrossEntropy()
    loss = loss_function.calculate(activation2.output, Y)  # Y returns 0 for first 100 and 1 and 2 for next hundreds

    accuracy_function = Accuracy()
    accuracy = accuracy_function.calculate(activation2.output, Y)

    return dense1, dense2, activation1, activation2, loss, accuracy


def derivative_ReLU(Z):
    return Z > 0  # in relu the left part d is 0 and right part is a y=x so derivative is 1


def backward_propagate(Z1, A1, Z2, A2, W1, W2):
    m = Y.size  # 300 in this case
    print()
    one_hot_y = one_hot(Y)
    dZ2 = A2 - one_hot_y  # just calculating the difference
    dW2 = 1 / m * dZ2.T.dot(A1)  # dz2 to dw2 to a1
    dB2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = W2.dot(dZ2.T) * derivative_ReLU(Z1.T)  # apply the weights in reverse and derivative of ReLU reverse of it
    dW1 = 1 / m * dZ1.dot(X)
    dB1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, dB1, dW2, dB2


def update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):  # alpha is just the learning rate
    W1 = W1 - alpha * dW1.T
    W2 = W2 - alpha * dW2
    B1 = B1 - alpha * dB1.T
    B2 = B2 - alpha * dB2
    return W1, W2, B1, B2


def train(iterations, alpha):
    Z1 = LayerDense(2, 3)
    Z2 = LayerDense(3, 3)
    W1 = Z1.weights
    W2 = Z2.weights
    B1 = Z1.biases
    B2 = Z2.biases
    loss = 0
    for i in range(iterations):
        Z1, Z2, A1, A2, loss, accuracy = forward_propagate(Z1, Z2)
        dW1, dB1, dW2, dB2 = backward_propagate(Z1.output, A1.output, Z2.output, A2.output, W1, W2)
        W1, W2, B1, B2 = update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, 0.1)
        Z1.weights = W1
        Z2.weights = W2
        Z1.biases = B1
        Z2.biases = B2
        print("Iteration {}".format(i+1))
        print("Loss: {}".format(loss))
        print("Accuracy: {}".format(accuracy))

    return W1, W2, B1, B2, loss


learningRateX = [-1,
                 -0.9,
                 -0.8,
                 -0.6,
                 -0.5,
                 -0.4,
                 -0.3,
                 -0.2,
                 -0.1,
                 0,
                 0.1,
                 0.2,
                 0.3,
                 0.4,
                 0.5,
                 0.6,
                 0.7,
                 0.8,
                 0.9,
                 1]
learningRateY = [0.25946102,
                 0.3501676,
                 0.9640169,
                 0.0066454783,
                 0.010988098,
                 0.59635836,
                 0.8748962,
                 0.46325165,
                 0.22478895,
                 0.6899418,
                 0.20339519,
                 0.44577387,
                 0.46853596,
                 0.0029323117,
                 0.46220347,
                 1.0986122,
                 0.28997412,
                 0.5952785,
                 0.42187908,
                 0.15039048]

coefficients = np.polyfit(learningRateX, learningRateY, 10)
polynomial = np.poly1d(coefficients)

plt.scatter(learningRateX, learningRateY)

plt.title("Learning rate (alpha) vs. loss for 20000 iterations, spiral data")

x_values = np.linspace(min(learningRateX), max(learningRateX), 100)
y_values = polynomial(x_values)

plt.plot(x_values, y_values, color='red')

plt.show()
