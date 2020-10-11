import numpy as np

np.set_printoptions(precision=10)


class NeuronNetwork():
    def __init__(self, alpha, hiddenSize):
        self.alpha = alpha
        self.hiddenSize = hiddenSize

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)


alpha = 10
hiddenSize = 32

net = NeuronNetwork(10, 16)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)


train_input = np.array([[0, 0, 1],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])

train_output = np.array([[0],
                         [1],
                         [1],
                         [0]])


np.random.seed(1)

synapse_0 = 2 * np.random.random((3, hiddenSize)) - 1
synapse_1 = 2 * np.random.random((hiddenSize, 1)) - 1

for j in range(100000):

    layer_0 = train_input
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))

    layer_2_error = layer_2 - train_output
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
    synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))

print(layer_2)
