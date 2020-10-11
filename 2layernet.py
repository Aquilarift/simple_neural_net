import numpy as np

np. set_printoptions(suppress=True)


class NeuronNetwork():
    def __init__(self, alpha=1, hiddenSize=4):
        self.alpha = alpha
        self.hiddenSize = hiddenSize

        self._synapse_0 = 2 * np.random.random((3, hiddenSize)) - 1
        self._synapse_1 = 2 * np.random.random((hiddenSize, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)

    def feedForward(self):
        self.layer_0 = train_input
        self.layer_1 = self.sigmoid(np.dot(self.layer_0, self._synapse_0))
        self.layer_2 = self.sigmoid(np.dot(self.layer_1, self._synapse_1))

    def backpropogation(self):
        self.layer_2_error = self.layer_2 - self.train_output
        self.layer_2_delta = self.layer_2_error * \
            self.sigmoid_derivative(self.layer_2)
        self.layer_1_error = self.layer_2_delta.dot(self._synapse_1.T)
        self.layer_1_delta = self.layer_1_error * \
            self.sigmoid_derivative(self.layer_1)

        self._synapse_1 -= self.alpha * \
            (self.layer_1.T.dot(self.layer_2_delta))
        self._synapse_0 -= self.alpha * \
            (self.layer_0.T.dot(self.layer_1_delta))

    def train(self, input, output, iterations):
        self.train_input = input
        self.train_output = output

        for _ in range(iterations):
            self.feedForward()
            self.backpropogation()

        print("Result:\n", self.layer_2)


train_input = np.array([[0, 0, 1],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])

train_output = np.array([[0],
                         [1],
                         [1],
                         [0]])

net = NeuronNetwork(10, 32)
net.train(train_input, train_output, 100000)
