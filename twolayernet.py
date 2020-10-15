import numpy as np

np.set_printoptions(suppress=True)


class NeuralNetwork():
    def __init__(self, inSize=3, hiddenSize=4, outSize=1, alpha=1):
        self.alpha = alpha
        self.hiddenSize = hiddenSize
        self.insize = inSize
        self.outsize = outSize

        self._synapse_0 = 2 * np.random.random((self.insize, hiddenSize)) - 1
        self._synapse_1 = 2 * np.random.random((hiddenSize, self.outsize)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self):
        self.layer_0 = self.train_input
        self.layer_1 = self.sigmoid(np.dot(self.layer_0, self._synapse_0))
        self.layer_2 = self.sigmoid(np.dot(self.layer_1, self._synapse_1))

    def backpropogation(self):
        self.layer_2_error = self.layer_2 - self.train_output
        self.layer_2_delta = self.layer_2_error * \
            self.sigmoid_derivative(self.layer_2)
        self.layer_1_error = self.layer_2_delta.dot(self._synapse_1.T)
        self.layer_1_delta = self.layer_1_error * \
            self.sigmoid_derivative(self.layer_1)

        #self.layer_1 = np.array([self.layer_1])
        #self.layer_0 = np.array([self.layer_0])

        self._synapse_1 -= self.alpha * \
            (self.layer_1.T.dot(self.layer_2_delta))
        self._synapse_0 -= self.alpha * \
            (self.layer_0.T.dot(self.layer_1_delta))

    def train(self, input, output, iterations, backpr=True):
        self.train_input = input
        self.train_output = output

        for i in range(iterations):
            if (i % 1 == 0):
                print(i/iterations*100, "%")
            self.feedforward()
            if (backpr == True):
                self.backpropogation()

        print("Result:\n", self.layer_2)


if __name__ == "__main__":
    train_input = np.array([[0, 0, 1],
                            [0, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1],
                            [0, 0, 0]])

    train_output = np.array([[0, 1],
                             [1, 0],
                             [1, 0],
                             [0, 1],
                             [0, 0]])

    net = NeuralNetwork(3, 16, 2, 10)
    net.train(train_input, train_output, 100000)

    net.train(np.array([[1, 1, 0]]), np.array([[0, 0]]), 1, False)
