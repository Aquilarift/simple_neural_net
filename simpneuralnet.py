import numpy as np

np.set_printoptions(suppress=True)
# np.random.seed(1)


class NeuralNetwork():
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def __init__(self, inSize=3, outSize=1, hidden=[16]):
        self.alpha = 1
        self.hidden = hidden
        self.hiddensize = len(hidden)
        self.insize = inSize
        self.outsize = outSize

        self._synapses = [None] * (self.hiddensize+1)
        self.layers = [None] * (len(self._synapses)+1)

        self.layers_err = [None] * (self.hiddensize+1)
        self.layers_delta = [None] * (len(self.layers_err))

        self._synapses[0] = 2 * \
            np.random.random((self.insize, self.hidden[0])) - 1
        self._synapses[self.hiddensize] = 2 * \
            np.random.random(
                (self.hidden[self.hiddensize-1], self.outsize)) - 1

        for i in range(1, self.hiddensize):
            self._synapses[i] = 2 * \
                np.random.random((self.hidden[i-1], self.hidden[i])) - 1

    def feedforward(self):
        self.layers[0] = self.train_input
        for i in range(1, len(self.layers)):
            self.layers[i] = self.sigmoid(
                np.dot(self.layers[i-1], self._synapses[i-1]))

    def backpropogation(self):
        self.layers_err[self.hiddensize] = self.layers[self.hiddensize+1] - \
            self.train_output
        self.layers_delta[self.hiddensize] = self.layers_err[self.hiddensize] * \
            self.sigmoid_derivative(self.layers[self.hiddensize+1])

        for i in reversed(range(self.hiddensize)):
            self.layers_err[i] = self.layers_delta[i +
                                                   1].dot(self._synapses[i+1].T)
            self.layers_delta[i] = self.layers_err[i] * \
                self.sigmoid_derivative(self.layers[i+1])

        for i in reversed(range(self.hiddensize+1)):
            self._synapses[i] -= self.alpha * \
                (self.layers[i].T.dot(self.layers_delta[i]))

    def train(self, input, output, iterations, alpha):
        self.alpha = alpha
        self.train_input = input
        self.train_output = output

        for i in range(iterations):
            if (i % 1000 == 0):
                print(i / iterations * 100, "%")

            self.feedforward()
            self.backpropogation()

        print("Result:\n", self.layers[self.hiddensize+1])

    def recognize(self, input):
        self.train_input = input
        self.feedforward()

        print("Result:\n", self.layers[self.hiddensize+1])


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

    net = NeuralNetwork(3, 2, [32, 8])
    net.train(train_input, train_output, 100000, 10)

    net.recognize(np.array([[1, 1, 0]]))
