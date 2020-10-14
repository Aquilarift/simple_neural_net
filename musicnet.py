from twolayernet import NeuralNetwork
import numpy as np

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
