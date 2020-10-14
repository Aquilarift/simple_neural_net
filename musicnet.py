from twolayernet import NeuralNetwork
import pydub
import numpy as np


def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(
        y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


'''train_input = np.array([[0, 0, 1],
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

net.train(np.array([[1, 1, 0]]), np.array([[0, 0]]), 1, False)'''

sr, test = read("test.wav", True)
print(test)

net = NeuralNetwork(2, 4, 1, 10)
net.train(test, np.array([[1]]), 2)
