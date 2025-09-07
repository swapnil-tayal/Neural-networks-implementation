import random
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidPrime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:

    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.weights = []
        self.biases = []
        for y in sizes[1:]:
            self.biases.append(np.random.randn(y, 1))
        for x, y in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(y, x))

    def forwardPropagate(self, activation):
        for baise, weight in zip(self.biases, self.weights):
            activation = np.dot(weight, activation) + baise
            activation = sigmoid(activation)
        return activation

    def evaluate(self, test_data):
        correct_results = 0
        for (x, y) in test_data:
            result = np.argmax(self.forwardPropagate(x))
            correct_results += int(result == y)
        return correct_results

    def stochasticGradientDescent(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
        training_data_len = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in range(0, training_data_len, mini_batch_size):
                mini_batches.append(training_data[k:k+mini_batch_size])
            for mini_batch in mini_batches:
                self.processMiniBatch(mini_batch, learning_rate)
            if test_data:
                test_date_len = len(test_data)
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), test_date_len))
            else:
                print("Epoch {0}".format(epoch))

    def processMiniBatch(self, mini_batch, learning_rate):
        nabla_b = []
        nabla_w = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))
        for (x, y) in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backProp(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backProp(self, x, y):
        nabla_b = []
        nabla_w = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))
        activation = x
        activations = [x]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z_vector = np.dot(w, activation) + b
            z_vectors.append(z_vector)
            activation = sigmoid(z_vector)
            activations.append(activation)
        delta = (activations[-1] - y) * sigmoidPrime(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.numLayers):
            z_sigmoid = sigmoidPrime(z_vectors[-layer])
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * z_sigmoid
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return nabla_b, nabla_w
