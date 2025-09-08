import mnistLoader
import neuralNetwork

training_data, validation_data, test_data = mnistLoader.load_data_wrapper()
network = neuralNetwork.NeuralNetwork([784, 100, 10])
network.stochasticGradientDescent(training_data, 30, 10, 0.5, test_data = test_data)