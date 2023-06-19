import numpy as np

from algorithms.util.activation_functions import ActivationSigmoid


class CasCorUltraHiddenUnitLayer:
    def __init__(self, n_inputs: int, n_outputs: int, learning_rate: float):
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.weights, self.dweights = None, None
        self.activation_function = ActivationSigmoid()
        self.weight_multiplier = 10
        self.current_learning_rate = learning_rate
        self.init_weights()

    def init_weights(self):
        self.weights = self.weight_multiplier * np.random.randn(self.n_inputs, self.n_outputs)

    def forward(self, x):
        linear_combination = np.dot(x, self.weights)
        self.activation_function.forward(linear_combination)
        return self.activation_function.output

    def backward(self, inputs, outputs):
        self.activation_function.backward(outputs)
        self.dweights = np.dot(inputs.T, self.activation_function.dinputs)

    def update_weights(self):
        weight_updates = -self.current_learning_rate * self.dweights
        self.weights += weight_updates
        self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.epoch))

