
import numpy as np


def sigmoid(x, derivative=False):
    return 1 / (1 + np.exp(-x)) if not derivative else x * (1 - x)


def tanh(x, derivative=False):
    return np.tanh(x) if not derivative else 1 - x * x


def relu(x, derivate=False):
    if not derivate:
        output = np.copy(x)
        output[output < 0] = 0.05
        return output

    if derivate:
        output = np.copy(x)
        output[output > 0] = 1
        output[output <= 0] = 0.05
        return output


# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class ActivationSoftmaxLossCategoricalCrossEntropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = ActivationSoftMax()
        self.loss = LossCategoricalCrossEntropy()
        self.output = None

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded -> turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class ActivationReLU:
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # set gradients zero where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftMax:
    def __init__(self):
        self.output = None
        self.inputs = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class ActivationSigmoid:
    def __init__(self):
        self.output = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


class Loss:
    def calculate(self, output, y):
        losses = self.forward(output, y)
        return np.mean(losses)

    def forward(self, y_pred, y_true) -> np.ndarray:
        pass

    def backward(self, y_pred, y_true):
        pass


class LossCategoricalCrossEntropy(Loss):
    def __init__(self):
        self.dinputs = None

    def forward(self, y_pred, y_true) -> np.ndarray:
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
