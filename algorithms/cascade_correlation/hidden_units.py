import numpy as np

from algorithms.util.activation_functions import ActivationSigmoid


class HiddenUnits:
    def __init__(self, n_candidates: int, learning_rate: float):
        self.n_units = 0
        self.activation_function = ActivationSigmoid()
        self.weights_per_unit = []

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.n_candidates = n_candidates
        self.candidate_weights = None
        self.weight_multiplier = 10

    def get_hidden_node_output(self, inputs):
        current_inputs = inputs
        for weight_vector in self.weights_per_unit:
            self.activation_function.forward(np.dot(current_inputs, weight_vector))
            current_inputs = np.column_stack((current_inputs, self.activation_function.output))
        return current_inputs

    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.candidate_weights)
        self.activation_function.forward(weighted_sum)
        return self.activation_function.output

    def backward(self, hidden_node_inputs, hidden_node_output, output_covariances, error_term):
        dweights = np.zeros(self.candidate_weights.shape)
        self.activation_function.backward(hidden_node_output)  # f_p'

        for k in range(self.n_candidates):  # maybe this could work without a loop
            tmp1 = np.multiply(self.activation_function.dinputs[:, k][:, np.newaxis], hidden_node_inputs)
            tmp2 = np.dot(tmp1.T, error_term)
            dweights[:, k] = np.dot(np.sign(output_covariances[k, :]), tmp2.T)

        return dweights

    def update_weights(self, dweights):
        self.candidate_weights += self.current_learning_rate * dweights
        self.current_learning_rate = 0.98 * self.current_learning_rate

    def has_units(self):
        return self.n_units > 0

    def init_candidate_weights(self, candidate_n_inputs):
        self.candidate_weights = self.weight_multiplier * np.random.randn(candidate_n_inputs, self.n_candidates)
        self.current_learning_rate = self.learning_rate

    def add_and_train_new_unit(self, current_inputs_and_hidden_activations, errors):
        self.n_units += 1
        candidate_n_inputs = current_inputs_and_hidden_activations.shape[1]
        self.init_candidate_weights(candidate_n_inputs)

        iteration = 0
        max_iterations = 500
        has_converged = False

        mean_covariance_timespan = 15
        mean_covariance_diff_tolerance = 1e-3

        covariances = []
        best_weights = None

        while not has_converged:
            iteration += 1
            candidate_outputs = self.forward(current_inputs_and_hidden_activations)
            complete_covariance, ouput_covariances, error_term = self.calculate_covariance(candidate_outputs, errors)
            dweights = self.backward(current_inputs_and_hidden_activations, candidate_outputs, ouput_covariances, error_term)
            self.update_weights(dweights)
            best_covariance = np.max(complete_covariance)
            best_weights = self.get_best_candidate_weights(complete_covariance)

            if not iteration % 50:
                print(f'iteration: {iteration}, ' +
                      f'covariance: {np.max(complete_covariance):.3f}')

            covariances.append(best_covariance)
            mean_covariances = sum(covariances[-mean_covariance_timespan:]) / len(covariances[-mean_covariance_timespan:])

            if iteration > max_iterations or (len(covariances) > mean_covariance_timespan and best_covariance - mean_covariances <= mean_covariance_diff_tolerance):
                has_converged = True

        self.weights_per_unit.append(best_weights)

    def get_best_candidate_weights(self, covariance):
        return self.candidate_weights[:, np.argmax(covariance)]

    def calculate_covariance(self, hidden_node_outputs, errors):
        value_term = hidden_node_outputs - np.mean(hidden_node_outputs, axis=0)
        error_term = errors - np.mean(errors, axis=0)
        output_covariances = np.dot(value_term.T, error_term)
        complete_covariance = np.sum(np.abs(output_covariances), axis=1)
        return complete_covariance, output_covariances, error_term

    def pruning_step(self, pruning_threshold):
        hidden_pruning_threshold = pruning_threshold * self.weight_multiplier
        for weights in self.weights_per_unit:
            weights[np.abs(weights) < hidden_pruning_threshold] = 0

    def get_pruned_parameters_amount(self, pruning_threshold):
        hidden_pruning_threshold = pruning_threshold * self.weight_multiplier
        pruned_parameters = 0
        for weights in self.weights_per_unit:
            pruned_parameters += np.count_nonzero(np.abs(weights) < hidden_pruning_threshold)
        return pruned_parameters

    def get_last_node_pruned_parameters_amount(self, pruning_threshold):
        hidden_pruning_threshold = pruning_threshold * self.weight_multiplier
        pruned_parameters = 0
        if len(self.weights_per_unit) > 0:
            pruned_parameters += np.count_nonzero(np.abs(self.weights_per_unit[-1]) < hidden_pruning_threshold)
        return pruned_parameters
