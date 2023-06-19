import numpy as np

from algorithms.common.constructive_algorithm import ConstructiveAlgorithm
from algorithms.util.activation_functions import ActivationSoftmaxLossCategoricalCrossEntropy, ActivationSoftMax
from algorithms.cascor_ultra.cascor_ultra_hidden_units import CasCorUltraHiddenUnits
from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.pruning_config import PruningConfig
from data.dataset import Dataset
from evaluation.evaluators.constants import Constants
from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class CasCorUltraNetwork(ConstructiveAlgorithm):
    """Implementation of the CasCorUltra algorithm"""
    def __init__(self, algorithm_type: AlgorithmTypes, n_inputs: int, n_outputs: int, logger: Logger, hyperparameters: {}, pruning_config: PruningConfig):
        super().__init__(algorithm_type=algorithm_type, logger=logger, pruning_config=pruning_config)
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.weights, self.dweights, self.dinputs = None, None, None

        self.activation_and_loss = ActivationSoftmaxLossCategoricalCrossEntropy()
        self.activation_softmax = ActivationSoftMax()

        n_candidates = hyperparameters.get('n_candidates') if 'n_candidates' in hyperparameters else 50
        learning_rate = hyperparameters.get('learning_rate') if 'learning_rate' in hyperparameters else 0.05
        candidate_correlation_threshold = hyperparameters.get('candidate_correlation_threshold') if 'candidate_correlation_threshold' in hyperparameters else 0.3
        max_add_candidates = hyperparameters.get('max_add_candidates') if 'max_add_candidates' in hyperparameters else 5
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = hyperparameters.get('decay') if 'decay' in hyperparameters else 1e-4
        self.momentum = hyperparameters.get('momentum') if 'momentum' in hyperparameters else 0.6
        self.weight_momentums = None
        self.n_corrective_steps = hyperparameters.get('n_corrective_steps') if 'n_corrective_steps' in hyperparameters else 10
        self.epoch = 0

        self.cur_X, self.cur_y = None, None

        self.hidden_units = CasCorUltraHiddenUnits(n_candidates=n_candidates,
                                                   learning_rate=learning_rate,
                                                   candidate_correlation_threshold=candidate_correlation_threshold,
                                                   max_add_candidates=max_add_candidates)

        self.reset_weights()

    def construction_step(self, x, y):
        pass

    def uses_pytorch(self):
        return False

    def reset_weights(self):
        self.weights = 0.01 * np.random.randn(self.n_inputs + 1 + self.hidden_units.n_units, self.n_outputs)  # +1 for bias input

    def forward(self, inputs):
        hidden_nodes_out = self.hidden_units.get_hidden_node_output(inputs)
        out = np.dot(hidden_nodes_out, self.weights)
        self.activation_softmax.forward(out)
        return self.activation_softmax.output

    def forward_output_layer(self, inputs, is_training=False):
        weighted_sums = np.dot(inputs, self.weights)
        if is_training:
            loss = self.activation_and_loss.forward(weighted_sums, self.cur_y)
            return self.activation_and_loss.output, loss
        else:
            self.activation_softmax.forward(weighted_sums)
            return self.activation_softmax.output

    def backward(self, inputs, outputs):
        self.activation_and_loss.backward(outputs, self.cur_y)
        self.dweights = np.dot(inputs.T, self.activation_and_loss.dinputs)
        self.dinputs = np.dot(self.activation_and_loss.dinputs, self.weights.T)

    def update_weights(self):
        if self.momentum:
            if self.weight_momentums is None or self.weight_momentums.shape != self.weights.shape:
                self.weight_momentums = np.zeros_like(self.weights)
            weight_updates = self.momentum * self.weight_momentums - self.current_learning_rate * self.dweights
            self.weight_momentums = weight_updates
        else:
            weight_updates = -self.current_learning_rate * self.dweights

        self.weights += weight_updates
        self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.epoch))

    def train(self, dataset: Dataset, target_accuracy):
        X, y = dataset.train_x.data.numpy(), dataset.train_y.data.numpy()

        splits_percent = (0.7, 0.3)
        data_length = len(y)
        split_index = [int(splits_percent[0] * data_length)]
        train_X, eval_X = np.split(X, split_index, axis=0)
        train_y, eval_y = np.split(y, split_index, axis=0)

        self.cur_X = train_X
        self.cur_y = train_y

        iterations = 0
        max_iterations = 50
        has_finished = False
        inputs_and_bias = np.column_stack((train_X, np.ones(train_X.shape[0])))
        current_errors = None
        current_inputs_and_hidden_activations = inputs_and_bias
        best_accuracy: float = 0
        worse_count: int = 0

        while not has_finished and iterations < max_iterations:

            iterations += 1
            self.logger.log_scalar(LogTag.ConstructionStep, iterations)

            current_best_accuracy = 0
            has_plateaued = False
            self.current_learning_rate = self.learning_rate

            mean_loss_timespan = 100
            mean_loss_diff_tolerance = 1e-4

            losses = []
            self.epoch = 0
            max_epochs = 5000

            if iterations > 1:
                self.hidden_units.add_and_train_new_unit(current_inputs_and_hidden_activations, current_errors)

            current_inputs_and_hidden_activations = self.hidden_units.get_hidden_node_output(inputs_and_bias)
            self.reset_weights()

            print(f'\n --------------------- \n'
                  f'iterations: {iterations}, ' +
                  f'hidden nodes: {self.hidden_units.n_units}')

            while not has_plateaued:
                output_activations, loss = self.forward_output_layer(current_inputs_and_hidden_activations, is_training=True)
                correct_output_node_values = np.zeros(output_activations.shape)
                correct_output_node_values[range(len(train_y)), train_y] = 1
                errors = output_activations - correct_output_node_values
                self.backward(current_inputs_and_hidden_activations, self.activation_and_loss.output)
                self.update_weights()

                predictions = np.argmax(self.activation_and_loss.output, axis=1)
                if len(train_y.shape) == 2:
                    train_y = np.argmax(train_y, axis=1)
                accuracy = np.mean(predictions == train_y)

                self.epoch += 1
                if not self.epoch % 50:
                    print(f'epoch: {self.epoch}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.5f}, ' +
                          f'lr: {self.current_learning_rate:.5f}, ')

                losses.append(loss)
                last_mean_loss = sum(losses[-mean_loss_timespan:]) / len(losses[-mean_loss_timespan:])
                current_mean_loss = sum(losses[-(mean_loss_timespan // 3):]) / len(losses[-(mean_loss_timespan // 3):])

                if self.epoch > max_epochs or (len(losses) > mean_loss_timespan and last_mean_loss - current_mean_loss <= mean_loss_diff_tolerance):
                    has_plateaued = True
                    current_errors = errors
                    current_best_accuracy = accuracy

                    dataset.visualize(X, y, logger=self.logger, model=self)
                    self.logger.log_scalar(LogTag.ConstructionLoss, loss)
                    self.logger.log_scalar(LogTag.ConstructionAccuracy, accuracy)
                    self.logger.log_scalar(LogTag.ConstructionTotalParameters, self.get_parameter_amount())
                    self.logger.log_scalar(LogTag.ConstructionPrunedParameters, self.get_pruned_parameters_amount())
                    self.logger.log_scalar(LogTag.ConstructionTrainableParameters, self.get_trainable_parameter_amount())
                    self.logger.log_scalar(LogTag.ConstructionStepEpochs, self.epoch)
                    self.logger.log_figure(LogTag.ResultHistoryPlot)
                    self.logger.log_figure(LogTag.ResultHistoryPlot)

            # corrective step
            corrective_losses = []
            correction_has_converged = False
            use_corrective_step = True
            if use_corrective_step:
                while not correction_has_converged:
                    current_inputs_and_hidden_activations = self.hidden_units.get_hidden_node_output(inputs_and_bias)
                    output_activations, loss = self.forward_output_layer(current_inputs_and_hidden_activations, is_training=True)
                    self.backward(current_inputs_and_hidden_activations, self.activation_and_loss.output)
                    self.update_weights()
                    self.hidden_units.corrective_step(self.dinputs[:, -self.hidden_units.n_units:])
                    current_inputs_and_hidden_activations = self.hidden_units.get_hidden_node_output(inputs_and_bias)
                    output_activations, loss = self.forward_output_layer(current_inputs_and_hidden_activations, is_training=True)
                    corrective_losses.append(loss)
                    print(loss)
                    if len(corrective_losses) > self.n_corrective_steps and sum(corrective_losses[-self.n_corrective_steps//2:]) + 0.0001 >= sum(corrective_losses[-self.n_corrective_steps:-self.n_corrective_steps//2]):
                        break

            if self.pruning_config.is_pruning_active:
                self.pruning_step()

            if current_best_accuracy > best_accuracy:
                best_accuracy = current_best_accuracy
                worse_count = 0
            else:
                worse_count += 1

            has_finished = best_accuracy >= target_accuracy or worse_count >= Constants.convergence_check_range

    def test(self, dataset: Dataset):
        X, y = dataset.test_x.data.numpy(), dataset.test_y.data.numpy()
        x_and_bias = np.column_stack((X, np.ones(X.shape[0])))

        inputs_and_hidden_activations = self.hidden_units.get_hidden_node_output(x_and_bias)
        self.forward_output_layer(inputs_and_hidden_activations, is_training=False)

        test_pred = np.argmax(self.activation_softmax.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(test_pred == y)

        self.logger.log_scalar(LogTag.TestAccuracy, accuracy)
        #dataset.visualize(dataset.test_x.data.numpy(), dataset.test_y.data.numpy(), logger=self.logger, model=self, save=True)

    def specific_pruning_step(self):
        self.weights[np.abs(self.weights) < self.pruning_config.magnitude_threshold] = 0
        self.hidden_units.pruning_step(self.pruning_config.magnitude_threshold)

    def get_parameter_amount(self):
        n_inputs_and_bias = self.n_inputs + 1
        parameter_amount = 0
        layer_index = 1
        n_previous_hidden_nodes = 0

        for node_weights_in_layer in self.hidden_units.weights_per_unit:
            n_hidden_nodes_in_layer = node_weights_in_layer.shape[1]
            parameter_amount += (n_inputs_and_bias + n_previous_hidden_nodes) * n_hidden_nodes_in_layer
            n_previous_hidden_nodes += n_hidden_nodes_in_layer
            layer_index += 1

        parameter_amount += (n_inputs_and_bias + self.hidden_units.n_units) * self.n_outputs
        return parameter_amount

    def get_pruned_parameters_amount(self) -> int:
        pruned_parameters = 0
        pruned_parameters += np.count_nonzero(np.abs(self.weights) < self.pruning_config.magnitude_threshold)
        pruned_parameters += self.hidden_units.get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        return pruned_parameters

    def get_trainable_parameter_amount(self) -> int:
        n_inputs_and_bias = self.n_inputs + 1
        parameter_amount = 0

        if self.hidden_units.n_units > 0:
            parameter_amount += (n_inputs_and_bias + self.hidden_units.n_units - 1) * self.hidden_units.n_candidates
            if self.pruning_config.is_pruning_active:
                parameter_amount -= self.hidden_units.get_last_node_pruned_parameters_amount(self.pruning_config.magnitude_threshold)

        parameter_amount += (n_inputs_and_bias + self.hidden_units.n_units) * self.n_outputs
        if self.pruning_config.is_pruning_active:
            parameter_amount -= np.count_nonzero(np.abs(self.weights) < self.pruning_config.magnitude_threshold)

        return parameter_amount