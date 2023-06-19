import numpy as np

from algorithms.common.constructive_algorithm import ConstructiveAlgorithm
from algorithms.util.activation_functions import ActivationSoftmaxLossCategoricalCrossEntropy, ActivationSoftMax
from algorithms.cascade_correlation.hidden_units import HiddenUnits
from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.pruning_config import PruningConfig
from data.dataset import Dataset
from evaluation.evaluators.constants import Constants
from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class CascadeCorrelationNetwork(ConstructiveAlgorithm):
    """Implementation of the CascadeCorrelation architecture"""
    def __init__(self, algorithm_type: AlgorithmTypes, n_inputs: int, n_outputs: int, logger: Logger, hyperparameters: {}, pruning_config: PruningConfig):
        super().__init__(algorithm_type=algorithm_type, logger=logger, pruning_config=pruning_config)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = None
        self.dweights = None

        self.activation_and_loss = ActivationSoftmaxLossCategoricalCrossEntropy()
        self.activation_softmax = ActivationSoftMax()

        self.cur_X = None
        self.cur_y = None

        n_candidates = hyperparameters.get('n_candidates') if 'n_candidates' in hyperparameters else 15
        learning_rate = hyperparameters.get('learning_rate') if 'learning_rate' in hyperparameters else 1
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = hyperparameters.get('decay') if 'decay' in hyperparameters else 1e-4
        self.epoch = 0

        self.hidden_units = HiddenUnits(n_candidates=n_candidates, learning_rate=learning_rate)

        self.current_errors = None
        self.current_best_accuracy = 0

        self.reset_weights()

    def reset_weights(self):
        self.weights = 0.01 * np.random.randn(self.n_inputs + 1 + self.hidden_units.n_units, self.n_outputs)  # +1 for bias input

    def forward_with_loss(self, inputs_and_hidden_outputs):
        out = np.dot(inputs_and_hidden_outputs, self.weights)
        loss = self.activation_and_loss.forward(out, self.cur_y)
        return self.activation_and_loss.output, loss

    def forward(self, x):
        hidden_nodes_out = self.hidden_units.get_hidden_node_output(x)
        out = np.dot(hidden_nodes_out, self.weights)
        self.activation_softmax.forward(out)
        return self.activation_softmax.output

    def backward(self, inputs, outputs):
        self.activation_and_loss.backward(outputs, self.cur_y)
        self.dweights = np.dot(inputs.T, self.activation_and_loss.dinputs)

    def update_weights(self):
        weight_updates = -self.current_learning_rate * self.dweights
        self.weights += weight_updates
        self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.epoch))

    def construction_step(self, x, y):
        pass

    def train(self, dataset: Dataset, target_accuracy):
        X, y = dataset.train_x.data.numpy(), dataset.train_y.data.numpy()

        splits_percent = (0.8, 0.2)
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
        current_inputs_and_hidden_activations = inputs_and_bias
        best_accuracy: float = 0
        worse_count: int = 0

        while not has_finished and iterations < max_iterations:

            iterations += 1
            self.logger.log_scalar(LogTag.ConstructionStep, iterations)

            if iterations > 1:
                # ---- train new layer -----
                self.hidden_units.add_and_train_new_unit(current_inputs_and_hidden_activations, self.current_errors)

            current_inputs_and_hidden_activations = self.hidden_units.get_hidden_node_output(inputs_and_bias)
            self.reset_weights()

            print(f'\n --------------------- \n'
                  f'iterations: {iterations}, ' +
                  f'hidden nodes: {self.hidden_units.n_units}')

            self.train_output_layer(dataset, current_inputs_and_hidden_activations, train_y)
            if self.pruning_config.is_pruning_active:
                self.pruning_step()

            if self.current_best_accuracy > best_accuracy:
                best_accuracy = self.current_best_accuracy
                worse_count = 0
            else:
                worse_count += 1

            has_finished = best_accuracy >= target_accuracy or worse_count >= Constants.convergence_check_range

    def train_output_layer(self, dataset: Dataset, current_inputs_and_hidden_activations, train_y):
        X, y = dataset.train_x.data.numpy(), dataset.train_y.data.numpy()

        self.current_learning_rate = 1

        mean_loss_timespan = 25
        mean_loss_diff_tolerance = 1e-4

        losses = []
        self.epoch = 0
        max_epochs = 5000

        has_plateaued = False

        while not has_plateaued:
            output_activations, loss = self.forward_with_loss(current_inputs_and_hidden_activations)
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
                self.current_errors = errors
                self.current_best_accuracy = accuracy

                dataset.visualize(X, y, logger=self.logger, model=self)
                self.logger.log_scalar(LogTag.ConstructionLoss, loss)
                self.logger.log_scalar(LogTag.ConstructionAccuracy, accuracy)
                self.logger.log_scalar(LogTag.ConstructionTotalParameters, self.get_parameter_amount())
                self.logger.log_scalar(LogTag.ConstructionPrunedParameters, self.get_pruned_parameters_amount())
                self.logger.log_scalar(LogTag.ConstructionTrainableParameters, self.get_trainable_parameter_amount())
                self.logger.log_scalar(LogTag.ConstructionStepEpochs, self.epoch)
                self.logger.log_figure(LogTag.ResultHistoryPlot)

    def test(self, dataset: Dataset):
        X, y = dataset.test_x.data.numpy(), dataset.test_y.data.numpy()
        # --- GET TEST SET RESULTS ---
        X = np.column_stack((X, np.ones(X.shape[0])))

        y_predicted = self.forward(X)

        test_pred = np.argmax(y_predicted, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(test_pred == y)

        self.logger.log_scalar(LogTag.TestAccuracy, accuracy)

    def uses_pytorch(self):
        return False

    def specific_pruning_step(self):
        self.weights[np.abs(self.weights) < self.pruning_config.magnitude_threshold] = 0
        self.hidden_units.pruning_step(self.pruning_config.magnitude_threshold)

    def get_parameter_amount(self):
        n_inputs_and_bias = self.n_inputs + 1
        parameter_amount = 0

        for i in range(1, self.hidden_units.n_units + 1):  # run through all construction steps
            parameter_amount += n_inputs_and_bias + i - 1

        parameter_amount += (n_inputs_and_bias + self.hidden_units.n_units) * self.n_outputs
        return parameter_amount

    def get_pruned_parameters_amount(self) -> int:
        pruned_parameters = 0
        pruned_parameters += np.count_nonzero(np.abs(self.weights) < self.pruning_config.magnitude_threshold)
        pruned_parameters += self.hidden_units.get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        return pruned_parameters

    def get_trainable_parameter_amount(self):
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