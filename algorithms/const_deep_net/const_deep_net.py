from typing import List

import torch

from algorithms.const_deep_net.const_deep_net_layer import ConstDeepNetLayer
from algorithms.common.constructive_algorithm import ConstructiveAlgorithm
from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.input_output_config import InputOutputConfig
from algorithms.util.pruning_config import PruningConfig
from data.dataset import Dataset
from evaluation.logger.logger import Logger


class ConstDeepNet(ConstructiveAlgorithm):
    """Implementation of the ConstDeepNet algorithm"""
    def __init__(self, algorithm_type: AlgorithmTypes, logger: Logger, in_out_config: InputOutputConfig, hyperparameters: {}, pruning_config: PruningConfig):
        super().__init__(algorithm_type=algorithm_type, logger=logger, pruning_config=pruning_config)
        self.max_layer_size = hyperparameters.get('max_layer_size') if 'max_layer_size' in hyperparameters else 20
        self.learning_rate = hyperparameters.get('learning_rate') if 'learning_rate' in hyperparameters else 0.001
        self.n_inputs, self.n_outputs = in_out_config.n_inputs, in_out_config.n_outputs
        self.layers: List[ConstDeepNetLayer] = []

    def forward(self, x):
        current_output = x
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def forward_frozen_layers(self, x):
        current_output = x
        for layer in self.layers:
            if layer.is_frozen:
                current_output = layer.forward(current_output)
        return current_output

    def construction_step(self, x, y):
        n_inputs_layer = self.n_inputs if len(self.layers) == 0 else self.layers[-1].n_hidden_nodes

        should_create_new_layer = len(self.layers) == 0 or self.layers[-1].n_hidden_nodes > self.max_layer_size
        if should_create_new_layer:
            if len(self.layers) > 0:
                self.layers[-1].freeze_weights()
                self.layers[-1].remove_output_layer()

            layer_inputs = self.forward(x)
            new_layer = ConstDeepNetLayer(n_inputs=n_inputs_layer, n_outputs=self.n_outputs, learning_rate=self.learning_rate, max_layer_size=self.max_layer_size, logger=self.logger)
            new_layer.train_layer(layer_inputs, y)
            self.layers.append(new_layer)

        else:
            layer_inputs = self.forward_frozen_layers(x)
            self.layers[-1].expand_layer()
            self.layers[-1].train_layer(layer_inputs, y)

    def specific_pruning_step(self):
        threshold = self.pruning_config.magnitude_threshold
        for name, param in self.layers[-1].named_parameters():
            param.data = torch.where(torch.abs(param.data) < threshold, torch.zeros_like(param.data), param.data)

    def train(self, dataset: Dataset, target_accuracy):
        pass

    def get_parameter_amount(self) -> int:
        parameter_amount = 0
        for layer in self.layers:
            parameter_amount += layer.get_parameter_amount()
        return parameter_amount

    def get_pruned_parameters_amount(self) -> int:
        pruned_parameter_amount = 0
        for layer in self.layers:
            pruned_parameter_amount += layer.get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        return pruned_parameter_amount

    def get_trainable_parameter_amount(self) -> int:
        parameter_amount = 0
        if len(self.layers) > 0:
            parameter_amount += self.layers[-1].get_parameter_amount()
            if self.pruning_config.is_pruning_active:
                parameter_amount -= self.layers[-1].get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        return parameter_amount
