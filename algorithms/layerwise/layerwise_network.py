from typing import List

import torch

from algorithms.common.constructive_algorithm import ConstructiveAlgorithm
from algorithms.layerwise.layerwise_layer import LayerwiseLayer
from algorithms.util.input_output_config import InputOutputConfig
from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.pruning_config import PruningConfig
from data.dataset import Dataset
from evaluation.logger.logger import Logger


class LayerwiseNetwork(ConstructiveAlgorithm):
    """Implementation of the Layerwise algorithm"""
    def __init__(self,
                 algorithm_type: AlgorithmTypes,
                 logger: Logger,
                 in_out_config: InputOutputConfig,
                 hyperparameters: {},
                 pruning_config: PruningConfig):
        super().__init__(algorithm_type=algorithm_type, logger=logger, pruning_config=pruning_config)
        self.n_inputs, self.n_outputs = in_out_config.n_inputs, in_out_config.n_outputs
        self.n_starting_hidden_nodes = hyperparameters.get('n_starting_hidden_nodes', 8)
        self.n_max_layer_expansions = hyperparameters.get('n_max_layer_expansions', 3)
        self.layers: List[LayerwiseLayer] = []

    def forward(self, x: torch.Tensor):
        current_output = x
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def forward_frozen_layers(self, x: torch.Tensor):
        current_output = x
        for layer in self.layers:
            if layer.is_frozen:
                current_output = layer.forward(current_output)
        return current_output

    def construction_step(self, x: torch.Tensor, y: torch.Tensor):
        n_inputs_layer = self.n_inputs if len(self.layers) == 0 else self.n_starting_hidden_nodes * 2 ** self.n_max_layer_expansions

        should_create_new_layer = len(self.layers) == 0 or self.layers[-1].is_layer_maximum_size()
        if should_create_new_layer:
            for layer in self.layers:
                layer.freeze_layer()
                layer.remove_output_layer()

            layer_inputs = self.forward(x)
            new_layer = LayerwiseLayer(n_inputs=n_inputs_layer, n_hidden_nodes=self.n_starting_hidden_nodes, n_outputs=self.n_outputs, max_n_expansions=self.n_max_layer_expansions, logger=self.logger)
            new_layer.train_layer(layer_inputs, y)
            self.layers.append(new_layer)

        else:
            layer_inputs = self.forward_frozen_layers(x)
            self.layers[-1].expand_layer()
            self.layers[-1].train_layer(layer_inputs, y)

    def train(self, dataset: Dataset, target_accuracy: float):
        # unused
        pass

    def specific_pruning_step(self):
        threshold = self.pruning_config.magnitude_threshold
        for name, param in self.layers[-1].named_parameters():
            param.data = torch.where(torch.abs(param.data) < threshold, torch.zeros_like(param.data), param.data)

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
