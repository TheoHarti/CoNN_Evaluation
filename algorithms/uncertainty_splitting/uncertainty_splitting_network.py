
from typing import List

import torch

from algorithms.common.constructive_algorithm import ConstructiveAlgorithm
from algorithms.uncertainty_splitting.uncertainty_spitting_layer import UncertaintySplittingLayer
from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.input_output_config import InputOutputConfig
from algorithms.util.pruning_config import PruningConfig
from data.dataset import Dataset
from evaluation.logger.logger import Logger


class UncertaintySplittingNetwork(ConstructiveAlgorithm):
    """Implementation of the UncertaintySplitting algorithm"""
    def __init__(self,
                 algorithm_type: AlgorithmTypes,
                 logger: Logger,
                 in_out_config: InputOutputConfig,
                 hyperparameters: {},
                 pruning_config: PruningConfig):
        super().__init__(algorithm_type=algorithm_type, logger=logger, pruning_config=pruning_config)
        self.n_inputs, self.n_outputs = in_out_config.n_inputs, in_out_config.n_outputs
        self.n_starting_hidden_nodes = hyperparameters.get('n_starting_hidden_nodes', 4)
        self.max_layer_size = hyperparameters.get('max_layer_size', 10)
        self.n_uncertain_nodes = hyperparameters.get('n_uncertain_nodes', 3)
        self.n_replacement_nodes = hyperparameters.get('n_replacement_nodes', 3)
        self.layers: List[UncertaintySplittingLayer] = []

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
        should_create_new_layer = len(self.layers) == 0 or self.layers[-1].n_hidden_nodes >= self.max_layer_size
        if should_create_new_layer:
            for layer in self.layers:
                layer.freeze_layer()
                layer.remove_output_layer()

            layer_inputs = self.forward(x)
            new_layer = UncertaintySplittingLayer(n_inputs=n_inputs_layer, n_starting_hidden_nodes=self.n_starting_hidden_nodes, n_outputs=self.n_outputs, max_layer_size=self.max_layer_size, n_uncertain_nodes=self.n_uncertain_nodes, n_replacement_nodes=self.n_replacement_nodes, logger=self.logger)
            new_layer.train_layer(layer_inputs, y)
            self.layers.append(new_layer)

        else:
            layer_inputs = self.forward_frozen_layers(x)
            self.layers[-1].expand_layer()
            self.layers[-1].train_layer(layer_inputs, y)

    def train(self, dataset: Dataset, target_accuracy):
        pass

    def specific_pruning_step(self):
        threshold = self.pruning_config.magnitude_threshold
        hidden_weights_prune_map = None
        hidden_bias_prune_map = None
        out_weights_prune_map = None
        out_bias_prune_map = None

        for name, param in self.layers[-1].named_parameters():
            if name == "hidden.weight_mu":
                hidden_weights_prune_map = torch.abs(param.data) < threshold
            if name == "hidden.bias_mu":
                hidden_bias_prune_map = torch.abs(param.data) < threshold
            if name == "out.weight_mu":
                out_weights_prune_map = torch.abs(param.data) < threshold
            if name == "out.bias_mu":
                out_bias_prune_map = torch.abs(param.data) < threshold

        for name, param in self.layers[-1].named_parameters():
            if name == "hidden.weight_mu":
                param.data = torch.where(hidden_weights_prune_map, torch.zeros_like(param.data), param.data)
            if name == "hidden.weight_log_sigma":
                param.data = torch.where(hidden_weights_prune_map, torch.mul(torch.ones_like(param.data), -10000000), param.data)
            if name == "hidden.bias_mu":
                param.data = torch.where(hidden_bias_prune_map, torch.zeros_like(param.data), param.data)
            if name == "hidden.bias_log_sigma":
                param.data = torch.where(hidden_bias_prune_map, torch.mul(torch.ones_like(param.data), -10000000), param.data)
            if name == "out.weight_mu":
                param.data = torch.where(out_weights_prune_map, torch.zeros_like(param.data), param.data)
            if name == "out.weight_log_sigma":
                param.data = torch.where(out_weights_prune_map, torch.mul(torch.ones_like(param.data), -10000000), param.data)
            if name == "out.bias_mu":
                param.data = torch.where(out_bias_prune_map, torch.zeros_like(param.data), param.data)
            if name == "out.bias_log_sigma":
                param.data = torch.where(out_bias_prune_map, torch.mul(torch.ones_like(param.data), -10000000), param.data)

    def get_parameter_amount(self):
        parameter_amount = 0
        for layer in self.layers:
            parameter_amount += layer.get_parameter_amount()
        return parameter_amount

    def get_pruned_parameters_amount(self) -> int:
        parameter_amount = 0
        for layer in self.layers:
            parameter_amount += layer.get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        return parameter_amount

    def get_trainable_parameter_amount(self) -> int:
        parameter_amount = 0
        if len(self.layers) > 0:
            parameter_amount += self.layers[-1].get_parameter_amount()
            if self.pruning_config.is_pruning_active:
                parameter_amount -= self.layers[-1].get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        return parameter_amount
