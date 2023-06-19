from typing import List

import torch

from algorithms.ccg_dlnn.ccg_dlnn_layer import CCG_DLNN_Layer
from algorithms.ccg_dlnn.ccg_dlnn_outputs import CCG_DLNN_Outputs
from algorithms.common.constructive_algorithm import ConstructiveAlgorithm
from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.input_output_config import InputOutputConfig
from algorithms.util.pruning_config import PruningConfig
from data.dataset import Dataset
from evaluation.logger.logger import Logger


class CCG_DLNN(ConstructiveAlgorithm):
    """Implementation of the CCG_DLNN algorithm"""
    def __init__(self, algorithm_type: AlgorithmTypes, logger: Logger, in_out_config: InputOutputConfig, hyperparameters: {}, pruning_config: PruningConfig):
        super().__init__(algorithm_type=algorithm_type, logger=logger, pruning_config=pruning_config)
        self.n_inputs, self.n_outputs = in_out_config.n_inputs, in_out_config.n_outputs
        self.weight_multiplier = 3
        self.max_layer_size = hyperparameters.get('max_layer_size') if 'max_layer_size' in hyperparameters else 32
        self.n_candidates = hyperparameters.get('n_candidates') if 'n_candidates' in hyperparameters else 25
        self.learning_rate_out = hyperparameters.get('learning_rate_out') if 'learning_rate_out' in hyperparameters else 0.005
        self.learning_rate_hidden = hyperparameters.get('learning_rate_hidden') if 'learning_rate_hidden' in hyperparameters else 0.03
        self.layers: List[CCG_DLNN_Layer] = []
        self.output: CCG_DLNN_Outputs = CCG_DLNN_Outputs(logger=self.logger, n_inputs=self.n_inputs, n_outputs=self.n_outputs, weight_multiplier=self.weight_multiplier, learning_rate=self.learning_rate_out)
        self.output_error_correlations = None

    def forward(self, x):
        last_hidden_layer_outputs = self.forward_hidden_layers(x)
        return self.output.forward(last_hidden_layer_outputs)

    def forward_hidden_layers(self, x):
        current_output = x
        for layer in self.layers:
            current_output = torch.cat((current_output, layer(current_output)), dim=1)
        return current_output

    def forward_frozen_layers(self, x):
        current_output = x
        for layer in self.layers[0:-1]:  # check if range is right
            current_output = torch.cat((current_output, layer(current_output)), dim=1)
        return current_output

    def construction_step(self, x, y):
        if len(self.layers) == 0:
            self.train_output_layer(x, y)

        self.add_node(torch.clone(x))
        self.train_output_layer(x, y)

    def train_output_layer(self, x, y):
        cloned_x = torch.clone(x)
        output_layer_inputs = self.forward_hidden_layers(cloned_x)
        self.output.train_layer(output_layer_inputs, y)

    def add_node(self, x):
        should_create_new_layer = len(self.layers) == 0 or len(self.layers[-1].layer_nodes) > self.max_layer_size
        if should_create_new_layer:
            layer_inputs = self.forward_hidden_layers(x)
            new_layer = CCG_DLNN_Layer(n_inputs=layer_inputs.shape[1], max_layer_size=self.max_layer_size, logger=self.logger, n_candidates=self.n_candidates, weight_multiplier=self.weight_multiplier, learning_rate=self.learning_rate_hidden)
            new_layer.expand_layer(layer_inputs, self.output.get_errors())
            self.layers.append(new_layer)
            self.output.change_inputs(layer_inputs.shape[1] + 1)

        else:
            layer_inputs = self.forward_frozen_layers(x)
            self.layers[-1].expand_layer(layer_inputs, self.output.get_errors())
            self.output.change_inputs(layer_inputs.shape[1] + len(self.layers[-1].layer_nodes))

    def specific_pruning_step(self):
        for layer in self.layers:
            layer.pruning_step(self.pruning_config.magnitude_threshold)
        self.output.pruning_step(self.pruning_config.magnitude_threshold)

    def train(self, dataset: Dataset, target_accuracy):
        pass

    def get_parameter_amount(self) -> int:
        parameter_amount = 0
        for layer in self.layers:
            parameter_amount += layer.get_parameter_amount()
        parameter_amount += self.output.get_parameter_amount()
        return parameter_amount

    def get_pruned_parameters_amount(self) -> int:
        parameter_amount = 0
        for layer in self.layers:
            parameter_amount += layer.get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        parameter_amount += self.output.get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        return parameter_amount

    def get_trainable_parameter_amount(self) -> int:
        parameter_amount = 0
        if len(self.layers) > 0:
            parameter_amount += self.layers[-1].get_trainable_parameter_amount()

        parameter_amount += self.output.get_parameter_amount()
        if self.pruning_config.is_pruning_active:
            parameter_amount -= self.output.get_pruned_parameters_amount(self.pruning_config.magnitude_threshold)
        return parameter_amount
