from typing import List

import torch
import torch.nn.functional

from algorithms.ccg_dlnn.ccg_dlnn_node import CCG_DLNN_Node
from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class CCG_DLNN_Layer(torch.nn.Module):
    def __init__(self, logger: Logger, n_inputs, max_layer_size, n_candidates, weight_multiplier, learning_rate):
        super(CCG_DLNN_Layer, self).__init__()
        self.logger = logger
        self.learning_rate = learning_rate
        self.n_candidates = n_candidates
        self.max_layer_size = max_layer_size
        self.weight_multiplier = weight_multiplier
        self.n_inputs, self.n_outputs = n_inputs, 1
        self.layer_nodes: List[CCG_DLNN_Node] = []

    def forward(self, x):
        return torch.cat([node(x) for node in self.layer_nodes], dim=1)

    def expand_layer(self, layer_inputs, output_layer_errors):
        best_correlation = 0
        best_candidate = None

        for i in range(self.n_candidates):
            candidate_node = CCG_DLNN_Node(logger=self.logger, n_inputs=self.n_inputs, weight_multiplier=self.weight_multiplier, learning_rate=self.learning_rate)
            candidate_node.train_node(layer_inputs, output_layer_errors)
            if candidate_node.get_best_correlation() > best_correlation:
                best_correlation = candidate_node.get_best_correlation()
                best_candidate = candidate_node

        self.layer_nodes.append(best_candidate)
        self.logger.log_scalar(log_tag=LogTag.ConstructionStepEpochs, scalar=best_candidate.get_trained_epochs())

    def pruning_step(self, prune_threshold: float):
        self.layer_nodes[-1].pruning_step(prune_threshold)

    def get_node_amount(self):
        return len(self.layer_nodes)

    def get_parameter_amount(self) -> int:
        parameters = 0
        for node in self.layer_nodes:
            parameters += node.get_parameter_amount()
        return parameters

    def get_pruned_parameters_amount(self, prune_threshold: float) -> int:
        parameters = 0
        for node in self.layer_nodes:
            parameters += node.get_pruned_parameters_amount(prune_threshold)
        return parameters

    def get_trainable_parameter_amount(self) -> int:
        return self.layer_nodes[-1].get_parameter_amount() * self.n_candidates
