import numpy as np
import torch
import torch.nn.functional
import torchbnn

from algorithms.util.convergence_checker import has_converged
from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class UncertaintySplittingLayer(torch.nn.Module):
    def __init__(self, logger: Logger, n_inputs: int, n_starting_hidden_nodes: int, n_outputs: int, max_layer_size: int, n_uncertain_nodes: int, n_replacement_nodes: int):
        super(UncertaintySplittingLayer, self).__init__()
        self.logger = logger
        self.max_layer_size = max_layer_size
        self.is_frozen = False
        self.n_inputs, self.n_hidden_nodes, self.n_outputs = n_inputs, n_starting_hidden_nodes, n_outputs
        self.n_uncertain_nodes, self.n_replacement_nodes = n_uncertain_nodes, n_replacement_nodes
        self.hidden = torchbnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=n_inputs, out_features=n_starting_hidden_nodes)
        self.out = torchbnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=n_starting_hidden_nodes, out_features=n_outputs)
        self.output_layer_active: bool = True

    def forward(self, x):
        hidden_layer_out = torch.nn.functional.relu(self.hidden(x))  # csv_test if F.relu is needed
        return self.out(hidden_layer_out) if self.output_layer_active else hidden_layer_out

    def expand_layer(self):
        uncertainty_node_indices = self.get_uncertain_nodes()
        old_layer = self.hidden
        self.n_hidden_nodes = old_layer.out_features + (self.n_uncertain_nodes-1) * len(uncertainty_node_indices)
        new_hidden_layer = torchbnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=self.n_inputs, out_features=self.n_hidden_nodes)

        # Copy the non-swapped weights and biases from the old layer
        new_hidden_layer.weight_mu.data[:old_layer.out_features, :old_layer.in_features] = old_layer.weight_mu.data
        new_hidden_layer.weight_log_sigma.data[:old_layer.out_features, :old_layer.in_features] = old_layer.weight_log_sigma.data
        new_hidden_layer.bias_mu.data[:old_layer.out_features] = old_layer.bias_mu.data
        new_hidden_layer.bias_log_sigma.data[:old_layer.out_features] = old_layer.bias_log_sigma.data

        self.hidden = new_hidden_layer
        self.out = torchbnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=self.n_hidden_nodes, out_features=self.n_outputs)

    def get_uncertain_nodes(self):
        # Uncertainty is measured by the standard deviation of the weights
        weight_stds = torch.exp(self.hidden.weight_log_sigma.data).detach().numpy()
        flattened_weight_stds = weight_stds.flatten()
        # The nodes with the highest uncertainty are those with the highest standard deviations
        uncertain_node_indices = np.argsort(-flattened_weight_stds)[:self.n_uncertain_nodes]
        # Reshape the indices to match the original array dimensions
        row_indices, col_indices = np.unravel_index(uncertain_node_indices, weight_stds.shape)
        uncertain_node_index_pairs = list(zip(row_indices, col_indices))
        return uncertain_node_index_pairs

    def freeze_layer(self):
        self.is_frozen = True

    def train_layer(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.03)
        loss_function = torch.nn.CrossEntropyLoss()
        convergence_check_range = 25
        losses = []
        epoch = 0
        is_finished = False

        while not is_finished:
            out = self.forward(x)
            loss = loss_function(out, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
            epoch += 1
            if epoch % 25 == 0:
                is_finished = has_converged(losses, convergence_check_range)
                print('Epoch: ' + str(epoch))

        self.logger.log_scalar(log_tag=LogTag.ConstructionStepEpochs, scalar=epoch)

    def remove_output_layer(self):
        self.output_layer_active = False

    def get_parameter_amount(self):
        parameters = self.n_inputs * self.n_hidden_nodes + self.n_hidden_nodes
        if self.output_layer_active:
            parameters += self.n_hidden_nodes * self.n_outputs + self.n_outputs
        parameters *= 2  # each weight/bias has mean and std
        return parameters

    def get_pruned_parameters_amount(self, prune_threshold: float) -> int:
        pruned_parameters = 0

        for name, param in self.hidden.named_parameters():
            if "_mu" in name:
                pruned_parameters += torch.numel(param.data[abs(param.data) < prune_threshold])

        if self.output_layer_active:
            for name, param in self.out.named_parameters():
                if "_mu" in name:
                    pruned_parameters += torch.numel(param.data[abs(param.data) < prune_threshold])

        return pruned_parameters * 2  # times two because sigma is also pruned
