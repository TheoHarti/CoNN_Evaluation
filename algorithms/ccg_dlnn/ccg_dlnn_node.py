import torch
import torch.nn.functional
from torch import nn

from algorithms.common import covariance_loss_function
from algorithms.util.convergence_checker import has_converged
from evaluation.logger.logger import Logger


class CCG_DLNN_Node(torch.nn.Module):
    def __init__(self, logger: Logger, n_inputs, weight_multiplier, learning_rate):
        super(CCG_DLNN_Node, self).__init__()
        self.logger = logger
        self.is_frozen = False
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.weights = torch.nn.Linear(self.n_inputs, 1)
        self.weight_multiplier = weight_multiplier
        nn.init.uniform_(self.weights.weight, a=-self.weight_multiplier, b=self.weight_multiplier)
        nn.init.uniform_(self.weights.bias, a=-self.weight_multiplier, b=self.weight_multiplier)
        self.best_correlation = 0
        self.trained_epochs = 0

    def forward(self, x):
        return torch.sigmoid(self.weights(x))

    def train_node(self, x, residual_errors):
        self.best_correlation = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = covariance_loss_function.covariance
        convergence_check_range = 25
        losses = []
        epoch = 0
        is_finished = False

        while not is_finished:
            epoch += 1
            out = self.forward(x)
            loss = loss_function(out, residual_errors)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
            correlation = abs(loss.item())
            if correlation > self.best_correlation:
                self.best_correlation = correlation

            if len(losses) % 25 == 0:
                print(f'Epoch: {str(epoch)} | CCG DLNN Node Correlation: {correlation}')
                is_finished = has_converged(losses, convergence_check_range) or epoch > 1000

        self.trained_epochs = epoch

    def get_trained_epochs(self):
        return self.trained_epochs

    def get_best_correlation(self):
        return self.best_correlation

    def freeze(self):
        self.is_frozen = True
        for param in self.weights.parameters():
            param.requires_grad = False

    def pruning_step(self, prune_threshold: float):
        for name, param in self.weights.named_parameters():
            param.data = torch.where(torch.abs(param.data) < prune_threshold * self.weight_multiplier, torch.zeros_like(param.data), param.data)

    def get_parameter_amount(self) -> int:
        return self.n_inputs + 1  # +1 for bias

    def get_pruned_parameters_amount(self, prune_threshold: float) -> int:
        pruned_parameters = 0
        for name, param in self.weights.named_parameters():
            pruned_parameters += torch.numel(param.data[abs(param.data) < prune_threshold * self.weight_multiplier])

        return pruned_parameters