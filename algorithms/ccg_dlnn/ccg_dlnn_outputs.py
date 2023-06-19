import torch
from torch import nn
import torch.nn.functional as torch_fun

from algorithms.util.convergence_checker import has_converged
from evaluation.logger.logger import Logger


class CCG_DLNN_Outputs(torch.nn.Module):
    def __init__(self, logger: Logger, n_inputs, n_outputs, weight_multiplier, learning_rate):
        super(CCG_DLNN_Outputs, self).__init__()
        self.logger = logger
        self.learning_rate = learning_rate
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.weights = nn.Linear(self.n_inputs, self.n_outputs)
        self.weight_multiplier = 0.01
        nn.init.uniform_(self.weights.weight, a=-self.weight_multiplier, b=self.weight_multiplier)
        nn.init.uniform_(self.weights.bias, a=-self.weight_multiplier, b=self.weight_multiplier)
        self.error_correlation = None
        self.errors = None

    def forward(self, x):
        return torch.softmax(self.weights(x), dim=1)

    def train_layer(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()
        convergence_check_range = 50
        losses = []
        outputs = None

        while not has_converged(losses, convergence_check_range):
            outputs = self.forward(x)
            loss = loss_function(outputs, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
            outputs.detach()

            if len(losses) % 25 == 0:
                print(f'Output Layer Loss: {loss}')

        with torch.no_grad():  # remove? output variable has requires grad true?!?!
            # Calculate the residuals for correlation
            one_hot_y = torch_fun.one_hot(y)
            self.errors = outputs - one_hot_y

    def get_error_correlations(self):
        return self.error_correlation

    def get_errors(self):
        return self.errors

    def change_inputs(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = torch.nn.Linear(self.n_inputs, self.n_outputs)

    def pruning_step(self, prune_threshold: float):
        for name, param in self.weights.named_parameters():
            param.data = torch.where(torch.abs(param.data) < prune_threshold * self.weight_multiplier, torch.zeros_like(param.data), param.data)

    def get_parameter_amount(self) -> int:
        return self.n_inputs * self.n_outputs + self.n_outputs

    def get_pruned_parameters_amount(self, prune_threshold: float) -> int:
        pruned_parameters = 0
        for name, param in self.weights.named_parameters():
            pruned_parameters += torch.numel(param.data[abs(param.data) < prune_threshold * self.weight_multiplier])

        return pruned_parameters
