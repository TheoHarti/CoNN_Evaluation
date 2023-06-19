import torch
import torch.nn.functional

from algorithms.util.convergence_checker import has_converged
from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class ConstDeepNetLayer(torch.nn.Module):
    def __init__(self, logger: Logger, n_inputs: int, n_outputs: int, max_layer_size: int, learning_rate: float):
        super(ConstDeepNetLayer, self).__init__()
        self.logger = logger
        self.max_layer_size = max_layer_size
        self.learning_rate = learning_rate
        self.is_frozen = False
        self.n_inputs, self.n_hidden_nodes, self.n_outputs = n_inputs, 1, n_outputs
        self.hidden = torch.nn.Linear(self.n_inputs, self.n_hidden_nodes)
        self.out = torch.nn.Linear(self.n_hidden_nodes, self.n_outputs)
        self.output_layer_active: bool = True

    def forward(self, x):
        hidden_layer_out = torch.nn.functional.relu(self.hidden(x))  # csv_test if F.relu is needed
        return self.out(hidden_layer_out) if self.output_layer_active else hidden_layer_out

    def expand_layer(self):
        self.n_hidden_nodes += 1
        self.hidden = torch.nn.Linear(self.n_inputs, self.n_hidden_nodes)
        self.out = torch.nn.Linear(self.n_hidden_nodes, self.n_outputs)

    def freeze_weights(self):
        self.is_frozen = True
        for param in self.hidden.parameters():
            param.requires_grad = False
        for param in self.out.parameters():
            param.requires_grad = False

    def train_layer(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        loss_function = torch.nn.CrossEntropyLoss()
        convergence_check_range = 25
        losses = []
        epoch = 0
        is_finished = False

        while not is_finished:
            out = self.forward(x)
            loss = loss_function(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch += 1
            if epoch % 25 == 0:
                is_finished = has_converged(losses, convergence_check_range)
                print('Epoch: ' + str(epoch))

        self.logger.log_scalar(log_tag=LogTag.ConstructionStepEpochs, scalar=epoch)

    def remove_output_layer(self):
        self.output_layer_active = False

    def get_parameter_amount(self) -> int:
        parameters = self.n_inputs * self.n_hidden_nodes + self.n_hidden_nodes
        if self.output_layer_active:
            parameters += self.n_hidden_nodes * self.n_outputs + self.n_outputs
        return parameters

    def get_pruned_parameters_amount(self, prune_threshold: float) -> int:
        pruned_parameters = 0

        for name, param in self.hidden.named_parameters():
            pruned_parameters += torch.numel(param.data[abs(param.data) < prune_threshold])

        if self.output_layer_active:
            for name, param in self.out.named_parameters():
                pruned_parameters += torch.numel(param.data[abs(param.data) < prune_threshold])

        return pruned_parameters
