from abc import ABC, abstractmethod

import torch

from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.pruning_config import PruningConfig
from data.dataset import Dataset
from evaluation.evaluators.constants import Constants
from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class ConstructiveAlgorithm(ABC):
    """Base Class for all Constructive Algorithms"""
    def __init__(self, algorithm_type: AlgorithmTypes, logger: Logger, pruning_config: PruningConfig):
        self.algorithm_type: AlgorithmTypes = algorithm_type
        self.logger: Logger = logger
        self.pruning_config: PruningConfig = pruning_config

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """performs a forward pass through the network"""
        raise NotImplementedError()

    def construct(self, dataset: Dataset, target_accuracy: float):
        """constructs/trains the neural network using x, y as data until the target_accuracy is reached."""
        x, y = dataset.train_x, dataset.train_y
        is_finished: bool = False
        best_test_accuracy: float = 0.0
        worse_count: int = 0
        max_construction_steps = 50
        n_construction_steps = 0

        splits_percent = (0.8, 0.2)
        n_data_points = len(y)

        n_train_points = int(splits_percent[0] * n_data_points)
        splits_sizes = (n_train_points, n_data_points - n_train_points)
        x_train, x_test = torch.split(x, splits_sizes)
        y_train, y_test = torch.split(y, splits_sizes)

        while not is_finished:
            self.construction_step(x_train, y_train)
            self.pruning_step()
            n_construction_steps += 1

            test_out = self.forward(x_test)

            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(test_out, y_test)
            accuracy = self.calculate_accuracy(test_out, y_test)
            if accuracy > best_test_accuracy:
                best_test_accuracy = accuracy
                worse_count = 0
            else:
                worse_count += 1

            is_finished = accuracy > target_accuracy \
                or worse_count >= Constants.convergence_check_range \
                or n_construction_steps >= max_construction_steps

            print("\n\n---------------------")
            print(f"Step: {n_construction_steps}")
            print(f"Accuracy: {accuracy}")

            dataset.visualize(x, y, logger=self.logger, model=self)
            self.logger.log_scalar(LogTag.ConstructionStep, n_construction_steps)
            self.logger.log_scalar(LogTag.ConstructionLoss, loss)
            self.logger.log_scalar(LogTag.ConstructionAccuracy, accuracy)
            self.logger.log_scalar(LogTag.ConstructionTotalParameters, self.get_parameter_amount())
            self.logger.log_scalar(LogTag.ConstructionPrunedParameters, self.get_pruned_parameters_amount() if self.pruning_config.is_pruning_active else 0)
            self.logger.log_scalar(LogTag.ConstructionTrainableParameters, self.get_trainable_parameter_amount())
            self.logger.log_figure(LogTag.ResultHistoryPlot)

            if n_construction_steps > 70:
                is_finished = True

    def calculate_accuracy(self, test_out: torch.Tensor, y_test: torch.Tensor) -> float:
        prediction = torch.max(test_out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y_test.data.numpy()
        return float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

    @abstractmethod
    def construction_step(self, x: torch.Tensor, y: torch.Tensor):
        """performs on step of network construction using x and y as data"""
        raise NotImplementedError()

    def pruning_step(self):
        """performs on step of network pruning"""
        if not self.pruning_config.is_pruning_active:
            return

        self.specific_pruning_step()

    @abstractmethod
    def specific_pruning_step(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self, dataset: Dataset, target_accuracy: float):
        """Trains and constructs the neural network using X, y as data until the target_accuracy is reached."""
        raise NotImplementedError()

    @torch.no_grad()
    def test(self, dataset: Dataset):
        """Tests the performance of the network on X, y as testing data.
        The dataset_type variable is used to perform specialized logging for certain dataset types"""
        loss_function = torch.nn.CrossEntropyLoss()
        test_out = self.forward(dataset.test_x)
        loss = loss_function(test_out, dataset.test_y)
        accuracy = self.calculate_accuracy(test_out, dataset.test_y)
        print(f"TEST Loss -> {loss}")
        print(f"TEST Accuracy -> {accuracy}")
        self.logger.log_scalar(LogTag.TestLoss, loss)
        self.logger.log_scalar(LogTag.TestAccuracy, accuracy)
        # dataset.visualize(dataset.test_x, dataset.test_y, logger=self.logger, model=self, save=True)

    def uses_pytorch(self):
        """Shows if this algorithm uses pytorch or not.
        Should be overriden by algorithms that don't use it"""
        return True

    @abstractmethod
    def get_parameter_amount(self) -> int:
        """Returns the amount of parameters in the current network"""
        raise NotImplementedError()

    @abstractmethod
    def get_pruned_parameters_amount(self) -> int:
        """Returns the amount of pruned parameters in the current network"""
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_parameter_amount(self) -> int:
        """Returns the amount of parameters in the current network"""
        raise NotImplementedError()
