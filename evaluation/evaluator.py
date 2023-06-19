from datetime import datetime

import numpy as np
import torch

from sklearn import datasets

from algorithms.const_deep_net.const_deep_net import ConstDeepNet
from algorithms.cascade_correlation.cascade_correlation_network import CascadeCorrelationNetwork
from algorithms.ccg_dlnn.ccg_dlnn import CCG_DLNN
from algorithms.common.constructive_algorithm import ConstructiveAlgorithm
from algorithms.layerwise.layerwise_network import LayerwiseNetwork
from algorithms.uncertainty_splitting.uncertainty_splitting_network import UncertaintySplittingNetwork
from algorithms.cascor_ultra.cascor_ultra_network import CasCorUltraNetwork
from algorithms.util.input_output_config import InputOutputConfig
from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.pruning_config import PruningConfig
from data.dataset import Dataset
from data.datasets.corner_data import corner_data
from data.datasets.spheres_data import moons_data
from data.datasets.vertical_data import vertical_data
from data.util.dataset_types import DatasetTypes
from data.datasets.spiral_data import spiral_data
from evaluation.logger.console.console_logger import ConsoleLogger
from evaluation.logger.csv.csv_logger import CsvLogger
from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger
from evaluation.logger.tensorboard.tensorboard_logger import TensorboardLogger
from evaluation.logger.logger_types import LoggerTypes


class Evaluator:
    """Main evaluator class that uses a specific evaluation configuration, sets up the needed objects, runs the evaluations, and finalizes the results"""
    def __init__(self,
                 algorithm_type: AlgorithmTypes,
                 dataset_type: DatasetTypes,
                 random_element: int,
                 target_accuracy: float,
                 logger_type: LoggerTypes,
                 pruning_config: PruningConfig,
                 hyperparameters: dict,
                 ui_reference=None):
        self.set_random_seed(random_element)
        self.algorithm_type = algorithm_type
        self.random_element = random_element
        self.target_accuracy = target_accuracy
        self.ui_reference = ui_reference
        self.pruning_config: PruningConfig = pruning_config
        self.hyperparameters = hyperparameters
        self.dataset: Dataset = self.load_dataset(dataset_type)
        self.logger = self.create_logger(logger_type)
        self.constructive_network = self.create_algorithm()

    def evaluate(self):
        """evaluates a specific algorithm for a dataset"""
        print(f"Evaluation -> Algorithm: {self.algorithm_type.name}, Dataset: {self.dataset.dataset_type.name}, Params: {self.hyperparameters}, Pruning: {self.pruning_config.is_pruning_active}, Seed: {self.random_element}")
        if self.constructive_network.uses_pytorch():
            self.constructive_network.construct(self.dataset, self.target_accuracy)
        else:
            self.constructive_network.train(self.dataset, self.target_accuracy)
        self.constructive_network.test(self.dataset)
        self.logger.finalize()

    def create_algorithm(self) -> ConstructiveAlgorithm:
        """creates the algorithm for the specified algorithm type"""
        n_inputs: int = self.dataset.get_amount_inputs()
        n_outputs: int = self.dataset.get_amount_outputs()
        in_out_config: InputOutputConfig = InputOutputConfig(n_inputs=n_inputs, n_outputs=n_outputs)

        if self.algorithm_type == AlgorithmTypes.CasCor:
            return CascadeCorrelationNetwork(algorithm_type=self.algorithm_type, n_inputs=n_inputs, n_outputs=n_outputs, logger=self.logger, hyperparameters=self.hyperparameters, pruning_config=self.pruning_config)
        elif self.algorithm_type == AlgorithmTypes.CasCorUltra:
            return CasCorUltraNetwork(algorithm_type=self.algorithm_type, n_inputs=n_inputs, n_outputs=n_outputs, logger=self.logger, hyperparameters=self.hyperparameters, pruning_config=self.pruning_config)
        elif self.algorithm_type == AlgorithmTypes.Layerwise:
            return LayerwiseNetwork(algorithm_type=self.algorithm_type, logger=self.logger, in_out_config=in_out_config, hyperparameters=self.hyperparameters, pruning_config=self.pruning_config)
        elif self.algorithm_type == AlgorithmTypes.UncertaintySplitting:
            return UncertaintySplittingNetwork(algorithm_type=self.algorithm_type, logger=self.logger, in_out_config=in_out_config, hyperparameters=self.hyperparameters, pruning_config=self.pruning_config)
        elif self.algorithm_type == AlgorithmTypes.ConstDeepNet:
            return ConstDeepNet(algorithm_type=self.algorithm_type, logger=self.logger, in_out_config=in_out_config, hyperparameters=self.hyperparameters, pruning_config=self.pruning_config)
        elif self.algorithm_type == AlgorithmTypes.CCG_DLNN:
            return CCG_DLNN(algorithm_type=self.algorithm_type, logger=self.logger, in_out_config=in_out_config, hyperparameters=self.hyperparameters, pruning_config=self.pruning_config)

    def create_logger(self, logger_type: LoggerTypes) -> Logger:
        logger_parameters = {
            LogTag.DateTime.name: datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            LogTag.AlgorithmType.name: self.algorithm_type.name,
            LogTag.DatasetType.name: self.dataset.get_dataset_name(),
            LogTag.RandomElement.name: self.random_element,
            LogTag.TargetAccuracy.name: self.target_accuracy,
            LogTag.SampleAmount.name: self.dataset.get_sample_amount(),
            LogTag.Hyperparameters.name: self.hyperparameters,
            LogTag.PruningActive.name: self.pruning_config.is_pruning_active,
        }
        if logger_type == LoggerTypes.Console:
            return ConsoleLogger(ui_reference=self.ui_reference)
        elif logger_type == LoggerTypes.Tensorboard:
            logger_parameters['log_directory'] = f'logs/{self.dataset.dataset_type.name}/{self.algorithm_type.name}/{str(self.hyperparameters.get("learning_rate"))}/{str(self.hyperparameters.get("n_candidates"))}/{datetime.now().strftime("%d-%m-%y_%H-%M")}'
            return TensorboardLogger(logger_parameters=logger_parameters, ui_reference=self.ui_reference)
        elif logger_type == LoggerTypes.Csv:
            logger_parameters['log_directory'] = f'../../../logs'
            return CsvLogger(logger_parameters=logger_parameters, ui_reference=self.ui_reference)

    def load_dataset(self, dataset_type: DatasetTypes) -> Dataset:
        """loads a specific dataset depending on the DatasetTypes enum value"""
        x, y = None, None
        n_points = 3000
        is_classification = True
        if dataset_type == DatasetTypes.Corner:
            x, y = corner_data()
        elif dataset_type == DatasetTypes.Vertical:
            n_classes = 3
            x, y = vertical_data(n_samples=int(n_points / n_classes), n_classes=n_classes)
        elif dataset_type == DatasetTypes.Spirals2:
            n_spirals = 2
            x, y = spiral_data(n_points=int(n_points / n_spirals), n_classes=n_spirals)
        elif dataset_type == DatasetTypes.Spirals3:
            n_spirals = 3
            x, y = spiral_data(n_points=int(n_points / n_spirals), n_classes=n_spirals)
        elif dataset_type == DatasetTypes.Spirals4:
            n_spirals = 4
            x, y = spiral_data(n_points=int(n_points / n_spirals), n_classes=n_spirals)
        elif dataset_type == DatasetTypes.Curves:
            x, y = datasets.make_moons(n_samples=n_points, noise=0.05, random_state=self.random_element)
        elif dataset_type == DatasetTypes.Spheres:
            x, y = moons_data(n_samples=n_points, n_dimensions=4, n_clusters=3)
        elif dataset_type == DatasetTypes.Compound:
            x, y = datasets.make_classification(n_samples=n_points, n_features=6, n_informative=5, n_redundant=1, n_classes=4, n_clusters_per_class=2, class_sep=1.5, random_state=self.random_element)
        elif dataset_type == DatasetTypes.Wine:
            data = datasets.load_wine()
            x, y = data.data, data.target
        elif dataset_type == DatasetTypes.Digits:
            data = datasets.load_digits()
            x, y = data.data, data.target
        elif dataset_type == DatasetTypes.OlivettiFaces:
            data = datasets.fetch_olivetti_faces()
            x, y = data.data, data.target
        elif dataset_type == DatasetTypes.BreastCancer:
            data = datasets.load_breast_cancer()
            x, y = data.data, data.target

        return Dataset(dataset_type=dataset_type, is_classification=is_classification, x=x, y=y)

    def set_random_seed(self, seed: int):
        """sets as seed for various random elements to make results reproducible"""
        np.random.seed(seed)
        torch.manual_seed(seed)
