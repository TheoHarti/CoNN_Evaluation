import json

from algorithms.util.algorithm_types import AlgorithmTypes
from algorithms.util.pruning_config import PruningConfig
from data.util.dataset_types import DatasetTypes
from evaluation.evaluator import Evaluator
from evaluation.logger.logger_types import LoggerTypes


class ConsoleEvaluator:
    """Evaluator class to read a config file from the command line and run the specified evaluations automatically"""
    def __init__(self, config_file_path: str):
        self.config = self.parse_config(config_file_path=config_file_path)
        self.run()

    def parse_config(self, config_file_path):
        with open(config_file_path) as file:
            return json.load(file)

    def run(self):
        datasets = self.config['datasets']
        random_numbers = self.config['random_numbers']

        for algorithm in self.config['algorithms']:
            algorithm_type = AlgorithmTypes[algorithm['algorithm_type']]
            hyperparameter_combinations = algorithm['hyperparameter_combinations']
            pruning_modes = algorithm['pruning_modes']

            for hyperparameters in hyperparameter_combinations:
                for dataset in datasets:
                    for pruning_mode in pruning_modes:
                        for random_number in random_numbers:

                            evaluator = Evaluator(
                                algorithm_type=algorithm_type,
                                dataset_type=DatasetTypes[dataset['dataset_type']],
                                random_element=random_number,
                                target_accuracy=dataset['target_accuracy'],
                                logger_type=LoggerTypes.Csv,
                                pruning_config=PruningConfig(**pruning_mode),
                                hyperparameters=hyperparameters,
                            )
                            evaluator.evaluate()
