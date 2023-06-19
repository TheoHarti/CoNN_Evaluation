import os.path
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any, Dict
from evaluation.logger.logger import Logger, LogTag


class CsvLogger(Logger):
    """Logger implementation that logs results to a CSV file"""
    def __init__(self, logger_parameters: Dict[str, str], ui_reference=None):
        super().__init__(ui_reference=ui_reference)
        self.logger_parameters = logger_parameters
        self.csv_file_name: str = os.path.join(pathlib.Path().resolve(), logger_parameters['log_directory'], 'Log.csv')

        init_log_tags = [LogTag.DateTime, LogTag.AlgorithmType, LogTag.DatasetType,
                         LogTag.RandomElement, LogTag.TargetAccuracy, LogTag.PruningActive,
                         LogTag.SampleAmount, LogTag.Hyperparameters]

        run_log_tags = [LogTag.ConstructionStep, LogTag.ConstructionLoss,
                        LogTag.ConstructionAccuracy, LogTag.ConstructionTotalParameters,
                        LogTag.ConstructionPrunedParameters, LogTag.ConstructionTrainableParameters,
                        LogTag.ConstructionTime, LogTag.ConstructionStepEpochs, LogTag.TestLoss, LogTag.TestAccuracy]

        dataframe_log_tags = [log_tag.name for log_tag in init_log_tags + run_log_tags]
        self.data_frame = pd.DataFrame(columns=dataframe_log_tags)
        self.data_frame.loc[0] = [None] * len(self.data_frame.columns)
        for init_log_tag in init_log_tags:
            self.data_frame[init_log_tag.name][0] = logger_parameters[init_log_tag.name]

    def log_scalar(self, log_tag: LogTag, scalar: Any, step: int = None):
        Logger.log_ui_scalar(self, log_tag=log_tag, scalar=scalar)
        if log_tag == LogTag.ConstructionLoss \
                or log_tag == LogTag.ConstructionAccuracy \
                or log_tag == LogTag.ConstructionTotalParameters \
                or log_tag == LogTag.ConstructionPrunedParameters \
                or log_tag == LogTag.ConstructionTrainableParameters \
                or log_tag == LogTag.ConstructionStep \
                or log_tag == LogTag.ConstructionStepEpochs:
            if type(self.data_frame[log_tag.name][0]) is str:
                self.data_frame[log_tag.name][0] += f', {scalar}'
            else:
                self.data_frame[log_tag.name][0] = f'{scalar}'

        elif log_tag == LogTag.TestLoss \
                or log_tag == LogTag.TestAccuracy:
            self.data_frame[log_tag.name][0] = f'{scalar}'

    def log_figure(self, log_tag: LogTag, figure: plt.Figure = None):
        if log_tag == LogTag.ResultHistoryPlot:
            if self.data_frame[LogTag.ConstructionAccuracy.name][0] is not None:
                accuracy_values = [0.0] + [float(accuracy) for accuracy in self.data_frame[LogTag.ConstructionAccuracy.name][0].split(',')]
                construction_steps = [0] + [int(step) for step in self.data_frame[LogTag.ConstructionStep.name][0].split(',')]
                fig = plt.figure(figsize=(5, 5), dpi=100)
                ax = fig.add_subplot(111)
                ax.set_title('accuracy at each construction step')
                ax.set_xlabel('construction step')
                ax.set_ylabel('accuracy')
                ax.set_xticks(construction_steps)
                ax.set_yticks(np.arange(0, 1, 0.1).tolist())
                ax.plot(construction_steps, accuracy_values, '--bo', label='accuracy per construction step')
                Logger.log_ui_figure(self, log_tag=log_tag, figure=ax)
                plt.close(fig)
        else:
            Logger.log_ui_figure(self, log_tag=log_tag, figure=figure)

    def finalize(self):
        is_new_file = not os.path.isfile(self.csv_file_name)
        if not os.path.exists(self.csv_file_name):
            os.makedirs(self.logger_parameters['log_directory'], exist_ok=True)

        self.data_frame.to_csv(path_or_buf=self.csv_file_name, sep=";", mode='a', index=False, header=is_new_file)
