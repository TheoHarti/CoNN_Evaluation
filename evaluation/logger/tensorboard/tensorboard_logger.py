from typing import Any, Dict

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class TensorboardLogger(Logger):
    """Logger implementation that logs results using TensorBoard"""
    def __init__(self, logger_parameters: Dict[str, str], ui_reference=None):
        super().__init__(ui_reference=ui_reference)
        self.summary_writer = SummaryWriter(log_dir=logger_parameters['log_directory'])

    def log_scalar(self, log_tag: LogTag, scalar: Any, step: int = None):
        Logger.log_ui_scalar(self, log_tag=log_tag, scalar=scalar)
        self.summary_writer.add_scalar(tag=log_tag.name, scalar_value=scalar, global_step=step)

    def log_figure(self, log_tag: LogTag, figure: plt.Figure = None):
        if log_tag == LogTag.VisualizationPlot:
            Logger.log_ui_figure(self, log_tag=log_tag, figure=figure)

    def finalize(self):
        self.summary_writer.flush()
        self.summary_writer.close()