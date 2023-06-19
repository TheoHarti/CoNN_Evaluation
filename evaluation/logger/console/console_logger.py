from datetime import date
from typing import Any, Dict

from matplotlib import pyplot as plt

from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class ConsoleLogger(Logger):
    """Logger implementation that logs results to the console"""
    def __init__(self, ui_reference=None):
        super().__init__(ui_reference=ui_reference)

    def log_scalar(self, log_tag: LogTag, scalar: Any, step: int = None):
        Logger.log_ui_scalar(self, log_tag=log_tag, scalar=scalar)
        print(f"{date.today().strftime('%d.%m.%Y %H:%M:%S')}  |  TAG: {log_tag}  |  STEP: {step}  |  VALUE: {scalar}")

    def log_figure(self, log_tag: LogTag, figure: plt.Figure = None):
        if log_tag == LogTag.VisualizationPlot:
            Logger.log_ui_figure(self, log_tag=log_tag, figure=figure)

    def finalize(self):
        pass
