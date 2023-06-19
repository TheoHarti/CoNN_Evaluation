from abc import ABC, abstractmethod
from typing import Any
import matplotlib.pyplot as plt

from evaluation.logger.log_tags import LogTag


class Logger(ABC):
    """Base Class for all Loggers (i.e. everything that handles the data produced by the algorithms)"""
    def __init__(self, ui_reference=None):
        self.ui_reference = ui_reference

    @abstractmethod
    def log_scalar(self, log_tag: LogTag, scalar: Any, step: int = None):
        """Base method to log a scalar for a specific tag and at a training step."""
        pass

    @abstractmethod
    def log_figure(self, log_tag: LogTag, figure: plt.Figure = None):
        """Base method to log a matplotlib.pyplot figure for a specific tag."""
        pass

    @abstractmethod
    def finalize(self):
        """Base method to handle finalization of the logger (e.g. flushing and closing)."""
        pass

    def log_ui_scalar(self, log_tag: LogTag, scalar: Any):
        """Handles the sending of the logged values to the UI"""
        if self.ui_reference is None:
            return

        if log_tag == LogTag.ConstructionStep:
            self.ui_reference.results_frame.update_construction_step(scalar)
        elif log_tag == LogTag.ConstructionLoss:
            self.ui_reference.results_frame.update_loss(scalar)
        elif log_tag == LogTag.ConstructionAccuracy:
            self.ui_reference.results_frame.update_accuracy(scalar)
        elif log_tag == LogTag.ConstructionTotalParameters:
            self.ui_reference.results_frame.update_parameter_amount(scalar)
        elif log_tag == LogTag.TestLoss:
            pass
        elif log_tag == LogTag.TestAccuracy:
            pass

    def log_ui_figure(self, log_tag: LogTag, figure: plt.Figure):
        if self.ui_reference is None:
            return

        self.ui_reference.graph_frame.set_figure(log_tag, figure)