from tkinter import ttk

from matplotlib import pyplot as plt

from evaluation.logger.log_tags import LogTag
from evaluation.evaluators.gui.graph_canvas import GraphCanvas
from evaluation.evaluators.gui.gui_constants import *


class GraphFrame:
    def __init__(self, parent):
        super().__init__()
        self.notebook = ttk.Notebook(master=parent, width=700)

        self.visualization_graph = GraphCanvas(parent=parent, log_tag=LogTag.VisualizationPlot)
        self.history_graph = GraphCanvas(parent=parent, log_tag=LogTag.ResultHistoryPlot)

        self.notebook.add(self.visualization_graph.frame, text='Visualization')
        self.notebook.add(self.history_graph.frame, text='History')

        self.notebook.pack(side="left", padx=(0, small_padding), pady=(medium_padding, small_padding))

    def set_figure(self, log_tag: LogTag, new_axes: plt.Axes):
        if log_tag == LogTag.VisualizationPlot:
            self.visualization_graph.set_figure(new_axes)
        elif log_tag == LogTag.ResultHistoryPlot:
            self.history_graph.set_figure(new_axes)
