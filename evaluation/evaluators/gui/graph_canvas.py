import tkinter as tk
from tkinter import ttk

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from evaluation.logger.log_tags import LogTag
from evaluation.evaluators.gui.gui_constants import small_padding


class GraphCanvas:
    def __init__(self, parent, log_tag: LogTag):
        super().__init__()
        self.log_tag = log_tag
        self.frame = ttk.Frame(master=parent, width=700, padding=small_padding)

        self.figure = plt.figure(figsize=(5, 5), dpi=100)
        self.axes: Axes = self.figure.add_subplot(111)
        self.chart = FigureCanvasTkAgg(self.figure, self.frame)
        self.chart.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        self.chart.draw()

    def set_figure(self, new_axes: plt.Axes):
        plt.close(self.figure)
        self.axes.remove()
        new_axes.remove()
        new_axes.figure = self.figure
        self.figure.axes.append(new_axes)
        self.figure.add_axes(new_axes)
        self.axes = new_axes
        self.chart.draw()