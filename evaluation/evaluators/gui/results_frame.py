import queue
from tkinter import ttk

from evaluation.evaluators.gui.labelled_value import LabelledValue
from evaluation.evaluators.gui.gui_constants import *


class ResultsFrame:
    def __init__(self, parent):
        self.event_queue = queue.Queue
        results_frame = ttk.LabelFrame(master=parent, text="Results", padding=small_padding)
        results_frame.pack(side="top", fill="both", expand=True, padx=small_padding, pady=small_padding)

        self.construction_step = LabelledValue(parent=results_frame, label="Construction Step:")
        ttk.Separator(results_frame, orient='horizontal').pack(side="top", fill='x')
        self.parameter_amount = LabelledValue(parent=results_frame, label="Parameter Amount:")
        self.loss = LabelledValue(parent=results_frame, label="Loss:")
        self.accuracy = LabelledValue(parent=results_frame, label="Accuracy:")

    def update_construction_step(self, construction_step: int):
        self.construction_step.set_value(str(construction_step))

    def update_loss(self, loss: float):
        self.loss.set_value(f'{loss:.2f}')

    def update_accuracy(self, accuracy: float):
        self.accuracy.set_value(f'{accuracy:.2f}')

    def update_parameter_amount(self, n_parameters: int):
        self.parameter_amount.set_value(str(n_parameters))