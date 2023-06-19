import tkinter as tk
from tkinter import ttk

from algorithms.util.algorithm_types import AlgorithmTypes
from data.util.dataset_types import DatasetTypes
from evaluation.logger.logger_types import LoggerTypes
from evaluation.evaluators.gui.labelled_checkbox import LabelledCheckbox
from evaluation.evaluators.gui.labelled_dropdown import LabelledDropdown
from evaluation.evaluators.gui.labelled_numeric_input import LabelledNumericInput
from evaluation.evaluators.gui.gui_constants import *


class SettingsFrame:
    def __init__(self, parent, on_run_pressed):
        super().__init__()
        settings_frame = ttk.LabelFrame(master=parent, text="Settings", padding=small_padding)
        settings_frame.pack(side="top", fill="both", expand=True, padx=small_padding, pady=small_padding)

        self.algorithm_dropdown = LabelledDropdown(parent=settings_frame, label="Algorithm:", options_enum=AlgorithmTypes, selected_option=AlgorithmTypes.CasCor)
        self.dataset_dropdown = LabelledDropdown(parent=settings_frame, label="Dataset:", options_enum=DatasetTypes, selected_option=DatasetTypes.Spirals3)
        self.logger_dropdown = LabelledDropdown(parent=settings_frame, label="Logger:", options_enum=LoggerTypes, selected_option=LoggerTypes.Csv)
        self.random_number_input = LabelledNumericInput(parent=settings_frame, label="Random Element:", value_range=(0, 10000), default_value=1)
        self.target_accuracy_input = LabelledNumericInput(parent=settings_frame, label="Target Accuracy [%]:", value_range=(0, 100), default_value=80)
        self.use_pruning_step_checkbox = LabelledCheckbox(parent=settings_frame, label="Use Pruning Step:", default_value=False)

        self.run_button = ttk.Button(master=settings_frame, text="RUN", command=on_run_pressed)
        self.run_button.pack(fill="both", pady=(small_padding, small_padding), padx=small_padding)

    def get_selected_algorithm(self):
        return AlgorithmTypes[self.algorithm_dropdown.get_selected_value()]

    def get_selected_dataset(self):
        return DatasetTypes[self.dataset_dropdown.get_selected_value()]

    def get_selected_logger(self):
        return LoggerTypes[self.logger_dropdown.get_selected_value()]

    def get_selected_random_number(self):
        return int(self.random_number_input.get_value())

    def get_selected_target_accuracy(self):
        return int(self.target_accuracy_input.get_value()) / 100.0

    def get_use_pruning_step(self) -> bool:
        return self.use_pruning_step_checkbox.get_value()

    def enable_inputs(self):
        self.run_button["state"] = tk.NORMAL
        self.algorithm_dropdown.enable()
        self.dataset_dropdown.enable()
        self.logger_dropdown.enable()
        self.random_number_input.enable()
        self.target_accuracy_input.enable()
        self.use_pruning_step_checkbox.enable()

    def disable_inputs(self):
        self.run_button["state"] = tk.DISABLED
        self.algorithm_dropdown.disable()
        self.dataset_dropdown.disable()
        self.logger_dropdown.disable()
        self.random_number_input.disable()
        self.target_accuracy_input.disable()
        self.use_pruning_step_checkbox.disable()
