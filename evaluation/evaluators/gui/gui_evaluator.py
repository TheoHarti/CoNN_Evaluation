import threading

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import sv_ttk

from algorithms.util.pruning_config import PruningConfig
from evaluation.evaluator import Evaluator
from evaluation.evaluators.gui.graph_frame import GraphFrame
from evaluation.evaluators.gui.results_frame import ResultsFrame
from evaluation.evaluators.gui.settings_frame import SettingsFrame


class GuiEvaluator(tk.Tk):
    """Evaluator class with a GUI to manually specify evaluation runs"""
    def __init__(self):
        super().__init__()
        self.is_running: bool = False
        plt.switch_backend('agg')

        self.title('Constructive Neural Network Algorithms')
        self.geometry('900x580')
        self.minsize(900, 580)
        self.maxsize(900, 580)

        sv_ttk.set_theme("dark")  # Set the initial theme

        left_frame = ttk.Frame(master=self, width=300)
        left_frame.pack(side="left", fill="both", expand=True)

        self.settings_frame = SettingsFrame(parent=left_frame, on_run_pressed=self.on_run_clicked)
        self.results_frame = ResultsFrame(parent=left_frame)
        self.graph_frame = GraphFrame(parent=self)

        self.mainloop()

    def on_run_clicked(self):
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        if self.is_running:
            return

        self.is_running = True
        self.settings_frame.disable_inputs()

        hyperparameters_layerwise = {
            'n_starting_hidden_nodes': 2,
            'n_max_layer_expansions': 4,
        }

        hyperparameters_cascor = {
            'learning_rate': 0.1,
            'n_candidates': 20,
        }

        hyperparameters_cascor_ultra = {
            "n_candidates": 10, "candidate_correlation_threshold": 0.4
        }

        hyperparameters_uncertainty_splitting = {
            'n_starting_hidden_nodes': 5,
            'max_layer_size': 64,
            'n_uncertain_nodes': 5,
            'n_replacement_nodes': 4,
        }

        hyperparameters_ccg_dlnn = {
            "n_candidates": 5,
            "learning_rate_out": 0.005,
            "learning_rate_hidden": 0.05,
            "max_layer_size": 6
        }

        hyperparameters = hyperparameters_ccg_dlnn

        evaluator = Evaluator(
            algorithm_type=self.settings_frame.get_selected_algorithm(),
            dataset_type=self.settings_frame.get_selected_dataset(),
            random_element=self.settings_frame.get_selected_random_number(),
            target_accuracy=self.settings_frame.get_selected_target_accuracy(),
            logger_type=self.settings_frame.get_selected_logger(),
            pruning_config=PruningConfig(is_pruning_active=self.settings_frame.get_use_pruning_step(), magnitude_threshold=0.01),
            ui_reference=self,
            hyperparameters=hyperparameters,
        )
        evaluator.evaluate()

        self.is_running = False
        self.settings_frame.enable_inputs()

