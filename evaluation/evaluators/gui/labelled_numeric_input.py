import tkinter as tk
from tkinter import ttk

from evaluation.evaluators.gui.gui_constants import *


class LabelledNumericInput:
    def __init__(self, parent, label, value_range: tuple, default_value: int):
        super().__init__()
        self.value_range = value_range
        frame = ttk.Frame(master=parent, padding=small_padding, width=44)
        frame.pack(fill='x')

        ttk.Label(master=frame, text=label, width=22).grid(row=1, column=1, sticky='E')
        range_validation = parent.register(self._validate)
        self.input = ttk.Spinbox(master=frame, from_=value_range[0], to=value_range[1], width=16, validate="key", validatecommand=(range_validation, "%P"))
        self.input.set(default_value)
        self.input.grid(row=1, column=2, sticky='W')

    def get_value(self):
        return self.input.get()

    def _validate(self, spinbox_input):
        if not spinbox_input.isdigit():
            return False
        if int(spinbox_input) < self.value_range[0] or int(spinbox_input) > self.value_range[1]:
            return False
        return True

    def enable(self):
        self.input.configure(state=tk.NORMAL)

    def disable(self):
        self.input.configure(state=tk.DISABLED)
