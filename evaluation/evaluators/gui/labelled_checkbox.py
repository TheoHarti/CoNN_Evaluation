import tkinter as tk
from tkinter import ttk

from evaluation.evaluators.gui.gui_constants import *


class LabelledCheckbox:
    def __init__(self, parent, label, default_value: bool):
        super().__init__()
        self.value: tk.BooleanVar = tk.BooleanVar(value=default_value)
        frame = ttk.Frame(master=parent, padding=small_padding, width=44)
        frame.pack(fill='x')

        ttk.Label(master=frame, text=label, width=22).grid(row=1, column=1, sticky='E')
        self.checkbox = ttk.Checkbutton(master=frame, width=16, variable=self.value)
        self.checkbox.grid(row=1, column=2, sticky='W')

    def get_value(self) -> bool:
        return self.value.get()

    def enable(self):
        self.checkbox.configure(state=tk.NORMAL)

    def disable(self):
        self.checkbox.configure(state=tk.DISABLED)
