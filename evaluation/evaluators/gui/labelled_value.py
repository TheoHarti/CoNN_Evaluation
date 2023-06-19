import tkinter as tk
from tkinter import ttk

from evaluation.evaluators.gui.gui_constants import *


class LabelledValue:
    def __init__(self, parent, label):
        super().__init__()
        frame = ttk.Frame(master=parent, padding=small_padding, width=44)
        frame.pack(fill='x')

        ttk.Label(master=frame, text=label, width=30).grid(row=1, column=1, sticky='E')
        self.value = tk.StringVar(value='0')
        ttk.Label(master=frame, textvariable=self.value, width=14).grid(row=1, column=2, sticky='W')

    def set_value(self, value: str):
        self.value.set(value)
