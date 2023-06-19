import tkinter as tk
from tkinter import ttk

from evaluation.evaluators.gui.gui_constants import *


class LabelledDropdown:
    def __init__(self, parent, label, options_enum, selected_option):
        super().__init__()
        frame = ttk.Frame(master=parent, padding=small_padding, width=44)
        frame.pack(fill='x')
        ttk.Label(master=frame, text=label, width=22).grid(row=1, column=1, sticky='E')
        self.selected_value = tk.StringVar(value=selected_option)
        self.option_menu = ttk.OptionMenu(frame, self.selected_value, selected_option.name, *[option.name for option in options_enum])
        self.option_menu.config(width=22)
        self.option_menu.grid(row=1, column=2, sticky='W')

    def get_selected_value(self):
        return self.selected_value.get()

    def enable(self):
        self.option_menu.configure(state=tk.NORMAL)

    def disable(self):
        self.option_menu.configure(state=tk.DISABLED)
