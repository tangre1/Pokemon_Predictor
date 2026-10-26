"""Pokemon Battle Outcome Visualizer.

GUI:  python main.py
CLI:  python main.py pikachu charizard      (opens a matplotlib window)
"""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import data
from battle import run_simulations
from visualize import make_figure

N_SIMS = 2000


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pokemon Battle Visualizer")
        self.geometry("1200x600")

        try:
            names = data.get_pokemon_names()
        except Exception as exc:  # network problems, etc.
            messagebox.showerror("Could not reach PokeAPI", str(exc))
            names = []

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        self.box_a = ttk.Combobox(top, values=names, width=22)
        self.box_b = ttk.Combobox(top, values=names, width=22)
        self.box_a.set("pikachu")
        self.box_b.set("charizard")
        ttk.Label(top, text="Pokemon 1:").pack(side="left")
        self.box_a.pack(side="left", padx=6)
        ttk.Label(top, text="vs   Pokemon 2:").pack(side="left")
        self.box_b.pack(side="left", padx=6)
        ttk.Button(top, text="Simulate", command=self.simulate).pack(side="left", padx=10)
        self.status = ttk.Label(top, text="")
        self.status.pack(side="left", padx=10)

        self.fig = Figure(figsize=(12, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def simulate(self) -> None:
        try:
            self.status.config(text="Loading data...")
            self.update_idletasks()
            a = data.get_pokemon(self.box_a.get())
            b = data.get_pokemon(self.box_b.get())
            chart = data.get_type_chart()
        except Exception as exc:
            self.status.config(text="")
            messagebox.showerror("Error", f"Could not load Pokemon data:\n{exc}")
            return
        sim = run_simulations(a, b, chart, n=N_SIMS)
        make_figure(sim, self.fig)
        self.canvas.draw()
        self.status.config(text="Done")


def cli(name_a: str, name_b: str) -> None:
    import matplotlib.pyplot as plt

    a, b = data.get_pokemon(name_a), data.get_pokemon(name_b)
    sim = run_simulations(a, b, data.get_type_chart(), n=N_SIMS)
    make_figure(sim, plt.figure(figsize=(12, 4.5)))
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        cli(sys.argv[1], sys.argv[2])
    else:
        App().mainloop()
