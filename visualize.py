"""Plot the simulation results in three charts."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from battle import SimulationSummary

COLOR_A = "#e4572e"
COLOR_B = "#2e86ab"


def make_figure(sim: SimulationSummary, fig: Figure | None = None,
                model_p: float | None = None) -> Figure:
    """If model_p is given, show the model's win chance for the first Pokemon in the title."""
    fig = fig or Figure(figsize=(12, 4.5))
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3)
    a, b = sim.a.name.title(), sim.b.name.title()

    # how often each Pokemon won
    pct = [100 * sim.wins_a / sim.n, 100 * sim.wins_b / sim.n]
    labels, colors = [a, b], [COLOR_A, COLOR_B]
    if sim.draws:
        pct.append(100 * sim.draws / sim.n)
        labels.append("Draw")
        colors.append("#999999")
    bars = ax1.bar(labels, pct, color=colors)
    ax1.bar_label(bars, fmt="%.1f%%")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Win probability (%)")
    ax1.set_title(f"Outcome of {sim.n:,} simulated battles")

    # how much HP the winners had left
    bins = np.linspace(0, 1, 11)
    if sim.hp_left_a:
        ax2.hist(sim.hp_left_a, bins=bins, alpha=0.7, color=COLOR_A, label=f"{a} wins")
    if sim.hp_left_b:
        ax2.hist(sim.hp_left_b, bins=bins, alpha=0.7, color=COLOR_B, label=f"{b} wins")
    ax2.set_xlabel("Winner's remaining HP (fraction)")
    ax2.set_ylabel("Battles")
    ax2.set_title("How close are the wins?")
    ax2.legend()

    # follow the HP changes from one battle
    turns = range(len(sim.sample_history))
    ax3.plot(turns, [h[0] / sim.a.hp * 100 for h in sim.sample_history],
             color=COLOR_A, marker="o", label=a)
    ax3.plot(turns, [h[1] / sim.b.hp * 100 for h in sim.sample_history],
             color=COLOR_B, marker="o", label=b)
    ax3.set_xlabel("Turn")
    ax3.set_ylabel("HP (%)")
    ax3.set_ylim(-2, 105)
    ax3.set_title("Sample battle")
    ax3.legend()

    if model_p is not None:
        fig.suptitle(f"Trained model: {a} {100 * model_p:.0f}% vs {b} {100 * (1 - model_p):.0f}%"
                     "  (raw simulator in the left panel is overconfident)", fontsize=11)
    fig.tight_layout()
    return fig
