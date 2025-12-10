from itertools import cycle
from typing import Iterable

from matplotlib.colors import TABLEAU_COLORS
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def _make_two_level_ticks(
    hierarchical_index: pd.MultiIndex,
) -> tuple[list[str], list[str]]:
    """Make ticks for two-level MultiIndex."""
    assert hierarchical_index.nlevels < 3
    if hierarchical_index.nlevels == 1:
        return hierarchical_index, None

    repeated_colours = cycle(TABLEAU_COLORS)
    groups = hierarchical_index.levels[0]
    colour_of_group = dict(zip(groups, repeated_colours))
    tick_labels = map(lambda x: ": ".join(x[-2:]), hierarchical_index)
    tick_colours = [colour_of_group[g] for g in hierarchical_index.get_level_values(0)]
    return list(tick_labels), tick_colours


def _set_coloured_xticks(
    ticks: Iterable,
    tick_colours: Iterable | None,
    axes: plt.Axes | None = None,
    **kwargs,
):
    """Color x-axis `ticks` with `tick_colours`."""
    ax = axes if axes is not None else plt.gca()
    range_x = np.arange(len(ticks))
    ax.set_xticks(range_x)
    ax.set_xticklabels(ticks, **kwargs)
    if tick_colours is not None:
        for xtick, color in zip(ax.get_xticklabels(), tick_colours):
            xtick.set_color(color)

        for tick, color in zip(ax.xaxis.get_major_ticks(), tick_colours):
            tick.tick1line.set_markeredgecolor(color)  # Color the left line
            tick.tick2line.set_markeredgecolor(color)  # Color the right line


def _set_coloured_yticks(
    ticks: Iterable,
    tick_colours: Iterable | None,
    axes: plt.Axes | None = None,
    **kwargs,
):
    """Color y-axis `ticks` with `tick_colours`."""
    ax = axes if axes is not None else plt.gca()
    range_y = np.arange(len(ticks))
    ax.set_yticks(range_y)
    ax.set_yticklabels(ticks, **kwargs)
    if tick_colours is not None:
        for ytick, color in zip(ax.get_yticklabels(), tick_colours):
            ytick.set_color(color)

        for tick, color in zip(ax.yaxis.get_major_ticks(), tick_colours):
            tick.tick1line.set_markeredgecolor(color)  # Color the left line
            tick.tick2line.set_markeredgecolor(color)  # Color the right line
