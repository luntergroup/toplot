"""Visualization of topic model weights with uncertainty estimates.

This module provides functions to visualize posterior samples from topic models. The input
data should be organized as a pandas DataFrame with a two-level column structure:

Level 1 (outer): Represents different multinomial groups (e.g., "demographics", "symptoms")
Level 2 (inner): Categories within each multinomial (e.g., ["male", "female"] for "sex")

Example DataFrame structure:
```python
            bmi                        sex
            underweight   overweight   male   female
sample_1    0.4           0.6          0.3      0.7
sample_2    0.5           0.5          0.2      0.8
...
```

Each row represents one posterior sample, and the values within each multinomial group
should sum to 1.0.
"""

from itertools import cycle
from typing import Literal

from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from toplot.scattermap import scattermap
from toplot.utils import (
    _make_two_level_ticks,
    _set_coloured_xticks,
    _set_coloured_yticks,
)


def _bar(names, sizes, offsets, error_bars, color, transpose: bool = False, ax=None):
    """Like matplotlib `bar` but can be transposed."""
    if ax is None:
        ax = plt.gca()

    if not transpose:
        return ax.bar(names, sizes, bottom=offsets, color=color, yerr=error_bars)

    ax.barh(names, sizes, left=offsets, color=color, xerr=error_bars)
    ax.invert_yaxis()
    return ax


def bar_plot_stacked(
    dataframe,
    quantile_range=(0.025, 0.975),
    height: Literal["mean", "median"] = "mean",
    ax=None,
    labels: bool = True,
    fontsize=None,
    transpose: bool = False,
):
    """Plot posterior of a topic as probability bars by stacking categories per set.

    Ags:
        dataframe: Posterior samples of a single topic organized as a DataFrame with
            two-level columns:
            - Level 1: Multinomial groups (e.g., "bmi", "sex")
            - Level 2: Categories within each group (e.g., ["male", "female"])
            Each row is one posterior sample, and values within each multinomial must
            sum to 1.
        quantile_range: Range of quantiles to plot as error bars.
        height: How to compute the height of the bars.
        ax: Matplotlib axes to plot on.
        labels: If `True`, annotate bars with category labels.
        fontsize: Font size for the category labels.
        transpose: If `True`, swap the x and y axes of the bar plot.

    Example:
        ```python
        from numpy.random import dirichlet
        import pandas as pd

        weights_bmi = dirichlet([16.0, 32.0, 32.0], size=1_000)
        weights_sex = dirichlet([8.1, 4.1], size=1_000)
        weights = pd.concat(
            {
                "BMI": pd.DataFrame(weights_bmi, columns=["Underweight", "Healthy Weight", "Overweight"]),
                "sex": pd.DataFrame(weights_sex, columns=["Male", "Female"]),
            },
            axis="columns",
        )
        bar_plot_stacked(weights)
        ```

    Returns: Reference to the axes.
    """
    if ax is None:
        ax = plt.gca()

    cmap = plt.get_cmap("PiYG")

    if dataframe.columns.nlevels < 2:
        raise ValueError("Dataframe must have two-level columns.")

    if len(dataframe.columns.unique()) < len(dataframe.columns):
        raise ValueError("Dataframe column names must be uniquely identifiable.")

    height_fn = np.mean
    if height == "mean":
        height_fn = np.mean
    elif height == "median":
        height_fn = np.median

    # Compute summary statistics of the posterior samples.
    estimate = dataframe.apply(height_fn, axis="rows")
    lower = dataframe.quantile(q=quantile_range[0], axis=0)
    upper = dataframe.quantile(q=quantile_range[1], axis=0)
    # The error bars are the distance from the mean to the quantiles.
    err = pd.concat([estimate - lower, upper - estimate], axis="columns")

    if np.any(err < -1e-6):
        msg = "Negative error bars detected."
        if height == "mean":
            msg += " Consider settings the height to the median."
        raise ValueError(msg)

    # Make a bar per leaf by stacking the categories on top of each other --> the per
    # category bar offsets (relative to y = 0) are the cumulative distribution.
    p_cum = estimate.groupby(level=0).cumsum()

    # For each leaf.
    for feature_name in dataframe.columns.unique(level=0):
        feature_weights = estimate.loc[feature_name]
        feature_err = err.loc[feature_name]
        feature_cum = p_cum.loc[feature_name]
        offsets = np.pad(
            feature_cum, pad_width=(1, 0), mode="constant", constant_values=0
        )

        feature_categories = dataframe[feature_name].columns
        n_categories = len(feature_weights)
        for j, category in enumerate(feature_categories):
            u = 1 - j / (n_categories - 1)
            color = cmap(0.1 + 0.8 * u)
            # Plot error bars for all but the last category.
            err_j = feature_err.loc[category].to_numpy().reshape(2, 1)
            if j == n_categories - 1:
                err_j = None

            _bar(
                feature_name,
                sizes=feature_weights.loc[category],
                offsets=offsets[j],
                error_bars=err_j,
                color=color,
                transpose=transpose,
                ax=ax,
            )
            if labels:
                text_properties = dict(
                    s=category, ha="center", va="center", fontsize=fontsize
                )
                position = offsets[j] + feature_weights.loc[category] / 2
                if not transpose:
                    ax.text(x=feature_name, y=position, **text_properties)
                else:
                    ax.text(x=position, y=feature_name, **text_properties)
    # Rotate the x-axis labels.
    if not transpose:
        ax.tick_params(axis="x", labelrotation=90, labelsize=fontsize)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Probability")
    else:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlabel("Probability")
    return ax


def bar_plot(
    dataframe: pd.DataFrame,
    quantile_range=(0.025, 0.975),
    height: Literal["mean", "median"] = "mean",
    label=None,
    ax=None,
    color_xlabels: bool = False,
    legend: bool = False,
):
    """Plot posterior of a topic weight as an unfolded array of probability bars.

    Args:
        dataframe: Posterior samples of a single topic organized as a DataFrame with
            two-level columns.
            - Level 1: Multinomial groups (e.g., "bmi", "sex")
            - Level 2: Categories within each group (e.g., ["male", "female"])
            Each row is one posterior sample, and values within each multinomial must
            sum to 1.
        quantile_range: Range of quantiles to plot as error bars.
        height: How to compute the height of the bars.
        label: A legend label for the plot.
        ax: Matplotlib axes to plot on.
        color_xlabels: If `True`, pair the colours of the x-axis labels with the bars.
        legend: If `True`, show a legend that explains the bar colors.

    Example:
        ```python
        from numpy.random import dirichlet
        import pandas as pd

        weights_bmi = dirichlet([16.0, 32.0, 32.0], size=1_000)
        weights_sex = dirichlet([8.1, 4.1], size=1_000)
        weights = pd.concat(
            {
                "BMI": pd.DataFrame(weights_bmi, columns=["Underweight", "Healthy Weight", "Overweight"]),
                "sex": pd.DataFrame(weights_sex, columns=["Male", "Female"]),
            },
            axis="columns",
        )
        bar_plot(weights)
        ```

    Returns:
        Reference to matplotlib bar axes.
    """
    if ax is None:
        ax = plt.gca()

    # TODO: Also allow for a single-level column (i.e., single multinomial) dataframe.
    if dataframe.columns.nlevels != 2:
        raise ValueError(
            "Dataframe must have two column levels: multinomial and category."
        )

    height_fn = np.mean
    if height == "mean":
        height_fn = np.mean
    elif height == "median":
        height_fn = np.median

    # Compute summary statistics of distribution.
    estimate = dataframe.apply(height_fn, axis="rows")
    lower = dataframe.apply(np.quantile, q=quantile_range[0], axis="rows")
    upper = dataframe.apply(np.quantile, q=quantile_range[1], axis="rows")
    err = np.stack([estimate - lower, upper - estimate], axis=0)

    assert np.all(
        err >= 0
    ), "(some) error values are negative, this might be because of the use of mean as height measure, using median solves this."

    if np.any(err < -1e-6):
        msg = "Negative error bars detected."
        if height == "mean":
            msg += " Consider settings the height to the median."
        raise ValueError(msg)

    # Give each category set (=first column level) a different colour.
    groups = dataframe.columns.levels[0]
    feature_names, colours = _make_two_level_ticks(dataframe.columns)

    ax.bar(feature_names, height=estimate, yerr=err, label=label, color=colours)
    ax.set_ylabel("Probability")
    ax.tick_params(axis="x", labelrotation=90)

    if color_xlabels:
        # Set the color of the x-tick labels to match the corresponding bar color.
        _set_coloured_xticks(feature_names, colours, ax)

    # Add a legend indicating what the colours mean.
    if legend:
        legend_handles = [
            Patch(color=c, label=gp) for c, gp in zip(cycle(TABLEAU_COLORS), groups)
        ]
        legend_title = "Groups"
        if dataframe.columns.names[0]:
            legend_title = dataframe.columns.names[0]
        legend_title = legend_title.capitalize()

        ax.legend(handles=legend_handles, title=legend_title, bbox_to_anchor=(1.0, 1))

    margin = 0.025
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0 - margin, 1 + margin)
    return ax


def scattermap_plot(
    dataframe,
    dataframe_counts,
    marker_scaler=10,
    scale_val_x_counts=2,
    scale_val_y_counts=2,
    ax=None,
):
    """Plot posterior of a topic weight as an multicolored scattermap plot, with dotsize representing frequency of occurrence, color the value of phi,
    with bars on the axes that show the axes total.

    Args:
        dataframe: dataframe in two levels of dimension [{feature: words}, n_components] containing phi, determines markers and their color
        dataframe_counts: another dataframe of same dimension containing counts (or any other characteristic you want to use), determines markersize and bars at the axes
        marker_scaler: this value scales the size of markers
        scale_val_x_counts: scale bar size on the x-axis
        scale_val_y_counts: scale bar size on the y-axis
        ax: figure axes to use, if none a new figure is created

    Returns:
        Reference to matplotlib bar axes.
    """

    topic_counts = dataframe_counts.sum(axis=1) / dataframe_counts.sum().sum()
    word_counts = dataframe_counts.sum() / dataframe_counts.sum().sum()
    topic_bar_positions = np.arange(start=0.5, stop=len(topic_counts), step=1)
    word_bar_positions = np.arange(start=0.5, stop=len(word_counts), step=1)

    if ax is None:
        ax = plt.gca()

    with sns.plotting_context():
        sns.set_theme(
            style="darkgrid",
            font_scale=1.5,
            rc={
                "axes.facecolor": "#F0E6EB",
                "grid.linestyle": "-",
                "grid.color": "#b0b0b0",
            },
        )
        scattermap(
            data=dataframe.T,
            cmap="YlGnBu",
            marker_size=marker_scaler * dataframe_counts.T,
            vmax=1,
            linecolor="black",
            linewidths=0.2,
            ax=ax,
        )

        # x axis on top
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.tick_params("x", labelrotation=90)

        # Add frequencies of attributes as barplot to y-axis
        ax.barh(
            list(word_bar_positions),
            -scale_val_y_counts * word_counts,
            0.6,
            alpha=1,
            edgecolor="none",
        )
        ax.axvline(x=0, color="k")
        ax.axhline(0, color="k")

        ax.set_xlim(-1, dataframe_counts.shape[0])

        # Add frequencies of diagnosis as barplot to x-axis
        ax.bar(
            topic_bar_positions,
            -scale_val_x_counts * topic_counts,
            0.6,
            color="#41b6c4",
            bottom=0,
            edgecolor="none",
        )

        ax.set_ylim([-1.5, dataframe_counts.shape[1]])
        return ax


def hinton(
    data: pd.DataFrame,
    max_weight: float | None = None,
    ax: plt.Axes | None = None,
    grid: bool = True,
):
    r"""Draw Hinton diagram for visualizing a the size and sign of a weight matrix.

    A red (blue) marker indicates a positive (negative) weight. The size scales as
    $\propto \sqrt{|w|}$.

    Args:
        data: Weights to plot.
        max_weight: The size that corresponds to a full width marker.
        ax: Axes to plot on.
        grid: Whether to draw a grid.
    """
    ax = ax if ax is not None else plt.gca()

    matrix = data.to_numpy().astype(float)
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    range_y = np.arange(data.shape[0])
    range_x = np.arange(data.shape[1])
    x, y = np.meshgrid(range_x, range_y)

    for (i, j), w in np.ndenumerate(matrix):
        color = "tab:red" if w > 0 else "tab:blue"
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle(
            [x[i, j] - size / 2, y[i, j] - size / 2],
            size,
            size,
            facecolor=color,
            edgecolor=color,
            zorder=3,
        )
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_xticks(range_x)
    ax.set_yticks(range_y)

    xticks, xtick_colours = _make_two_level_ticks(data.columns)
    _set_coloured_xticks(xticks, xtick_colours, ax, rotation=90)
    if grid:
        if xtick_colours is not None:
            for j, color in enumerate(xtick_colours):
                ax.axvline(range_x[j], color=color, linewidth=0.75, zorder=2)
        else:
            ax.xaxis.grid(True, zorder=1)

    yticks, ytick_colours = _make_two_level_ticks(data.index)
    _set_coloured_yticks(yticks, ytick_colours, ax)
    if grid:
        if ytick_colours is not None:
            for i, color in enumerate(ytick_colours):
                ax.axhline(range_y[i], color=color, linewidth=0.75, zorder=2)
        else:
            ax.yaxis.grid(True, zorder=1)

    return ax
