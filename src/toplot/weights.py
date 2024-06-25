"""Visualization of the weights/components/topics with uncertainty estimates."""

from functools import partial

from jax import tree_util
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.pyplot import cm, rcParams
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from .scattermap import scattermap
from .utils import dataframe_to_pytree


def bar_plot_stacked(dataframe, quantile_range=(0.025, 0.975), ax=None, fontsize=None):
    """Plot posterior topic weights as probability bars by stacking items per set.

    Ags:
        dataframe: Posterior samples (rows) of (normalized) topic weights. The dataframe
            consists of two-level columns to group categories that belong to the same
            item set (i.e., multinomial). Second level columns must sum to one.
        quantile_range: Range of quantiles to plot as error bars.
        ax: Matplotlib axes to plot on.
        fontsize: Font size for the category labels.

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
    weight_tree, metadata = dataframe_to_pytree(dataframe, to_numpy=True)

    if ax is None:
        ax = plt.gca()

    cmap = plt.get_cmap("PiYG")

    # Compute summary statistics of the posterior samples.
    avg = tree.map(partial(np.mean, axis=0), weight_tree)
    lower = tree.map(partial(np.quantile, q=quantile_range[0], axis=0), weight_tree)
    upper = tree.map(partial(np.quantile, q=quantile_range[1], axis=0), weight_tree)
    # The error bars are the distance from the mean to the quantiles.
    err = tree.map(lambda a, l, u: np.stack([a - l, u - a], axis=0), avg, lower, upper)

    # Make a bar per leaf by stacking the categories on top of each other --> the per
    # category bar offsets (relative to y = 0) are the cumulative distribution.
    p_cum = tree.map(partial(np.cumsum, axis=-1), avg)
    # The bar offset of the first category is 0.
    p_cum_zero_padded = tree.map(
        lambda cdf: np.pad(cdf, pad_width=(1, 0), mode="constant", constant_values=0),
        p_cum,
    )

    # For each leaf.
    for feature_name in weight_tree.keys():
        feature_weights = avg[feature_name]
        feature_err = err[feature_name]
        offsets = p_cum_zero_padded[feature_name]

        feature_categories = metadata[feature_name]
        n_categories = len(feature_weights)
        for j in range(n_categories):
            u = 1 - j / (n_categories - 1)
            color = cmap(0.1 + 0.8 * u)
            # Plot error bars for all but the last category.
            err_j = feature_err[..., [j]]
            if j == n_categories - 1:
                err_j = None
            ax.bar(
                feature_name,
                feature_weights[j],
                bottom=offsets[j],
                color=color,
                yerr=err_j,
            )
            ax.text(
                x=feature_name,
                y=offsets[j] + feature_weights[j] / 2,
                s=feature_categories[j],
                ha="center",
                va="center",
                fontsize=fontsize,
            )
    # Rotate the x-axis labels.
    ax.tick_params(axis="x", labelrotation=90, labelsize=fontsize)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Probability")
    return ax


def bar_plot(
    dataframe: pd.DataFrame, quantile_range=(0.025, 0.975), label=None, ax=None
):
    """Plot posterior of topic weights as an unfolded array of probability bars.

    Args:
        dataframe: Posterior samples (rows) of topic weights. This dataframe consists of
            two-level columns that group categories that belong to the same multinomial.
            Second level columns must sum to one.
        quantile_range: Range of quantiles to plot as error bars.
        label: A legend label for the plot.
        ax: Matplotlib axes to plot on.

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

    # Compute summary statistics of distribution.
    avg = dataframe.apply(np.mean, axis="rows")
    lower = dataframe.apply(np.quantile, q=quantile_range[0], axis="rows")
    upper = dataframe.apply(np.quantile, q=quantile_range[1], axis="rows")
    err = np.stack([avg - lower, upper - avg], axis=0)

    # Give each category set (=first column level) a different colour.
    multinomial_names = dataframe.columns.unique(level=0)
    repeated_colours = 5 * tuple(TABLEAU_COLORS)  # Five times should suffice.
    colour_of_multinomial = dict(zip(multinomial_names, repeated_colours))
    colours = [
        colour_of_multinomial[name] for name in dataframe.columns.get_level_values(0)
    ]
    feature_names = [
        f"{set_name}: {item_name}" for set_name, item_name in dataframe.columns
    ]

    ax.bar(feature_names, height=avg, yerr=err, label=label, color=colours)
    ax.set_ylabel("Probability")
    ax.tick_params(axis="x", labelrotation=90)
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
    """
    dataframe: dataframe in two levels {feature: words} containing phi, determines markers and their color
    dataframe_counts: another dataframe of same structure containing counts, determines markersize and bars at the axes
    marker_scaler: scale size of markers
    scale_val_x_counts: scale bar size
    scale_val_y_counts: scale bar size

    There are issues with figure size for very large dataframes (many topics), possibly fix for future use.
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

def top_words(dataframe, dataframe_counts):
        '''
        Visualize the most important 'words' of categories over featrures per latent variable.
        '''

    n_topics = 6
    color = iter(cm.rainbow(np.linspace(0, 1, n_topics)))
    rcParams['figure.figsize'] = 3, 20
    fig, ax = plt.subplots(nrows=6, ncols=1, sharex=True, sharey=False)
    # plt.title('Top 10 words in the topics with their count')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.tight_layout()
    for t in range(n_topics):
        kleur = next(color)
        for keys, vals in topic_top_words[t].items():
            ax[t].barh(keys, vals, color=kleur)
