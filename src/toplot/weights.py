"""Visualization of the weights/components/topics with uncertainty estimates."""

from functools import partial

from matplotlib import pyplot as plt
import numpy as np
from jax import tree

from .utils import dataframe_to_pytree


def bar_plot_folded(dataframe, quantile_range=(0.025, 0.975), ax=None, fontsize=None):
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
        plot_topic_posterior_folded(weights)
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
    ax.set_ylabel("Probability")
    return ax
