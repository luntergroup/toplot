"""Visualization of hidden units/topic proportions"""

from typing import Literal

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from python_tsp.heuristics import solve_tsp_local_search
from scipy.spatial.distance import pdist, squareform


def _sorted_topic_proportions(hidden):
    """Solve the travelling salesman problem to sort the hidden units by similarity."""
    if len(hidden) > 500:
        print(
            "Warning: More than 500 samples. Solving the traveling salesman problem may take a while."
        )
    sample_distances = pdist(hidden, metric="jensenshannon")
    distance_matrix = squareform(sample_distances)
    permutation, _ = solve_tsp_local_search(distance_matrix)
    return hidden[np.array(permutation)]


def plot_cohort(
    hidden, sort: Literal["travelling-salesman"] | None = "travelling-salesman", ax=None
):
    """Plot the topic proportions for a set of participants/examples/documents.

    Args:
        hidden: For each participant/document (rows), the proportion per topic
            (columns).
        sort: Whether to sort the participants according to the shortest Jensen-Shannon
            distance path ("travelling-salesman"), or no sorting (None).
        ax: Matplotlib axes to plot on.

    Returns: Reference to matplotlib axes.

    Example:
        ```python
        from pandas import DataFrame
        from numpy.random import dirichlet

        hidden = DataFrame(dirichlet([0.6, 0.8, 0.2], size=30), columns=['A', 'B', 'C'])
        plot_cohort(hidden)
        ```
    """
    if ax is None:
        ax = plt.gca()

    n_topics = hidden.shape[-1]

    topic_labels = [str(k + 1) for k in range(n_topics)]
    if isinstance(hidden, pd.DataFrame):
        topic_labels = hidden.columns
        hidden = hidden.to_numpy()

    if sort == "travelling-salesman":
        hidden_sorted = _sorted_topic_proportions(hidden)
    elif sort is None:
        hidden_sorted = hidden
    else:
        raise ValueError(f"Unknown sort method: {sort}")

    # Compute offsets for stacking bars.
    p_cum = np.cumsum(hidden_sorted, axis=1)
    p_cum_zero_padded = np.pad(
        p_cum, ((0, 0), (1, 0)), mode="constant", constant_values=0
    )
    x = np.arange(len(hidden_sorted)) + 1
    dx = x[1] - x[0]
    bar_width = dx

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Topic proportion")
    ax.set_ylim(0, 1)

    for k in range(n_topics):
        ax.bar(
            x,
            hidden_sorted[:, k],
            width=bar_width,
            label=topic_labels[k],
            bottom=p_cum_zero_padded[:, k],
            linewidth=0.0,
        )
    ax.legend(title="Topic:")
    return ax


def plot_polar_cohort(
    hidden, ax=None, sort: Literal["travelling-salesman"] | None = "travelling-salesman"
):
    """Make radial bar plots for a set of participants/examples/documents.

    Args:
        hidden: For each participant (rows), the proportion per topic (columns).
        sort: Whether to sort the participants according to the shortest Jensen-Shannon
            distance path ("travelling-salesman"), or no sorting (None).
        ax: Matplotlib axes to plot on.

    Returns: Reference to matplotlib axes.

    Example:
        ```python
        from pandas import DataFrame
        from numpy.random import dirichlet

        hidden = DataFrame(dirichlet([0.6, 0.8, 0.2], size=30), columns=['A', 'B', 'C'])
        plot_polar_cohort(hidden)
        ```
    """
    if ax is None:
        ax = plt.gcf().add_subplot(projection="polar")

    m_samples, n_topics = hidden.shape
    topic_labels = [str(k + 1) for k in range(n_topics)]
    if isinstance(hidden, pd.DataFrame):
        topic_labels = hidden.columns
        hidden = hidden.to_numpy()

    if sort == "travelling-salesman":
        hidden_sorted = _sorted_topic_proportions(hidden)
    elif sort is None:
        hidden_sorted = hidden
    else:
        raise ValueError(f"Unknown sort method: {sort}")

    p_cum = np.cumsum(hidden_sorted, axis=1)
    p_cum_zero_padded = np.pad(
        p_cum, ((0, 0), (1, 0)), mode="constant", constant_values=0
    )

    x = np.linspace(0, 2 * np.pi, m_samples, endpoint=False)
    dx = x[1] - x[0]
    bar_width = dx
    ax.set_rlim(0, 1.0)
    ax.set_rorigin(-0.5)

    for k in range(n_topics):
        ax.bar(
            x,
            hidden_sorted[:, k],
            width=bar_width,
            label=topic_labels[k],
            bottom=p_cum_zero_padded[:, k],
        )

    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(1.0, 0), title="Topic:")
    ax.axis("off")
    return ax
