import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.spatial.distance import cdist, pdist, squareform  # type: ignore
from sklearn.metrics import silhouette_score  # type: ignore


def _assign_clusters(topic_distr, centeroids, metric):
    n_chains = topic_distr.shape[0]
    n_components = topic_distr.shape[1]
    # For each topic in each chain, find the best matching centroid.
    cluster_assignments = np.zeros([n_chains, n_components], dtype=int)
    for k in range(n_chains):
        cost_matrix = cdist(centeroids, topic_distr[k], metric=metric)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cluster_assignments[k, col_ind] = row_ind

    return cluster_assignments


def invert_cluster_mapping(cluster_assignments):
    n_chains = cluster_assignments.shape[0]
    n_components = cluster_assignments.shape[1]
    # Invert cluster assignment mappings.
    cluster_assignments_inv = np.zeros_like(cluster_assignments)
    for k in range(n_components):
        for j in range(n_chains):
            cluster_assignments_inv[j, int(cluster_assignments[j, k])] = k
    return cluster_assignments_inv


def _update_clusters(topic_distr, cluster_assignments):
    n_chains = topic_distr.shape[0]

    cluster_assignments_inv = invert_cluster_mapping(cluster_assignments)
    # Per cluster, update the centroid as the centre of mass across the chains.
    centeroids = np.zeros(topic_distr.shape[1:])
    for j in range(n_chains):
        idx_c = cluster_assignments_inv[j]
        centeroids += topic_distr[j, idx_c]
    centeroids /= n_chains
    return centeroids


def cluster_latent_components(
    weights, metric="jensenshannon", n_iterations: int = 11, verbose=False
):
    r"""Solve cluster identifiability problem using the Hungarian algorithm.

    Given `s=1,..,S` sets of `K` latent components that are similar upto a permutation
    of the component index `k = 1,..,K` (e.g., from `S` model fits), match the
    components `k` between the chains `s` and determine the centroid (i.e., the
    consensus component). The algorith uses multiple restarts and selects the best
    clustering based on the silhouette score.

    Example:
        See the notebook `examples/identifiability.ipynb`.

    Args:
        weights: Per chain (leading axis), the components \( \pmb{W}_k \) (second axis)
            to cluster.
        metric: The metric to use for computing the pairwise distances between the
            components.
        n_iterations: Number of iterations to run the clustering algorithm.
        verbose: Print the silhouette score at each iteration.

    Returns: A pair of
        cluster_assignments: Per chain (leading axis), the cluster assignments (second
            axis).
        centeroids: The centre of mass of the clusters.
    """
    if metric == "jensenshannon":
        if not np.all(weights.sum(axis=1) == 1):
            raise ValueError(
                "Trailing axis of weights should sum to 1 when using Jenson-Shannon distance."
            )
    # Restart the clustering algorithm with different chain centroids.
    n_chains = n_restarts = weights.shape[0]
    n_components = weights.shape[1]

    best_score = -1
    for i in range(n_restarts):
        # Restart clustering algorithm with centroid initially at the i-th chain.
        centeroids = weights[i]

        for _ in range(n_iterations):
            # 1) Assignment step.
            cluster_assignments = _assign_clusters(weights, centeroids, metric)

            # 2) Update step.
            centeroids = _update_clusters(weights, cluster_assignments)

            # 3) Compute pairwise distances per chain.
            x_flat = weights.reshape([n_chains * n_components, -1])
            distances = pdist(x_flat, metric=metric)
            X_dist = squareform(distances)
            ss = silhouette_score(
                X_dist, cluster_assignments.flatten(), metric="precomputed"
            )

            if verbose:
                print("Silhouette score:", ss)
        if verbose:
            print("-------------------")
            print(f"Final silhouette score {i}:", ss)
        if ss > best_score:
            best_score = ss
            best_cluster_assignments = cluster_assignments
            best_centeroids = centeroids
    if verbose:
        print("Best score:", best_score)
    return best_cluster_assignments, best_centeroids
