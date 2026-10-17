"""Matplotlib figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402


def plot_elbow(inertia: pd.Series, silhouette: pd.Series, k: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(inertia.index, inertia.values, marker="o")
    ax1.axvline(k, color="tab:red", linestyle="--", label=f"elbow k = {k}")
    ax1.set(title="Elbow method", xlabel="Number of clusters (k)", ylabel="Inertia")
    ax1.legend()
    ax2.plot(silhouette.index, silhouette.values, marker="o", color="tab:green")
    ax2.set(title="Silhouette score", xlabel="Number of clusters (k)", ylabel="Mean silhouette")
    fig.tight_layout()
    return fig


def plot_dendrogram(linkage_matrix: np.ndarray, labels: list[str], k: int):
    # Cut the tree so that exactly k clusters get distinct colours.
    threshold = linkage_matrix[-(k - 1), 2] if k > 1 else linkage_matrix[-1, 2] + 1
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=8,
               color_threshold=threshold, ax=ax)
    ax.set(title=f"Ward dendrogram cut at k = {k}", ylabel="Distance")
    fig.tight_layout()
    return fig
