"""Core clustering routines. No plotting and no I/O here, so everything is unit-testable."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["mpg", "horsepower", "weight"]


def prepare_features(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Select numeric features, impute missing values with the median and standardize.

    Standardizing matters: K-Means and Ward linkage use Euclidean distance, so without it
    the feature with the largest scale (here ``weight``) would dominate the clustering.
    """
    features = features or DEFAULT_FEATURES
    missing = [c for c in features if c not in df]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    subset = df[features].astype(float)
    subset = subset.fillna(subset.median())
    scaled = StandardScaler().fit_transform(subset)
    return pd.DataFrame(scaled, index=subset.index, columns=features)


def elbow_inertia(x: pd.DataFrame | np.ndarray, k_max: int = 10, seed: int = 42) -> pd.Series:
    """Within-cluster sum of squares (inertia) of K-Means for k = 1..k_max."""
    k_max = min(k_max, len(x))
    inertia = {
        k: KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x).inertia_ for k in range(1, k_max + 1)
    }
    return pd.Series(inertia, name="inertia").rename_axis("k")


def suggest_k(inertia: pd.Series) -> int:
    """Automatic elbow: the point farthest from the line joining the first and last points."""
    if len(inertia) < 3:
        return int(inertia.index[0])
    ks = inertia.index.to_numpy(dtype=float)
    ys = inertia.to_numpy(dtype=float)
    # Normalise both axes so the distance is not dominated by the inertia scale.
    kx = (ks - ks.min()) / (ks.max() - ks.min())
    ky = (ys - ys.min()) / (ys.max() - ys.min())
    p1, p2 = np.array([kx[0], ky[0]]), np.array([kx[-1], ky[-1]])
    line = (p2 - p1) / np.linalg.norm(p2 - p1)
    points = np.column_stack([kx, ky]) - p1
    distances = np.abs(points[:, 0] * line[1] - points[:, 1] * line[0])
    return int(ks[int(np.argmax(distances))])


def silhouette_scores(x: pd.DataFrame | np.ndarray, k_max: int = 10, seed: int = 42) -> pd.Series:
    """Mean silhouette coefficient for k = 2..k_max (higher is better)."""
    k_max = min(k_max, len(x) - 1)
    scores = {
        k: silhouette_score(x, KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x))
        for k in range(2, k_max + 1)
    }
    return pd.Series(scores, name="silhouette").rename_axis("k")


def fit_kmeans(x: pd.DataFrame | np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Cluster labels from K-Means with ``k`` clusters."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)


def ward_linkage(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Hierarchical clustering linkage matrix (Ward)."""
    return linkage(x, method="ward")
