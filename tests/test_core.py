import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from clustering_demo import core
from clustering_demo.cli import main


@pytest.fixture
def blobs():
    x, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=0)
    return pd.DataFrame(x, columns=["a", "b"])


def test_prepare_features_standardizes_and_imputes():
    df = pd.DataFrame(
        {"mpg": [10, 20, None, 40], "horsepower": [100, 120, 90, 80], "weight": [3000, 2500, 2800, 2000]}
    )
    x = core.prepare_features(df)
    assert not x.isna().any().any()
    assert np.allclose(x.mean(), 0, atol=1e-9)
    assert np.allclose(x.std(ddof=0), 1, atol=1e-9)


def test_prepare_features_requires_columns():
    with pytest.raises(ValueError, match="Missing feature columns"):
        core.prepare_features(pd.DataFrame({"mpg": [1.0]}))


def test_inertia_is_non_increasing(blobs):
    inertia = core.elbow_inertia(blobs, k_max=8)
    assert list(inertia.index) == list(range(1, 9))
    assert (inertia.diff().dropna() <= 1e-9).all()


def test_suggest_k_recovers_true_number_of_blobs(blobs):
    assert core.suggest_k(core.elbow_inertia(blobs, k_max=9)) == 3


def test_silhouette_is_highest_at_true_k(blobs):
    assert core.silhouette_scores(blobs, k_max=8).idxmax() == 3


def test_fit_kmeans_is_deterministic(blobs):
    a = core.fit_kmeans(blobs, 3, seed=1)
    b = core.fit_kmeans(blobs, 3, seed=1)
    assert (a == b).all() and set(a) == {0, 1, 2}


def test_ward_linkage_shape(blobs):
    assert core.ward_linkage(blobs).shape == (len(blobs) - 1, 4)


def test_cli_end_to_end(tmp_path, blobs):
    csv = tmp_path / "data.csv"
    blobs.assign(name=[f"p{i}" for i in range(len(blobs))]).to_csv(csv, index=False)
    out = tmp_path / "out"
    assert main(["--csv", str(csv), "--features", "a", "b", "--sample", "60", "--out", str(out)]) == 0
    assert (out / "elbow.png").exists() and (out / "dendrogram.png").exists()
    assert (out / "summary.json").exists()
