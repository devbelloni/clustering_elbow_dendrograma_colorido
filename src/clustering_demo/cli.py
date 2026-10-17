"""Command line: python -m clustering_demo [--csv FILE] [--out DIR]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import core, plots


def load_data(csv: Path | None, sample: int, seed: int) -> pd.DataFrame:
    if csv is not None:
        df = pd.read_csv(csv)
    else:
        import seaborn as sns  # the bundled 'mpg' dataset is downloaded on first use

        df = sns.load_dataset("mpg")
    df = df.dropna(subset=["name"]) if "name" in df else df
    return df.sample(min(sample, len(df)), random_state=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="Custom CSV (default: seaborn 'mpg')")
    parser.add_argument("--features", nargs="+", default=core.DEFAULT_FEATURES)
    parser.add_argument("--sample", type=int, default=30, help="Rows to sample")
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    df = load_data(args.csv, args.sample, args.seed)
    x = core.prepare_features(df, args.features)
    inertia = core.elbow_inertia(x, args.k_max, args.seed)
    silhouette = core.silhouette_scores(x, args.k_max, args.seed)
    k = core.suggest_k(inertia)
    labels = core.fit_kmeans(x, k, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    plots.plot_elbow(inertia, silhouette, k).savefig(args.out / "elbow.png", dpi=150)
    names = df["name"].tolist() if "name" in df else [str(i) for i in df.index]
    plots.plot_dendrogram(core.ward_linkage(x), names, k).savefig(args.out / "dendrogram.png", dpi=150)

    summary = {
        "rows": len(df),
        "features": args.features,
        "suggested_k": k,
        "best_silhouette_k": int(silhouette.idxmax()),
        "cluster_sizes": {int(c): int((labels == c).sum()) for c in sorted(set(labels))},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
