# Clustering: Elbow Method, K-Means and Colored Dendrogram

Finding the right number of clusters instead of guessing it. The script uses the
**Elbow Method** to choose `k`, fits **K-Means**, and draws a **hierarchical-clustering
dendrogram** (Ward linkage) colored by distance.

Dataset: the `mpg` car dataset bundled with `seaborn` (no download needed). Features:
`mpg`, `horsepower`, `weight`. A random sample of 30 cars is used.

## How it works

1. `elbow.py` fits K-Means for `k = 1..10` and plots the inertia (within-cluster sum of
   squares). The "elbow" in the curve marks the best `k`; for this data it is **4**.
2. `main.py` fits K-Means with `k = 4` and builds a Ward-linkage dendrogram, coloring
   branches with a distance threshold so the four groups are visible.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

## Stack

Python, pandas, scikit-learn, SciPy, seaborn, Matplotlib.

## Author

**Márcio Belloni**, AI/ML Engineer.
[LinkedIn](https://www.linkedin.com/in/marcio-belloni/) · [GitHub](https://github.com/devbelloni)
