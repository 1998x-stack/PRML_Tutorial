"""PRML 4.3.4: full K-class softmax, correct gradient, stable log loss."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import MulticlassLogisticRegression


def generate_data(n_samples=300, seed=0):
    if not isinstance(n_samples, int) or n_samples < 6:
        raise ValueError("n_samples must be >= 6")
    rng = np.random.default_rng(seed)
    groups = np.array_split(np.arange(n_samples), 3)
    centers = np.array([[-2, -2], [2, -2], [0, 2]])
    X = np.empty((n_samples, 2))
    y = np.empty(n_samples, dtype=int)
    for index, inds in enumerate(groups):
        X[inds] = rng.normal(size=(len(inds), 2)) + centers[index]
        y[inds] = [10, 20, 30][index]
    return X, y


if __name__ == "__main__":
    X, y = generate_data()
    clf = MulticlassLogisticRegression(n_iter=800).fit(X, y)
    print("class labels:", clf.classes_)
    print("weights shape:", clf.W_.shape)
    print("training accuracy:", np.mean(clf.predict(X) == y))
    print("probability row sums:", clf.predict_proba(X[:3]).sum(axis=1))
