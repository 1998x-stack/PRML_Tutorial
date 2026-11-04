"""PRML 4.3.2: stable binary cross-entropy and backtracked gradient descent."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import BinaryLogisticRegression


class LogisticRegression(BinaryLogisticRegression):
    """Preserves lesson constructor and w_/cost_/predict interface."""


def generate_data(n_samples=100, seed=0):
    if not isinstance(n_samples, int) or n_samples < 4:
        raise ValueError("n_samples must be >= 4")
    rng = np.random.default_rng(seed)
    n_positive = n_samples // 2
    n_negative = n_samples - n_positive
    X = np.vstack((rng.normal(loc=1.5, size=(n_positive, 2)),
                   rng.normal(loc=-1.5, size=(n_negative, 2))))
    y = np.r_[np.ones(n_positive), np.zeros(n_negative)]
    return X, y


if __name__ == "__main__":
    X, y = generate_data(seed=0)
    clf = LogisticRegression(n_iter=700).fit(X, y)
    print("weights:", clf.w_)
    print("first/last mean cross entropy + regularization:", clf.cost_[0], clf.cost_[-1])
    print("training accuracy:", np.mean(clf.predict(X) == y))
