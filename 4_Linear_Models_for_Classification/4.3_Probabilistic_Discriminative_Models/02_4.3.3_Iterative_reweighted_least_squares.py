"""PRML 4.3.3: damped, regularized IRLS / Newton logistic regression."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import IRLSLogisticRegression


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 2))
    probabilities = 1 / (1 + np.exp(-(0.7 + 1.8 * X[:, 0] - X[:, 1])))
    y = rng.binomial(1, probabilities).astype(float)
    model = IRLSLogisticRegression(n_iter=80, tol=1e-9, regularization=1e-2)
    model.fit(X, y)
    print("weights:", model.w_)
    print("objective start/end:", model.cost_[0], model.cost_[-1])
    print("iterations accepted:", len(model.cost_) - 1)
