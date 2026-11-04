"""PRML 3.3.2: Bayesian predictive variance includes observation noise."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import BayesianLinearRegression


if __name__ == "__main__":
    X = np.array([[0.1], [0.4], [0.7], [1.0]])
    y = np.array([1.1, 1.9, 3.0, 4.2])
    fitted = BayesianLinearRegression(alpha=1, beta=25).fit(X, y)
    mean, variance = fitted.predict(np.array([[0.2], [0.5], [0.8]]))
    print("predictive mean:", mean)
    print("predictive std dev (includes observation noise):", np.sqrt(variance))
