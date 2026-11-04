"""PRML 3.5.1: log model evidence using Cholesky and log determinants."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import BayesianLinearRegression as _BayesianLinearRegression


class BayesianLinearRegression(_BayesianLinearRegression):
    """Preserve the legacy evidence(X, y) lesson API."""

    def evidence(self, X, t):
        return self.log_evidence(X, t)


if __name__ == "__main__":
    X = np.array([[0.1], [0.4], [0.7], [1.0]])
    y = np.array([1.1, 1.9, 3.0, 4.2])
    fitted = BayesianLinearRegression(alpha=1, beta=25).fit(X, y)
    print("log marginal likelihood:", fitted.evidence(X, y))
