"""PRML 3.1.1: Gaussian-noise MLE and least squares (NumPy).

python "3_Linear_Models_for_Regression/3.1_Linear_Basis_Function_Models/00_3.1.1_Maximum_likelihood_and_least_squares.py"
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import LinearLeastSquares


class LinearRegression(LinearLeastSquares):
    """Legacy lesson name; see stable LinearLeastSquares in prml.supervised."""


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2))
    y = 2 + 3 * X[:, 0] + 5 * X[:, 1] + rng.normal(0, 0.5, size=100)
    fitted = LinearRegression().fit(X, y)
    print("coefficient order [intercept, x1, x2]:", fitted.weights)
    print("training RMSE:", np.sqrt(np.mean(fitted.calculate_residuals(X, y) ** 2)))
