"""PRML 1.2.5 — Polynomial curve fitting from a probabilistic viewpoint.

This is an original, self-contained NumPy implementation of polynomial
least squares and Gaussian-prior MAP (ridge) estimation. It does NOT implement
full Bayesian posterior predictive uncertainty; see the paired Markdown.

Run: python "1_Introduction/1.2_Probability_Theory/04_1.2.5_Curve_fitting_re_visited.py"
Requires: numpy.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np


def _as_finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not arr.size:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


class PolynomialCurveFitting:
    """Fit y(x)=sum_j w[j] x**j using least squares or L2 regularization.

    Parameters
    ----------
    degree:
        Nonnegative integer polynomial degree.
    regularization_param:
        Nonnegative ridge penalty λ. By convention, the intercept w[0] is
        *not* penalized. λ is on the unscaled sum-of-squares objective.

    Notes
    -----
    Uses an augmented least-squares system instead of explicitly computing
    inverse(Phi.T @ Phi), which can square the condition number.
    The Vandermonde basis itself can still be ill-conditioned at high degree.
    """

    def __init__(self, degree: int, regularization_param: float = 0.0) -> None:
        if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        lam = float(regularization_param)
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("regularization_param must be finite and nonnegative")
        self.degree = int(degree)
        self.regularization_param = lam
        self.coefficients: np.ndarray | None = None

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        features = np.vander(x, N=self.degree + 1, increasing=True)
        if not np.all(np.isfinite(features)):
            raise ValueError("Polynomial features overflowed; rescale x or lower degree")
        return features

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """Estimate coefficients; x and t are finite, aligned 1D vectors."""
        x_data = _as_finite_vector(x, "x")
        targets = _as_finite_vector(t, "t")
        if x_data.size != targets.size:
            raise ValueError("x and t must have the same number of samples")

        phi = self._design_matrix(x_data)
        if self.regularization_param:
            # Penalize w[1:] but not the intercept w[0].
            penalty = np.eye(self.degree + 1, dtype=float)[1:, :]
            design = np.vstack((phi, np.sqrt(self.regularization_param) * penalty))
            observations = np.concatenate((targets, np.zeros(self.degree)))
        else:
            design, observations = phi, targets
        # Least-squares solution also handles rank-deficient design matrices.
        self.coefficients = np.linalg.lstsq(design, observations, rcond=None)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the fitted conditional mean, not Bayesian uncertainty."""
        if self.coefficients is None:
            raise ValueError("Call fit before predict")
        x_data = _as_finite_vector(x, "x")
        return self._design_matrix(x_data) @ self.coefficients

    def calculate_rmse(self, x: np.ndarray, t: np.ndarray) -> float:
        """Root mean squared error on the specified dataset."""
        targets = _as_finite_vector(t, "t")
        predicted = self.predict(x)
        if predicted.size != targets.size:
            raise ValueError("x and t must have the same number of samples")
        return float(np.sqrt(np.mean((predicted - targets) ** 2)))


def demonstration(seed: int = 42) -> None:
    """Compare model complexity and ridge regularization using fixed data."""
    rng = np.random.default_rng(seed)
    x_train = np.linspace(0.0, 1.0, 12)
    y_true = np.sin(2.0 * np.pi * x_train)
    t_train = y_true + rng.normal(0.0, 0.20, size=x_train.size)

    x_eval = np.linspace(0.0, 1.0, 201)
    y_eval = np.sin(2.0 * np.pi * x_eval)
    print("model | training RMSE (noisy targets) | evaluation RMSE (true function)")
    for degree, lam in [(1, 0.0), (9, 0.0), (9, 0.01)]:
        model = PolynomialCurveFitting(degree, lam)
        model.fit(x_train, t_train)
        train_rmse = model.calculate_rmse(x_train, t_train)
        eval_rmse = model.calculate_rmse(x_eval, y_eval)
        print(f"degree={degree:2d} λ={lam:4.2f} | {train_rmse:.4f} | {eval_rmse:.4f}")


if __name__ == "__main__":
    demonstration()
