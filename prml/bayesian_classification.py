"""PRML 4.5: Gaussian-prior Bayesian logistic regression via Laplace.

The posterior p(w|D) is non-conjugate. A local Gaussian approximation is
constructed at the MAP solution, with Hessian-inverse covariance.
Predictive probabilities integrate sigmoid over its Gaussian logit
approximation using deterministic Gauss-Hermite quadrature.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from .supervised import _bias, _binary_targets, _features, _positive, sigmoid


class BayesianLogisticLaplace:
    """Bayesian binary logistic regression, proper N(0, alpha^-1 I) prior.

    In contrast to the standard frequentist logistic lesson, the Gaussian
    prior includes the intercept coefficient. This makes the prior proper.
    All optimization is performed in original sample *sum* likelihood units.
    """

    def __init__(self, alpha=1.0, n_iter=80, tol=1e-9):
        self.alpha = _positive(alpha, "alpha")
        if not isinstance(n_iter, (int, np.integer)) or isinstance(n_iter, bool) or n_iter <= 0:
            raise ValueError("n_iter must be a positive integer")
        self.n_iter = int(n_iter)
        self.tol = _positive(tol, "tol")
        self.w_map_ = None
        self.covariance_ = None
        self.n_features_in_ = None
        self.objective_history_ = []

    def objective_gradient_hessian(self, X, y, w):
        """Negative log posterior (up to a constant), gradient and Hessian."""
        X = _features(X)
        y = _binary_targets(y, len(X))
        phi = _bias(X)
        w = np.asarray(w, dtype=float)
        if w.shape != (phi.shape[1],) or not np.all(np.isfinite(w)):
            raise ValueError("w must be a finite vector of length D+1")
        z = phi @ w
        p = sigmoid(z)
        weights = p * (1.0 - p)
        loss = (np.logaddexp(0.0, z) - y * z).sum()
        loss += 0.5 * self.alpha * float(w @ w)
        gradient = phi.T @ (p - y) + self.alpha * w
        hessian = phi.T @ (weights[:, None] * phi) + self.alpha * np.eye(len(w))
        return float(loss), gradient, hessian

    def fit(self, X, y):
        X = _features(X)
        y = _binary_targets(y, len(X))
        w = np.zeros(X.shape[1] + 1, dtype=float)
        self.objective_history_ = []
        loss, grad, hessian = self.objective_gradient_hessian(X, y, w)
        self.objective_history_.append(loss)
        for _ in range(self.n_iter):
            if np.linalg.norm(grad, ord=np.inf) < self.tol:
                break
            # Proper prior precision ensures SPD even under complete separation.
            delta = np.linalg.solve(hessian, grad)
            slope = float(grad @ delta)
            step = 1.0
            accepted = False
            while step >= 1e-14:
                new_w = w - step * delta
                next_loss, next_grad, next_h = self.objective_gradient_hessian(X, y, new_w)
                if next_loss <= loss - 1e-4 * step * slope:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            w, loss, grad, hessian = new_w, next_loss, next_grad, next_h
            self.objective_history_.append(loss)

        # Laplace covariance = inverse Hessian at MAP, evaluated at final w.
        # Use Cholesky solves rather than explicit inverse.
        _, _, hessian = self.objective_gradient_hessian(X, y, w)
        L = np.linalg.cholesky(hessian)
        identity = np.eye(len(w))
        covariance = np.linalg.solve(L.T, np.linalg.solve(L, identity))
        self.w_map_ = w
        self.covariance_ = 0.5 * (covariance + covariance.T)
        self.n_features_in_ = X.shape[1]
        return self

    def logit_mean_variance(self, X):
        """Gaussian approximation to the latent logit at each new example."""
        if self.w_map_ is None:
            raise ValueError("Call fit before predict")
        X = _features(X, self.n_features_in_)
        phi = _bias(X)
        mean = phi @ self.w_map_
        variance = np.einsum("ni,ij,nj->n", phi, self.covariance_, phi)
        return mean, np.maximum(variance, 0.0)

    def predict_proba(self, X, quadrature_points=32):
        """Approximate E_{w|D}[sigmoid(phi(x)^T w)] using Gauss-Hermite nodes."""
        if not isinstance(quadrature_points, (int, np.integer)) or isinstance(quadrature_points, bool):
            raise ValueError("quadrature_points must be an integer")
        if not 4 <= quadrature_points <= 128:
            raise ValueError("quadrature_points must be between 4 and 128")
        mean, variance = self.logit_mean_variance(X)
        nodes, weights = hermgauss(int(quadrature_points))
        logits = mean[:, None] + np.sqrt(2.0 * variance[:, None]) * nodes[None, :]
        p = sigmoid(logits) @ weights / np.sqrt(np.pi)
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack((1.0 - p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
