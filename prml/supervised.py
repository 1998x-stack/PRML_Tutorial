"""Reference numerical kernels for PRML chapters 3 and 4.

Original educational implementations based on NumPy. The public input
contract for every estimator is a finite 2D feature matrix (N, D), with N>0
and D>0, and a matching 1D target vector. No side effects on import.
"""
from __future__ import annotations

import numpy as np


def _features(X, n_features=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError("X must be a nonempty 2D array of shape (N, D)")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite numbers")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features; got {X.shape[1]}")
    return X


def _targets(y, n):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.shape[0] != n or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite 1D vector aligned with X")
    return y


def _bias(X):
    return np.column_stack((np.ones(X.shape[0]), X))


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def sigmoid(z):
    """Stable sigmoid with no exp overflow for very large logits."""
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def softmax(z):
    """Row-wise, shift-invariant softmax (shape N,K)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("z must be a matrix with at least two classes")
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class LinearLeastSquares:
    """Ordinary least squares via SVD-based lstsq (no normal-equation inverse)."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = bool(fit_intercept)
        self.weights = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X = _features(X)
        y = _targets(y, len(X))
        phi = _bias(X) if self.fit_intercept else X
        self.weights = np.linalg.lstsq(phi, y, rcond=None)[0]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")
        X = _features(X, self.n_features_in_)
        return (_bias(X) if self.fit_intercept else X) @ self.weights

    def calculate_residuals(self, X, y):
        predicted = self.predict(X)
        return _targets(y, len(predicted)) - predicted


class BayesianLinearRegression:
    """Conjugate Gaussian linear regression.

    p(w) = N(0, alpha^-1 I) INCLUDING the intercept coefficient.
    p(t|X,w) = N(Phi w, beta^-1 I).
    alpha and beta denote precisions, not variances.
    """

    def __init__(self, alpha=1.0, beta=25.0):
        self.alpha = _positive(alpha, "alpha")
        self.beta = _positive(beta, "beta")
        self.m_N = None
        self.S_N = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X = _features(X)
        y = _targets(y, len(X))
        phi = _bias(X)
        precision = self.alpha * np.eye(phi.shape[1]) + self.beta * (phi.T @ phi)
        # alpha>0 makes the posterior precision positive definite.
        L = np.linalg.cholesky(precision)
        rhs = self.beta * phi.T @ y
        self.m_N = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        identity = np.eye(phi.shape[1])
        self.S_N = np.linalg.solve(L.T, np.linalg.solve(L, identity))
        self.S_N = 0.5 * (self.S_N + self.S_N.T)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Return predictive mean and OBSERVATION variance (including noise)."""
        if self.m_N is None:
            raise ValueError("Model must be fitted before prediction")
        X = _features(X, self.n_features_in_)
        phi = _bias(X)
        mean = phi @ self.m_N
        variance = 1.0 / self.beta + np.einsum("ni,ij,nj->n", phi, self.S_N, phi)
        return mean, variance

    def log_evidence(self, X, y):
        """Log marginal likelihood of the model's Gaussian prior/noise assumptions."""
        X = _features(X)
        y = _targets(y, len(X))
        phi = _bias(X)
        precision = self.alpha * np.eye(phi.shape[1]) + self.beta * (phi.T @ phi)
        L = np.linalg.cholesky(precision)
        m = np.linalg.solve(L.T, np.linalg.solve(L, self.beta * phi.T @ y))
        e = 0.5 * (self.beta * np.sum((y - phi @ m) ** 2) +
                   self.alpha * np.dot(m, m))
        log_det_precision = 2.0 * np.log(np.diag(L)).sum()
        n, k = phi.shape
        return float(0.5 * (k * np.log(self.alpha) + n * np.log(self.beta) -
                            log_det_precision - n * np.log(2.0 * np.pi)) - e)


def _binary_targets(y, n):
    y = _targets(y, n)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Binary targets must contain only 0 and 1")
    return y


class BinaryLogisticRegression:
    """L2-regularized binary logistic regression using backtracked gradient descent.

    Objective: mean(logaddexp(0, Phi@w) - y*(Phi@w))
               + 0.5 * regularization * ||w[1:]||^2
    """

    def __init__(self, learning_rate=1.0, n_iter=1000, tol=1e-8,
                 regularization=1e-3):
        self.learning_rate = _positive(learning_rate, "learning_rate")
        if not isinstance(n_iter, (int, np.integer)) or isinstance(n_iter, bool) or n_iter <= 0:
            raise ValueError("n_iter must be a positive integer")
        self.n_iter = int(n_iter)
        self.tol = _positive(tol, "tol")
        self.regularization = _positive(regularization, "regularization", allow_zero=True)
        self.w_ = None
        self.cost_ = []
        self.n_features_in_ = None

    def loss_and_grad(self, X, y, w):
        X = _features(X)
        y = _binary_targets(y, len(X))
        phi = _bias(X)
        w = np.asarray(w, dtype=float)
        if w.shape != (phi.shape[1],):
            raise ValueError("w has incompatible shape")
        z = phi @ w
        regularized = w.copy()
        regularized[0] = 0.0
        loss = (np.mean(np.logaddexp(0.0, z) - y * z) +
                0.5 * self.regularization * regularized @ regularized)
        gradient = phi.T @ (sigmoid(z) - y) / len(y) + self.regularization * regularized
        return float(loss), gradient

    def fit(self, X, y):
        X = _features(X)
        y = _binary_targets(y, len(X))
        w = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        loss, grad = self.loss_and_grad(X, y, w)
        self.cost_.append(loss)
        for _ in range(self.n_iter):
            if np.linalg.norm(grad, ord=np.inf) < self.tol:
                break
            step = self.learning_rate
            g2 = float(grad @ grad)
            accepted = False
            while step >= 1e-14:
                candidate = w - step * grad
                next_loss, next_grad = self.loss_and_grad(X, y, candidate)
                if next_loss <= loss - 1e-4 * step * g2:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            w, loss, grad = candidate, next_loss, next_grad
            self.cost_.append(loss)
        self.w_ = w
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        if self.w_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = _features(X, self.n_features_in_)
        p = sigmoid(_bias(X) @ self.w_)
        return np.column_stack((1 - p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class IRLSLogisticRegression(BinaryLogisticRegression):
    """Damped Newton/IRLS solver for the same regularized logistic objective."""

    def __init__(self, n_iter=100, tol=1e-8, regularization=1e-3):
        super().__init__(learning_rate=1.0, n_iter=n_iter, tol=tol,
                         regularization=regularization)

    def fit(self, X, y):
        X = _features(X)
        y = _binary_targets(y, len(X))
        phi = _bias(X)
        w = np.zeros(phi.shape[1])
        loss, grad = self.loss_and_grad(X, y, w)
        self.cost_ = [loss]
        for _ in range(self.n_iter):
            if np.linalg.norm(grad, ord=np.inf) < self.tol:
                break
            z = phi @ w
            p = sigmoid(z)
            weight = p * (1 - p)
            hessian = phi.T @ (phi * weight[:, None]) / len(y)
            penalty = self.regularization * np.eye(phi.shape[1])
            penalty[0, 0] = 0.0
            # Small jitter makes Newton direction well-defined on collinear data.
            delta = np.linalg.solve(hessian + penalty + 1e-10 * np.eye(phi.shape[1]), grad)
            step = 1.0
            accepted = False
            while step >= 1e-14:
                candidate = w - step * delta
                next_loss, next_grad = self.loss_and_grad(X, y, candidate)
                if next_loss <= loss - 1e-4 * step * float(grad @ delta):
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            w, loss, grad = candidate, next_loss, next_grad
            self.cost_.append(loss)
        self.w_ = w
        self.n_features_in_ = X.shape[1]
        return self


class MulticlassLogisticRegression:
    """Multiclass softmax using vectorized cross-entropy and gradient descent.

    Labels can be arbitrary comparable integers or strings; classes_ records
    the sorted label map. No invalid scalar-Hessian shortcut is used.
    """

    def __init__(self, learning_rate=1.0, n_iter=1500, tol=1e-8,
                 regularization=1e-3):
        self.learning_rate = _positive(learning_rate, "learning_rate")
        if not isinstance(n_iter, (int, np.integer)) or isinstance(n_iter, bool) or n_iter <= 0:
            raise ValueError("n_iter must be positive")
        self.n_iter = int(n_iter)
        self.tol = _positive(tol, "tol")
        self.regularization = _positive(regularization, "regularization", allow_zero=True)
        self.W_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.cost_ = []

    def _objective(self, phi, indices, W):
        logits = phi @ W
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_den = np.log(np.exp(shifted).sum(axis=1))
        loss = np.mean(log_den - shifted[np.arange(len(indices)), indices])
        reg = W.copy()
        reg[0, :] = 0
        loss += 0.5 * self.regularization * np.sum(reg ** 2)
        proba = softmax(logits)
        proba[np.arange(len(indices)), indices] -= 1
        grad = phi.T @ proba / len(indices) + self.regularization * reg
        return float(loss), grad

    def fit(self, X, y):
        X = _features(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(X):
            raise ValueError("Labels must be a 1D vector aligned with X")
        classes, indices = np.unique(labels, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Training requires at least two distinct classes")
        phi = _bias(X)
        W = np.zeros((phi.shape[1], len(classes)))
        self.cost_ = []
        loss, grad = self._objective(phi, indices, W)
        self.cost_.append(loss)
        for _ in range(self.n_iter):
            if np.linalg.norm(grad, ord=np.inf) < self.tol:
                break
            step = self.learning_rate
            g2 = float(np.sum(grad * grad))
            accepted = False
            while step >= 1e-14:
                candidate = W - step * grad
                next_loss, next_grad = self._objective(phi, indices, candidate)
                if next_loss <= loss - 1e-4 * step * g2:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            W, loss, grad = candidate, next_loss, next_grad
            self.cost_.append(loss)
        self.W_, self.classes_, self.n_features_in_ = W, classes, X.shape[1]
        return self

    def predict_proba(self, X):
        if self.W_ is None:
            raise ValueError("Model must be fitted before prediction")
        return softmax(_bias(_features(X, self.n_features_in_)) @ self.W_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class GaussianDiscriminant:
    """Shared-covariance Gaussian generative classifier (LDA).

    Fits class means, pooled maximum-likelihood covariance and empirical class
    priors. Covariance regularization is numerical, not shrinkage estimation.
    """

    def __init__(self, jitter=1e-6):
        self.jitter = _positive(jitter, "jitter")
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_ = None
        self.precision_means_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X = _features(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(X):
            raise ValueError("Labels must be aligned with X")
        classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if len(classes) < 2:
            raise ValueError("At least two classes required")
        means = np.vstack([X[inverse == k].mean(axis=0) for k in range(len(classes))])
        residuals = X - means[inverse]
        covariance = (residuals.T @ residuals) / len(X)
        covariance += self.jitter * np.eye(X.shape[1])
        precision_means = np.linalg.solve(covariance, means.T).T
        self.classes_, self.means_ = classes, means
        self.priors_ = counts / len(X)
        self.covariance_ = covariance
        self.precision_means_ = precision_means
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        if self.means_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = _features(X, self.n_features_in_)
        log_scores = (X @ self.precision_means_.T
                      - 0.5 * np.sum(self.means_ * self.precision_means_, axis=1)
                      + np.log(self.priors_))
        return softmax(log_scores)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def bic(log_likelihood, n_parameters, n_observations):
    """Schwarz BIC = k log N - 2 log L_hat; lower is better.

    Valid as an asymptotic model-selection approximation under regularity
    conditions. Not an exact Bayes factor, nor valid for arbitrary penalties.
    """
    log_likelihood = float(log_likelihood)
    if not np.isfinite(log_likelihood):
        raise ValueError("log_likelihood must be finite")
    if not isinstance(n_parameters, (int, np.integer)) or n_parameters < 0:
        raise ValueError("n_parameters must be nonnegative integer")
    if not isinstance(n_observations, (int, np.integer)) or n_observations < 1:
        raise ValueError("n_observations must be positive integer")
    return float(n_parameters * np.log(n_observations) - 2 * log_likelihood)
