"""PRML 4.4.1: BIC approximation (not exact model evidence).

Use maximum likelihood fits, not regularized MAP fits, with BIC.
This small example compares intercept-only and one-feature Bernoulli models.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import BinaryLogisticRegression, bic


def comparison(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 1))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.2 + 1.5 * X[:, 0]))))
    results = {}
    for label, features in (("intercept only", np.zeros_like(X)), ("one feature", X)):
        # Both fit with negligible ridge for an identifiable stable example;
        # report BIC from *unpenalized* likelihood. The intercept-only model
        # has one free parameter; the one-feature model has two.
        clf = BinaryLogisticRegression(n_iter=2000, tol=1e-10, regularization=0.0)
        clf.fit(features, y)
        logits = np.column_stack((np.ones(len(features)), features)) @ clf.w_
        log_likelihood = -np.sum(np.logaddexp(0, logits) - y * logits)
        k = 1 if label == "intercept only" else 2
        results[label] = bic(log_likelihood, k, len(y))
    return results


if __name__ == "__main__":
    scores = comparison()
    for model, score in scores.items():
        print(f"{model}: BIC={score:.4f}")
    print("lower BIC preferred:", min(scores, key=scores.get))
