"""PRML 3.3.1: conjugate Gaussian parameter posterior.

alpha = prior precision, beta = noise precision; intercept is regularized.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import BayesianLinearRegression


if __name__ == "__main__":
    X = np.array([[0.1], [0.4], [0.7], [1.0]])
    t = np.array([1.1, 1.9, 3.0, 4.2])
    fitted = BayesianLinearRegression(alpha=1, beta=25).fit(X, t)
    print("posterior mean:", fitted.m_N)
    print("posterior covariance:\n", fitted.S_N)
