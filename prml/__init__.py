"""Small, original reference implementations accompanying the PRML curriculum.

Do not confuse these learning kernels with production-scale model packages.
"""
from .bayesian_classification import BayesianLogisticLaplace
from .supervised import (
    BayesianLinearRegression,
    BinaryLogisticRegression,
    GaussianDiscriminant,
    IRLSLogisticRegression,
    LinearLeastSquares,
    MulticlassLogisticRegression,
    bic,
    sigmoid,
    softmax,
)

__all__ = [
    "BayesianLinearRegression",
    "BayesianLogisticLaplace",
    "BinaryLogisticRegression",
    "GaussianDiscriminant",
    "IRLSLogisticRegression",
    "LinearLeastSquares",
    "MulticlassLogisticRegression",
    "bic",
    "sigmoid",
    "softmax",
]
