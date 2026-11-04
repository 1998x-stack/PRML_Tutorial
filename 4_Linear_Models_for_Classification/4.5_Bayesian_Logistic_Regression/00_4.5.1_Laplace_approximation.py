"""PRML 4.5.1: Laplace approximation to Bayesian logistic regression."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.bayesian_classification import BayesianLogisticLaplace


if __name__ == "__main__":
    rng = np.random.default_rng(14)
    X = rng.normal(size=(90, 2))
    p = 1 / (1 + np.exp(-(.2 + 1.3 * X[:, 0] - .8 * X[:, 1])))
    y = rng.binomial(1, p)
    model = BayesianLogisticLaplace(alpha=1.5).fit(X, y)
    print("MAP parameter vector:", model.w_map_)
    print("Laplace posterior covariance:\n", model.covariance_)
    print("negative log posterior start/end:",
          model.objective_history_[0], model.objective_history_[-1])
