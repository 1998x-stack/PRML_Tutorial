"""PRML 4.5.2: integrate logistic probabilities under Laplace posterior."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.bayesian_classification import BayesianLogisticLaplace
from prml.supervised import sigmoid


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    X = rng.normal(size=(70, 1))
    p = sigmoid(.2 + 1.6 * X[:, 0])
    y = rng.binomial(1, p)
    model = BayesianLogisticLaplace(alpha=1.).fit(X, y)
    x_new = np.array([[-2.], [0.], [2.]])
    plug_in = sigmoid(np.c_[np.ones(len(x_new)), x_new] @ model.w_map_)
    integrated = model.predict_proba(x_new)[:, 1]
    print("plug-in probabilities:", plug_in)
    print("posterior predictive probabilities (Gauss-Hermite):", integrated)
    print("latent logit variance:", model.logit_mean_variance(x_new)[1])
