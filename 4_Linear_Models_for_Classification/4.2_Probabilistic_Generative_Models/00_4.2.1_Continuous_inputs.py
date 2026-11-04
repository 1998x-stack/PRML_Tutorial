"""PRML 4.2.1: shared-covariance Gaussian generative classifier (LDA).

Training: estimate class means, pooled covariance and class priors.
Prediction: apply Bayes rule in the log domain to obtain posteriors.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from prml.supervised import GaussianDiscriminant


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    X = np.vstack((rng.normal(loc=-1.5, size=(40, 2)),
                   rng.normal(loc=1.5, size=(40, 2))))
    y = np.array(["negative"] * 40 + ["positive"] * 40)
    clf = GaussianDiscriminant(jitter=1e-6).fit(X, y)
    print("class labels:", clf.classes_)
    print("training accuracy:", np.mean(clf.predict(X) == y))
    print("P(class | first sample):", clf.predict_proba(X[:1]))
