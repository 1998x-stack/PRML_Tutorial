"""V0.2 closed-form, numerical, stability, and direct-execution tests."""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from prml.supervised import (
    BayesianLinearRegression, BinaryLogisticRegression, GaussianDiscriminant,
    IRLSLogisticRegression, LinearLeastSquares, MulticlassLogisticRegression,
    bic, sigmoid, softmax,
)

ROOT = Path(__file__).resolve().parents[1]


class RegressionTests(unittest.TestCase):
    def test_least_squares_exact_and_collinear(self):
        x = np.array([[-2, 1], [0, 3], [3, 0], [4, 4], [-1, -3.]])
        y = 4 + 2 * x[:, 0] - 3 * x[:, 1]
        model = LinearLeastSquares().fit(x, y)
        np.testing.assert_allclose(model.weights, [4, 2, -3], atol=1e-12)
        np.testing.assert_allclose(model.calculate_residuals(x, y), 0, atol=1e-12)
        x2 = np.ones((5, 2))
        np.testing.assert_allclose(LinearLeastSquares().fit(x2, np.full(5, 3)).predict(x2), 3)
        with self.assertRaises(ValueError):
            LinearLeastSquares().fit(x, y[:2])
        with self.assertRaises(ValueError):
            LinearLeastSquares().predict(x)

    def test_bayes_posterior_and_predictive(self):
        x = np.array([[-1], [0], [1], [2.]])
        y = np.array([-1.2, 0.4, 2.1, 3.2])
        phi = np.column_stack((np.ones(len(x)), x))
        precision = 2 * np.eye(2) + 5 * phi.T @ phi
        expected_cov = np.linalg.inv(precision)  # independent test oracle
        expected_mean = 5 * expected_cov @ phi.T @ y
        model = BayesianLinearRegression(2, 5).fit(x, y)
        np.testing.assert_allclose(model.m_N, expected_mean, atol=1e-12)
        np.testing.assert_allclose(model.S_N, expected_cov, atol=1e-12)
        mean, variance = model.predict(x)
        np.testing.assert_allclose(mean, phi @ expected_mean)
        np.testing.assert_allclose(variance, 0.2 + np.einsum("ni,ij,nj->n", phi, expected_cov, phi))

    def test_log_evidence_independent_covariance_oracle(self):
        x = np.array([[-1.0], [0.5], [2.0]])
        y = np.array([0.2, -0.6, 2.1])
        phi = np.column_stack((np.ones(len(x)), x))
        C = np.eye(len(x)) / 8 + phi @ phi.T / 1.5
        _, logdet = np.linalg.slogdet(C)
        oracle = -.5 * (len(y) * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(C, y))
        self.assertAlmostEqual(BayesianLinearRegression(1.5, 8).log_evidence(x, y), oracle, places=11)

    def test_bad_hyperparameters(self):
        for a, b in [(0, 1), (1, -1), (float("nan"), 1)]:
            with self.assertRaises(ValueError):
                BayesianLinearRegression(a, b)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.normal(size=(100, 2))
        p = sigmoid(0.4 + 1.4 * self.X[:, 0] - 0.7 * self.X[:, 1])
        self.y = rng.binomial(1, p).astype(float)

    def test_sigmoid_and_softmax_extremes(self):
        self.assertTrue(np.isfinite(sigmoid(np.array([-1000., 0, 1000.]))).all())
        z = np.array([[10000, 10001, 9999], [-10000, -9999, -10001]])
        np.testing.assert_allclose(softmax(z).sum(axis=1), 1, atol=1e-13)
        np.testing.assert_allclose(softmax(z), softmax(z + 300), atol=1e-13)

    def test_binary_gradient_finite_difference(self):
        model = BinaryLogisticRegression(regularization=0.2)
        w = np.array([0.1, -0.3, 0.4])
        _, grad = model.loss_and_grad(self.X, self.y, w)
        eps = 1e-6
        numeric = []
        for j in range(len(w)):
            d = np.zeros_like(w)
            d[j] = eps
            numeric.append((model.loss_and_grad(self.X, self.y, w + d)[0] -
                            model.loss_and_grad(self.X, self.y, w - d)[0]) / (2 * eps))
        np.testing.assert_allclose(grad, numeric, atol=2e-8, rtol=1e-6)

    def test_binary_and_newton_agree(self):
        gd = BinaryLogisticRegression(n_iter=1800, tol=1e-10, regularization=.02).fit(self.X, self.y)
        nt = IRLSLogisticRegression(n_iter=80, tol=1e-11, regularization=.02).fit(self.X, self.y)
        self.assertLess(gd.cost_[-1], gd.cost_[0])
        self.assertTrue(np.all(np.diff(gd.cost_) <= 1e-12))
        self.assertTrue(np.all(np.diff(nt.cost_) <= 1e-12))
        np.testing.assert_allclose(gd.w_, nt.w_, atol=3e-4)
        np.testing.assert_allclose(nt.predict_proba(self.X).sum(axis=1), 1)
        with self.assertRaises(ValueError):
            gd.fit(self.X, self.y + .2)

    def test_softmax_gradient_and_classes(self):
        rng = np.random.default_rng(123)
        X = np.vstack((rng.normal(size=(30, 2)) + [2, 0],
                       rng.normal(size=(30, 2)) + [-2, 0],
                       rng.normal(size=(30, 2)) + [0, 3]))
        y = np.array([20] * 30 + [10] * 30 + [99] * 30)
        model = MulticlassLogisticRegression(regularization=.1)
        phi = np.column_stack((np.ones(len(X)), X))
        _, inds = np.unique(y, return_inverse=True)
        W = np.arange(9, dtype=float).reshape(3, 3) / 20
        _, grad = model._objective(phi, inds, W)
        eps = 1e-6
        num = np.zeros_like(W)
        for i in range(3):
            for j in range(3):
                d = np.zeros_like(W)
                d[i, j] = eps
                num[i, j] = (model._objective(phi, inds, W + d)[0] -
                             model._objective(phi, inds, W - d)[0]) / (2 * eps)
        np.testing.assert_allclose(grad, num, atol=2e-8)
        model.fit(X, y)
        np.testing.assert_array_equal(model.classes_, [10, 20, 99])
        self.assertGreater(np.mean(model.predict(X) == y), .8)
        self.assertTrue(np.all(np.diff(model.cost_) <= 1e-12))
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
        with self.assertRaises(ValueError):
            model.fit(X, np.full(len(X), 10))

    def test_lda_and_singular_covariance(self):
        x = np.array([[0., 0.], [0., 0.], [1., 1.], [1., 1.]])
        y = np.array(["A", "A", "B", "B"])
        model = GaussianDiscriminant(jitter=1e-4).fit(x, y)
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)
        self.assertTrue(np.linalg.eigvalsh(model.covariance_).min() > 0)
        np.testing.assert_array_equal(model.predict(x), y)

    def test_bic(self):
        self.assertAlmostEqual(bic(-10, 3, 100), 20 + 3 * np.log(100))
        for args in [(-10, -1, 100), (float("inf"), 1, 10), (-10, 1, 0)]:
            with self.assertRaises(ValueError):
                bic(*args)


class DirectScriptTests(unittest.TestCase):
    paths = [
        "3_Linear_Models_for_Regression/3.1_Linear_Basis_Function_Models/00_3.1.1_Maximum_likelihood_and_least_squares.py",
        "3_Linear_Models_for_Regression/3.3_Bayesian_Linear_Regression/00_3.3.1_Parameter_distribution.py",
        "3_Linear_Models_for_Regression/3.3_Bayesian_Linear_Regression/01_3.3.2_Predictive_distribution.py",
        "3_Linear_Models_for_Regression/3.5_The_Evidence_Approximation/00_3.5.1_Evaluation_of_the_evidence_function.py",
        "4_Linear_Models_for_Classification/4.2_Probabilistic_Generative_Models/00_4.2.1_Continuous_inputs.py",
        "4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models/01_4.3.2_Logistic_regression.py",
        "4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models/02_4.3.3_Iterative_reweighted_least_squares.py",
        "4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models/03_4.3.4_Multiclass_logistic_regression.py",
        "4_Linear_Models_for_Classification/4.4_The_Laplace_Approximation/00_4.4.1_Model_comparison_and_BIC.py",
        "4_Linear_Models_for_Classification/4.5_Bayesian_Logistic_Regression/00_4.5.1_Laplace_approximation.py",
        "4_Linear_Models_for_Classification/4.5_Bayesian_Logistic_Regression/01_4.5.2_Predictive_distribution.py",
    ]

    def test_all_nine_direct_scripts(self):
        for path in self.paths:
            with self.subTest(path=path):
                run = subprocess.run([sys.executable, str(ROOT / path)], cwd=ROOT,
                                     capture_output=True, text=True, timeout=40)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertTrue(run.stdout.strip())



class LaplaceClassificationTests(unittest.TestCase):
    def setUp(self):
        from prml.bayesian_classification import BayesianLogisticLaplace
        self.Cls = BayesianLogisticLaplace
        rng = np.random.default_rng(2026)
        self.X = rng.normal(size=(75, 2))
        probs = sigmoid(0.3 + 0.8 * self.X[:, 0] - 1.1 * self.X[:, 1])
        self.y = rng.binomial(1, probs).astype(float)

    def test_map_gradient_and_hessian_numerically(self):
        model = self.Cls(alpha=0.9)
        w = np.array([0.2, -0.5, 0.1])
        _, grad, hess = model.objective_gradient_hessian(self.X, self.y, w)
        eps = 1e-5
        numerical_grad = np.empty_like(grad)
        numerical_hess = np.empty_like(hess)
        for j in range(len(w)):
            bump = np.zeros_like(w)
            bump[j] = eps
            lp, gp, _ = model.objective_gradient_hessian(self.X, self.y, w + bump)
            lm, gm, _ = model.objective_gradient_hessian(self.X, self.y, w - bump)
            numerical_grad[j] = (lp - lm) / (2 * eps)
            numerical_hess[:, j] = (gp - gm) / (2 * eps)
        np.testing.assert_allclose(grad, numerical_grad, atol=2e-8)
        np.testing.assert_allclose(hess, numerical_hess, atol=2e-8)

    def test_positive_covariance_and_probabilities(self):
        model = self.Cls(alpha=1.2).fit(self.X, self.y)
        self.assertTrue(np.linalg.eigvalsh(model.covariance_).min() > 0)
        self.assertTrue(np.all(np.diff(model.objective_history_) <= 1e-9))
        self.assertLess(model.objective_history_[-1], model.objective_history_[0])
        probs = model.predict_proba(self.X[:10])
        self.assertEqual(probs.shape, (10, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-12)
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))

    def test_gaussian_hermite_quadrature_consistency(self):
        model = self.Cls(alpha=2.0).fit(self.X, self.y)
        test_x = np.array([[-2., .3], [0., 0.], [1.5, -1.]])
        p32 = model.predict_proba(test_x, quadrature_points=32)
        p64 = model.predict_proba(test_x, quadrature_points=64)
        np.testing.assert_allclose(p32, p64, atol=1e-6)
        means, vars_ = model.logit_mean_variance(test_x)
        self.assertTrue(np.all(vars_ >= 0))
        self.assertEqual(means.shape, (3,))

    def test_priors_and_input_validation(self):
        for alpha in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                self.Cls(alpha=alpha)
        model = self.Cls().fit(self.X, self.y)
        for n in (3, 129, 2.5):
            with self.assertRaises(ValueError):
                model.predict_proba(self.X[:1], quadrature_points=n)
        with self.assertRaises(ValueError):
            model.predict_proba(np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
