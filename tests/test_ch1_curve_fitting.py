"""Numerical and input-contract tests for PRML 1.2.5 (NumPy only)."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


LESSON = (
    Path(__file__).resolve().parents[1]
    / "1_Introduction"
    / "1.2_Probability_Theory"
    / "04_1.2.5_Curve_fitting_re_visited.py"
)
SPEC = importlib.util.spec_from_file_location("curve_fitting_lesson", LESSON)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Model = MODULE.PolynomialCurveFitting


class PolynomialCurveFittingTest(unittest.TestCase):
    def test_recovers_known_polynomial(self):
        x = np.linspace(-0.9, 0.9, 31)
        target = 1.0 - 2.0 * x + 3.0 * x ** 2
        model = Model(degree=2)
        model.fit(x, target)
        np.testing.assert_allclose(model.coefficients, [1.0, -2.0, 3.0], atol=1e-11)
        self.assertLess(model.calculate_rmse(x, target), 1e-11)

    def test_regularization_does_not_penalize_intercept(self):
        x = np.linspace(-1.0, 1.0, 25)
        target = np.full(x.shape, 4.25)
        model = Model(degree=5, regularization_param=1e5)
        model.fit(x, target)
        self.assertAlmostEqual(model.coefficients[0], 4.25, places=9)
        np.testing.assert_allclose(model.coefficients[1:], 0.0, atol=1e-9)

    def test_rank_deficient_design_is_supported(self):
        x = np.ones(5)
        target = np.full(5, 2.5)
        model = Model(degree=5)
        model.fit(x, target)
        np.testing.assert_allclose(model.predict(x), target, atol=1e-11)

    def test_invalid_inputs_raise_explanatory_errors(self):
        for degree in (-1, 1.5, True):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                Model(degree=degree)
        for lam in (-0.1, float("nan"), float("inf")):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                Model(degree=2, regularization_param=lam)

        model = Model(2)
        with self.assertRaises(ValueError):
            model.predict(np.array([1.0]))
        with self.assertRaises(ValueError):
            model.fit(np.array([0.0, 1.0]), np.array([1.0]))
        with self.assertRaises(ValueError):
            model.fit(np.array([0.0, np.nan]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            model.fit(np.array([[1.0]]), np.array([1.0]))
        with self.assertRaises(ValueError):
            model.fit(np.array([]), np.array([]))

    def test_demonstration_is_deterministic(self):
        import contextlib
        import io

        streams = []
        for _ in range(2):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                MODULE.demonstration(seed=42)
            streams.append(output.getvalue())
        self.assertEqual(streams[0], streams[1])
        self.assertEqual(streams[0].count("degree="), 3)


if __name__ == "__main__":
    unittest.main()
