# V0.2 Chapter 3–4 Development Status

> The repository's 9-phase curriculum plan lists 34 lesson pairs for chapters 3–4. This file distinguishes completed **implementation work** from final chapter accreditation.

## Initial technical audit

1. Chapter 3 OLS used pseudo-inverse of normal equations; Bayesian regression explicitly inverted posterior precision; evidence used log(det(A)).
2. Chapter 4 logistic regression used a potentially overflowing sigmoid and log(p), IRLS materialized a dense diagonal matrix, softmax tried an incorrect Hessian shortcut.
3. Chapter 4 contained placeholders for Gaussian generative classification, model comparison/BIC, and the two Bayesian logistic sections.
4. These findings are sample-based and do **not** represent a complete correctness audit of every other lesson.

## Reviewed implementation slice

| Subsection | Work delivered | Verification |
|---|---|---|
| 3.1.1 | Gaussian MLE / SVD-based least squares | Exact coefficients; rank deficiency |
| 3.3.1 | Proper Gaussian prior and posterior | Independent closed-form covariance & mean |
| 3.3.2 | Predictive mean and observation variance | Analytic variance expression |
| 3.5.1 | Marginal log evidence via Cholesky | Data-covariance oracle |
| 4.2.1 | Shared-covariance Gaussian generative LDA | Posterior normalization; singular covariance regularization |
| 4.3.2 | Stable binary logistic objective + GD | Gradient finite difference; monotone loss |
| 4.3.3 | Damped IRLS / Newton | Agreement with first-order solver |
| 4.3.4 | Multiclass Softmax cross entropy | Full-matrix gradient; label map; normalization |
| 4.4.1 | Standard unpenalized BIC demonstration | Formula/parameter count edge cases |
| 4.5.1 | Proper Gaussian-prior Laplace MAP | Gradient/Hessian finite differences, SPD covariance |
| 4.5.2 | Laplace posterior predictive quadrature | Quadrature node-count agreement; probability normalization |

**Count:** 11 implemented lesson pairs out of 34 in V0.2. These represent rigorously scoped technical examples. A green CI does not imply every derivation in the rest of the course is correct.

## Architecture and validation

- Portable NumPy reference code lives in [prml/supervised.py](../prml/supervised.py) and [prml/bayesian_classification.py](../prml/bayesian_classification.py).
- Lessons stay in their original chapter paths and can be executed directly from the repository root.
- [tests/test_v02_supervised.py](../tests/test_v02_supervised.py) tests independent numerical oracles, finite differences, reproducible models and direct script execution.
- [the nine-phase roadmap](ROADMAP_9_PHASES.md) remains the source of the rest of the work.

## Remaining V0.2 content

23 pairs in chapters 3–4 still need section-specific implementation, mathematics review and tests. Prioritize 4.2.2 maximum-likelihood fitting, 4.2.3 discrete features, 4.3.1 fixed basis functions, 3.3.3 equivalent kernels, 3.5.2/3.5.3 empirical Bayes hyperparameter evidence, and the discriminant/perceptron sections. Do not collapse distinct tasks into one "finished" badge.

## Phase acceptance

- [ ] All 34 lesson pairs manually reviewed for assumptions, equations and numerical implementation
- [ ] Each substantive section has an executable example and its own correctness checks
- [ ] At least two independent reviewers verify technical pedagogical explanations
- [x] Representative core reference kernels implemented and cross-checked by unit tests
- [x] Existing chapter paths retained
- [ ] Full legacy-script execution verified (not included in this narrow smoke suite)
