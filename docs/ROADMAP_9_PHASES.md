# PRML Tutorial: Nine-Phase Curriculum & Engineering Roadmap

> Status: planned; this document is a work breakdown, **not** a completion claim.  
> Baseline: `main` inspected on 2026-11-04. Preserve the existing chapter paths and section numbering until redirects and link checks exist.

## 0. Evidence-backed baseline

The repository currently has 14 chapter directories, **197 Markdown + Python pairs** (394 chapter files), and a directory-builder script `main.py`.

| Chapter | Markdown | Python | Initial diagnosis |
|---|---:|---:|---|
| 1. Introduction | 14 | 14 | Notes developed; most scripts are still tiny stubs |
| 2. Probability Distributions | 16 | 16 | Some substantial examples; mathematical/numerical review required |
| 3. Linear Models for Regression | 14 | 14 | Text/code available; validate correctness and numerical conditioning |
| 4. Linear Models for Classification | 20 | 20 | Mixed implementation depth; verify probabilities and calibration |
| 5. Neural Networks | 26 | 26 | Developed notes; **all 26 Python scripts are very short** |
| 6. Kernel Methods | 10 | 10 | Developed notes; **all 10 Python scripts are very short** |
| 7. Sparse Kernel Machines | 8 | 8 | Partial implementations; remainder are stubs |
| 8. Graphical Models | 18 | 18 | Markdown and Python are short placeholders |
| 9. Mixture Models and EM | 8 | 8 | Markdown and Python are short placeholders |
| 10. Approximate Inference | 19 | 19 | Markdown and Python are short placeholders |
| 11. Sampling Methods | 13 | 13 | Markdown and Python are short placeholders |
| 12. Continuous Latent Variables | 12 | 12 | Markdown and Python are short placeholders |
| 13. Sequential Data | 11 | 11 | Markdown and Python are short placeholders |
| 14. Combining Models | 8 | 8 | Markdown and Python are short placeholders |
| **Total** | **197** | **197** | **89 pairs in chapters 8–14 have very short content** |

This is an inventory, not proof that the larger scripts are correct or runnable. The README claims 198 runnable scripts; reconcile that claim with the actual 197 chapter `.py` files and publish real CI results before asserting universal execution.

**Safety finding:** Running the current `main.py` writes `README.md` and rewrites chapter files. Do not run it as a normal tutorial entrypoint; convert it to an explicit, non-destructive scaffold generator in Phase 1.

## 1. Shared learning-content contract

For each existing `<section>.md` and `<section>.py` pair:

1. **Learning objectives:** 2–4 verifiable outcomes and precise prerequisites.
2. **Mathematical formulation:** explicit random variables, dimensions, assumptions, distributions, and symbol glossary.
3. **Derivation:** derive the key result from stated assumptions, show intermediate transformations, state boundary cases and limits. Never cite an equation without defining its symbols.
4. **Intuition:** explain what changes when model assumptions fail; compare with related methods.
5. **From-scratch algorithm:** NumPy/SciPy reference implementation with clear inputs, outputs, shape validation, deterministic RNG, numerical stability, and no side effects on import.
6. **Reproducible experiment:** fixed seed, synthetic or permissively licensed dataset, CPU-compatible command, metric(s), expected behavior, and a meaningful plot only where it clarifies the phenomenon.
7. **Verification:** unit test(s) for invariant/shape/edge cases and independent reference test (closed form, finite differences, or a standard library).
8. **Learning checks:** at least two original exercises including one computational exercise; solutions or validation criteria.
9. **References & credit:** distinguish original exposition from textbook references; avoid reproducing copyrighted book prose/figures.

A section is **done** only when its mathematical claim is supported by a verified experiment or explicitly labelled as theoretical, and when tests and links pass. See [CHAPTER_QUALITY_STANDARD.md](CHAPTER_QUALITY_STANDARD.md).

## 2. Nine phases

### Phase 1 — Probability foundations & safe repository baseline (chapters 1–2, 30 pairs)

**Objectives:** make this project safe to run and mathematically precise at the beginning of the learning path.

- Chapter 1: polynomial curve fitting; probability densities, expectation/covariance and Bayes' rule; Gaussian conditioning; model selection; decision-theoretic loss; entropy/KL/MI.
- Chapter 2: Bernoulli/Beta, categorical/Dirichlet, Gaussian conditioning and marginalization, exponential families, conjugate priors, KDE and nearest-neighbor density estimation.
- Target examples: `1_Introduction/1.2_Probability_Theory/04_1.2.5_Curve_fitting_re_visited.*`, `2_Probability_Distributions/2.3_The_Gaussian_Distribution/00_2.3.1_Conditional_Gaussian_distributions.*`.
- Numeric quality gates: prefer `lstsq`/`solve` to forming explicit inverse or unstable normal equations; exclude intercept from L2 penalty by an explicit choice; check covariance symmetry/positive definiteness.
- Repository: make `main.py` opt-in/non-destructive, introduce `pyproject.toml`, pinned lock approach, `tests/`, `scripts/`, and representative smoke tests. Audit full inventory before further changes.
- **Exit:** 30 paired lessons substantively reviewed; key derivations dimensionally checked; fixed-seed examples and chapter-specific tests pass; README claims match evidence.

### Phase 2 — Supervised probabilistic learning (chapters 3–4, 34 pairs)

- Chapter 3: least-squares geometry, ridge/regularized solutions, bias–variance, Bayesian posterior/predictive distributions, model evidence and effective degrees of freedom.
- Chapter 4: linear discriminants, LDA/QDA, softmax/logistic regression, IRLS, Laplace approximation, Bayesian logistic regression.
- Key correctness tests: gradients versus finite differences; Bayesian posterior versus closed form; log-sum-exp stability; multiclass probabilities sum to 1; calibrations measured on held-out data.
- **Exit:** 34 sections trace formulas to code, with numerical and statistical validation.

### Phase 3 — Neural networks from first principles (chapter 5, 26 pairs)

- Build an instructional `Linear -> activation -> loss -> backprop` stack using NumPy, with optional PyTorch *only as a reference comparator*.
- Derive chain rule and backprop; test Jacobians and gradients by central differences; compare optimizers, regularization and Hessian-vector products.
- For CNNs, distinguish equivariance from invariance, test convolution weight sharing and padding behavior; document computational complexity.
- Target example: `5_Neural_Networks/5.5_Regularization_in_Neural_Networks/05_5.5.6_Convolutional_networks.*` (current Python file is just a stub).
- **Exit:** 26 functioning lessons; a reproducible gradient-check matrix and a small CPU training demonstration.

### Phase 4 — Kernels and sparse predictors (chapters 6–7, 18 pairs)

- Chapter 6: dual formulations, valid positive-semidefinite kernels, RBF, Gaussian-process regression and classification, marginal likelihood and ARD.
- Chapter 7: maximum-margin classifiers, primal/dual SVM, support vectors, regression SVM and relevance vector machines.
- Test Gram matrix symmetry/PSD, GP posterior covariance, jitter behavior, optimization/KKT conditions and sparsity differences.
- **Exit:** all 18 lessons include implementations or justified analytic examples and validated benchmarks.

### Phase 5 — Graphical models, clustering and EM (chapters 8–9, 26 pairs)

- Chapter 8: Bayesian networks, conditional independence and d-separation, MRFs, factor graphs, sum-product/max-sum and loopy belief propagation.
- Chapter 9: k-means, Gaussian mixture EM, EM lower bound, responsibilities, convergence and degenerate covariance handling.
- Build tiny enumerable graphical models as ground truth for message passing; compare EM likelihoods over iterations without overstating convergence to a global optimum.
- Target example: `8_Graphical_Models/8.2_Conditional_Independence/01_8.2.2_D_separation.*` (both stubs).
- **Exit:** 26 placeholders replaced with independent explanations, reference algorithms and executable tests.

### Phase 6 — Variational and approximate inference (chapter 10, 19 pairs)

- Derive ELBO from Jensen's inequality and the KL identity; explain coordinate-ascent mean-field updates, variational GMM, variational linear/logistic regression and expectation propagation.
- Track ELBO against optimization iteration on models where monotonic improvement is expected; distinguish approximations from exact posteriors and use toy posterior checks.
- Target example: `10_Approximate_Inference/10.1_Variational_Inference/00_10.1.1_Factorized_distributions.*` (both stubs).
- **Exit:** 19 lessons; tests compare analytic/simple exact inference against approximations and document failure modes.

### Phase 7 — Monte Carlo inference and continuous latent variables (chapters 11–12, 25 pairs)

- Chapter 11: rejection/importance sampling, ESS, Metropolis–Hastings, Gibbs, slice sampling and Hamiltonian Monte Carlo; diagnose autocorrelation, burn-in/warm-up and Monte Carlo error.
- Chapter 12: PCA (variance/reconstruction equivalence), probabilistic PCA, factor analysis, kernel PCA, ICA and nonlinear manifolds.
- Test MCMC stationary statistics on tractable distributions; PCA subspaces versus SVD; account for sign and rotation non-identifiability.
- **Exit:** 25 placeholders replaced; quantified approximation error and numerical diagnostics are included.

### Phase 8 — Sequential models and ensembles (chapters 13–14, 19 pairs)

- Chapter 13: Markov chains, forward–backward, Viterbi, Baum–Welch, linear dynamical systems, Kalman filtering/smoothing and particle filters.
- Chapter 14: Bayesian model averaging, committees, AdaBoost and exponential loss, trees, mixture-of-experts and gating networks.
- Test HMM probabilities against brute-force path enumeration; log-domain numerical stability; Kalman predictions versus simulation; boosting sample-weight normalization and training curves.
- **Exit:** 19 placeholders replaced; end-to-end sequential and ensemble examples reproducible on CPU.

### Phase 9 — Global verification, curriculum UX and release (all 14 chapters, 197 pairs)

- Normalize section navigation, notation index, prerequisites and learning paths; correct typos and broken links; keep old section URLs stable or provide redirects.
- Introduce/finish `pytest`, `ruff`, type checks where useful, deterministic smoke tests, CI per chapter and full-suite checks on supported Python versions.
- Publish a generated coverage manifest distinguishing `stub`, `draft`, `validated` and `reviewed`; no self-declared "complete" without evidence.
- Build an optional static documentation site with readable maths, search and diagrams; treat GitHub Pages publishing as a separate verified deployment.
- Conduct a content review for mathematical accuracy, citations and figure licensing; check notebook/script dependencies and headless runs.
- **Exit:** all 197 chapter pairs independently reviewed, documented, reproducibly verified, linked from a clean README, and accompanied by published evidence.

## 3. Implementation rules

- **Branch & PR isolation:** one phase per branch/PR (or smaller chapter PRs) with `main` protected; do not merge while CI is red.
- **No bulk filler:** do not make every lesson "complete" by copying a generic explanation into it. Each section needs its own derivation, experiment and tests.
- **Preserve links:** avoid mass-renaming chapter paths before inventorying incoming links and redirects.
- **Test quality not prose length:** a correct, concise proof or code may be better than a lengthy narrative. Manual reviewers determine scientific correctness.
- **Numerical hygiene:** no matrix inversion when `solve` suffices; explain regularization, Cholesky/jitter and log-domain computation as appropriate.
- **Reproducibility:** default headless/CPU, fixed seed, bounded runtime, deterministic small datasets; store only licensed assets.
- **Truthful reporting:** make status checks separate from generated content. Never claim all scripts run merely because files exist.

## 4. Phase PR checklist

- [ ] Scoped inventory and before/after file counts recorded
- [ ] Every modified derivation checked with dimensions/assumptions
- [ ] Each implementation has deterministic expected behavior
- [ ] Edge-case and numeric-stability tests added
- [ ] Representative reference comparison recorded
- [ ] Documentation links and commands checked
- [ ] Independent reviewer approves math/code/educational accuracy
- [ ] Actual CI run and final commit SHA linked in PR

## 5. Risks already confirmed by inspection

1. `main.py` is a **destructive scaffold generator** when launched in a populated repository.
2. README makes a universal `198`-script execution claim without a visible test pipeline; actual chapter pair count is 197.
3. Several first-half lessons contain explanatory text without a substantive paired implementation.
4. The second half is mostly directory scaffolding, not teaching material.
5. Some existing implementations use `np.linalg.inv` or equations through `X.T @ X` where stability deserves an explicit review.

This document is a plan based on repository inspection. **No phase is marked completed by creating this roadmap.**
