# Chapter Quality Standard / 章节质量规范

This is an **acceptance rubric**, not a requirement to produce a fixed number of words. Every lesson must be independently understandable and verifiably correct.

## 1. Lesson template

```markdown
# <chapter.section> <中文主题> / <English topic>

## Learning objectives / 学习目标
- State what a reader can calculate, derive, or implement afterwards.

## Prerequisites / 前置知识
- Link to prior lessons; introduce any nonstandard notation.

## Problem and assumptions / 问题定义和假设
- Data-generating process, dimensions, independence assumptions, model domain.

## Derivation / 数学推导
- Define every variable.
- Give intermediate steps, not just the final equation.
- Explain mathematical and numerical conditions for validity.

## Algorithm / 算法
- Stable pseudocode or concise NumPy implementation.
- Complexity, degenerate inputs, identifiability, stopping criteria.

## Experiment / 可复现实验
- CPU command, dependency version, seed, known expected behavior and metrics.
- State whether output is exact, stochastic or approximate.
- Link to the paired .py program.

## Sanity checks and failure cases / 校验和失效情况
- Compare to a closed-form result, brute force, finite differences, or trusted reference.

## Exercises / 练习
- One conceptual, one mathematical or computational.

## Further reading / 参考文献
- Cite references without reproducing their prose or figures.
```

## 2. Mathematical review rubric

| Criterion | Required evidence |
|---|---|
| Correct definitions | Variables, conditioning events, units, dimensions and supports are defined |
| Derivation | Uses valid equalities and states assumptions (rank, positivity, differentiability, etc.) |
| Boundary conditions | Numerical/analytic behavior at zero counts, singular matrices, extreme logits, small samples |
| Interpretation | Distinguishes posterior, likelihood, predictive, optimization objective and test risk |
| Scientific accuracy | No universal statements about convergence/generalization unless conditions are met |

## 3. Python implementation rubric

- Importing a lesson must not train, write files, print results or open windows.
- Expose small pure functions or well-defined classes; place demonstrations under `if __name__ == "__main__":`.
- Use `numpy.random.default_rng(seed)` or explicit RNG injection; never rely on implicit global randomness in tests.
- Use `scipy.linalg.solve`, `numpy.linalg.lstsq`, `scipy.special.logsumexp` and Cholesky/Jitter where appropriate.
- Fail with meaningful `ValueError` for invalid shape/domain instead of using `assert` for user input validation.
- Favor synthetic, generated or openly licensed small data, no mandatory GPU or network.
- Include function docstrings for inputs/outputs and algorithm limits.

## 4. Evidence & testing rubric

| Level | Meaning | Minimum evidence |
|---|---|---|
| `stub` | Heading, metadata or code scaffold only | Automatic inventory detection (heuristic, not a proof) |
| `draft` | Substantive explanation or implementation exists | Chapter owner acknowledges it is not yet verified |
| `validated` | Independently checked technical correctness | Passing deterministic tests, reference comparison, documented assumptions |
| `reviewed` | Ready for use as a high-quality lesson | Second-pass math, pedagogy, reproducibility and citations review |

Never equate file presence, byte count, or nonzero test execution exit with pedagogical correctness.

Per lesson tests should check:
1. Simple exact/reference examples;
2. Edge cases and numerical stability;
3. Shape/invariant properties;
4. The actual teaching demonstration via a short, headless smoke test;
5. Stochastic behavior with tolerances and fixed seeds, not exact floating-point snapshots.

## 5. Deliverable format for phase PRs

- Changed section inventory and the reasons for every substantial correction.
- Independent math review notes for major formulas.
- Commands actually executed, results, operating system, and versions.
- Backward compatibility for established paths.
- Any known limitation, rather than calling incomplete areas "done".

## 6. Copyright and attribution

This repository is a companion to Christopher M. Bishop's *Pattern Recognition and Machine Learning*. Author original explanations, derivations, figures and experiments. Do not copy substantial textbook content, illustrations or exercise solutions wholesale. Cite source concepts and indicate independent modifications clearly.
