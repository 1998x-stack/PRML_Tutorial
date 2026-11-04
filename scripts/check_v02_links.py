#!/usr/bin/env python3
"""Check the Markdown file links in the revised V0.2 lessons and status page.

This is deliberately a scoped checker; it is NOT a claim of repo-wide link
validity. It does not fetch internet URLs.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

LESSONS = [
    "3_Linear_Models_for_Regression/3.1_Linear_Basis_Function_Models/00_3.1.1_Maximum_likelihood_and_least_squares.md",
    "3_Linear_Models_for_Regression/3.3_Bayesian_Linear_Regression/00_3.3.1_Parameter_distribution.md",
    "3_Linear_Models_for_Regression/3.3_Bayesian_Linear_Regression/01_3.3.2_Predictive_distribution.md",
    "3_Linear_Models_for_Regression/3.5_The_Evidence_Approximation/00_3.5.1_Evaluation_of_the_evidence_function.md",
    "4_Linear_Models_for_Classification/4.2_Probabilistic_Generative_Models/00_4.2.1_Continuous_inputs.md",
    "4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models/01_4.3.2_Logistic_regression.md",
    "4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models/02_4.3.3_Iterative_reweighted_least_squares.md",
    "4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models/03_4.3.4_Multiclass_logistic_regression.md",
    "4_Linear_Models_for_Classification/4.4_The_Laplace_Approximation/00_4.4.1_Model_comparison_and_BIC.md",
    "4_Linear_Models_for_Classification/4.5_Bayesian_Logistic_Regression/00_4.5.1_Laplace_approximation.md",
    "4_Linear_Models_for_Classification/4.5_Bayesian_Logistic_Regression/01_4.5.2_Predictive_distribution.md",
    "docs/V0_2_SUPERVISED_LEARNING.md",
]

MD_LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")


def check() -> list[str]:
    problems = []
    for relative in LESSONS:
        path = ROOT / relative
        if not path.is_file():
            problems.append(f"missing lesson: {relative}")
            continue
        for ref in MD_LINK.findall(path.read_text(encoding="utf-8")):
            url = ref.strip().split()[0].strip("<>")
            if url.startswith(("https://", "http://", "mailto:", "#")):
                continue
            local_path = unquote(url.split("#", 1)[0])
            if local_path and not (path.parent / local_path).exists():
                problems.append(f"broken link: {relative} -> {url}")
    return problems


if __name__ == "__main__":
    errors = check()
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"OK: checked local links in {len(LESSONS)} V0.2 Markdown documents.")
