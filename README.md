# GP/RKHS--MInt--QCLE thesis pipeline

This is the verified final repository for:

**Gaussian-Process Reconstruction of the Mapping-QCLE Excess Term: A
Moving-Cloud Formulation and Failure Analysis**

The scientific result is deliberately negative. The tested product-GP
moving-cloud MIDPOINT discretization does not satisfy the joint operator,
projection, stochastic-stability, and raw-conservation requirements and does
not demonstrate systematic improvement over PBME. This claim is restricted to
the tested discretization, not the continuum QCLE excess term.

## Final artifacts

- thesis/Thesis.pdf -- final thesis.
- Reviewer_Response.pdf -- point-by-point examiner response.
- pipeline/ -- exact flat source used by the final analysis.
- src/gp_mint_qcle/ -- installable package mirror of the scientific modules.
- audit/ -- evidence generation, verification, and document-build scripts.
- final_reviewer_closure/ -- compact generated CSV/table/figure evidence.
- [Release thesis-final-2026-08-01](https://github.com/sahandgit/gaussian_process_guided_mapping_quantumclassical_liouville_dynamics/releases/tag/thesis-final-2026-08-01) -- frozen numerical-evidence archive,
  checksums, manifests, environment, and complete downloadable closure.

Raw trajectory arrays are kept in the release asset rather than ordinary Git
history. This avoids GitHub's per-file limits while preserving exact retrieval
through a versioned release.

## Reproduce the accepted evidence

    python -m pip install -r requirements.txt
    python -m pytest -q tests/test_pipeline_core.py tests/test_math_expressions.py tests/test_master_table.py tests/test_reviewer_closure.py tests/test_regularization_selection.py tests/test_thesis_modules.py
    python audit/reviewer_final_closure.py --mode analyze
    python audit/reviewer_final_closure.py --mode verify
    python audit/final_acceptance_check.py --stage final

The final evidence inventory is 24 paired time-step configurations (48
individual PBME/MIDPOINT method executions) using seeds 11, 29, 47, and 73.
The absolute-plus-relative numerical-noise rule and all rejection reasons are
stored in the time-step and reference CSVs.

## Environment

Python 3.10+ is recommended. Exact captured versions are in
final_reviewer_closure/environment.json and
final_reviewer_closure/environment/pip_freeze.txt.

## Citation and license

See CITATION.cff. Code is MIT licensed. Thesis text and numerical data retain
their stated scholarly authorship and citation requirements.
