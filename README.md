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
- release/response_item_audit.csv -- exact 58-row gate and I/M/L crosswalk.
- [Release thesis-final-2026-08-01-r4](https://github.com/sahandgit/gaussian_process_guided_mapping_quantumclassical_liouville_dynamics/releases/tag/thesis-final-2026-08-01-r4) -- frozen numerical-evidence archive,
  checksums, manifests, clean-room verification, environment, and complete
  downloadable closure.

Raw trajectory arrays are kept in the release asset rather than ordinary Git
history. This avoids GitHub's per-file limits while preserving exact retrieval
through a versioned release.

The versioned GitHub release is public and immutable by tag convention, but it
is not a DOI or an institutional persistent identifier. No DOI has been
assigned.

## Clean source build

The repository/tag archive is the self-contained source package. Do not detach
`thesis/Thesis.tex` from its sibling bibliography, class, generated tables, and
figures. From a clean extraction of the repository archive, with Tectonic on
`PATH`, compile both documents as follows:

    cd thesis
    tectonic Thesis.tex
    cd ..
    tectonic Reviewer_Response.tex

All LaTeX inputs resolve within the extracted repository root. The release
asset `CLEAN_ROOM_VERIFICATION.json` records a public download, checksum,
extraction, manifest-presence, embedded-checksum, and clean-compilation test.

## Reproduce the accepted evidence

    python -m pip install -r requirements.txt
    python -m pytest -q tests/test_pipeline_core.py tests/test_math_expressions.py tests/test_master_table.py tests/test_reviewer_closure.py tests/test_regularization_selection.py tests/test_thesis_modules.py tests/test_run_cli_contract.py
    python audit/reviewer_final_closure.py --mode analyze
    python audit/reviewer_final_closure.py --mode verify
    python audit/final_acceptance_check.py --stage final

The final evidence inventory is 24 paired time-step configurations (48
individual PBME/MIDPOINT method executions) using seeds 11, 29, 47, and 73.
The absolute-plus-relative numerical-noise rule and all rejection reasons are
stored in the time-step and reference CSVs. For both stochastic moving-cloud
methods, PBME and MIDPOINT, the time-step hierarchy is numerical floor, finite
output, physical admissibility, and then same-seed paired contraction. Raw
cross-seed observable spread is retained as a descriptive cloud-variability
diagnostic and is not used as an order or uncertainty gate.

Cloud-size decisions are hierarchical: numerical resolution is checked first,
then physical admissibility (including negative signed central second moments),
and only then seed dispersion. The low-momentum grid-QCLE results fail the
stated three-level reference-tolerance screen for six observables and are used
only as numerical-sensitivity references. The decisive controlled negative
conclusion instead rests on TDSE benchmarking, raw-versus-projected diagnostics,
and independent replication.

## Environment

Python 3.10+ is recommended. Exact captured versions are in
final_reviewer_closure/environment.json and
final_reviewer_closure/environment/pip_freeze.txt.

## Citation and license

See CITATION.cff. Code is MIT licensed. Thesis text and numerical data retain
their stated scholarly authorship and citation requirements.
