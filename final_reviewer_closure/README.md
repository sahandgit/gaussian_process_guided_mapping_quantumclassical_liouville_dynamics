# Final reviewer-closure evidence

This directory is generated from the final GP/RKHS--MInt--QCLE thesis
pipeline. Reused simulations are accepted only after their manifest, expected
physical endpoint, finite stored arrays, and PBME/MIDPOINT cloud pairing have
been verified.

## Public record

Repository:
https://github.com/sahandgit/gaussian_process_guided_mapping_quantumclassical_liouville_dynamics

Release:
https://github.com/sahandgit/gaussian_process_guided_mapping_quantumclassical_liouville_dynamics/releases/tag/thesis-final-2026-08-01

The release asset frozen_numerical_evidence_payload.zip contains this complete
directory, including the raw accepted trajectory arrays that are omitted from
ordinary Git history. PAYLOAD_SHA256SUMS.csv inside the ZIP records the size
and SHA-256 digest of every archived file.

## Reproduce the analysis

From the repository root:

    python -m pip install -r requirements.txt
    python pipeline/reviewer_final_closure.py --mode analyze
    python pipeline/reviewer_final_closure.py --mode verify
    python audit/final_acceptance_check.py --stage final

The six focused verification tests are:

    python -m pytest -q tests/test_pipeline_core.py tests/test_math_expressions.py tests/test_master_table.py tests/test_reviewer_closure.py tests/test_regularization_selection.py tests/test_thesis_modules.py

The analysis reads the accepted run artifacts; it does not modify or rerun
their dynamics. To regenerate the numerical campaign from scratch, inspect
pipeline/reviewer_final_closure.py --help and execute its plan and execute
modes before analyze and verify.

## Evidence map

- FINAL_RUN_MANIFEST.json: accepted job inventory and source hashes.
- validation_inventory.csv: paired and individual method-run counts.
- timestep/timestep_run_by_run.csv: three endpoint values, both differences,
  numerical-noise floor, seed guard, observed order, and final verdict.
- reference_tdse/tdse_three_level.csv and
  reference_grid_qcle/qcle_three_level.csv: exact three-level settings and
  guarded order reasons.
- reference_settings_by_method_and_momentum.csv: eight exact Table 6.12/F.1
  configuration rows.
- TABLE_DATA_CROSSWALK.csv and figures/FIGURE_DATA_CROSSWALK.csv: displayed
  artifact to full-precision source mapping.
- environment.json and environment/pip_freeze.txt: platform and dependencies.
- the per-campaign and per-run JSON manifests: inputs, endpoints, and hashes.

Independent support clouds are never interpreted as deterministic convergence.
Rows below the declared absolute-plus-relative numerical-noise threshold are
labelled roundoff- or saturation-limited and are not assigned an order.
