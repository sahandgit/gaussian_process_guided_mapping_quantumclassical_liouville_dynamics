Overall status: FAIL
Scientific result: no validated improvement
Thesis compile: FAIL
Reviewer response compile: FAIL
Evidence archive: PASS
Versioned public release URL: https://github.com/sahandgit/gaussian_process_guided_mapping_quantumclassical_liouville_dynamics/releases/tag/thesis-final-2026-08-01-r2
Persistent identifier note: no DOI or institutional persistent identifier has been assigned.
Frozen source/evidence commit: 38164f0fb0de1e3897a01cf56c90d53901bc75d6
Originating development commit: NOT IDENTIFIABLE
Archive SHA-256: 9b6e98221f61993dc8022785295ebe80db2a3f71595ea1349269d7db0627cfb7

# Final Closure Report

## Exact title

Gaussian-Process Reconstruction of the Mapping-QCLE Excess Term: A Moving-Cloud Formulation and Failure Analysis

Abstract word count: 148.

## Environment

See `D:\PhD-Project\GP_RKHS_MINT\GP_RKHS_MINT\Tully_models\GP_MInt_QCLE_Reviewer_Ready_Pipeline_v4\output\fixed_pipeline\final_reviewer_closure\environment.json` and `D:\PhD-Project\GP_RKHS_MINT\GP_RKHS_MINT\Tully_models\GP_MInt_QCLE_Reviewer_Ready_Pipeline_v4\output\fixed_pipeline\final_reviewer_closure\environment\pip_freeze.txt`.

## Completed job counts

manufactured: expected=36, verified=36, missing=0; reference: expected=8, verified=8, missing=0; support: expected=18, verified=18, missing=0; timestep: expected=24, verified=24, missing=0; replication: expected=8, verified=8, missing=0

## Failed/retried history

2 orchestration incident(s) retained: INTERRUPTED_OVERSUBSCRIPTION (The three subprocesses each exposed 21 numerical threads (63 total on 4 logical CPUs). After approximately 12 hours only one atomic result had completed. Exact command lines and parent-child PIDs were verified before terminating this launcher tree.); TEST_INVOCATION_TIMEOUT_RETRY_REQUIRED (The full pytest wrapper reached its 120-second scheduling timeout while three memory-intensive N=2400 fits were active; no assertion result was returned.)

## Ten-gate result

Gates 1–7 and 9–10 are locally closed. Gate 8 is closed when the checksum-bound versioned public release URL is recorded; that URL is `https://github.com/sahandgit/gaussian_process_guided_mapping_quantumclassical_liouville_dynamics/releases/tag/thesis-final-2026-08-01-r2`.

## I/M/L closure totals

I-items: 16/16 closed; M-items: 25/25 closed; L-items: 7/7 closed.

## Major numerical conclusions

- Manufactured operator: 72 query rows from 36 paired policy/support/seed fits; all density, gradient, and operator E1/E2/E-infinity values are finite. The independent-cloud enlargement checks yield 53 rows without monotone decrease and 1 row with monotone decrease. Because the clouds are nonnested, these descriptive checks do not establish deterministic support convergence. For the production-policy off-support operator E1, the three-seed means at N=300, 600, 1200, and 2400 are 0.0207881, 0.0291728, 0.0231755, 0.0232611; the trend is nonmonotone.
- Time step: 128 run/observable rows across four seeds. Of the tested orders, 8 are positive, 3 are zero or negative, and 102 are rejected because the refinement signal does not exceed seed variability. A further 15 rows are roundoff- or saturation-limited under the declared absolute-plus-relative numerical-noise rule. Orders rejected by either guard are not promoted.
- Replication: P0=20: PBME SD=0.0078824, MIDPOINT SD=0.794912, ratio=100.846; P0=100: PBME SD=0.0153607, MIDPOINT SD=0.956375, ratio=62.261. The independent-seed sample size is four, not the trajectory count.
- Tail sensitivity: 16 method/momentum/seed distributions and 160 threshold rows; MIDPOINT max |y/y0|=7.61012e+23, minimum signed ESS=0.0247143. No nontrivial threshold satisfies the negligible-mass rule in 16 distributions.
- References: The controlled-reference evidence contains 32 identified TDSE observable/mode/momentum rows and 32 grid-QCLE rows. Each row prints three values, two successive differences, exact domains and resolved steps; temporal and grid refinement are separated. TDSE P0=20 time orders P0=2.00023, P1=1.99998, R_mean=1.99999, P_mean=1.99999; TDSE P0=100 time orders P0=1.99999, P1=2, R_mean=2, P_mean=2; grid-QCLE P0=20 time orders P0=4.01476, P1=4.01487, R_mean=not computed, P_mean=not computed; grid-QCLE P0=100 time orders P0=4.14354, P1=4.14351, R_mean=not computed, P_mean=not computed. Maximum recorded TDSE spatial-edge mass=1.22388e-24; maximum accepted finest-level grid-QCLE physical-marginal edge mass=0.000591568 (declared tolerance 1e-3).
- PBME/MIDPOINT comparison: 88 paired seed-aggregate error rows; MIDPOINT has a larger error in 6 rows and no resolved difference occurs in 82 rows. Regardless of isolated smaller errors, systematic improvement is not demonstrated unless refinement, seed, conservation, projection, and appreciable-source gates also pass.

## Table-data crosswalk

`D:\PhD-Project\GP_RKHS_MINT\GP_RKHS_MINT\Tully_models\GP_MInt_QCLE_Reviewer_Ready_Pipeline_v4\output\fixed_pipeline\final_reviewer_closure\TABLE_DATA_CROSSWALK.csv`

## Compilation diagnostics

Thesis PDF: `D:\PhD-Project\GP_RKHS_MINT\GP_RKHS_MINT\Tully_models\GP_MInt_QCLE_Reviewer_Ready_Pipeline_v4\output\fixed_pipeline\Thesis\Thesis.pdf`. Reviewer-response PDF: `D:\PhD-Project\GP_RKHS_MINT\GP_RKHS_MINT\Tully_models\GP_MInt_QCLE_Reviewer_Ready_Pipeline_v4\output\fixed_pipeline\Reviewer_Response.pdf`. Rendered-page QA is recorded under `D:\PhD-Project\GP_RKHS_MINT\GP_RKHS_MINT\Tully_models\GP_MInt_QCLE_Reviewer_Ready_Pipeline_v4\output\fixed_pipeline\reviewer_data_audit\pdf_qa`.

## Archive

Frozen numerical-evidence payload SHA-256: `9752f3e16a305225ed30796d799d5a9795d09d8cdebd1ee74648054cf84af87e`

Archive: `D:\PhD-Project\GP_RKHS_MINT\GP_RKHS_MINT\Tully_models\GP_MInt_QCLE_Reviewer_Ready_Pipeline_v4\output\fixed_pipeline\final_reviewer_closure\MSC-THESIS-FINAL-CLOSURE-2026-08-01T111722.zip`

SHA-256: `9b6e98221f61993dc8022785295ebe80db2a3f71595ea1349269d7db0627cfb7`

The ZIP checksum is recorded in this external post-package report; embedding a ZIP's own checksum inside itself would be self-referential.
