# Pipeline Revision Log Against the Examiner/Reviewer Report

## Executive summary

The uploaded pipeline was normalized into a clean module set, audited, and revised so every numerical item in the reviewer report has an explicit implementation path and reproducible output. The work separates three states that were previously conflated:

- **implemented**: code and output schema exist;
- **executed and verified**: a numerical run has produced metrics;
- **thesis claim ready**: the executed metrics support the wording used in the thesis.

This package completes the implementation layer. Lightweight NumPy/SciPy tests were executed successfully in the available environment. The full GP/operator campaign was not executed here because this runtime does not contain PyTorch or JAX; consequently, this log does not invent convergence, leakage, or manufactured-operator numbers.

## Windows and current-NumPy portability correction

The first packaged test assumed `test_pipeline_core.py` would always remain inside the `tests` directory and used an eagerly evaluated `np.trapz` fallback. Running a copied test directly beside the modules on Windows therefore searched one directory too high, while current NumPy releases that remove `np.trapz` raised an exception before the available `np.trapezoid` implementation could be selected. The test now discovers the module root from file existence, every affected integration call uses a lazy NumPy-version branch, and a top-level `test_pipeline_core.py` launcher supports direct PowerShell execution from any working directory.

## Long-run adaptive-refit singularity correction (v4)

The reported `P0=20` PBME failure at step 4081 was isolated to the adaptive GP breathing optimizer, not the MInt dynamics: raw normalization and physical energy remained conserved immediately before the exception. The previous strong-Wolfe L-BFGS path evaluated unconstrained temporary hyperparameters inside its line search. Those trials bypassed the post-step length-scale/noise bounds, and the fallback restored only `log_lengthscales`, leaving `log_sigma_n` vulnerable to contamination.

The breathing update is now transactional projected L-BFGS. Each outer update is projected into the global bounds and anchor trust region before its objective is evaluated; the accepted state contains both the full length-scale vector and noise parameter. Any non-finite loss, parameter, covariance, or failed Cholesky rejects only that breathing burst, restores the complete last-known-good state, and allows the dynamics step to continue. Cholesky now distinguishes finite round-off from an invalid optimizer state: finite matrices receive monotonically increasing positive jitter up to a documented cap, while non-finite matrices raise immediately and enter the transactional recovery path. Per-step outputs expose `adapt_refit_failed`, a reason code, and the cumulative failure count. A PyTorch regression test injects a failed post-update candidate and asserts exact restoration of both length scales and noise.

## Reviewer-item comparison

| Reviewer item | Uploaded pipeline | Revision implemented | Evidence/output | Current state |
|---|---|---|---|---|
| Exact problem, importance, actual outcome, proof-of-concept claims | Thesis prose reportedly revised; not a Python concern | No scientific claim was expanded by the code changes | Thesis source must retain its proof-of-concept wording | Source-level item; unchanged here |
| Eq. 3.26 projection or leakage | Leakage was discussed but not measured | Four-function real SEO projection at fixed bath anchors; relative (L_2) and absolute RMS residuals | `ReviewerValidation.py projection`; `projection_leakage.json` | Implemented; execute on final snapshots |
| Manufactured (Q[\rho]), on/off support | Operator existed but no analytic manufactured run | Analytic Gaussian-bath × SEO-profile density, exact gradient and (Q); GP/product comparison on support and independent off-support points | `manufactured_operator_metrics.json` | Implemented; requires PyTorch/JAX execution |
| Δt convergence | No controlled refinement campaign | Fixed-endpoint Δt and Δt/2 cases with identical (N)/seed/config | `campaign_plan.json`, `endpoint_metrics.csv`, `campaign_metrics.json` | Implemented; campaign execution required |
| Support refinement | No (N\to2N) campaign | Finer-Δt comparison at (N) and (2N) | Same campaign outputs | Implemented; campaign execution required |
| Independent clouds/seeds | Comparisons relied on one cloud | At least three fixed-config seeds; mean/sample standard deviation reported | Replication block in `campaign_metrics.json` | Implemented; campaign execution required |
| Conservation: raw drift vs self-normalization | One-step/normalized diagnostics could conceal cumulative raw drift | Step-0-referenced raw norm, energy, trace, and mapping-radius drifts plus relative drifts; `lw_*` retained and explicitly labeled self-normalized | `raw_*_drift` arrays in both NPZ files | Implemented in production path |
| KDE vs GP on identical support | A physical 2D cloud KDE was compared with an unconstrained 6D GP mapping integral; focused PBME does not identify that GP off its mapping manifold | One projected-density contract: saved frozen geometric measure, effective labels, identical support, Scott/Silverman bandwidth, grid and raw mass. A sparse 2D GP is conditioned on the all-trajectory KDE field; PBME has an explicit (E_1\le0.02) acceptance rule. The 6D integral is separated as leakage, not physical density | `KDEDensity.ProjectedNuclearGP`; `kde_gp_identical_support.{json,npz}` | Implemented and synthetic PBME regression-tested; execute on final run |
| TDSE/grid-QCLE convergence | Reference solvers existed but convergence was not documented | Two-level TDSE time/grid and QCLE time/support refinements with endpoint differences | `reference_convergence.json` | Implemented; reference execution required |
| Stand-alone figure metadata | Long sentence-like headers and scattered configuration | All visible figure/subplot headers removed; axes, legends and colorbar labels retained; method/time/configuration moved to per-figure JSON sidecars and thesis-ready caption catalog | `THESIS_FIGURE_CAPTIONS.md`, `figure_catalog.json`; AST no-header test | Implemented and tested |
| Common phase-space color scale | Per-method quantiles could visually equalize unequal densities | One symmetric quantile scale shared by all methods in each comparison panel | `Compare_gp_se_qcle.py` | Implemented |
| Excess-term diagnosis | (Q)-only plot; yellow/broken curves; undefined signed-normalized mean could create gaps | Two panels: operator magnitude/source and MIDPOINT−PBME (P_0/P_1)/raw-norm response; solid blue/vermillion/green/black curves; undefined mean removed | `plot_qcle_correction_diagnostics` | Implemented |
| Signed denominator guard for \(\bar Q_y\) | Effectively absolute (10^{-30}) guard; scale unsafe | \(\tau_N=\max(10^{-15},\sqrt{\epsilon}\sum_i|w_i|)\), saved denominator, threshold, and defined flag | `cs_q_weight_denominator*`, `cs_q_weighted_mean_defined` | Implemented |
| Same cloud for PBME/MIDPOINT | Deep copy occurred but was not auditable | SHA-256 content fingerprint in both run manifests and explicit paired comparison contract | `initial_cloud_sha256`, `paired_initial_cloud_sha256` | Implemented |
| GP breathing-policy mismatch | Comments/defaults alternated between frozen, Adam, fixed noise, and adaptive behavior | Production default documented as bounded L-BFGS breathing; σf anchored; σn floats unless explicitly fixed; focused mode's adaptive trigger is documented and recorded | `GP_Density.py`, `run.py`, run manifest | Implemented |
| Product-surrogate moments | `ProductMoments` was imported but absent; product modes could fail | New exact Gaussian-polynomial moment engine; energy/adiabatic quantities use deterministic GH quadrature | `ProductMoments.py`; core tests | Implemented and lightweight-tested |
| Product snapshot plots | Inner modulation \(\mu\) could be plotted as physical \(g\mu\) | Product metadata survives snapshots; static product prediction and 1D/2D marginals include (g); transported off-cloud operations fail explicitly | `Collector.py`, `Visualization.py` | Implemented |
| Operator posterior variance wording | Mean operator was available, posterior derivative variance was not | Every step records `operator_variance_computed=0`; code/documentation cannot silently imply otherwise | NPZ observable | Disclaimer implemented; variance itself remains uncomputed |
| Density-space LOO wording | GP-internal LOO could be mistaken for a physical-density validation | Validation protocol distinguishes GP-internal diagnostics from manufactured/off-support density/operator tests | `VALIDATION_PROTOCOL.md` | Documentation clarified; thesis wording still must match |
| Figure normalization/floor metadata | Not systematically preserved | Full config/model/environment manifest, noise floor, product floor and affected fraction, raw/self-normalized policy | JSON sidecars and run metadata | Implemented |
| Duplicate derivative modules | Two byte-identical editable files | `GPDerivatives.py` is canonical; `GP_Derivatives.py` is a compatibility re-export | Source audit | Implemented |
| Source compile/duplicate definitions | Uploaded files compiled, but runtime package had no automated audit | All Python files compile; no duplicate top-level definitions or duplicate full-file hashes; unit tests included | Test commands below | Verified to available scope |

## File-by-file disposition

| File | Disposition |
|---|---|
| `Collector.py` | Added full run metadata and product/density-difference snapshot round-trip fields. |
| `Compare_gp_se_qcle.py` | Removed every visible figure/subplot header and top method label, replaced yellow/broken method styling, enforced shared phase-space scales, routed trajectory R-P panels through the projected GP contract, and added figure metadata sidecars. |
| `Dynamics.py` | Added reproducibility metadata, raw cumulative drifts, product-floor audit, operator-variance disclaimer, product snapshot metadata, corrected KDE diagnostics to use effective midpoint labels rather than stale raw labels, and exported adaptive-refit rejection diagnostics. |
| `GP_Density.py` | Reconciled breathing implementation/documentation/defaults, connected the product moment engine, and made adaptive refits transaction-safe with bounded projected L-BFGS, complete `(ell, sigma_n)` restoration, and finite-matrix jitter escalation. |
| `GP_DensityDiff.py` | Preserved after syntax/source audit; the explicit product+diff incompatibility remains guarded rather than silently composed. |
| `GPDerivatives.py` | Canonical derivative implementation, preserved after source audit. |
| `GP_Derivatives.py` | Converted to a compatibility re-export to prevent future divergence. |
| `KDEDensity.py` | Added the shared projected-density bandwidth authority and sparse `ProjectedNuclearGP`, deterministic inducing selection, common raw-mass constraint, chunked prediction, 1D analytic marginals, and physical-coordinate training-center recovery. |
| `Mint.py` | Preserved; energy and mapping-radius conservation checked numerically. |
| `Models.py` | Preserved; analytic Tully derivatives checked against centered finite differences. |
| `Monodromy.py` | Preserved after compilation/source audit; full JAX runtime check remains part of `run.py`. |
| `Observables.py` | Added scale-aware denominator guard, guard observability, correct geometric weights in correction statistics, and safe transported-product dispatch. |
| `Operator.py` | Preserved analytic product/vanilla operator; now exercised by the manufactured driver. |
| `qcle_grid_tully.py` | Preserved; trace conservation checked on a small grid and refinement exposed by the validation driver. |
| `run.py` | Added paired-cloud proof, full manifest, product-floor CLI, no-figure campaign mode, consistent policy text, and caption-catalog generation. |
| `Sampling.py` | Preserved after audit; deterministic seeds and sampling metadata are now carried into manifests/campaigns. |
| `Visualization.py` | Removed all visible headers, added a source-level guard against their return, redesigned the excess-term plot, revised palette/line policy, made snapshot plotting independent of eager PyTorch imports, and routed all production low-dimensional marginals through the physical cloud-projected GP instead of the unidentified 6D mapping integral. |
| `ReviewerValidation.py` | Added reviewer-facing convergence, manufactured-operator, projection-leakage, reference, and baseline commands; the baseline imports the same projected-density implementation as the publication figures and records the PBME acceptance decision. |

New files are `ReviewerValidation.py`, `Reproducibility.py`, `ProductMoments.py`, `FigureCatalog.py`, `VALIDATION_PROTOCOL.md`, `requirements.txt`, and `tests/test_pipeline_core.py`.

## Checks executed in this environment

```text
python -m py_compile output/fixed_pipeline/*.py tests/*.py    PASS
AST duplicate top-level definition scan                      PASS (none)
SHA-256 duplicate Python-file scan                            PASS (none)
unittest core suite                                           PASS (11 runnable, 1 PyTorch-only skipped)
synthetic PBME projected-GP/KDE visual QA                     PASS
```

The executed tests cover Tully analytic derivatives, one-step MInt energy/Casimir conservation, closed-form product normalization/trace/radius, Collector metadata/snapshot round trip, support fingerprints, SEO basis rank, campaign design, one-step grid-QCLE trace conservation, the PBME projected-GP/KDE (E_1<0.02) contract, the prohibition on visible figure/subplot headers, and a source guard requiring full-state transactional breathing recovery. The additional PyTorch test injects a failed adaptive candidate and verifies exact restoration of both length scales and noise; it is included for the production environment and was skipped here because PyTorch is not installed.

## Required final numerical closure before thesis submission

1. Install `requirements.txt` in the production environment.
2. Execute the full reviewer campaign at the actual thesis endpoint and production (N).
3. Run manufactured, projection, KDE/GP, and reference-convergence commands.
4. Insert the produced numbers—not statements that the tests merely exist—into the thesis validation table.
5. Regenerate figures and use `THESIS_FIGURE_CAPTIONS.md` to update the LaTeX captions.
6. Compile the final LaTeX source and separately check undefined references, duplicate labels, and overfull boxes; those source-level checks cannot be inferred from Python outputs.
