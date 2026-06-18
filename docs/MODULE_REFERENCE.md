# Module Reference

This document expands the README module table for readers who need to modify or audit the pipeline.

## Core physics modules

### `Models.py`
Defines Tully-model diabatic potentials, Hamiltonian matrices, traceless decompositions, and derivatives.

### `Sampling.py`
Builds the initial nuclear Gaussian Wigner distribution and MMST mapping-variable samples.

### `Mint.py`
Contains packed state conventions and PBME-MInt propagation of the extended state.

### `Monodromy.py`
Computes pullback geometry for midpoint evaluation: Jacobians, Hessians, and third-derivative tensor slices.

### `Operator.py`
Evaluates the midpoint mapping-QCLE excess-density correction using pulled-back GP derivatives.

## Surrogate modules

### `GP_Density.py`
Full-density GP/RKHS representation. This is the default production surrogate.

### `GPDerivatives.py`
Analytic ARD-RBF GP derivative routines.

### `GP_Derivatives.py`
Compatibility wrapper retained for older imports.

### `GP_DensityDiff.py`
Optional density-difference representation for ablation/diagnostic studies.

### `KDEDensity.py`
Signed KDE diagnostic estimator for comparing cloud structure against GP reconstructions.

## Runtime modules

### `Dynamics.py`
Owns simulation state, scheme stepping, GP refitting, diagnostics, effective-label bookkeeping, and snapshot creation.

### `Observables.py`
Computes physical observables from GP moments and cloud-weighted estimators.

### `Collector.py`
Serializes diagnostics and snapshots.

### `Visualization.py`
Creates research-quality figures for saved runs.

## Reference and command modules

### `qcle_grid_tully.py`
Grid-QCLE reference solver.

### `Compare_gp_se_qcle.py`
Comparison driver for SE/grid-QCLE/PBME/GP-midpoint results.

### `run.py`
Main production driver behind `gp-mqcld-run`.

### `cli.py`
Unified command-line interface behind `gp-mqcld`.

### `cli_smoke.py`
Tiny executable smoke test.
