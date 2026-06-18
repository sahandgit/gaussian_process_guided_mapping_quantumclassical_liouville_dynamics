# Research Validation Checklist

This checklist defines the minimum validation expected before using this repository to support scientific claims.

## Software validity

- [ ] The package installs in a fresh virtual environment.
- [ ] All tests pass with `pytest`.
- [ ] The installed commands `gp-mqcld`, `gp-mqcld-run`, `gp-mqcld-compare`, and `gp-mqcld-smoke` work.
- [ ] Windows PowerShell launchers work from a clean checkout.
- [ ] Saved `.npz` and `.json` outputs can be reloaded without code modification.

## Numerical consistency

- [ ] Tully potential derivatives match finite-difference checks.
- [ ] PBME-MInt conserves energy in short-step tests.
- [ ] Monodromy/JAX pullback tensors pass finite-difference spot checks.
- [ ] GP first, second, and third derivatives pass finite-difference spot checks.
- [ ] The QCLE correction sign convention is tested.
- [ ] The live effective labels used for GP refits are also used for diagnostics and snapshots.

## Scientific convergence

- [ ] Run at multiple time steps, e.g. `dt = 1.0, 0.5, 0.25`.
- [ ] Run at multiple training-cloud sizes, e.g. `n_train = 500, 1000, 2000, 4000`.
- [ ] Run at multiple random seeds.
- [ ] Compare full-density GP midpoint results against PBME, grid-QCLE, and TDSE.
- [ ] Report population, coherence, nuclear, trace, and energy errors.

## Repository readiness

- [ ] README explains purpose, installation, module roles, and execution.
- [ ] Notebooks reproduce installation and workflow.
- [ ] Example commands are tested.
- [ ] Generated artifacts are not committed unless intentionally curated.
- [ ] The repository does not claim a completed thesis or paper.
