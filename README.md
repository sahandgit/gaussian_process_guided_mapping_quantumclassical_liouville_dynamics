# GP-MQCLD

**Gaussian-Process-Based Mapping Quantum-Classical Liouville Dynamics**

GP-MQCLD is a research-software pipeline for studying nonadiabatic dynamics in the mapping representation using Gaussian-process/RKHS density surrogates, PBME-MInt characteristic transport, midpoint quantum-classical Liouville corrections, and reference comparisons against PBME, grid-QCLE, and split-operator TDSE.

The repository is designed to stand on its own as a public scientific-computing package. It does **not** depend on a finished thesis or paper. Thesis-specific discussion, final figures, tables, and citations can be added later after the thesis text and validation campaign are complete.

---

## 1. Name and abbreviation

The public name is **GP-MQCLD**.

The abbreviation means:

| Abbreviation | Meaning |
|---|---|
| **GP** | Gaussian-process-based density representation. |
| **M** | Mapping-basis/MMST representation of the electronic subsystem. |
| **QC** | Quantum-classical mixed dynamics. |
| **LD** | Liouville dynamics, i.e. density-level phase-space propagation. |

So the full title is:

> **Gaussian-Process-Based Mapping Quantum-Classical Liouville Dynamics**

This name is intentionally broader and more scientifically descriptive than implementation-specific names such as “MInt-GP” or “LiouvilleGP-MInt”. The method is not only a MInt propagator and not only a GP regression model. The object of the repository is the complete density-level pipeline:

```text
mapping-basis phase-space density
      -> PBME-MInt characteristic transport
      -> full-density GP/RKHS surrogate
      -> midpoint QCLE excess-density correction
      -> physical observables and benchmark comparison
```

---

## 2. What this repository is

GP-MQCLD is a complete Python pipeline for the following scientific problem:

> How can a smooth differentiable density surrogate be used to approximate mapping-basis quantum-classical Liouville dynamics along PBME/MInt characteristics?

The package provides:

- Tully-model Hamiltonians and analytic derivatives;
- nuclear Wigner and MMST mapping-basis sampling;
- PBME-MInt propagation of an extended mapping phase-space cloud;
- full-density Gaussian-process/RKHS density surrogates;
- analytic GP derivatives through third order;
- midpoint evaluation of the mapping-QCLE excess-density correction;
- live density-label updates and GP refitting;
- physical observables, conservation diagnostics, and snapshot serialization;
- visualization tools for populations, coherence, nuclear moments, conservation, corrections, and density slices/marginals;
- split-operator TDSE and grid-QCLE comparison workflows for Tully benchmarks;
- installation notebooks, Windows scripts, pytest tests, and validation documents.

This repository is meant to be pushed to GitHub as the **pipeline** first. Later, once the thesis is finalized, a thesis-specific layer can be added without changing the scientific identity of the package.

---

## 3. Why we are doing this

Mixed quantum-classical dynamics is often formulated as an evolution equation for a partially Wigner-transformed density. In the mapping representation, the electronic subsystem is represented by continuous mapping variables, producing an extended phase-space density depending on

```text
z = (R, P, r0, r1, p0, p1).
```

The PBME/MInt dynamics transports support points efficiently, but the mapping-QCLE density correction contains derivatives of the density with respect to both nuclear and mapping coordinates. A Monte Carlo cloud alone does not provide stable analytic derivatives. A Gaussian-process/RKHS surrogate does.

The purpose of this pipeline is therefore to test the following numerical idea:

1. propagate a support cloud with PBME-MInt characteristics;
2. represent the transported density by a differentiable GP/RKHS surrogate;
3. evaluate the mapping-QCLE excess-density correction using analytic GP derivatives and midpoint pullback geometry;
4. update the density labels along the transported cloud;
5. compare the resulting midpoint GP dynamics against PBME, grid-QCLE, and TDSE references.

The package is designed to answer scientific and numerical questions such as:

- Does the full-density GP surrogate follow the transported cloud?
- Are the GP-derived QCLE corrections stable over long propagation times?
- Does the midpoint correction improve over PBME for populations and coherences?
- Are normalization, trace, and energy conserved to acceptable tolerance?
- How sensitive are the results to time step, cloud size, random seed, and GP hyperparameters?
- How does the method compare to grid-QCLE and exact TDSE on Tully models?

---

## 4. What the production method is

The production/default method is the **full-density GP surrogate**.

```powershell
gp-mqcld-run --density_mode full ...
```

In this mode, the GP directly represents the live mapping-basis density on the transported cloud. This is the method that should be used for the main pipeline results unless a future validation campaign explicitly changes the default.

The density-difference mode remains available only as an optional diagnostic/ablation path:

```powershell
gp-mqcld-run --density_mode diff ...
```

The density-difference mode is useful for studying alternative stabilization strategies, but it is **not** the production default in this repository.

---

## 5. How the pipeline works

A typical simulation follows this execution graph:

```text
run.py / gp-mqcld-run
    |
    |-- Models.py
    |     defines Tully Hamiltonian h(R), gradients, and traceless components
    |
    |-- Sampling.py
    |     samples nuclear Wigner density and MMST mapping variables
    |
    |-- Mint.py
    |     packs z=(R,P,r,p) and propagates PBME-MInt characteristics
    |
    |-- GP_Density.py
    |     fits the full-density GP/RKHS surrogate with moment constraints
    |
    |-- Dynamics.py
    |     runs PBME and midpoint schemes from identical initial state
    |       |
    |       |-- Monodromy.py
    |       |     computes midpoint pullback geometry
    |       |
    |       |-- Operator.py
    |       |     evaluates the mapping-QCLE excess-density correction Q
    |       |
    |       |-- Observables.py
    |       |     computes trace, populations, coherence, energy, diagnostics
    |       |
    |       |-- Collector.py
    |             saves diagnostics and snapshots
    |
    |-- Visualization.py
          generates standard figure panels
```

The comparison workflow is separate:

```text
Compare_gp_se_qcle.py / gp-mqcld-compare
    |
    |-- loads saved PBME and midpoint GP runs
    |-- runs/loads split-operator TDSE reference
    |-- runs/loads grid-QCLE reference
    |-- generates comparison panels across methods
```

---

## 6. Repository layout

```text
gp_mqcld/
├── src/gp_mint_qcle/              # implementation namespace retained from the original research code
├── src/gp_mqcld/                  # public import alias for GP-MQCLD
├── src/liouvillegp_mint/          # legacy import alias retained for older local notes/scripts
├── scripts/                       # Python script entry points
│   └── windows/                   # PowerShell and CMD helper scripts
├── tests/                         # pytest tests for imports, CLI, labels, model derivatives, and MInt
├── notebooks/                     # tutorial notebooks: installation, run, diagnostics, comparison
├── configs/                       # example configuration files for P0 cases
├── docs/                          # method, module, validation, and future-thesis documentation
├── examples/                      # minimal Python examples
├── benchmarks/                    # reserved for convergence and timing studies
├── artifacts/                     # generated figures/tables/results; ignored by git except .gitkeep
├── pyproject.toml                 # install metadata and console scripts
├── requirements.txt               # pip dependency list
├── environment.yml                # conda-style environment file
├── CITATION.cff                   # software citation metadata
├── CONTRIBUTING.md
├── ROADMAP.md
├── LICENSE
└── README.md
```

The implementation namespace is still `gp_mint_qcle` because the uploaded code was originally organized around GP + MInt + QCLE modules. The public-facing import alias is now:

```python
import gp_mqcld
```

For backward compatibility, this still works:

```python
import gp_mint_qcle
```

and the older alias is retained:

```python
import liouvillegp_mint
```

New code should prefer `gp_mqcld`.

---

## 7. Installation

### 7.1 Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,notebooks]"
pytest
gp-mqcld-smoke
```

### 7.2 Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,notebooks]"
pytest
gp-mqcld-smoke
```

### 7.3 Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,notebooks]"
pytest
gp-mqcld-smoke
```

### 7.4 Installed commands

| Command | Role |
|---|---|
| `gp-mqcld` | Unified command with `run`, `compare`, and `smoke` subcommands. |
| `gp-mqcld-run` | Runs one PBME + midpoint GP simulation. |
| `gp-mqcld-compare` | Compares saved GP runs against SE, grid-QCLE, PBME, and midpoint. |
| `gp-mqcld-smoke` | Runs a tiny installation smoke test. |

The package also keeps compatibility aliases:

```text
gp-mint, gp-mint-run, gp-mint-compare, gp-mint-smoke
liouvillegp, liouvillegp-run, liouvillegp-compare, liouvillegp-smoke
```

Module execution is also supported:

```powershell
python -m gp_mqcld --version
python -m gp_mqcld smoke
python -m gp_mqcld run --help
python -m gp_mqcld compare --help
```

---

## 8. Quick start

### 8.1 Installation smoke test

```powershell
gp-mqcld-smoke
```

This runs a tiny full-density simulation and checks that the pipeline can produce:

```text
pbme.npz
pbme.json
midpoint.npz
midpoint.json
```

### 8.2 Full-density run for P0 = 10

```powershell
gp-mqcld-run --density_mode full --sampling_mode focused --P0 10 --R0 -15 --sigma_R 1.0 --dt 0.5 --n_steps 6000 --n_train 2000 --snapshot_every 20 --panel_every 200 --seed 0 --out ".\runs\P0_10"
```

### 8.3 Full-density run for P0 = 100

```powershell
gp-mqcld-run --density_mode full --sampling_mode focused --P0 100 --R0 -15 --sigma_R 1.0 --dt 0.5 --n_steps 6000 --n_train 2000 --snapshot_every 20 --panel_every 200 --seed 0 --out ".\runs\P0_100"
```

### 8.4 Compare P0 = 10 and P0 = 100

```powershell
gp-mqcld-compare ".\runs" --p0-list 10 100 --R0 -15 --sigma_R 1.0 --dt 0.5 --n_steps 6000 --density-times "0,500,800,1500" --out ".\runs\comparison_P0_10_100"
```

The comparison script expects folders such as:

```text
runs/
  P0_10/
    pbme.npz
    pbme.json
    midpoint.npz
    midpoint.json
  P0_100/
    pbme.npz
    pbme.json
    midpoint.npz
    midpoint.json
```

---

## 9. Module-by-module explanation

### `Models.py`

Defines the one-dimensional two-state Tully models. It provides diabatic Hamiltonian matrix elements, gradients, and traceless decompositions used by PBME, QCLE, GP moment constraints, and reference solvers.

### `Sampling.py`

Builds the initial distribution. It samples the nuclear Gaussian Wigner density and the MMST mapping variables. This module determines the initial support cloud and the initial density labels used by the GP.

### `Mint.py`

Defines the packed state convention

```text
z = (R, P, r0, r1, p0, p1)
```

and implements PBME-MInt propagation. This is the characteristic transport layer used by both the PBME baseline and the midpoint GP-QCLE correction scheme.

### `GP_Density.py`

Implements the production full-density GP/RKHS surrogate. It handles kernel construction, hyperparameter optimization, moment constraints, GP coefficient solves, prediction, and analytic moment machinery. This is the default density representation.

### `GPDerivatives.py`

Provides analytic derivatives of the ARD-RBF GP density surrogate. These derivatives are required because the mapping-QCLE excess operator contains high-order density derivatives.

### `GP_Derivatives.py`

Compatibility wrapper retained for older imports. New code should use `GPDerivatives.py`.

### `GP_DensityDiff.py`

Optional density-difference surrogate. It is useful for diagnostics and ablation studies but is not the default production method.

### `KDEDensity.py`

Provides a signed KDE diagnostic estimator. It is not used as the production density representation. Its role is to compare the local cloud structure with GP reconstructions.

### `Monodromy.py`

Computes midpoint pullback geometry: Jacobians, Hessians, and third-derivative tensor slices of the backward MInt map. These tensors allow the QCLE correction to be evaluated at pulled-back midpoint coordinates.

### `Operator.py`

Evaluates the pulled-back midpoint mapping-QCLE excess-density operator. This is the module that turns GP derivatives and pullback geometry into the correction term `Q` used by the midpoint density update.

### `Dynamics.py`

Coordinates the runtime simulation. It owns the simulation state, builds PBME and midpoint schemes, performs time stepping, updates live density labels, refits the GP, computes diagnostics, and triggers snapshots.

### `Observables.py`

Computes physical observables from GP-analytic moment formulas and cloud-weighted estimators. It reports normalization, trace, populations, coherences, nuclear moments, energy, and correction diagnostics.

### `Collector.py`

Serializes run data. Each run produces `.npz` numerical arrays and `.json` metadata/sidecar files. These saved outputs are used by plotting, diagnostics, and comparison scripts.

### `Visualization.py`

Generates standard research-quality figures: conservation, populations, coherences, nuclear moments, mapping moments, correction diagnostics, fit quality, and density marginals/slices.

### `qcle_grid_tully.py`

Implements the grid-QCLE reference solver for the one-dimensional Tully model. This provides a quantum-classical reference against which the GP midpoint method can be compared.

### `Compare_gp_se_qcle.py`

Runs the comparison workflow across TDSE, grid-QCLE, PBME, and GP midpoint dynamics. This is the primary script for method benchmarking.

### `run.py`

Main simulation driver behind `gp-mqcld-run`. It samples the initial cloud, fits the initial GP, constructs PBME and midpoint simulations, runs both, saves results, and produces figures.

### `cli.py`

Unified command-line interface behind `gp-mqcld`.

### `cli_smoke.py`

Tiny end-to-end smoke test used to check that installation and execution work.

---

## 10. Notebooks

| Notebook | Purpose |
|---|---|
| `00_installation_and_smoke_test.ipynb` | Shows environment setup, editable installation, command checks, and smoke testing. |
| `01_pipeline_walkthrough.ipynb` | Explains the conceptual and code-level pipeline flow. |
| `02_run_full_density_p0_cases.ipynb` | Gives command templates for full-density `P0=10`, `P0=40`, and `P0=100` runs. |
| `03_load_results_and_basic_diagnostics.ipynb` | Loads `.npz` results and inspects diagnostic arrays. |
| `04_compare_se_qcle_gp.ipynb` | Shows how to run SE/QCLE/PBME/GP comparisons from saved run folders. |

The notebooks are intentionally tutorial-oriented. They are not thesis notebooks yet. Thesis-specific notebooks can be added later once the final figures and text are fixed.

---

## 11. Outputs

A production run writes a directory such as:

```text
runs/P0_100/
  pbme.npz
  pbme.json
  midpoint.npz
  midpoint.json
  fig_conservation.png
  fig_populations.png
  fig_coherences.png
  fig_nuclear.png
  fig_mapping_moments.png
  fig_local_energy.png
  fig_correction.png
  fig_fit_quality.png
  fig_marginal_*.png
```

The comparison workflow writes method-comparison figures into the selected output directory, for example:

```text
runs/comparison_P0_10_100/
  P0_10/
  P0_100/
```

Generated results should usually stay outside git or under `artifacts/` if they are small demonstration outputs.

---

## 12. Validation status

The repository includes automated tests for:

- package importability;
- public and legacy import aliases;
- console-script declarations;
- valid tutorial notebooks;
- Tully derivative finite-difference consistency;
- short-step PBME-MInt energy conservation;
- midpoint effective-label semantics;
- default CLI contract confirming `--density_mode full` is the default.

Run:

```powershell
pytest
```

Before using the pipeline for thesis or paper claims, complete the validation campaign described in:

```text
docs/RESEARCH_VALIDATION_CHECKLIST.md
```

At minimum, the scientific validation should include:

- convergence with time step;
- convergence with training-cloud size;
- seed variability;
- comparison with TDSE and grid-QCLE;
- conservation of normalization, trace, and energy;
- diagnostic checks that the GP surrogate follows the transported support cloud;
- ablation studies for optional modes such as density-difference or clipping.

---

## 13. Development workflow

Recommended local workflow:

```powershell
git status
pytest
gp-mqcld-smoke
gp-mqcld-run --help
gp-mqcld-compare --help
```

Before committing scientific changes:

```powershell
pytest
python -m compileall -q src tests scripts examples
```

For larger numerical changes, run at least one short physical simulation and verify that diagnostics are finite and saved outputs reload correctly.

---

## 14. GitHub publication strategy

The recommended strategy is:

1. publish this repository now as **GP-MQCLD**, a standalone research-software pipeline;
2. keep the README and docs focused on the method and code, not on an unfinished thesis;
3. use the repository to generate validated results;
4. after the thesis is finished, add a thesis-specific branch, folder, or documentation layer with exact figure provenance;
5. after thesis completion, decide whether to prepare a paper-focused branch or archived software release.

Suggested first push:

```powershell
git init
git add .
git commit -m "Initial GP-MQCLD research pipeline"
git branch -M main
git remote add origin https://github.com/sahandgit/gp-mqcld.git
git push -u origin main
```

---

## 15. Citation placeholder

Until a thesis or paper exists, cite the software repository itself using `CITATION.cff`. After the thesis is complete, update `CITATION.cff` with the thesis citation and optionally archive a release on Zenodo or a similar service.

---

## 16. Current status

This is a research pipeline under active development. It is suitable for public GitHub development, reproducibility work, and validation studies. It should not yet be described as a finalized thesis or paper artifact until the corresponding scientific validation and written interpretation are complete.
