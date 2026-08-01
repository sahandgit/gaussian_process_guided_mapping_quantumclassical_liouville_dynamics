# Reviewer-closure runbook — Phase 1 (audit before compute)

Generated 2026-07-28. Companion to `PIPELINE_REVIEWER_CLOSURE_SPECIFICATION.pdf`.

---

## 0. Read this first — honest status

The specification is an **execution contract**, not evidence. It requires
eleven campaigns (A–K), 22 tests, a 16-subcommand orchestration interface, a
frozen checksummed release, and a closure matrix over 48 reviewer items
(I-1…I-16, M-1…M-25, L-1…L-7).

**What exists as of this document:**

| Spec section | Requirement | Status |
|---|---|---|
| 3 | Campaign dir + starting-state SHA-256 + `campaign_manifest.json` | implemented (`closure_audit.py`) |
| 4 | `pipeline_inventory.md`, duplicate-module + name/manifest conflict detection | implemented |
| 4.1 | `existing_data_index.csv`, `required_run_gap_analysis.csv` | implemented |
| 5 | `reviewer-closure-1.0` artifact schema + atomic JSON write | implemented |
| 7 | `acceptance_contract.yaml` | implemented |
| 6, 8–18 | Module changes and campaigns A–K | **not implemented** |
| 19–22 | 22 tests, frozen release, closure matrix | **not implemented** |

Phase 1 is implemented because the specification forbids doing anything else
first (§4.1: *"Do not launch an expensive simulation until this table proves
that existing data cannot answer the corresponding reviewer item"*). Given the
~38 completed dynamics runs, the manufactured factorial, the projection and
KDE/GP snapshot trees, and the three-level references already on disk, a large
fraction of the campaign matrix is expected to resolve to REUSE or REANALYZE.
Running the audit first is what prevents another multi-day compute cycle
producing data you already have.

**Two blockers I cannot clear myself:**

1. `Sahand_Nikzat_Revised_Thesis_Review.docx` is unreadable to me (zip
   container, sandbox down). §1 names it *the scientific requirements
   authority* and forbids building the closure matrix "against memory or an
   earlier summary." **Export it to PDF.** Until then, item identifiers
   I-1…I-16 / M-1…M-25 / L-1…L-7 in the gap table are provisional.
2. My Linux sandbox has been unavailable, so `closure_audit.py` is **written
   but not executed**. Step 1 below is therefore not optional.

---

## 1. Verify the new module (2 seconds)

```powershell
cd D:\PhD-Project\GP_RKHS_MINT\GP_RKHS_MINT\Tully_models\GP_MInt_QCLE_Reviewer_Ready_Pipeline_v4\output\fixed_pipeline
python closure_audit.py --self-test
```

Expected final line:

```
[self-test] closure_audit checks passed (atomic write, schema, cell
enumeration, inventory, name/manifest conflict, gap dispositions,
campaign creation, acceptance contract).
```

If it raises, paste the traceback — do not proceed. The self-test covers the
atomic writer, schema validation, campaign-cell enumeration (36 manufactured
cells, 48 timestep cells, no duplicates), snapshot time classification for both
momenta, provenance resolution through the `source` field, and the
directory-name-vs-manifest conflict detector.

---

## 2. Run the audit (minutes, no physics)

```powershell
python closure_audit.py inspect --root .
```

This creates `validation\reviewer_closure_<UTC stamp>\` containing:

- `campaign_manifest.json` — environment, package versions, SHA-256 of every
  source file at campaign start
- `starting_state\source_hashes_before.csv`
- `pipeline_inventory.md` — module roles, missing modules, the
  `GPDerivatives.py` / `GP_Derivatives.py` duplication, artifact-family counts,
  and **every directory whose name contradicts its manifest**
- `existing_data_index.csv` — one row per `.npz` / `.json` / `.csv` artifact
  with method, P0, seed, N, dt, GP policy, normalization status, SHA-256,
  readability and compatibility group (figure sidecars excluded; there are
  ~6,400 and they are catalogued separately)
- `required_run_gap_analysis.csv` — the decision table
- `acceptance_contract.yaml` — thresholds fixed **before** any run
- `inspect_summary.json`

It hashes several GB of `.npz`, so it prints a progress line every 250 files.
It is not hung.

---

## 3. Extract the decision (this is the deliverable of Phase 1)

Open `required_run_gap_analysis.csv`. Every one of the ~230 campaign cells
carries exactly one of:

| Disposition | Meaning | Action |
|---|---|---|
| `REUSE - COMPLETE AND COMPATIBLE` | finished, compatible run exists | analysis only |
| `REANALYZE - RAW DATA PRESENT` | raw arrays exist, metric not extracted | post-process, no simulation |
| `RERUN - MISSING` | no such run | must compute |
| `RERUN - INCOMPATIBLE` | run exists but violates a contract | must recompute |
| `REPAIR THEN RERUN - TECHNICAL FAILURE` | provenance/technical defect first | fix, then compute |

Summary counts print to the console and land in `inspect_summary.json`.

**Predictions to check against the actual output** — these are expectations
from inspecting the tree by hand, and the CSV is what settles them:

- **E (timestep)** — largely `REUSE`. Both momenta × seeds 11/29 × Δt
  {0.5, 0.25, 0.125} exist in `reviewer_closure_20260723_194254` and
  `..._20260726_174927`. Seeds 47 and 73 at Δt are likely `RERUN - MISSING`.
- **D (projection)** and **I (KDE/GP)** — mostly `REUSE`.
  `reviewer_data_audit\derived_validations\snapshots\` holds both diagnostics
  for P0 ∈ {20, 100} × seeds {11, 29, 47, 73} × {pbme, midpoint} × three
  snapshot steps, each recording `snapshot_step` and a `source` run directory.
  For P0 = 20 those steps are t = 0, t_c, 2t_c (= t_final). For P0 = 100 they
  are t = 0, t_c, 2t_c, so the **t_final snapshot is absent** — expect
  `RERUN - MISSING` on exactly those four cells.
- **C (manufactured)** — `REUSE` at ℓ₂ = 1e-6 for the full N × seed grid;
  `RERUN - MISSING` at ℓ₂ ∈ {0.01, 0.05}. The factorial is one-third complete.
- **F (nested support)** — `RERUN - INCOMPATIBLE` throughout. Existing N-series
  clouds were sampled **independently** per N; no manifest records a
  parent/prefix hash. §14 requires N = 500/1000 to be deterministic prefixes of
  a master N = 2000 cloud. Until that holds, no deterministic support-order
  claim is admissible.
- **B (regularization)** — `RERUN - INCOMPATIBLE`. `l2_selection.json` selected
  ℓ₂\* = 0.01 at N ≈ 300–350 pilot support. §9 requires N ∈ {300, 600, 1200},
  ≥ 3 folds, and a confirmation seed at the production contract. Rule 8
  forbids relabelling the pilot value as production-selected.
- **G (conservation)** — `REANALYZE`. Raw arrays are in the `.npz`; the tables
  have not been extracted.
- **H (tail/source audit)** — `RERUN - MISSING`. `Dynamics.py` does not
  currently save `source_raw` / `source_after_floor` / `source_after_clip` /
  `source_applied`, so the six-threshold sensitivity study cannot be done from
  existing data. This needs a code change before it needs compute.
- **K (matched physical errors)** — `RERUN - MISSING`. Matched L1/L2/L∞ field
  errors against the converged grid-QCLE field are not implemented anywhere.
- **PROVENANCE row** — if any snapshot artifact cannot be tied back to a run
  manifest, a `REPAIR` row appears. Do not ignore it; unresolvable provenance
  is a defect to report, not an absence to fill.

---

## 4. What Phase 1 does *not* do

The audit tells you what to run. It does not run it, and it does not write the
thesis. Remaining, in dependency order:

1. **Provenance repairs (no compute).** `results\P0_20\run_manifest.json`
   records `P0: 40.0`; every manifest records `sigma_R: 1.0` while the thesis
   states σ_R(P₀) = 10/P₀. Both must be resolved before any figure sourced from
   those directories is admissible. The inventory lists them explicitly.
2. **Code changes for H** — instrument the source pipeline in `Dynamics.py`.
3. **Code changes for F** — nested prefix sampling in `Sampling.py` with a
   parent-cloud hash in the manifest.
4. **Campaign B rerun** at the production contract.
5. **Campaign C completion** — the two missing ℓ₂ levels.
6. **Campaign K implementation** — matched field errors.
7. **Analysis + closure matrix** against the review PDF, once you send it.

Only items 4–6 cost significant compute, and the gap table will size them
exactly.

---

## 5. Standing rules that constrain every claim

From §2 of the specification, and directly relevant to what the review already
objected to:

- A two-level difference is **not** a convergence order. Report p_obs only from
  three levels, and suppress it when the denominator falls below the
  independent-seed noise floor or 100 ε_mach × solution scale.
- Self-normalization is **not** conservation evidence. Report raw integrals.
- Never relabel the unconstrained product surrogate as projected. Its leakage
  (≈ 0.97 relative L2 in the snapshots inspected) is a diagnostic, and the
  four-field branch is the only one that may claim ≤ 1e-10.
- Default verdict on method comparison is **"No validated improvement"** unless
  the paired-difference interval excludes zero on matched data.
- A negative result, fully characterised, is a complete result — and that is
  what this dataset currently supports.
