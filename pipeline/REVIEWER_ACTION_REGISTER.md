# Reviewer action register

Source of authority: **Revised MSc Thesis Review, 28 July 2026** (major revision;
not acceptable as the final submitted PDF). 48 numbered items in §8, ten
non-negotiable acceptance gates in §9.

---

## 1. The headline finding

The reviewer is **not** asking for a large new computational campaign.

| Item class | Count | What closes it |
|---|---:|---|
| Pure thesis-text correction | **33** | Editing `Thesis.tex`. No compute. |
| Table extraction from data already on disk | **9** | Running the extractors below. |
| Figure regeneration / removal | **1** (I-6) | The submission blocker. |
| Archival act | **1** (I-15) | Freeze a release, record hashes/DOI. |
| Genuinely new computation | **4** | I-9, I-13, M-12, M-24 — and **three of the four have a reviewer-sanctioned escape hatch.** |

This matters because `required_run_gap_analysis.csv` reports 118 `RERUN - MISSING`
cells. Those come from the *pipeline specification*, which is a strict superset
of what the reviewer demands. **Do not run 118 cells.** The specification's
campaign matrix is an engineering ideal; §9 of the review is the actual bar.

### The escape hatches the reviewer explicitly offers

These are quotes, not my interpretation:

- **I-13 / gate 7** — "report seed-resolved L₁/L₂ field errors … **or write only
  that no validated improvement over PBME was demonstrated.**"
- **I-11** — "add a table of raw and normalized analytic-GP and cloud estimates
  … **or remove the claim.**"
- **Gate 6 (support)** — "either perform a nested/coupled study **or state
  plainly that support convergence remains untested.**"
- **§4.2 / M-24** — repeat the operator test at ℓ₂ = 0.05 and 0.01, "**or justify
  mathematically why the operator conclusion is insensitive to this change.**"
- **I-9** — until the tail study exists, "**qualify every multiplier and
  cloud-weight claim as potentially tail-sensitive.**"

Taking every escape hatch reduces the required new computation to **zero** and
still satisfies all ten gates — provided the thesis wording is narrowed to match.
That is the fastest defensible path to submission. It is also exactly consistent
with the negative result the data already supports.

---

## 2. The ten acceptance gates and how each closes

| Gate | Requirement | Route |
|---|---|---|
| 1 | Six-question clarity in every chapter | Text. Chapter-opening contract (§2.4): Question / Importance / Approach / Outcome. |
| 2 | Novelty positioning | Text + targeted literature reading. Use §3.3's five-step chain verbatim as the contribution. |
| 3 | **Figure provenance** | **Blocker.** Regenerate each retained figure from a verified manifest, or delete it. |
| 4 | Complete manufactured evidence | `tables` extractor (I-1) + qualify ℓ₂ per M-24. |
| 5 | Transparent convergence tables | `tables` extractor (I-3) + timestep table (I-2). |
| 6 | Support claim corrected | Text — state plainly that support convergence is untested. |
| 7 | Reference comparison | Text — "no validated improvement demonstrated." |
| 8 | Immutable archive | Freeze release, SHA-256 every artifact, reproduce settings in App. F. |
| 9 | Title/metadata consistency | Text. Adopt one of the reviewer's two suggested titles. |
| 10 | Response-letter accuracy | Text. Replace inflated "Resolved" labels; cite exact pages. |

Seven of ten are closed by writing. Gate 3 is the one that requires real work.

---

## 3. Two items I can close from data right now

**I-3 — "reference-convergence orders are unassigned."** They are not
unassigned; they were simply never tabulated. `thesis_closure_out/reference_convergence_3level.json`
identifies every one of the numbers the reviewer lists:

| Reported value | Actually is |
|---|---|
| 1.872 | TDSE, P₀ = 20, refine mode `both`, observable **P₀** |
| 2.000 | same block, observable **P₁** |
| 2.152 | same block, observable **energy** |
| 2.006 | same block, observable **R_mean** |
| **−3.391** | same block, observable **trace** |

The −3.391 outlier is not a negative convergence order. Its successive
differences are **1.82 × 10⁻¹⁴ → 1.91 × 10⁻¹³** on a quantity of magnitude 1 —
i.e. at and below the roundoff floor 100·ε_mach ≈ 2.2 × 10⁻¹⁴. The trace is
conserved to machine precision, so the ratio is pure noise. The extractor
applies the guard and reports it as `REJECTED: coarse-to-fine difference at or
below roundoff floor`, which is the honest identified row the reviewer asked
for.

The same file also carries the exact settings I-14 wants inside Appendix F:
grids 2048/4096/8192, dt 0.2/0.1/0.05, t_final = 40.0, boundary rule
"periodic (split-operator FFT)".

> ### CORRECTION (after running the extractor)
>
> The **table** is closable by extraction. The **scientific claim** is not.
> 45 of 72 reference rows fall at or below the roundoff floor, and 24 of them
> have successive differences of *exactly* `0.000e+00`.
>
> Root cause, verified in `thesis_closure.py` lines 93 and 146: the reference
> study runs at `R0 = -10.0`, `dt = 0.2`, `n_steps = 200`, i.e. **t_final = 40**.
> The collision time is t_c = M|R₀|/P₀ = 1000 (P₀ = 20) and 200 (P₀ = 100).
> The packet travels 0.4 and 2.0 bohr respectively — it **never reaches the
> avoided crossing**. In the flat asymptotic region the split-operator FFT
> propagator is *exact* for free motion (kinetic step exact in momentum space,
> potential locally constant), so refining dt or the grid changes the endpoints
> by nothing. The refinement test has no power by construction.
>
> Consequences:
> - The orders **18.36, 17.78, 18.34, 17.75** previously emitted for
>   `qcle_P020_*` are noise ratios, not convergence orders. They must not enter
>   the thesis.
> - Only `tdse_P020_both` yields usable orders (1.872–2.152) and even those come
>   from residual differences of order 1e-12.
> - Production uses R₀ = −15; the reference used R₀ = −10. The two are **not a
>   matched comparison pair**.
>
> Gate 5 therefore needs the reference study re-run over a window that reaches
> the crossing. This is the only new computation I recommend, and TDSE is cheap.

**I-1 — "complete three-seed manufactured result is not shown."** All of
N ∈ {300, 600, 1200, 2400} × seeds {123, 124, 125} exist, with on- and
off-support density / gradient / operator_Q errors in three metrics each. The
extractor emits every row plus per-N seed means and spreads, and flags any
(N, seed) pair whose duplicates disagree rather than silently merging them.

---

## 4. What to run

```powershell
cd D:\PhD-Project\...\output\fixed_pipeline

python reviewer_closure_matrix.py --self-test
python reviewer_closure_matrix.py all --root . --out reviewer_closure_out
```

Produces in `reviewer_closure_out\`:

- `I1_manufactured_complete.csv` — every seed × N × support × quantity
- `I1_manufactured_seed_statistics.csv` — means, SDs, min/max per N
- `I3_I14_reference_convergence_identified.csv` — every order uniquely
  identified with exact numerical settings and guard verdict
- `closure_matrix.csv` — all 48 items with evidence status and action
- `closure_matrix_summary.json`

Run the self-test first; the sandbox has been unavailable, so this module has
not been executed.

---

## 4b. I-10 is closed by code inspection — no computation

The review asks for "the exact formula and enabled controls for the production
branch, **or** state unambiguously that applied source equals raw source and
remove the unexplained stabilization language."

`Dynamics.py` settles it:

- `apply_q_clip: bool = False` (default), documented as
  *"LEGACY — accepted for back-compat with older configs. Still unused."*
- `omega_clip_quantile: Optional[float] = None` (disabled by default)
- Every production `run_manifest.json` records `apply_q_clip: false`,
  `omega_clip_quantile: null`, `abs_target: false`.

**Applied source = raw source.** No stabilization operation acts on the excess
source in the production branch. The thesis text describing "stabilization
procedures" refers to code paths that are switched off and, in the case of
`q_clip`, not wired in at all. Delete that language and state the identity.

---

## 5. The reference re-run (the one computation worth doing)

```powershell
python thesis_closure.py reference `
    --out thesis_closure_out_v2 `
    --P0 20 100 --R0 -15 --dt 0.2 --n-steps 15000 `
    --modes both time grid --methods tdse
```

`t_final = 0.2 x 15000 = 3000`, and t_c(P₀ = 20, R₀ = −15) = 1500, so the packet
passes through the crossing and out the far side. The runner now prints a
warning whenever the requested window is shorter than t_c, and records
`R0`, `t_final`, `collision_time` and `window_reaches_crossing` in every block.

TDSE first, deliberately: it is a 1-D two-state split-operator FFT, so the
finest level is ~6 x 10⁴ steps on an 8192-point grid — minutes, not hours.
**Grid QCLE is a different matter** — `both` refinement scales the phase-space
grid by 4x in each of R and P, i.e. 16x the points, over 4x the steps. Estimate
that separately before launching it; `--methods qcle` runs it alone.

Then re-extract:

```powershell
python reviewer_closure_matrix.py tables --root . --out reviewer_closure_out_v2
```

---

## 6. The real blocker: gate 3

The reviewer is unambiguous: *"Every retained figure must be regenerated from a
verified run or tied to a complete immutable manifest. Otherwise remove it.
This applies equally to 'qualitative' and quantitative figures."* He rejects the
response letter's "Resolved with stated limitation" and re-labels it
**Unresolved — submission blocker**.

There is no escape hatch for this one. The decision is per figure:

1. **Regenerate** from a run whose manifest is complete — cheap for figures
   backed by the 38 verified runs.
2. **Remove** the figure *and every sentence that depends on it* (I-7, M-9).

Chapter 6 currently carries 41 figures across 67 pages. The reviewer separately
asks for compression (L-6, §7.4), so removal is not a loss — deleting
unprovenanced legacy figures serves gates 3 and the length criticism at once.
My recommendation is to keep only figures regenerable from the verified runs and
delete the rest, rather than attempting to trace 3,217 legacy sidecars.

One provenance defect must be fixed regardless, because the reviewer cites it
twice: `results\P0_20\run_manifest.json` records `P0: 40.0`. Any figure sourced
from that directory is currently mislabelled.

---

## 6. What I need from you

1. **The thesis `.tex`.** 33 of 48 items are text edits and I cannot make them
   without the source. This is now the critical path.
2. **A decision on the escape hatches** — take them (fast, honest, zero new
   compute) or spend compute on I-13/M-24 to make a stronger positive statement.
   My recommendation is to take them: the data does not support an improvement
   claim, and the reviewer has already said a well-documented negative result is
   an acceptable MSc contribution.
3. **A decision on Chapter 6 figures** — regenerate-and-keep, or delete-and-cut.

---

## 7. Standing constraint

§3.2 lists what the thesis may **not** claim: a validated solution of the
complete mapping-basis QCLE; improvement over PBME; convergence of the
production MIDPOINT discretization; convergence with support size; general
reliability of GP density reconstruction; accuracy beyond the 1-D two-state
benchmark; multidimensional scalability.

Title, abstract, every chapter opening, the conclusions, the response letter and
the defense answers must all stay inside those limits.
