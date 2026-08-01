"""Synchronize the verified flat pipeline into the public GitHub worktree."""
from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "reviewer_data_audit" / "github_main_staging"
RELEASE_TAG = "thesis-final-2026-08-01"
REPO_URL = (
    "https://github.com/sahandgit/"
    "gaussian_process_guided_mapping_quantumclassical_liouville_dynamics"
)

CORE = (
    "Collector.py", "Compare_gp_se_qcle.py", "conservative_excess.py",
    "Dynamics.py", "FigureCatalog.py", "GP_Density.py", "GP_DensityDiff.py",
    "GP_Derivatives.py", "GPDerivatives.py", "KDEDensity.py", "Mint.py",
    "Models.py", "Monodromy.py", "Observables.py", "Operator.py",
    "ProductMoments.py", "qcle_grid_tully.py", "Reproducibility.py",
    "ReviewerValidation.py", "run.py", "Sampling.py",
    "select_regularization.py", "seo_coefficient_gp.py", "Visualization.py",
)
AUDIT_MODULES = (
    "closure_audit.py", "final_acceptance_check.py",
    "reviewer_closure_campaign.py", "reviewer_closure_matrix.py",
    "reviewer_final_closure.py", "thesis_analysis.py", "thesis_closure.py",
)
EVIDENCE_DIRS = (
    "commands", "environment", "figures", "manufactured",
    "physical_comparison", "preserved_evidence", "reference_grid_qcle",
    "reference_tdse", "replication", "smoke", "support", "tables",
    "tail_sensitivity", "timestep",
)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def package_text(path: Path, local_modules: set[str]) -> str:
    text = path.read_text(encoding="utf-8")
    names = "|".join(sorted(map(re.escape, local_modules), key=len, reverse=True))
    return re.sub(
        rf"(?m)^(\s*)from ({names})(?=\.|\s+import)",
        r"\1from .\2",
        text,
    )


def public_readme() -> str:
    release = f"{REPO_URL}/releases/tag/{RELEASE_TAG}"
    return f"""# GP/RKHS--MInt--QCLE thesis pipeline

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
- [Release {RELEASE_TAG}]({release}) -- frozen numerical-evidence archive,
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
"""


def main() -> int:
    if not (STAGE / ".git").is_dir():
        raise FileNotFoundError(f"Git worktree absent: {STAGE}")

    pipeline = STAGE / "pipeline"
    reset_dir(pipeline)
    for name in (*CORE, *AUDIT_MODULES):
        copy_file(ROOT / name, pipeline / name)
        copy_file(ROOT / name, STAGE / name)
    for name in (
        "acceptance_contract.yaml", "l2_selection.json",
        "requirements.txt", "CLOSURE_PHASE1_RUNBOOK.md",
        "PIPELINE_REVISION_LOG.md", "REVIEWER_ACTION_REGISTER.md",
    ):
        if (ROOT / name).exists():
            copy_file(ROOT / name, pipeline / name)

    package = STAGE / "src" / "gp_mint_qcle"
    package.mkdir(parents=True, exist_ok=True)
    local_modules = {Path(name).stem for name in CORE}
    for name in CORE:
        (package / name).write_text(
            package_text(ROOT / name, local_modules), encoding="utf-8"
        )

    tests = STAGE / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    for source in ROOT.glob("test_*.py"):
        copy_file(source, tests / source.name)
    (tests / "conftest.py").write_text(
        "from pathlib import Path\n"
        "import sys\n\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "sys.path.insert(0, str(ROOT / 'pipeline'))\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n",
        encoding="utf-8",
    )

    audit = STAGE / "audit"
    reset_dir(audit)
    for name in AUDIT_MODULES:
        copy_file(ROOT / name, audit / name)
    shutil.copytree(
        ROOT / "reviewer_data_audit" / "scripts",
        audit / "scripts",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    thesis = STAGE / "thesis"
    reset_dir(thesis)
    for source in (ROOT / "Thesis").iterdir():
        if source.is_file() and source.suffix.lower() in {
            ".tex", ".bib", ".cls", ".pdf"
        }:
            copy_file(source, thesis / source.name)
    for name in ("Reviewer_Response.tex", "Reviewer_Response.pdf"):
        copy_file(ROOT / name, STAGE / name)

    evidence_target = STAGE / "final_reviewer_closure"
    reset_dir(evidence_target)
    evidence_source = ROOT / "final_reviewer_closure"
    for source in evidence_source.iterdir():
        if source.is_file() and source.suffix.lower() != ".zip":
            copy_file(source, evidence_target / source.name)
    for name in EVIDENCE_DIRS:
        shutil.copytree(
            evidence_source / name,
            evidence_target / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.npz"),
        )
    if (ROOT / "reviewer_closure_out").exists():
        shutil.copytree(
            ROOT / "reviewer_closure_out",
            STAGE / "reviewer_closure_out",
            dirs_exist_ok=True,
        )

    (STAGE / "README.md").write_text(public_readme(), encoding="utf-8")
    print(f"Synchronized verified repository: {STAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
