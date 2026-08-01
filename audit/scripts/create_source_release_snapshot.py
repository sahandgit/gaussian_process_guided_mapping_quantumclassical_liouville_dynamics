"""Create an audit-owned Git commit for the final executable source snapshot.

This does not reconstruct or claim the unavailable originating development
commit. It creates a new, explicitly labelled release commit from the source
files used for the final audit so the deposited evidence has an exact
version-control identifier.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reviewer_data_audit"
TARGET = AUDIT / "source_release_snapshot"
MANIFEST = AUDIT / "source_release_snapshot_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{subprocess.list2cmdline(command)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def remove_readonly(function, path: str, exception) -> None:
    """Permit deterministic replacement of a prior Git snapshot on Windows."""
    del exception
    os.chmod(path, stat.S_IWRITE)
    function(path)


def main() -> int:
    resolved_target = TARGET.resolve()
    if AUDIT.resolve() not in resolved_target.parents:
        raise RuntimeError(f"Refusing to manage target outside audit tree: {resolved_target}")
    if TARGET.exists():
        shutil.rmtree(TARGET, onexc=remove_readonly)
    (TARGET / "pipeline").mkdir(parents=True)
    (TARGET / "audit_scripts").mkdir(parents=True)

    copied: list[tuple[Path, Path]] = []
    for source in sorted(ROOT.glob("*.py")):
        target = TARGET / "pipeline" / source.name
        shutil.copy2(source, target)
        copied.append((source, target))
    for source in sorted((AUDIT / "scripts").glob("*.py")):
        target = TARGET / "audit_scripts" / source.name
        shutil.copy2(source, target)
        copied.append((source, target))
    for relative in (
        Path("requirements.txt"),
        Path("Thesis") / "ut-thesis.cls",
    ):
        source = ROOT / relative
        if source.exists():
            target = TARGET / "support" / relative.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append((source, target))

    hash_rows = [
        {
            "snapshot_path": str(target.relative_to(TARGET)),
            "original_path": str(source.resolve()),
            "sha256": sha256(target),
            "size_bytes": target.stat().st_size,
        }
        for source, target in copied
    ]
    hashes = TARGET / "SOURCE_FILE_HASHES.csv"
    with hashes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(hash_rows[0]))
        writer.writeheader()
        writer.writerows(hash_rows)
    (TARGET / "README.md").write_text(
        "# Audit-created final code snapshot\n\n"
        "Thesis: *Gaussian-Process Reconstruction of the Mapping-QCLE Excess "
        "Term: A Moving-Cloud Formulation and Failure Analysis*\n\n"
        "This repository was created by the reviewer audit from the executable "
        "source files used for final calculations. It is a release snapshot, "
        "not the unavailable originating development history.\n",
        encoding="utf-8",
    )

    run(["git", "init"], TARGET)
    run(["git", "config", "user.name", "Reviewer Data Audit"], TARGET)
    run(["git", "config", "user.email", "reviewer-audit@invalid.local"], TARGET)
    run(["git", "add", "--all"], TARGET)
    run(["git", "commit", "-m", "Final reviewer closure code snapshot"], TARGET)
    commit = run(["git", "rev-parse", "HEAD"], TARGET)
    status = run(["git", "status", "--porcelain"], TARGET)
    if status:
        raise RuntimeError(f"Release snapshot is unexpectedly dirty: {status}")
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_commit": commit,
        "repository": str(TARGET.resolve()),
        "file_count": len(hash_rows),
        "hash_index": str(hashes.resolve()),
        "hash_index_sha256": sha256(hashes),
        "working_tree_clean": True,
        "provenance_qualification": (
            "Audit-created final code release commit; originating development "
            "commit remains NOT IDENTIFIABLE because the supplied workspace "
            "contains no parent .git metadata. This archival qualification is "
            "kept outside the scientific thesis narrative."
        ),
    }
    MANIFEST.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
