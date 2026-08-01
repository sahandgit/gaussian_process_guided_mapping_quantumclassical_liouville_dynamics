"""Refresh the legacy full-audit checksum index after final source corrections.

The frozen numerical payload and the audit-created Git source snapshot have
their own complete manifests.  This utility keeps the earlier full-directory
audit index internally consistent without rerunning the old table generator,
which would overwrite the final reviewer-closure tables.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "reviewer_data_audit" / "checksums_sha256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not INDEX.exists():
        raise FileNotFoundError(INDEX)
    with INDEX.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != ("scope", "path", "sha256", "bytes"):
        raise ValueError("Unexpected checksum-index schema")

    refreshed = []
    removed_transient = 0
    updated = 0
    for row in rows:
        relative = Path(row["path"])
        if "__pycache__" in relative.parts or relative.suffix.lower() == ".pyc":
            removed_transient += 1
            continue
        path = (ROOT / relative).resolve()
        if path == INDEX.resolve() or not path.exists():
            refreshed.append(row)
            continue
        if ROOT.resolve() not in path.parents:
            raise RuntimeError(f"Refusing to hash path outside workspace: {path}")
        new_hash = sha256(path)
        new_size = str(path.stat().st_size)
        if new_hash != row["sha256"] or new_size != row["bytes"]:
            updated += 1
        refreshed.append(
            {
                "scope": row["scope"],
                "path": row["path"],
                "sha256": new_hash,
                "bytes": new_size,
            }
        )

    temporary = INDEX.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("scope", "path", "sha256", "bytes")
        )
        writer.writeheader()
        writer.writerows(refreshed)
    os.replace(temporary, INDEX)
    print(
        f"Refreshed {updated} checksum row(s); removed "
        f"{removed_transient} transient bytecode row(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
