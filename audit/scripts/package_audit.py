#!/usr/bin/env python3
"""Create the requested ZIP archive without modifying source simulation data."""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reviewer_data_audit"
OUT = ROOT / "reviewer_data_audit_complete.zip"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    required = (
        "PIPELINE_DATA_AUDIT_AND_THESIS_EVIDENCE.md",
        "THESIS_EVIDENCE_TABLES.tex",
        "verification_result.json",
        "frozen_numerical_evidence_payload.zip",
        "frozen_numerical_evidence_payload_manifest.json",
    )
    missing = [name for name in required if not (AUDIT/name).exists()]
    if missing:
        raise SystemExit(f"Refusing to package incomplete audit; missing: {missing}")
    tmp = ROOT / "reviewer_data_audit_complete.zip.tmp"
    if tmp.exists():
        tmp.unlink()
    result_path = AUDIT / "package_result.json"
    if result_path.exists():
        result_path.unlink()
    files = sorted(p for p in AUDIT.rglob("*") if p.is_file())
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            arc = Path("reviewer_data_audit") / path.relative_to(AUDIT)
            zf.write(path, arc.as_posix())
    tmp.replace(OUT)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(OUT),
        "bytes": OUT.stat().st_size,
        "sha256": sha(OUT),
        "files": len(files),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
