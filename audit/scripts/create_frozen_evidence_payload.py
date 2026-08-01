"""Create and identify a deterministic frozen numerical-evidence payload."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "final_reviewer_closure"
AUDIT = ROOT / "reviewer_data_audit"
ARCHIVE = AUDIT / "frozen_numerical_evidence_payload.zip"
MANIFEST = AUDIT / "frozen_numerical_evidence_payload_manifest.json"
THESIS = ROOT / "Thesis" / "Thesis.tex"
REPOSITORY_URL = (
    "https://github.com/sahandgit/"
    "gaussian_process_guided_mapping_quantumclassical_liouville_dynamics"
)
RELEASE_BEGIN = "% BEGIN VERSIONED RELEASE IDENTIFIERS"
RELEASE_END = "% END VERSIONED RELEASE IDENTIFIERS"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def breakable_hash(value: str) -> str:
    chunks = [value[i:i + 8] for i in range(0, len(value), 8)]
    return r"\texttt{" + r"\allowbreak{}".join(chunks) + "}"


def update_thesis_release_block(
    archive_sha: str,
    index_sha: str,
    source_commit: str,
    release_asset_url: str,
    release_tag: str,
) -> None:
    """Keep the printed release identifiers synchronized with the archive."""
    block = "\n".join([
        RELEASE_BEGIN,
        r"\begingroup\small\sloppy",
        "Public source repository:",
        r"\url{" + REPOSITORY_URL + r"}.\\",
        "Public release asset:",
        r"\url{" + release_asset_url + r"}.\\",
        "Final release tag (identifies the final tagged document commit):",
        r"\path{" + release_tag + r"}.\\",
        "Archive filename:",
        r"\path{" + ARCHIVE.name + r"}.\\",
        "Frozen source/evidence commit (retrievable at the repository above):",
        breakable_hash(source_commit) + r".\\",
        "Frozen numerical-evidence archive SHA-256:",
        breakable_hash(archive_sha) + r".\\",
        "Embedded checksum-index SHA-256:",
        breakable_hash(index_sha) + ".",
        r"The archive contains \path{final_reviewer_closure/README.md} "
        r"(reproduction instructions), \path{environment.json} and "
        r"\path{environment/pip_freeze.txt} (environment), "
        r"\path{FINAL_RUN_MANIFEST.json} and the table/figure and per-run "
        r"manifests, plus \path{PAYLOAD_SHA256SUMS.csv} for every archived "
        r"file. The release also supplies \path{CLEAN_ROOM_VERIFICATION.json}, "
        r"which records public download, checksum, extraction, manifest, and "
        r"clean source-compilation checks. The public release asset and cited "
        r"frozen source/evidence commit make this record independently "
        r"retrievable and checksum-verifiable. The versioned GitHub release is "
        r"not a DOI or an institutional persistent identifier.",
        r"\endgroup",
        RELEASE_END,
    ])
    text = THESIS.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(RELEASE_BEGIN) + ".*?" + re.escape(RELEASE_END), flags=re.S
    )
    if not pattern.search(text):
        raise RuntimeError(
            f"Versioned-release markers absent from {THESIS}; cannot embed "
            "the archive identifiers."
        )
    THESIS.write_text(pattern.sub(lambda _: block, text, count=1),
                      encoding="utf-8")
    print(f"Updated versioned-release identifiers in {THESIS}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", default="NOT_IDENTIFIABLE")
    parser.add_argument("--release-asset-url", default="PENDING_PUBLIC_RELEASE")
    parser.add_argument("--release-tag", default="thesis-final-2026-08-01-r2")
    args = parser.parse_args()
    if (EVIDENCE / "EXECUTION.lock").exists():
        raise RuntimeError("Refusing to freeze evidence while an execution lock exists")
    verification = EVIDENCE / "VERIFY_PASSED.json"
    if not verification.exists():
        raise FileNotFoundError(
            "Run reviewer_final_closure.py --mode verify successfully first"
        )
    verified = json.loads(verification.read_text(encoding="utf-8"))
    if verified.get("status") != "PASSED":
        raise RuntimeError("Numerical verification status is not PASSED")

    files = sorted(
        path
        for path in EVIDENCE.rglob("*")
        if path.is_file()
        and "release" not in path.relative_to(EVIDENCE).parts
        and path.suffix.lower() != ".zip"
    )
    if not files:
        raise RuntimeError("No verified evidence files found")
    index_rows = [
        {
            "relative_path": (
                Path("final_reviewer_closure")
                / path.relative_to(EVIDENCE)
            ).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    index_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        index_buffer, fieldnames=("relative_path", "sha256", "size_bytes")
    )
    writer.writeheader()
    writer.writerows(index_rows)
    index_bytes = index_buffer.getvalue().encode("utf-8")

    temp = AUDIT / "frozen_numerical_evidence_payload.zip.tmp"
    AUDIT.mkdir(parents=True, exist_ok=True)
    if temp.exists():
        temp.unlink()
    with zipfile.ZipFile(
        temp,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as bundle:
        for path, row in zip(files, index_rows):
            with path.open("rb") as source, bundle.open(
                zip_info(row["relative_path"]), "w", force_zip64=True
            ) as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        bundle.writestr(
            zip_info("final_reviewer_closure/PAYLOAD_SHA256SUMS.csv"),
            index_bytes,
        )
    temp.replace(ARCHIVE)
    archive_sha = sha256(ARCHIVE)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(ARCHIVE.resolve()),
        "archive_sha256": archive_sha,
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_filename": ARCHIVE.name,
        "public_repository_url": REPOSITORY_URL,
        "public_release_asset_url": args.release_asset_url,
        "final_release_tag": args.release_tag,
        "frozen_source_evidence_commit": args.source_commit,
        "source_root": str(EVIDENCE.resolve()),
        "source_file_count": len(files),
        "embedded_checksum_index": (
            "final_reviewer_closure/PAYLOAD_SHA256SUMS.csv"
        ),
        "embedded_checksum_index_sha256": sha256_bytes(index_bytes),
        "verification_source": str(verification.resolve()),
        "verification_source_sha256": sha256(verification),
        "deterministic_zip_timestamp": FIXED_ZIP_TIME,
        "excluded": ["release/**", "*.zip"],
        "scope_note": (
            "This checksum identifies the frozen numerical evidence payload. "
            "It is printed in the thesis release record together with the "
            "embedded checksum-index digest and public asset URL."
        ),
    }
    MANIFEST.write_text(json.dumps(record, indent=2), encoding="utf-8")
    update_thesis_release_block(
        archive_sha,
        sha256_bytes(index_bytes),
        args.source_commit,
        args.release_asset_url,
        args.release_tag,
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
