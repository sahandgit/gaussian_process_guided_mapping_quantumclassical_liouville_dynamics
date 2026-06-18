from __future__ import annotations

"""Tiny installed smoke test for the GP-MQCLD package."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny one-step GP-MQCLD smoke test.")
    parser.add_argument("--out", default="runs/smoke_full", help="Output directory for the smoke run.")
    args = parser.parse_args()

    from .run import main as run_main

    out = Path(args.out)
    argv = [
        "gp-mqcld-smoke",
        "--density_mode", "full",
        "--sampling_mode", "focused",
        "--n_train", "32",
        "--n_steps", "1",
        "--snapshot_every", "1",
        "--panel_every", "0",
        "--n_opt_steps", "1",
        "--dt", "0.1",
        "--out", str(out),
        "--quiet",
        "--skip_figures",
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        run_main()
    finally:
        sys.argv = old_argv

    required = [out / "pbme.npz", out / "midpoint.npz", out / "pbme.json", out / "midpoint.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Smoke run finished but expected files are missing: " + ", ".join(missing))

    print(f"Smoke test passed. Outputs written to: {out}")


if __name__ == "__main__":
    main()
