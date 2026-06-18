from __future__ import annotations

"""Unified command-line interface for GP-MQCLD."""

import argparse
import sys
from typing import Sequence

from . import __version__


def _dispatch(module_main, argv: Sequence[str], prog: str) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [prog, *argv]
        module_main()
    finally:
        sys.argv = old_argv


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="gp-mqcld",
        description="Unified CLI for GP-MQCLD: Gaussian-process-based mapping quantum-classical Liouville dynamics.",
    )
    parser.add_argument("--version", action="store_true", help="Print package version and exit.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Run one PBME + midpoint GP-MInt simulation. Remaining args are passed to gp-mqcld-run.")
    sub.add_parser("compare", help="Compare saved GP runs against SE/QCLE/PBME/midpoint. Remaining args are passed to gp-mqcld-compare.")
    sub.add_parser("smoke", help="Run a tiny installation smoke test.")

    if not argv:
        parser.print_help()
        return

    if argv[0] == "--version":
        print(__version__)
        return

    command = argv[0]
    rest = argv[1:]

    if command == "run":
        from .run import main as run_main
        _dispatch(run_main, rest, "gp-mqcld run")
        return
    if command == "compare":
        from .Compare_gp_se_qcle import main as compare_main
        _dispatch(compare_main, rest, "gp-mqcld compare")
        return
    if command == "smoke":
        from .cli_smoke import main as smoke_main
        _dispatch(smoke_main, rest, "gp-mqcld smoke")
        return

    parser.print_help()
    raise SystemExit(f"Unknown gp-mqcld command: {command!r}")


if __name__ == "__main__":
    main()
