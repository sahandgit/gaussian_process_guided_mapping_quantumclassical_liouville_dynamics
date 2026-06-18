"""Legacy public import alias. Prefer ``gp_mqcld`` for new code.

The implementation package is ``gp_mint_qcle`` because the original research
modules use explicit GP/MInt/QCLE terminology.  This namespace provides the
legacy project name while preserving backwards-compatible imports.
"""
from __future__ import annotations

import importlib
import sys

_impl = importlib.import_module("gp_mint_qcle")
__version__ = getattr(_impl, "__version__", "0.0.0")

# Register common submodule aliases so that, for example,
# ``from gp_mqcld.Models import TullyModel`` works.
_MODULES = [
    "Collector",
    "Compare_gp_se_qcle",
    "Dynamics",
    "GP_Density",
    "GP_DensityDiff",
    "GP_Derivatives",
    "GPDerivatives",
    "KDEDensity",
    "Mint",
    "Models",
    "Monodromy",
    "Observables",
    "Operator",
    "Sampling",
    "Visualization",
    "qcle_grid_tully",
    "run",
    "cli",
    "cli_smoke",
]

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"gp_mint_qcle.{_name}")

__all__ = ["__version__"]
