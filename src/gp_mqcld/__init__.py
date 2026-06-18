"""Public import alias for GP-MQCLD.

GP-MQCLD abbreviates **Gaussian-Process-Based Mapping Quantum-Classical
Liouville Dynamics**. The implementation namespace remains ``gp_mint_qcle``
for backward compatibility with the original research modules.
"""
from __future__ import annotations

import importlib
import sys

_impl = importlib.import_module("gp_mint_qcle")
__version__ = getattr(_impl, "__version__", "0.0.0")

_MODULES = ['Collector', 'Compare_gp_se_qcle', 'Dynamics', 'GP_Density', 'GP_DensityDiff', 'GP_Derivatives', 'GPDerivatives', 'KDEDensity', 'Mint', 'Models', 'Monodromy', 'Observables', 'Operator', 'Sampling', 'Visualization', 'qcle_grid_tully', 'run', 'cli', 'cli_smoke']

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"gp_mint_qcle.{_name}")

from gp_mint_qcle import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", []) + ["__version__"]
