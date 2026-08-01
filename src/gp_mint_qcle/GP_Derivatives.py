"""Backward-compatible import alias for :mod:`GPDerivatives`.

The upload contained two byte-identical derivative implementations under
different names.  Keeping two editable copies invites silent divergence;
``GPDerivatives.py`` is now canonical and this legacy filename re-exports it.
"""

from .GPDerivatives import *  # noqa: F401,F403
