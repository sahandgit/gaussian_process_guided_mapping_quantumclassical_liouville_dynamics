import numpy as np
from types import SimpleNamespace

from gp_mint_qcle.Dynamics import _effective_labels


def test_effective_labels_use_raw_labels_when_no_correction_weight():
    state = SimpleNamespace(y=np.array([1.0, -2.0, 3.0]), correction_weight=None)
    np.testing.assert_allclose(_effective_labels(state), [1.0, -2.0, 3.0])


def test_effective_labels_use_weighted_labels_when_weight_exists():
    state = SimpleNamespace(
        y=np.array([1.0, -2.0, 3.0]),
        correction_weight=np.array([0.5, 2.0, -1.0]),
    )
    np.testing.assert_allclose(_effective_labels(state), [0.5, -4.0, -3.0])
