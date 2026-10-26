"""Small unit tests for the simulation-to-encoder boundary."""
import numpy as np
import pytest

from mujoco_adapter import normalize_forces


def test_n_to_normalized():
    np.testing.assert_allclose(normalize_forces([0.0, 0.5, 1.0, 2.0], 1.0), [0, 0.5, 1, 1])


def test_bad_scale():
    with pytest.raises(ValueError):
        normalize_forces([0, 0, 0, 0], 0)
