"""Tests for stateful EMA, Butterworth, and contact hysteresis."""
import numpy as np
import pytest

from config import EncoderConfig
from preprocessing import ForcePreprocessor
from pipeline import ContactPulser


def test_ema_step():
    p = ForcePreprocessor(EncoderConfig(smoothing_alpha=0.5))
    p.process([0, 0, 0, 0])
    values, _ = p.process([1, 0, 0, 0])
    np.testing.assert_allclose(values, [0.5, 0, 0, 0])


def test_hysteresis_keeps_contact_on_below_on_threshold():
    p = ForcePreprocessor(EncoderConfig(smoothing_alpha=1.0))
    assert p.process([0.2, 0, 0, 0])[1][0]
    assert p.process([0.12, 0, 0, 0])[1][0]
    assert not p.process([0.09, 0, 0, 0])[1][0]


def test_butterworth_filters_without_bogus_initial_zero():
    p = ForcePreprocessor(EncoderConfig(filter_kind="butterworth", sample_rate_hz=50))
    first, _ = p.process([0.6, 0.4, 0.2, 0.0])
    np.testing.assert_allclose(first, [0.6, 0.4, 0.2, 0.0], atol=1e-8)
    for _ in range(10):
        filtered, _ = p.process([1, 1, 1, 1])
    assert all(0 <= value <= 1 for value in filtered)


def test_nonfinite_raises():
    with pytest.raises(ValueError):
        ForcePreprocessor(EncoderConfig()).process([float("nan"), 0, 0, 0])


def test_pulse_duration_uses_whole_samples_without_rounding_up():
    pulser = ContactPulser(EncoderConfig(sample_rate_hz=50.0, pulse_max_s=0.05))
    assert pulser.n_steps == 2
    assert pulser.update([True, False, False, False]) == [True, False, False, False]
    assert pulser.update([True, False, False, False]) == [True, False, False, False]
    assert pulser.update([True, False, False, False]) == [False, False, False, False]


def test_pulser_rejects_non_boolean_contacts():
    pulser = ContactPulser(EncoderConfig())
    with pytest.raises(ValueError, match="Contact flags must be booleans"):
        pulser.update([0, 1, 0, 1])
