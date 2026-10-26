"""Unit tests for the four-to-twelve encoding strategy (run with pytest)."""
import numpy as np
import pytest

from config import EncoderConfig
from encoding import encode_feedback
from pipeline import FeedbackPipeline


def test_all_twelve_slots_are_returned():
    commands = encode_feedback([0.2, 0.8, 0.5, 0.1], [True] * 4, mode="all")
    assert [len(v) for v in commands.values()] == [4, 4, 2, 2]
    np.testing.assert_allclose(commands["mechanotactile"], [0.5, 0.3])
    np.testing.assert_allclose(commands["skin_stretch"], [-0.6, 0.4])


def test_lra_mode_disables_unselected_outputs():
    commands = encode_feedback([0.2, 0.8, 0.5, 0.1], [True] * 4)
    np.testing.assert_allclose(commands["lra"], [0.2, 0.8, 0.5, 0.1])
    assert commands["erm"] == [0] * 4
    assert commands["mechanotactile"] == [0] * 2
    assert commands["skin_stretch"] == [0] * 2


def test_contact_thresholds():
    pipeline = FeedbackPipeline(EncoderConfig(smoothing_alpha=1.0))
    result = pipeline.step([0.1, 0.2, 0.0, 0.0])
    assert result["contacts"] == [False, True, False, False]
    np.testing.assert_allclose(result["commands"]["lra"], [0, 0.2, 0, 0])


def test_no_force_generates_no_feedback():
    result = FeedbackPipeline().step([0, 0, 0, 0], mode="all")
    assert all(all(value == 0 for value in arr) for arr in result["commands"].values())


def test_four_inputs_required():
    with pytest.raises(ValueError):
        FeedbackPipeline().step([0.3, 0.4, 0.5])


def test_reset_clears_history():
    pipeline = FeedbackPipeline()
    pipeline.step([1, 1, 1, 1])
    pipeline.reset()
    assert pipeline.step([0, 0, 0, 0])["contacts"] == [False] * 4


def test_unselected_contact_is_zero():
    commands = encode_feedback([1, 1, 1, 1], [True, False, True, False], mode="all")
    assert commands["lra"] == [1, 0, 1, 0]


def test_lra_force_intensity_respects_pulse_and_vibration_gain():
    commands = encode_feedback(
        [0.2, 0.8, 0.5, 0.1],
        [True] * 4,
        mode="lra",
        config=EncoderConfig(vibration_gain=0.5),
        pulses=[True, False, True, True],
    )
    np.testing.assert_allclose(commands["lra"], [0.1, 0.0, 0.25, 0.05])


def test_invalid_mode_does_not_advance_pipeline_state():
    pipeline = FeedbackPipeline(EncoderConfig(smoothing_alpha=0.5))
    with pytest.raises(ValueError, match="mode must"):
        pipeline.step([1, 1, 1, 1], mode="unknown")
    result = pipeline.step([0, 0, 0, 0])
    assert result["filtered_forces"] == [0.0] * 4


def test_contact_flags_must_be_booleans():
    with pytest.raises(ValueError, match="Contact flags must be booleans"):
        encode_feedback([0.2] * 4, [0, 1, 0, 1])


def test_pulse_flags_must_be_booleans():
    with pytest.raises(ValueError, match="Pulse flags must be booleans"):
        encode_feedback([0.2] * 4, [True] * 4, pulses=[0, 1, 0, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"erm_curve": "log", "erm_curve_k": 0.0},
        {"sample_rate_hz": float("inf")},
        {"butterworth_cutoff_hz": 25.0},
        {"contact_on": 0.05, "contact_off": 0.1},
        {"vibration_gain": -1.0},
        {"erm_levels": 1},
        {"mujoco_full_scale_newtons": 0.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EncoderConfig(**kwargs)
