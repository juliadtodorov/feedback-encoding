"""Configuration shared by preprocessing and feedback encoding."""
from dataclasses import dataclass
import math
from typing import Literal

FilterKind = Literal["ema", "butterworth"]
Curve = Literal["linear", "sqrt", "log"]


@dataclass(frozen=True)
class EncoderConfig:
    """Validated tuning parameters for the streaming feedback pipeline.

    The pipeline consumes four normalized force values. Gains and outputs are
    dimensionless mapping values; this configuration does not describe a
    hardware drive voltage, actuator displacement, or force calibration.
    """

    # Inputs are assumed to arrive at this rate. It determines the Butterworth
    # filter's frequency interpretation and the discrete LRA pulse duration.
    filter_kind: FilterKind = "ema"
    sample_rate_hz: float = 50.0
    # EMA trade-off: 1.0 follows each sample immediately; smaller values smooth
    # more strongly and therefore introduce more lag.
    smoothing_alpha: float = 0.4
    butterworth_cutoff_hz: float = 5.0
    butterworth_order: int = 2

    # Hysteresis prevents rapid switching: contact starts at contact_on and
    # remains active until the filtered force falls to contact_off or below.
    contact_on: float = 0.15
    contact_off: float = 0.10

    # Non-negative scale factors for the abstract command groups. Values above
    # 1 are allowed and saturate at the output range's upper bound.
    vibration_gain: float = 1.0
    pressure_gain: float = 1.0
    stretch_gain: float = 1.0

    # LRA commands are force-scaled during a contact-onset pulse. The duration
    # is quantized to whole samples using sample_rate_hz.
    pulse_max_s: float = 0.3

    # ERM commands rescale force above contact_off, apply this curve, then
    # optionally quantize to erm_levels steps (0 means continuous).
    erm_curve: Curve = "sqrt"
    erm_curve_k: float = 3.0
    erm_levels: int = 0

    # MuJoCo adapter scale: this many newtons map to normalized input 1.0.
    mujoco_full_scale_newtons: float = 1.5

    def __post_init__(self) -> None:
        """Reject values that would make filters or mappings undefined."""
        if self.filter_kind not in ("ema", "butterworth"):
            raise ValueError("filter_kind must be 'ema' or 'butterworth'")
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if not math.isfinite(self.smoothing_alpha) or not 0 < self.smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be finite and in (0, 1]")
        if (
            isinstance(self.butterworth_order, bool)
            or not isinstance(self.butterworth_order, int)
            or self.butterworth_order < 1
        ):
            raise ValueError("butterworth_order must be a positive integer")
        if (
            not math.isfinite(self.butterworth_cutoff_hz)
            or not 0 < self.butterworth_cutoff_hz < self.sample_rate_hz / 2
        ):
            raise ValueError("Butterworth cutoff must be finite and below Nyquist")
        if not (
            math.isfinite(self.contact_off)
            and math.isfinite(self.contact_on)
            and 0 <= self.contact_off < self.contact_on <= 1
        ):
            raise ValueError("Require finite thresholds: 0 <= contact_off < contact_on <= 1")
        for name, gain in (
            ("vibration_gain", self.vibration_gain),
            ("pressure_gain", self.pressure_gain),
            ("stretch_gain", self.stretch_gain),
        ):
            if not math.isfinite(gain) or gain < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if not math.isfinite(self.pulse_max_s) or self.pulse_max_s <= 0:
            raise ValueError("pulse_max_s must be positive")
        if self.erm_curve not in ("linear", "sqrt", "log"):
            raise ValueError("erm_curve must be 'linear', 'sqrt' or 'log'")
        if not math.isfinite(self.erm_curve_k) or self.erm_curve_k <= 0:
            raise ValueError("erm_curve_k must be positive and finite")
        if (
            isinstance(self.erm_levels, bool)
            or not isinstance(self.erm_levels, int)
            or self.erm_levels < 0
            or self.erm_levels == 1
        ):
            raise ValueError("erm_levels must be 0 (continuous) or >= 2")
        if (
            not math.isfinite(self.mujoco_full_scale_newtons)
            or self.mujoco_full_scale_newtons <= 0
        ):
            raise ValueError("mujoco_full_scale_newtons must be positive and finite")
