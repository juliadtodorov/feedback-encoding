"""Pure strategy: four processed forces -> twelve abstract command slots.

This module owns the force-to-command mapping. Stateful behavior such as pulse
timing is computed by pipeline.py and supplied through the pulses argument.
"""
from collections.abc import Sequence

import numpy as np

from config import EncoderConfig

MODES = frozenset({"lra", "erm", "mechanotactile", "skin_stretch", "all"})


def _erm_levels(active: np.ndarray, cfg: EncoderConfig) -> np.ndarray:
    """Map active force to ERM intensity using a curve and optional quantization.

    Rescaling starts at contact_off so an active contact has no ERM output at
    that boundary. Hysteresis can keep contact active below contact_on during
    release; the intensity then follows its lower filtered force.
    """
    floor = cfg.contact_off
    n = np.clip((active - floor) / (1.0 - floor), 0.0, 1.0)
    if cfg.erm_curve == "sqrt":
        n = np.sqrt(n)
    elif cfg.erm_curve == "log":
        n = np.log1p(cfg.erm_curve_k * n) / np.log1p(cfg.erm_curve_k)
    if cfg.erm_levels:
        # Any contact gets at least the lowest step; no contact stays silent.
        steps = np.minimum(np.ceil(n * cfg.erm_levels), cfg.erm_levels) / cfg.erm_levels
        n = np.where(n > 0, steps, 0.0)
    return np.clip(cfg.vibration_gain * n, 0.0, 1.0)


def encode_feedback(
    forces: Sequence[float],
    contacts: Sequence[bool],
    mode: str = "lra",
    config: EncoderConfig | None = None,
    pulses: Sequence[bool] | None = None,
) -> dict[str, list[float]]:
    """Map four processed forces and contact states into abstract commands.

    Returns four LRA and four ERM values in [0, 1], two mechanotactile values
    in [0, 1], and two signed skin-stretch values in [-1, 1]. Unselected modes
    are zero-filled. These are dimensionless mappings, not hardware signals.

    LRA intensity follows the active force during a contact pulse. If pulses
    are omitted, contact flags act as the pulse flags, which is useful for
    stateless direct calls.
    """
    forces_array = np.asarray(forces, dtype=float)
    contact_array = np.asarray(contacts)
    if forces_array.shape != (4,) or contact_array.shape != (4,):
        raise ValueError("Expected four forces and four contact flags")
    if not np.isfinite(forces_array).all():
        raise ValueError("Forces must be finite")
    if not np.issubdtype(contact_array.dtype, np.bool_):
        raise ValueError("Contact flags must be booleans")
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")

    cfg = config or EncoderConfig()
    # Ignore residual/noisy force when the preprocessor says there is no contact.
    active = np.where(contact_array, np.clip(forces_array, 0.0, 1.0), 0.0)
    f1, f2, f3, f4 = active

    # LRA: per-finger force intensity, enabled only during its contact pulse.
    if pulses is None:
        pulse_array = contact_array
    else:
        pulse_array = np.asarray(pulses)
        if pulse_array.shape != (4,):
            raise ValueError("Expected four pulse flags")
        if not np.issubdtype(pulse_array.dtype, np.bool_):
            raise ValueError("Pulse flags must be booleans")
    # A pulse can never outlive contact; gain and saturation apply consistently
    # to both vibration modalities, while ERM remains graded by its own curve.
    lra = np.clip(cfg.vibration_gain * active * pulse_array, 0.0, 1.0)

    # ERM: graded force.
    erm = _erm_levels(active, cfg)

    pressure = np.clip(
        cfg.pressure_gain * np.array([(f1 + f2) / 2, (f3 + f4) / 2]),
        0.0, 1.0,
    )
    # Pairwise force difference is only a heuristic direction/magnitude cue;
    # it is not a measured or validated estimate of slip.
    stretch = np.clip(
        cfg.stretch_gain * np.array([f1 - f2, f3 - f4]), -1.0, 1.0
    )

    return {
        "lra": lra.tolist() if mode in ("lra", "all") else [0.0] * 4,
        "erm": erm.tolist() if mode in ("erm", "all") else [0.0] * 4,
        "mechanotactile": pressure.tolist() if mode in ("mechanotactile", "all") else [0.0] * 2,
        "skin_stretch": stretch.tolist() if mode in ("skin_stretch", "all") else [0.0] * 2,
    }
