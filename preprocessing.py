"""Validate/smooth 4 normalized forces and estimate contact via hysteresis."""


from collections.abc import Sequence

import numpy as np
from scipy import signal

from config import EncoderConfig


class ForcePreprocessor:
    """Smooth normalized sensor samples and track per-finger contact state.

    Each instance owns its filter history and contact flags, so independent
    trials should call reset() or use a fresh instance.
    """

    def __init__(self, config: EncoderConfig):
        # EncoderConfig validates these shared values at construction, keeping
        # invalid filter parameters from reaching SciPy or state initialization.
        self.config = config
        self._sos = signal.butter(
            config.butterworth_order,
            config.butterworth_cutoff_hz,
            fs=config.sample_rate_hz,
            output="sos",
        ) if config.filter_kind == "butterworth" else None
        self.reset()

    def reset(self) -> None:
        """Forget smoothing/contact history before starting a new trial."""
        self._filtered: np.ndarray | None = None
        self._zi: np.ndarray | None = None
        self._contact = np.zeros(4, dtype=bool)

    def process(self, sample: Sequence[float]) -> tuple[list[float], list[bool]]:
        """Filter one four-force sample and return values plus contact flags.

        Finite out-of-range values are clipped intentionally, matching the
        simulation adapter's saturation behavior. Callers remain responsible
        for ensuring their readings have already been meaningfully calibrated.
        """
        arr = np.asarray(sample, dtype=float)
        if arr.shape != (4,):
            raise ValueError("Expected exactly four sensor values (shape: 4,)")
        if not np.isfinite(arr).all():
            raise ValueError("Sensor values must be finite")
        arr = np.clip(arr, 0.0, 1.0)

        if self.config.filter_kind == "ema":
            if self._filtered is None:
                # Seed from the first sample rather than ramping up from zero.
                self._filtered = arr.copy()
            else:
                a = self.config.smoothing_alpha
                self._filtered = a * arr + (1 - a) * self._filtered
        else:
            if self._sos is None:
                raise RuntimeError("Butterworth coefficients were not initialized")
            if self._zi is None:
                # Initialize at the first reading to avoid a spurious startup transient.
                self._zi = signal.sosfilt_zi(self._sos)[:, :, None] * arr[None, None, :]
            filtered, self._zi = signal.sosfilt(
                self._sos, arr[np.newaxis, :], axis=0, zi=self._zi
            )
            self._filtered = filtered[0].copy()

        self._filtered = np.clip(self._filtered, 0.0, 1.0)
        # The previous contact state selects which threshold applies: rising
        # into contact uses contact_on; an active contact releases at contact_off.
        self._contact = np.where(
            self._contact,
            self._filtered > self.config.contact_off,
            self._filtered >= self.config.contact_on,
        )
        return self._filtered.tolist(), self._contact.tolist()
