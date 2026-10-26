"""Connect filtering, contact pulses, and encoding into a streaming pipeline."""
from collections.abc import Sequence
import math

import numpy as np

from config import EncoderConfig
from encoding import MODES, encode_feedback
from preprocessing import ForcePreprocessor


class ContactPulser:
    """Turns contact flags into capped pulses, one per fingertip.

    A pulse starts on contact onset and lasts as long as contact does, up to
    pulse_max_s. It re-arms only after contact is released.
    """

    def __init__(self, config: EncoderConfig):
        # A pulse is represented in whole samples. Floor keeps a pulse at or
        # below its requested duration, except when at least one sample is
        # required to represent a positive duration shorter than one interval.
        self.n_steps = max(1, math.floor(config.pulse_max_s * config.sample_rate_hz))
        self.reset()

    def reset(self) -> None:
        self._prev = np.zeros(4, dtype=bool)
        self._remaining = np.zeros(4, dtype=int)

    def update(self, contacts: Sequence[bool]) -> list[bool]:
        c = np.asarray(contacts)
        if c.shape != (4,):
            raise ValueError("Expected four contact flags")
        if not np.issubdtype(c.dtype, np.bool_):
            raise ValueError("Contact flags must be booleans")
        onset = c & ~self._prev
        # A new onset starts a fresh pulse; release cancels it and rearms the
        # finger so its next contact can produce another pulse.
        self._remaining = np.where(onset, self.n_steps, self._remaining)
        self._remaining = np.where(c, self._remaining, 0)   # release cancels
        pulse = c & (self._remaining > 0)
        self._remaining = np.maximum(self._remaining - 1, 0)
        self._prev = c
        return pulse.tolist()


class FeedbackPipeline:
    """Stateful streaming composition of the preprocessor and encoder."""

    def __init__(self, config: EncoderConfig | None = None):
        self.config = config or EncoderConfig()
        self.preprocessor = ForcePreprocessor(self.config)
        self.pulser = ContactPulser(self.config)

    def reset(self) -> None:
        """Clear filter, hysteresis, and pulse history before a new trial."""
        self.preprocessor.reset()
        self.pulser.reset()

    def step(self, normalized_forces: Sequence[float], mode: str = "lra") -> dict:
        """Process one four-element normalized sample and return its result.

        Validate the mode before touching state: a caller typo must not advance
        filters or consume a pulse sample.
        """
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        filtered, contacts = self.preprocessor.process(normalized_forces)
        pulses = self.pulser.update(contacts)
        commands = encode_feedback(filtered, contacts, mode, self.config, pulses=pulses)
        return {
            "filtered_forces": filtered,
            "contacts": contacts,
            "pulses": pulses,
            "commands": commands,
        }
