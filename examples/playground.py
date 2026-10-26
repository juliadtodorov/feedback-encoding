"""Interactive popup to explore 4 normalized FSR inputs -> 12 logical outputs.

Run from the repository root:
    python -m examples.playground

A native Matplotlib window opens with four force sliders and modality selector.
For clarity each slider move is treated as an independent snapshot (no EMA
history). Use plot_demo.py to visualize temporal filtering over time.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

from config import EncoderConfig
from pipeline import FeedbackPipeline

# Change these settings to experiment with contact thresholds or output gains.
CONFIG = EncoderConfig(smoothing_alpha=1.0)
STARTING_FORCES = [0.2, 0.8, 0.5, 0.1]
STARTING_MODE = "all"

CHANNELS = [
    ("LRA 1", "lra", 0), ("LRA 2", "lra", 1),
    ("LRA 3", "lra", 2), ("LRA 4", "lra", 3),
    ("ERM 1", "erm", 0), ("ERM 2", "erm", 1),
    ("ERM 3", "erm", 2), ("ERM 4", "erm", 3),
    ("Pressure 1", "mechanotactile", 0),
    ("Pressure 2", "mechanotactile", 1),
    ("Stretch 1", "skin_stretch", 0),
    ("Stretch 2", "skin_stretch", 1),
]
COLORS = {
    "lra": "tab:blue", "erm": "tab:green",
    "mechanotactile": "tab:orange", "skin_stretch": "tab:purple",
}


def snapshot(forces: list[float], mode: str) -> dict:
    """One independent test input; reset filter state between slider changes."""
    pipeline = FeedbackPipeline(CONFIG)
    return pipeline.step(forces, mode=mode)


def main() -> None:
    fig = plt.figure(figsize=(12, 7.5))
    fig.suptitle("Feedback encoding playground | 4 inputs → 12 outputs", fontsize=14)
    fig.text(0.06, 0.92, "Change FSR readings below (normalized 0–1)", fontsize=10)
    fig.text(0.06, 0.04,
             "Snapshot mode: each slider change is a fresh trial. "
             "Run plot_demo for time-dependent filtering.", fontsize=9)

    axis = fig.add_axes((0.44, 0.14, 0.52, 0.74))
    ys = np.arange(len(CHANNELS))
    bars = axis.barh(
        ys, np.zeros(len(CHANNELS)),
        color=[COLORS[key] for _, key, _ in CHANNELS], height=0.7,
    )
    axis.set_yticks(ys, [name for name, _, _ in CHANNELS])
    axis.invert_yaxis()
    axis.set_xlim(-1.06, 1.06)
    axis.axvline(0, color="gray", linewidth=1)
    axis.grid(axis="x", alpha=0.2)
    axis.set_xlabel("Logical command (signed for skin stretch)")
    axis.set_title("Output commands (not electrical signals)")
    labels = [axis.text(0.04, y, "0.00", va="center", fontsize=8) for y in ys]

    sliders = []
    for i, y in enumerate([0.77, 0.64, 0.51, 0.38]):
        slider_axis = fig.add_axes((0.1, y, 0.24, 0.035))
        sliders.append(Slider(slider_axis, f"FSR {i + 1}", 0, 1,
                              valinit=STARTING_FORCES[i], valstep=0.01))

    mode_axis = fig.add_axes((0.08, 0.08, 0.28, 0.22))
    mode_axis.set_title("Modality to preview", fontsize=10, pad=8)
    mode_buttons = RadioButtons(
        mode_axis, ["all", "lra", "erm", "mechanotactile", "skin_stretch"],
        active=0,
    )

    def update(_=None):
        forces = [slider.val for slider in sliders]
        result = snapshot(forces, mode_buttons.value_selected)
        commands = result["commands"]
        for j, (_, key, idx) in enumerate(CHANNELS):
            v = float(commands[key][idx])
            bars[j].set_width(v)
            labels[j].set_x(0.04 if v >= 0 else -0.04)
            labels[j].set_ha("left" if v >= 0 else "right")
            labels[j].set_text(f"{v:+.2f}" if key == "skin_stretch" else f"{v:.2f}")
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)
    mode_buttons.on_clicked(update)
    update()
    plt.show()  # GUI popup (requires a desktop/display-capable backend)


if __name__ == "__main__":
    main()
