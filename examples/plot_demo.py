"""Visualize a simulated force stream + feedback commands (no hardware needed).

Run: python plot_demo.py
Produces outputs/feedback_demo.png; no GUI/display required.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless: save plots, do not open a GUI window.
import matplotlib.pyplot as plt
import numpy as np

from config import EncoderConfig
from pipeline import FeedbackPipeline


def main() -> None:
    sample_rate = 50.0
    t = np.arange(0.0, 5.0, 1.0 / sample_rate)
    rng = np.random.default_rng(7)

    # Different grips over time, with gentle sensor-like noise.
    envelope = np.clip((t - 0.5) / 0.8, 0, 1) * np.clip((4.5 - t) / 0.8, 0, 1)
    four_forces = envelope[:, None] * np.array([0.35, 0.8, 0.55, 0.25])
    four_forces = np.clip(four_forces + rng.normal(0, 0.02, (len(t), 4)), 0, 1)

    pipeline = FeedbackPipeline(EncoderConfig(filter_kind="butterworth", sample_rate_hz=sample_rate))
    filtered, vibration, pressure = [], [], []
    for sample in four_forces:
        result = pipeline.step(sample, mode="all")
        filtered.append(result["filtered_forces"])
        vibration.append(result["commands"]["lra"])
        pressure.append(result["commands"]["mechanotactile"])

    filtered = np.asarray(filtered)
    vibration = np.asarray(vibration)
    pressure = np.asarray(pressure)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for finger in range(4):
        axes[0].plot(t, four_forces[:, finger], alpha=0.35)
        axes[0].plot(t, filtered[:, finger], label=f"FSR {finger + 1}")
        axes[1].plot(t, vibration[:, finger], label=f"LRA {finger + 1}")
    axes[2].plot(t, pressure[:, 0], label="Servo group FSR 1+2")
    axes[2].plot(t, pressure[:, 1], label="Servo group FSR 3+4")
    for ax, ylabel in zip(axes, ["Normalized force", "Logical LRA intensity", "Logical pressure"]):
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Prosthetic hand feedback: sample strategies")
    fig.tight_layout()
    target = Path("outputs/feedback_demo.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {target}")


if __name__ == "__main__":
    main()
