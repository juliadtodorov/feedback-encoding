# Prosthetic Hand Feedback Encoding — Python prototype 

An **encoding-strategy prototype** logic to translate input from sensors to output for actuators 
It takes **four normalized force inputs** (corresponding to four FSRs) and outputs **up to twelve logical feedback commands**: four LRA, four ERM, two mechanotactile and two skin-stretch
There is **no electronics or Arduino firmware** in this repository

**Start here:** Installation instructions are below. The repository includes NumPy, SciPy, Matplotlib, MuJoCo, and pytest, with executable examples and automated tests. You do **not** need the actual sensors, actuators, or robotic-hand model to get started.

## SET-UP in terminal
git clone https://github.com/USERNAME/feedback-encoding.git
cd feedback-encoding

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"


## Running the demos
'python -m examples.playground' --Opens an interactive Matplotlib window where users can adjust four simulated FSR inputs and visualize the resulting twelve actuator commands
`python -m examples.demo` --Prints how a few made-up four-channel force readings map to feedback commands
`python -m examples.plot_demo` --Saves `outputs/feedback_demo.png` with force, filtering, and encoding over time 
`python -m examples.demo_mujoco` --Runs a **headless** MuJoCo contact simulation and prints four contact forces + feedback commands
`python -m pytest -v` --Runs the automated tests and reports pass/fail |


## File map

```text
feedback-encoding/
├── README.md               # setup, usage and design
├── pyproject.toml
├── .gitignore             # Don't upload environments and generated outputs
|
├── config.py               # Thresholds, smoothing, gains, simulation scaling
├── preprocessing.py        # NumPy validation, EMA/SciPy Butterworth, contact detection
├── encoding.py             # MAIN LOGIC: map four force values to 12 command slots
├── pipeline.py             # Preprocessing -> encoder, maintains state
├── mujoco_adapter.py      # Read contact forces from named MuJoCo geoms
└── tests/
    ├── test_encoding.py
    ├── test_preprocessing.py
    └── test_mujoco_adapter.py
└── examples/
    ├── demo.py                 # Example hardcoded forces
    ├── plot_demo.py            # NumPy + Matplotlib: visualize a force time series
    ├── demo_mujoco.py          # Minimal runnable MuJoCo contact demonstration
    ├── playground.py           #Interactive input-> output
```


## 6. Data interface — what goes in and what comes out

Each call to `pipeline.step()` expects **one sample of four numbers**, **already normalized to `[0, 1]`**, e.g.:

can run this in demo.py:
```python
from pipeline import FeedbackPipeline

pipeline = FeedbackPipeline()
result = pipeline.step([0.2, 0.8, 0.5, 0.1], mode="all")
print(result["commands"])
```

**Do not pass uncalibrated FSR ADC counts, volts or newtons directly into `step()`.** Calibrate/normalize real sensor values first, or use the MuJoCo adapter for simulated normal-force data. The example simulator adapter divides forces (newtons) by a configurable full-scale force before clipping them to `[0, 1]`.

Processing:

```text
Four normalized readings [0, 1]
          |
          v
preprocessing.py: smoothing (EMA or SciPy Butterworth) + contact hysteresis
          |
          v
encoding.py: select feedback modality and map four values to twelve slots
          |
          v
{ "lra": [4], "erm": [4], "mechanotactile": [2], "skin_stretch": [2] }
```

Outputs are **plain Python lists**, easy to log or serialize:

- `lra` / `erm`: four unitless **intensities** in `[0, 1]`, one per sensor. LRA
  intensity follows the filtered force only during the contact-onset pulse;
  ERM intensity is a graded force mapping for the duration of contact.
- `mechanotactile`: two unitless **pressure / displacement requests** in `[0, 1]`, averaging FSRs 1+2 and 3+4.
- `skin_stretch`: two **signed differential-force requests** in `[-1, 1]`, computed as FSR1-FSR2 and FSR3-FSR4. These are **not measured slip**.

Modes: `"lra"`, `"erm"`, `"mechanotactile"`, `"skin_stretch"`, or `"all"`. The encoder always provides twelve command **slots**, but unselected outputs are zero. `"all"` is for **comparing encodings**, not a claim that the electronics can drive twelve actuators at the same time.

**These are experimental logical commands, not motor voltages, vibration frequency, servo angles, real pressure, validated sensory substitution, or a safety-certified controller.**

## Updating Strategy

`encoding.py` to change **what the user should feel**. Start with the commented sections for `vibration`, `pressure`, and `stretch`. 

`config.py` to change thresholds, mapping gains and filter settings.

The current mapping logic is:

- **LRA:** each fingertip's active filtered force, multiplied by `vibration_gain`,
  is emitted only during a pulse that starts on contact onset. The pulse stops
  when contact releases or its maximum sample-count duration expires.
- **ERM:** active force is rescaled from `contact_off` to 1, passed through the
  selected `linear`, `sqrt`, or `log` curve, optionally quantized, multiplied by
  `vibration_gain`, and clipped to `[0, 1]`.
- **Mechanotactile:** averages active forces for each configured sensor pair,
  scales by `pressure_gain`, and clips to `[0, 1]`.
- **Skin stretch:** subtracts the second active force from the first in each
  pair, scales by `stretch_gain`, and clips to `[-1, 1]`. This is a heuristic
  differential-force cue, not a validated stretch or slip estimate.

`EncoderConfig` validates shared filter, threshold, gain, pulse, curve, and
normalization-scale parameters at construction. The preprocessing and encoding
boundaries still clip finite sensor forces to `[0, 1]`; that saturation is not
a substitute for calibrating physical sensors. `sample_rate_hz` is assumed,
not measured, so the caller must provide samples at the configured rate for
filter cutoff and pulse duration to have their intended timing.

For example, to test a SciPy Butterworth low-pass filter in Python:

```python
from config import EncoderConfig
from pipeline import FeedbackPipeline

config = EncoderConfig(filter_kind="butterworth", sample_rate_hz=50.0,
                       butterworth_cutoff_hz=5.0)
pipeline = FeedbackPipeline(config)
for sample in [[0.0] * 4, [0.2, 0.8, 0.5, 0.1]]:
    result = pipeline.step(sample, mode="all")
    print(result["commands"])
```

The Butterworth filter assumes equally spaced samples at `sample_rate_hz`; the ordinary EMA uses `smoothing_alpha`. Contact threshold hysteresis means activation and release have **different thresholds**. **Call `pipeline.reset()` before each independent simulation trial** so stale filter/contact state does not leak into new trials.

## MuJoCo Example

`demo_mujoco.py` builds a small XML scene inside Python with four separate falling spheres as **mock fingertips** and a ground plane. It advances the physics simulation and calls `mujoco_adapter.contact_forces_n()` to collect the **normal contact force in newtons** on the named geometries. It then uses `normalize_forces(..., full_scale_n=1.5)` to feed the same encoding pipeline as `demo.py`.

Later replace the XML scene with your robotic-hand model, pass four fingertip **geom names**, and configure full-scale force/calibration appropriately. The demo runs physics at **500 Hz**, encodes at **50 Hz**, and prints every 0.2 seconds. These are just demonstration settings

## Tests and team workflow

Run tests **after changing the encoding logic**:

python -m pytest -v

Tests verify 4-to-12 output shapes, contact thresholds/hysteresis, filtering, reset behavior, and the normalization adapter. Add test cases in `tests/` for any new encoding strategy.


## Planned extensions

- Additional mappings: contact onset pulses, nonlinear intensity scaling, force-rate encoding, and modality comparisons.
- Unit-aware calibration for actual FSRs and integration with the prosthetic-hand MuJoCo model.
- Logging and experiment scripts to compare mapping strategies over multiple trials.
- Arduino / actuator output modules **in a separate hardware integration stage**.
