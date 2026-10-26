"""Headless MuJoCo smoke demo: simulated contact forces -> feedback commands.

Uses four separate falling spheres as mock fingertips (not a robotic hand).
Run: python demo_mujoco.py
"""

import mujoco

from config import EncoderConfig
from mujoco_adapter import contact_forces_n, normalize_forces
from pipeline import FeedbackPipeline

MODEL_XML = """
<mujoco model="four_mock_fingertips">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="1 1 0.1"/>
    <body name="finger1" pos="-0.18 0 0.08"><freejoint/>
      <geom name="fsr_1" type="sphere" size="0.03" mass="0.03"/>
    </body>
    <body name="finger2" pos="-0.06 0 0.10"><freejoint/>
      <geom name="fsr_2" type="sphere" size="0.03" mass="0.06"/>
    </body>
    <body name="finger3" pos="0.06 0 0.12"><freejoint/>
      <geom name="fsr_3" type="sphere" size="0.03" mass="0.09"/>
    </body>
    <body name="finger4" pos="0.18 0 0.14"><freejoint/>
      <geom name="fsr_4" type="sphere" size="0.03" mass="0.12"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    cfg = EncoderConfig()
    pipeline = FeedbackPipeline(cfg)
    sensor_geom_names = ["fsr_1", "fsr_2", "fsr_3", "fsr_4"]

    # Simulate physics at 500 Hz but run encoding at 50 Hz (every 10 steps).
    steps_per_encoder_sample = round(1.0 / (cfg.sample_rate_hz * model.opt.timestep))
    for step in range(500):
        mujoco.mj_step(model, data)
        if (step + 1) % steps_per_encoder_sample == 0:
            measured_n = contact_forces_n(model, data, sensor_geom_names)
            normalized = normalize_forces(measured_n, cfg.mujoco_full_scale_newtons)
            output = pipeline.step(normalized, mode="all")
            if (step + 1) % 100 == 0:
                print(
                    f"t={data.time:.3f} s | contact force (N) "
                    f"{measured_n.round(3).tolist()} | LRA: "
                    f"{[round(x, 3) for x in output['commands']['lra']]}"
                )


if __name__ == "__main__":
    main()
