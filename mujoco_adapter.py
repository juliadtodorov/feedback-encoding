"""Optional MuJoCo input adapter: measure normal contact force at four geoms.

Provides forces in newtons, then normalizes them for pipeline.step().
This is NOT a simulation of electrical FSR resistance or calibration.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco


def contact_forces_n(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_geom_names: Sequence[str],
) -> np.ndarray:
    """Sum absolute normal contact forces on four named 'fingertip' geoms.

    data must have been updated by mujoco.mj_step/mj_forward. Unit: newtons.
    Forces are summed if multiple contacts touch the same geom.
    """
    import mujoco  # Imported only when reading a live MuJoCo model.

    if len(sensor_geom_names) != 4 or len(set(sensor_geom_names)) != 4:
        raise ValueError("Specify four different fingertip geometry names")
    geom_ids = []
    for name in sensor_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"MuJoCo geom not found: {name}")
        geom_ids.append(geom_id)

    result = np.zeros(4, dtype=float)
    lookup = {geom_id: i for i, geom_id in enumerate(geom_ids)}
    wrench = np.zeros(6, dtype=float)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 not in lookup and contact.geom2 not in lookup:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        magnitude_n = abs(float(wrench[0]))
        if contact.geom1 in lookup:
            result[lookup[contact.geom1]] += magnitude_n
        if contact.geom2 in lookup:
            result[lookup[contact.geom2]] += magnitude_n
    return result


def normalize_forces(forces_n: Sequence[float], full_scale_n: float) -> list[float]:
    """Convert 4 normal forces (N) to values from 0 to 1 for the encoder."""
    forces = np.asarray(forces_n, dtype=float)
    if forces.shape != (4,) or not np.isfinite(forces).all():
        raise ValueError("Expected four finite normal forces in newtons")
    if not np.isfinite(full_scale_n) or full_scale_n <= 0:
        raise ValueError("full_scale_n must be positive and finite")
    return np.clip(forces / full_scale_n, 0.0, 1.0).tolist()
