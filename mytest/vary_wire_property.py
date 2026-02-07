"""
Two side‑by‑side rods to compare axial stretch vs. inextensible behavior.

- Both rods: length 1 m, 20 elements, aligned +x, one end fixed at the origin.
- Common settings: gravity, Young's modulus 1e6, damping 1e-2, dt = 2e-5.
- Rod B is made effectively inextensible by cranking its axial shear/stretch stiffness.
- Uses the shared 4‑view renderer from render_scripts/post_processing to output an MP4.

Run:
  python mytest/vary_wire_property.py
Outputs:
  - mytest/vary_wire_property_state.npz
  - mytest/vary_wire_property_4view.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import elastica as ea
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import GravityForces
from elastica.timestepper import integrate
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.callback_functions import CallBackBaseClass

# Make repo root importable so we can reuse the shared plotting helpers.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from render_scripts import post_processing as pp  # noqa: E402


class TwoRodSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping, ea.CallBacks
):
    """Minimal simulator mixin stack."""


class RodCallback(CallBackBaseClass):
    def __init__(self, step_skip: int, store: dict, key: str):
        super().__init__()
        self.step_skip = step_skip
        self.store = store
        self.key = key
        self.time: list[float] = []
        self.position: list[np.ndarray] = []

    def make_callback(self, system, time, current_step):
        if current_step % self.step_skip:
            return
        self.time.append(time)
        self.position.append(system.position_collection.copy())
        self.store[self.key] = {"time": self.time, "position": self.position}


def make_rod(
    sim: TwoRodSimulator,
    *,
    y_offset: float,
    dt: float,
    youngs_modulus: float,
    damping_constant: float,
    inextensible: bool,
) -> ea.CosseratRod:
    n_elem = 20
    base_length = 1.0
    base_radius = 0.01  # 1 cm radius keeps things numerically tame
    density = 1000.0

    start = np.array([0.0, y_offset, 0.0])
    direction = np.array([1.0, 0.0, 0.0])  # +x
    normal = np.array([0.0, 0.0, 1.0])

    rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
    )

    if inextensible:
        # Penalize axial stretch: amplify the 3rd (axial) diagonal of shear_matrix.
        stiffening = 1.0e5
        rod.shear_matrix[2, 2, :] *= stiffening

    sim.append(rod)

    # Fix the left end (node 0) in both position and orientation.
    sim.constrain(rod).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    # Gravity straight down the -y axis.
    sim.add_forcing_to(rod).using(
        GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
    )

    # Simple linear damping.
    sim.dampen(rod).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )

    return rod


def main():
    dt = 1.0e-5
    final_time = 2.0
    total_steps = int(final_time / dt)

    youngs_modulus = 1.0e6
    damping_constant = 5.0e-2

    sim = TwoRodSimulator()

    # Shared callback collector
    cb_store: dict[str, dict] = {}
    step_skip = 100  # record every 50 steps (~1e-3 s)

    # Offset the second rod slightly in +z for visual separation.
    rod_stretchy = make_rod(
        sim,
        y_offset=0.0,
        dt=dt,
        youngs_modulus=youngs_modulus,
        damping_constant=damping_constant,
        inextensible=False,
    )
    sim.collect_diagnostics(rod_stretchy).using(
        RodCallback, step_skip=step_skip, store=cb_store, key="stretch"
    )

    rod_inext = make_rod(
        sim,
        y_offset=0.02,
        dt=dt,
        youngs_modulus=youngs_modulus,
        damping_constant=damping_constant,
        inextensible=True,
    )
    sim.collect_diagnostics(rod_inext).using(
        RodCallback, step_skip=step_skip, store=cb_store, key="inext"
    )

    sim.finalize()

    timestepper = PositionVerlet()
    integrate(timestepper, sim, final_time, total_steps)

    # Stack trajectories for rendering: (frames, rods, 3, n_nodes)
    pos_a = np.asarray(cb_store["stretch"]["position"])
    pos_b = np.asarray(cb_store["inext"]["position"])
    times = np.asarray(cb_store["stretch"]["time"])

    n = min(len(pos_a), len(pos_b))
    pos_a = pos_a[:n]
    pos_b = pos_b[:n]
    times = times[:n]

    rod_positions = np.stack([pos_a, pos_b], axis=1)

    out_dir = Path(__file__).resolve().parent
    np.savez(
        out_dir / "vary_wire_property_state.npz",
        time=times,
        pos_stretch=pos_a,
        pos_inext=pos_b,
        dt=dt,
        final_time=final_time,
    )

    video_path = out_dir / "vary_wire_property_4view.mp4"
    pp.plot_rods_multiview(
        rod_positions,
        video_path=video_path,
        times=times,
        fps=100,
        speed=1.0,
        plane_z=0.0,
        colors=["#1f77b4", "#d62728"],
    )
    print(f"Saved {video_path}")


if __name__ == "__main__":
    main()
