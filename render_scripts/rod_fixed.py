"""
Minimal example: 1 m Cosserat rod pinned at both ends in a horizontal line,
allowed to sag under gravity, then rendered with four synchronized views.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure matplotlib can write its cache in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

# Make the repository importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import elastica as ea
from render_scripts import post_processing as pp


def run_fixed_rod(
    final_time: float = 2.0,
    dt: float = 1.0e-5,
    n_elem: int = 40,
    base_length: float = 1.0,
    base_radius: float = 4.0e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 5e5,
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 5e-3,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_fixed",
    output_interval: float = 0.01,
    render_speed: float = 1.0,
    render_fps: int | None = None,
) -> dict[str, object]:
    """
    Simulate a horizontally pinned Cosserat rod that droops under gravity.

    Returns a dictionary with recorded arrays and file paths.
    """

    class FixedRodSim(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks, ea.Damping
    ):
        """Simulation container for the pinned rod."""

    simulator = FixedRodSim()

    # Geometry: start at origin, rod laid along +X, initially at z = 0.25 m.
    start = np.array([0.0, 0.0, 0.25])
    direction = np.array([1.0, 0.0, 0.0])  # along x
    normal = np.array([0.0, 1.0, 0.0])  # y defines the normal; binormal points up

    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=start,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=youngs_modulus / (2.0 * shear_modulus_ratio),
    )
    simulator.append(rod)

    # Gravity and light damping for settling.
    simulator.add_forcing_to(rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
    )
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )

    # Pin both ends: positions and orientations fixed at indices 0 and -1.
    simulator.constrain(rod).using(
        ea.FixedConstraint,
        constrained_position_idx=(0, -1),
        constrained_director_idx=(0, -1),
    )

    # Diagnostics: record state every `output_interval`.
    collector: dict[str, RodCallback] = {}

    class RodCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.position: list[np.ndarray] = []
            self.director: list[np.ndarray] = []
            collector["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    step_skip = max(1, int(np.round(output_interval / dt)))
    simulator.collect_diagnostics(rod).using(RodCallback, step_skip=step_skip)

    simulator.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    cb = collector["cb"]
    time_arr = np.asarray(cb.time)
    pos_arr = np.asarray(cb.position)
    dir_arr = np.asarray(cb.director)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{output_name}_state.npz"
    np.savez(
        state_path,
        time=time_arr,
        position=pos_arr,
        director=dir_arr,
        dt=dt,
        final_time=final_time,
    )

    video_path_four = output_dir / f"{output_name}_4view.mp4"
    pp.plot_rods_multiview(
        pos_arr,
        video_path=video_path_four,
        times=time_arr,
        fps=render_fps,
        speed=render_speed,
        plane_z=0.0,
        colors=["#1f77b4"],
    )

    return {
        "state_path": state_path,
        "video_path_four": video_path_four,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr,
    }


if __name__ == "__main__":
    results = run_fixed_rod()
    print(
        f"Saved npz to {results['state_path']} and 4-view video to "
        f"{results['video_path_four']}."
    )
