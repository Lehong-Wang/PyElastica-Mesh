"""
Two Cosserat rods pinned at both ends, aligned along +X and separated along +/–Y.
Both rods droop under gravity, and the run saves state plus a four-view render.
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


def run_two_fixed_rods(
    final_time: float = 2.0,
    dt: float = 1.0e-5,
    n_elem: int = 20,
    base_length: float = 1.0,
    base_radius: float = 4.0e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 5e6,
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 1e-2,
    lateral_gap: float = 0.05,
    start_height: float = 0.25,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_hang_multi",
    output_interval: float = 0.01,
    render_speed: float = 1.0,
    render_fps: int | None = None,
) -> dict[str, object]:
    """
    Simulate two horizontally pinned Cosserat rods that hang side by side under gravity.

    Returns a dictionary with recorded arrays and file paths.
    """

    class DualFixedRodSim(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks, ea.Damping
    ):
        """Simulation container for two pinned rods."""

    simulator = DualFixedRodSim()

    # Geometry: rods start at z = start_height, aligned along +X, separated along Y.
    base_start = np.array([0.0, 0.0, start_height])
    direction = np.array([1.0, 0.0, 0.0])  # along x
    normal = np.array([0.0, 1.0, 0.0])  # y defines the normal; binormal points up

    rods: list[ea.CosseratRod] = []
    y_offsets = (-lateral_gap / 2.0, lateral_gap / 2.0)
    for y_off in y_offsets:
        start = base_start + np.array([0.0, y_off, 0.0])
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
        rods.append(rod)

        simulator.add_forcing_to(rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
        )
        simulator.dampen(rod).using(
            ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
        )
        simulator.constrain(rod).using(
            ea.FixedConstraint,
            constrained_position_idx=(0, -1),
            constrained_director_idx=(0, -1),
        )

    # Diagnostics: record state every `output_interval`.
    collector: dict[str, RodCallback] = {}

    class RodCallback(ea.CallBackBaseClass):
        def __init__(self, name: str, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.position: list[np.ndarray] = []
            self.director: list[np.ndarray] = []
            collector[name] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    step_skip = max(1, int(np.round(output_interval / dt)))
    callback_names: list[str] = []
    for idx, rod in enumerate(rods):
        name = f"rod_{idx}"
        simulator.collect_diagnostics(rod).using(
            RodCallback, name=name, step_skip=step_skip
        )
        callback_names.append(name)

    simulator.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    callbacks: list[RodCallback] = [collector[name] for name in callback_names]

    time_arr = np.asarray(callbacks[0].time)
    pos_arr = np.stack([np.asarray(cb.position) for cb in callbacks], axis=1)
    dir_arr = np.stack([np.asarray(cb.director) for cb in callbacks], axis=1)

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
    colors = ["#1f77b4", "#ff7f0e"]
    pp.plot_rods_multiview(
        pos_arr,
        video_path=video_path_four,
        times=time_arr,
        fps=render_fps,
        speed=render_speed,
        plane_z=0.0,
        colors=colors,
    )

    return {
        "state_path": state_path,
        "video_path_four": video_path_four,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr,
        "colors": colors,
    }


if __name__ == "__main__":
    results = run_two_fixed_rods()
    print(
        f"Saved npz to {results['state_path']} and 4-view video to "
        f"{results['video_path_four']}."
    )
