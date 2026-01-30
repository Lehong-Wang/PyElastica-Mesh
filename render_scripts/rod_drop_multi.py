"""
Drop multiple Cosserat rods from staggered heights onto the plane z=0 under gravity,
record the state, and render a color-coded video. Rod–rod contact is intentionally
disabled; only plane contact is enabled.
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

import matplotlib

# Headless rendering must be configured before importing pyplot.
matplotlib.use("Agg")
import numpy as np

import elastica as ea
from render_scripts import post_processing as pp


def run_multi_rod_drop(
    final_time: float = 3.0,
    dt: float = 1.0e-5,
    n_elem: int = 20,
    base_length: float = 1.0,
    base_radius: float = 5.0e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 1e6,
    shear_modulus_ratio: float = 1.5,
    contact_k: float = 1e4,
    contact_nu: float = 4.0,
    friction_coefficient: float = 1.0,
    damping_constant: float = 1e-2,
    num_rods: int = 8,
    height_gap: float = 0.05,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_drop_multi",
    output_interval: float = 0.01,
    seed: int | None = None,
    render_speed: float = 1.0,
    render_fps: int | None = None,
) -> dict[str, object]:
    """
    Simulate multiple free-falling Cosserat rods hitting a plane.

    Returns a dictionary with recorded arrays and file paths.
    """

    class MultiRodDropSim(
        ea.BaseSystemCollection, ea.Forcing, ea.Contact, ea.CallBacks, ea.Damping, 
    ):
        """Simulation container for multiple rod drops."""

    simulator = MultiRodDropSim()

    rng = np.random.default_rng(seed)

    base_start = np.array([0.0, 0.0, 0.1])
    direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    normal = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)

    rods: list[ea.CosseratRod] = []
    for i in range(num_rods):
        # Small random jitter in position and rotation about z
        start_i = base_start + np.array([0.0, 0.0, i * height_gap])
        jitter_pos = (rng.random(3) - 0.5) * 0.04  # ±0.02 m
        jitter_pos[2] = (rng.random() - 0.5) * 0.04  # keep vertical jitter within ±0.02
        start_i = start_i + jitter_pos

        rot_angle = (rng.random() - 0.5) * 0.1  # ±0.02 rad about z
        rot_mat = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle), 0.0],
                [np.sin(rot_angle), np.cos(rot_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dir_i = rot_mat @ direction
        norm_i = rot_mat @ normal

        rod = ea.CosseratRod.straight_rod(
            n_elements=n_elem,
            start=start_i,
            direction=dir_i,
            normal=norm_i,
            base_length=base_length,
            base_radius=base_radius,
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=youngs_modulus / (2.0 * shear_modulus_ratio),
        )
        simulator.append(rod)
        rods.append(rod)

    plane = ea.Plane(plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]))
    simulator.append(plane)

    static_mu = np.array(
        [friction_coefficient * 2.0, friction_coefficient * 2.0, friction_coefficient * 2.0]
    )
    kinetic_mu = np.array(
        [friction_coefficient, friction_coefficient, friction_coefficient]
    )

    for rod in rods:
        simulator.add_forcing_to(rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
        )

        simulator.dampen(rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=dt,
        )

        simulator.detect_contact_between(rod, plane).using(
            ea.RodPlaneContact, k=contact_k, nu=contact_nu
        )

        simulator.add_forcing_to(rod).using(
            ea.AnisotropicFrictionalPlane,
            k=contact_k,
            nu=contact_nu,
            plane_origin=np.zeros(3),
            plane_normal=np.array([0.0, 0.0, 1.0]),
            slip_velocity_tol=1e-6,
            static_mu_array=static_mu,
            kinetic_mu_array=kinetic_mu,
        )

    # Pairwise rod-rod contact
    for i in range(len(rods)):
        for j in range(i + 1, len(rods)):
            simulator.detect_contact_between(rods[i], rods[j]).using(
                ea.RodRodContact, k=contact_k, nu=contact_nu
            )

    collector: dict[str, ea.CallBackBaseClass] = {}

    class RodDropCallback(ea.CallBackBaseClass):
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

    step_skip = max(1, int(output_interval / dt))
    callback_names: list[str] = []
    for idx, rod in enumerate(rods):
        name = f"rod_{idx}"
        simulator.collect_diagnostics(rod).using(
            RodDropCallback, name=name, step_skip=step_skip
        )
        callback_names.append(name)

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    callbacks: list[RodDropCallback] = [collector[name] for name in callback_names]

    time_arr = np.asarray(callbacks[0].time)
    pos_arr = np.stack([np.asarray(cb.position) for cb in callbacks], axis=1)
    dir_arr = np.stack([np.asarray(cb.director) for cb in callbacks], axis=1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / f"{output_name}_state.npz"
    seed_val = np.int64(-1 if seed is None else int(seed))

    np.savez(
        state_path,
        time=time_arr,
        position=pos_arr,
        director=dir_arr,
        dt=dt,
        final_time=final_time,
    )

    video_path_four = output_dir / f"{output_name}_4view.mp4"
    colors = pp._color_cycle(num_rods)

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
    results = run_multi_rod_drop(
        final_time=3.0,
        dt=2.0e-5,
        base_radius=2.5e-3,
        damping_constant=1e-2,
        seed=45,
        youngs_modulus=5e6,
        contact_k=1e6,
        contact_nu=20.0,
        friction_coefficient=1.5,
        height_gap=0.05,
    )
    print(
        f"Saved npz to {results['state_path']} and video to "
        f"{results['video_path_four']} (4-view) (num_rods={len(results['colors'])})."
    )
