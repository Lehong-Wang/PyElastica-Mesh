"""
Up to four Cosserat rods: a random count of hanging rods (both ends fixed) plus
sweeping rods (one end fixed, translating in +Y). The counts sum to four.
Hanging rods have randomized fixed endpoints (not necessarily horizontal) with
endpoint spacing equal to the rod length. Sweep rods have randomized fixed roots
and slide in +Y at constant speed without rotation. Saves state and renders a
four-view video.
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


class SweepingBC(ea.ConstraintBase):
    """
    Time-dependent boundary condition: translate the base node at constant +Y speed
    while keeping its directors fixed (no rotation).
    """

    def __init__(
        self,
        fixed_position: np.ndarray,
        fixed_directors: np.ndarray,
        sweep_speed: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_position0 = np.asarray(fixed_position, dtype=float)
        self.director_reference = np.asarray(fixed_directors, dtype=float)
        self.sweep_speed = float(sweep_speed)

    def constrain_values(self, system, time: np.float64) -> None:
        pos = self.base_position0 + np.array([0.0, self.sweep_speed * float(time), 0.0])
        system.position_collection[..., 0] = pos
        system.director_collection[..., 0] = self.director_reference

    def constrain_rates(self, system, time: np.float64) -> None:  # noqa: ARG002
        system.velocity_collection[..., 0] = np.array([0.0, self.sweep_speed, 0.0])
        system.omega_collection[..., 0] = np.array([0.0, 0.0, 0.0])


class MovingEndsBC(ea.ConstraintBase):
    """
    Move both constrained end nodes of a hanging rod inward toward y=0
    at a constant speed (0, ±0.1, 0) depending on initial y sign.
    Preserves rod length because both ends shift by the same delta.
    """

    def __init__(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        dir_a: np.ndarray,
        dir_b: np.ndarray,
        inward_speed_range: tuple[float, float] = (0.0, 0.1),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        rng = np.random.default_rng()
        self.p0_a = np.asarray(pos_a, dtype=float)
        self.p0_b = np.asarray(pos_b, dtype=float)
        self.dir_a = np.asarray(dir_a, dtype=float)
        self.dir_b = np.asarray(dir_b, dtype=float)
        self.inward_speed = float(rng.uniform(*inward_speed_range))
        self.sign_a = -np.sign(self.p0_a[1]) if np.abs(self.p0_a[1]) > 1e-9 else 0.0
        self.sign_b = -np.sign(self.p0_b[1]) if np.abs(self.p0_b[1]) > 1e-9 else 0.0

    def _offset(self, sign: float, time: float) -> float:
        return sign * self.inward_speed * time

    def constrain_values(self, system, time: np.float64) -> None:
        dy_a = self._offset(self.sign_a, float(time))
        dy_b = self._offset(self.sign_b, float(time))
        pos = system.position_collection
        pos[..., 0] = self.p0_a + np.array([0.0, dy_a, 0.0])
        pos[..., -1] = self.p0_b + np.array([0.0, dy_b, 0.0])
        system.director_collection[..., 0] = self.dir_a
        system.director_collection[..., -1] = self.dir_b

    def constrain_rates(self, system, time: np.float64) -> None:  # noqa: ARG002
        vel = system.velocity_collection
        vel[..., 0] = np.array([0.0, self.sign_a * self.inward_speed, 0.0])
        vel[..., -1] = np.array([0.0, self.sign_b * self.inward_speed, 0.0])
        system.omega_collection[..., 0] = 0.0
        system.omega_collection[..., -1] = 0.0


class RandomImpulseForce(ea.NoForces):
    """Apply a constant impulse vector to one node for a fixed number of steps."""

    def __init__(self, node_index: int, impulse: np.ndarray, steps: int):
        super().__init__()
        self.node_index = int(node_index)
        self.impulse = np.asarray(impulse, dtype=float)
        self.steps_remaining = int(steps)

    def apply_forces(self, system, time=np.float64(0.0)):
        if self.steps_remaining <= 0:
            return
        system.external_forces[..., self.node_index] += self.impulse
        self.steps_remaining -= 1


def run_hang_with_sweep(
    final_time: float = 2.0,
    dt: float = 2.0e-5,
    n_elem: int = 20,
    base_length: float = 1.0,
    base_radius: float = 5.0e-3,  # unused for sampling; kept for backward compat
    density: float = 1_000.0,
    youngs_modulus: float = 5e6,  # unused for sampling; kept for backward compat
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 5e-2,
    contact_k: float = 2e4,
    contact_nu: float = 5.0,
    sweeper_length: float = 1.0,
    sweeper_radius: float = 5.0e-3,
    sweep_speed_range: tuple[float, float] = (0.2, 0.5),
    root_pos_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.2, 0.2),
        (-0.5, -0.2),
        (0.4, 0.6),
    ),
    center_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, 0.1),
    ),
    apply_random_impulse: bool = True,
    impulse_mag_range: tuple[float, float] = (0.1, 0.2),
    impulse_duration_range: tuple[float, float] = (0.01, 0.02),
    available_radii: tuple[float, ...] = (0.001, 0.003, 0.005),
    total_rods: int = 4,
    ym_low: float = 1e5,
    ym_high: float = 1e10,
    render: bool = True,
    save_state: bool = True,
    seed: int | None = None,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_hang_sweep",
    output_interval: float = 0.01,
    render_speed: float = 1.0,
    render_fps: int | None = None,
    enable_rod_rod_contact: bool = True,
) -> dict[str, object]:
    """
    Randomly choose hanging vs. sweeping rods (sum=4), randomize endpoints/root
    locations, and simulate gravity + rod–rod contact.

    Returns a dictionary with recorded arrays and file paths.
    """

    class SweepSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        """Simulation container for sweeping rod with pinned neighbors."""

    simulator = SweepSim()

    rng = np.random.default_rng(seed)

    def _rand_in_bounds(bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]):
        return np.array(
            [
                rng.uniform(low, high)
                for (low, high) in bounds
            ],
            dtype=float,
        )

    def _direction_within_cone(axis: np.ndarray, max_angle_deg: float):
        # Sample until within angle w.r.t. axis
        axis = axis / np.linalg.norm(axis)
        cos_max = np.cos(np.deg2rad(max_angle_deg))
        for _ in range(100):
            v = rng.normal(size=3)
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            v /= n
            if np.dot(v, axis) >= cos_max:
                return v
        return axis  # fallback

    def _normal_to(direction: np.ndarray):
        # Pick any vector not parallel, take cross.
        cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(direction, cand)) > 0.9:
            cand = np.array([0.0, 1.0, 0.0])
        n = np.cross(direction, cand)
        n /= np.linalg.norm(n)
        return n

    def _sample_radius():
        return float(rng.choice(available_radii))

    def _sample_ym():
        return float(np.exp(rng.uniform(np.log(ym_low), np.log(ym_high))))

    total_rods = max(2, int(total_rods))
    # Decide counts: at least 1 hanging and 1 sweeping.
    hanging_count = int(rng.integers(1, total_rods))
    sweep_count = total_rods - hanging_count
    if sweep_count == 0:
        hanging_count = max(1, total_rods - 1)
        sweep_count = 1

    hanging_rods: list[ea.CosseratRod] = []
    radii_all: list[float] = []
    yms_all: list[float] = []
    impulse_nodes_all: list[int] = []
    impulse_vectors_all: list[np.ndarray] = []
    impulse_steps_all: list[int] = []
    for idx in range(hanging_count):
        center = np.zeros(3) if idx == 0 else _rand_in_bounds(center_bounds)
        direction = _direction_within_cone(np.array([1.0, 0.0, 0.0]), 50.0)
        start = center - 0.5 * base_length * direction
        normal = _normal_to(direction)
        radius = _sample_radius()
        ym = _sample_ym()
        rod = ea.CosseratRod.straight_rod(
            n_elements=n_elem,
            start=start,
            direction=direction,
            normal=normal,
            base_length=base_length,
            base_radius=radius,
            density=density,
            youngs_modulus=ym,
            shear_modulus=ym / (2.0 * shear_modulus_ratio),
        )
        simulator.append(rod)
        hanging_rods.append(rod)
        radii_all.append(radius)
        yms_all.append(ym)

        simulator.add_forcing_to(rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
        )
        simulator.dampen(rod).using(
            ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
        )
        simulator.constrain(rod).using(
            MovingEndsBC,
            constrained_position_idx=(0, -1),
            constrained_director_idx=(0, -1),
        )
        if apply_random_impulse:
            node_idx = int(rng.integers(1, max(2, rod.n_nodes - 1)))
            direction_imp = _direction_within_cone(np.array([1.0, 0.0, 0.0]), 180.0)
            mag = float(rng.uniform(*impulse_mag_range))
            steps = max(1, int(np.ceil(rng.uniform(*impulse_duration_range) / dt)))
            impulse_vec = mag * direction_imp
            simulator.add_forcing_to(rod).using(
                RandomImpulseForce,
                node_index=node_idx,
                impulse=impulse_vec,
                steps=steps,
            )
            impulse_nodes_all.append(node_idx)
            impulse_vectors_all.append(impulse_vec)
            impulse_steps_all.append(steps)

    # ------------------------------------------------------------------ #
    # Sweeping rods (one end fixed, translate in +Y at constant speed).
    # ------------------------------------------------------------------ #
    sweep_rods: list[ea.CosseratRod] = []
    sweep_speeds: list[float] = []
    for _ in range(sweep_count):
        base = _rand_in_bounds(root_pos_bounds)
        # Ensure sweeper is at least 0.1–0.6 m above z=0 hanging zone
        # base[2] = rng.uniform(0.1, 0.6)
        sweep_direction = np.array([0.0, 0.0, -1.0])  # drop down
        sweep_normal = _normal_to(sweep_direction)
        speed = float(rng.uniform(*sweep_speed_range))

        radius = _sample_radius()
        ym = _sample_ym()
        sweeper = ea.CosseratRod.straight_rod(
            n_elements=n_elem,
            start=base,
            direction=sweep_direction,
            normal=sweep_normal,
            base_length=sweeper_length,
            base_radius=radius,
            density=density,
            youngs_modulus=ym,
            shear_modulus=ym / (2.0 * shear_modulus_ratio),
        )
        simulator.append(sweeper)
        sweep_rods.append(sweeper)
        sweep_speeds.append(speed)
        radii_all.append(radius)
        yms_all.append(ym)

        simulator.add_forcing_to(sweeper).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
        )
        simulator.dampen(sweeper).using(
            ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
        )
        simulator.constrain(sweeper).using(
            SweepingBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
            sweep_speed=speed,
        )
        if apply_random_impulse:
            node_idx = int(rng.integers(1, max(2, sweeper.n_nodes - 1)))
            direction_imp = _direction_within_cone(np.array([0.0, 1.0, 0.0]), 180.0)
            mag = float(rng.uniform(*impulse_mag_range))
            steps = max(1, int(np.ceil(rng.uniform(*impulse_duration_range) / dt)))
            impulse_vec = mag * direction_imp
            simulator.add_forcing_to(sweeper).using(
                RandomImpulseForce,
                node_index=node_idx,
                impulse=impulse_vec,
                steps=steps,
            )
            impulse_nodes_all.append(node_idx)
            impulse_vectors_all.append(impulse_vec)
            impulse_steps_all.append(steps)

    # ------------------------------------------------------------------ #
    # Contacts: rod–rod between all rods.
    # ------------------------------------------------------------------ #
    rods_all = hanging_rods + sweep_rods
    if enable_rod_rod_contact:
        for i in range(len(rods_all)):
            for j in range(i + 1, len(rods_all)):
                simulator.detect_contact_between(rods_all[i], rods_all[j]).using(
                    ea.RodRodContact, k=contact_k, nu=contact_nu
                )

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    collector: dict[str, ea.CallBackBaseClass] = {}

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
    for idx, rod in enumerate(rods_all):
        name = f"rod_{idx}"
        simulator.collect_diagnostics(rod).using(RodCallback, name=name, step_skip=step_skip)
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
    if save_state:
        np.savez(
            state_path,
            time=time_arr,
            position=pos_arr,
            director=dir_arr,
            dt=dt,
            final_time=final_time,
        )
    else:
        state_path = None

    video_path_four = output_dir / f"{output_name}_4view.mp4"
    colors = pp._color_cycle(len(rods_all))
    if render:
        pp.plot_rods_multiview(
            pos_arr,
            video_path=video_path_four,
            times=time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=0.0,
            colors=colors,
        )
    else:
        video_path_four = None

    return {
        "state_path": state_path,
        "video_path_four": video_path_four,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr,
        "colors": colors,
        "hanging_count": hanging_count,
        "sweep_count": sweep_count,
        "sweep_speeds": sweep_speeds,
        "radii": np.asarray(radii_all, dtype=float),
        "youngs_modulus": np.asarray(yms_all, dtype=float),
        "impulse_nodes": np.asarray(impulse_nodes_all, dtype=int),
        "impulse_vectors": np.asarray(impulse_vectors_all, dtype=float),
        "impulse_steps": np.asarray(impulse_steps_all, dtype=int),
    }


if __name__ == "__main__":
    results = run_hang_with_sweep()
    print(
        f"Saved npz to {results['state_path']} and 4-view video to "
        f"{results['video_path_four']}."
    )
