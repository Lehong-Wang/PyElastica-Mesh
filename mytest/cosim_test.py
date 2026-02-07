"""
Minimal co-simulation test:
- Frame command is updated every `isaac_dt`.
- PyElastica integrates at `py_dt`.
- Frame remains kinematic (not dynamically driven by rod loads).
- All saved/rendered data use real simulation time.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import elastica as ea
import matplotlib.pyplot as plt
from elastica.joint import get_relative_rotation_two_systems
from render_scripts import post_processing as pp


def _as_vec3(x: np.ndarray | list[float] | tuple[float, float, float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _as_rot3x3(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}.")
    return arr


@dataclass
class FrameState:
    """Frame kinematics from the external simulator in world coordinates."""

    position: np.ndarray
    director: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray

    def validated(self) -> "FrameState":
        return FrameState(
            position=_as_vec3(self.position, "position"),
            director=_as_rot3x3(self.director, "director"),
            velocity=_as_vec3(self.velocity, "velocity"),
            acceleration=_as_vec3(self.acceleration, "acceleration"),
            omega=_as_vec3(self.omega, "omega"),
            alpha=_as_vec3(self.alpha, "alpha"),
        )


@dataclass
class CoSimConfig:
    py_dt: float = 1.0e-5
    isaac_dt: float = 1.0e-2
    n_elem: int = 40
    base_length: float = 1.0
    base_radius: float = 2.5e-3
    density: float = 1_000.0
    youngs_modulus: float = 1.0e6
    shear_modulus_ratio: float = 1.5
    damping_constant: float = 1.0e-2
    joint_k: float = 5.0e2
    joint_nu: float = 20.0
    joint_kt: float = 1.0e1
    joint_nut: float = 0.0


class _FrameStateBuffer:
    """Mutable state buffer used by the kinematic frame constraint."""

    def __init__(self) -> None:
        self.position = np.zeros(3)
        self.director = np.eye(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.omega_world = np.zeros(3)
        self.alpha_world = np.zeros(3)

    def update(self, state: FrameState) -> None:
        s = state.validated()
        self.position[...] = s.position
        self.director[...] = s.director
        self.velocity[...] = s.velocity
        self.acceleration[...] = s.acceleration
        self.omega_world[...] = s.omega
        self.alpha_world[...] = s.alpha

    @property
    def omega_local(self) -> np.ndarray:
        # director maps world vectors to local vectors in PyElastica.
        return self.director @ self.omega_world

    @property
    def alpha_local(self) -> np.ndarray:
        return self.director @ self.alpha_world


class RateOnlyFrameBC(ea.ConstraintBase):
    """Constrain frame rates only; position/director are integrated by the stepper."""

    def __init__(self, state: _FrameStateBuffer, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def constrain_values(self, system, time: np.float64) -> None:
        # Leave position/director unconstrained so they evolve from rates.
        pass

    def constrain_rates(self, system, time: np.float64) -> None:
        np.copyto(system.velocity_collection[:, 0], self.state.velocity)
        np.copyto(system.omega_collection[:, 0], self.state.omega_local)


class RecordAndZeroFrameLoads(ea.NoForces):
    """
    Record frame loads and then clear them, so the frame remains kinematic.
    """

    def __init__(self, dt: float, accum: "_ImpulseAccumulator"):
        super().__init__()
        self.dt = float(dt)
        self.accum = accum

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        force_world = system.external_forces[:, 0].copy()
        self.accum.linear_impulse += force_world * self.dt
        system.external_forces[...] = 0.0

    def apply_torques(self, system, time: np.float64 = np.float64(0.0)) -> None:
        torque_local = system.external_torques[:, 0].copy()
        torque_world = system.director_collection[..., 0].T @ torque_local
        self.accum.angular_impulse += torque_world * self.dt
        system.external_torques[...] = 0.0


class _ImpulseAccumulator:
    def __init__(self) -> None:
        self.linear_impulse = np.zeros(3)
        self.angular_impulse = np.zeros(3)

    def reset(self) -> None:
        self.linear_impulse[...] = 0.0
        self.angular_impulse[...] = 0.0


class CoSimEngine:
    """
    Kinematic frame + rod response at fixed internal dt.
    """

    class _Simulator(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.Connections,
        ea.Constraints,
        ea.Damping,
    ):
        pass

    def __init__(self, config: CoSimConfig):
        self.config = config
        self.py_dt = float(config.py_dt)
        self.isaac_dt = float(config.isaac_dt)
        self.command_step = int(round(self.isaac_dt / self.py_dt))
        if self.command_step <= 0:
            raise ValueError("command_step must be positive.")
        if not np.isclose(self.command_step * self.py_dt, self.isaac_dt):
            raise ValueError(
                "isaac_dt must be an integer multiple of py_dt "
                f"(got isaac_dt={self.isaac_dt}, py_dt={self.py_dt})."
            )

        self.sim = self._Simulator()

        start = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])

        self.rod = ea.CosseratRod.straight_rod(
            n_elements=config.n_elem,
            start=start,
            direction=direction,
            normal=normal,
            base_length=config.base_length,
            base_radius=config.base_radius,
            density=config.density,
            youngs_modulus=config.youngs_modulus,
            shear_modulus=config.youngs_modulus / (2.0 * config.shear_modulus_ratio),
        )
        self.sim.append(self.rod)

        self.frame = ea.Cylinder(
            start=np.array([-0.05, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            base_length=0.1,
            base_radius=0.01,
            density=5_000.0,
        )
        self.sim.append(self.frame)

        self.sim.dampen(self.rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=config.damping_constant,
            time_step=self.py_dt,
        )

        rest_rot = get_relative_rotation_two_systems(self.frame, 0, self.rod, 0)
        self.sim.connect(self.frame, self.rod, first_connect_idx=0, second_connect_idx=0).using(
            ea.FixedJoint,
            k=config.joint_k,
            nu=config.joint_nu,
            kt=config.joint_kt,
            nut=config.joint_nut,
            rest_rotation_matrix=rest_rot,
        )

        self.frame_state = _FrameStateBuffer()
        self.frame_state.position[...] = self.frame.position_collection[:, 0]
        self.frame_state.director[...] = self.frame.director_collection[..., 0]
        self.frame_state.velocity[...] = self.frame.velocity_collection[:, 0]
        self.frame_state.acceleration[...] = self.frame.acceleration_collection[:, 0]
        self.frame_state.omega_world[...] = (
            self.frame.director_collection[..., 0].T @ self.frame.omega_collection[:, 0]
        )
        self.frame_state.alpha_world[...] = (
            self.frame.director_collection[..., 0].T @ self.frame.alpha_collection[:, 0]
        )
        self.sim.constrain(self.frame).using(
            RateOnlyFrameBC,
            state=self.frame_state,
        )

        # Register after joint so synchronize() sees joint loads first, then records+zeros.
        self.tick_impulse = _ImpulseAccumulator()
        self.sim.add_forcing_to(self.frame).using(
            RecordAndZeroFrameLoads,
            dt=self.py_dt,
            accum=self.tick_impulse,
        )

        self.sim.finalize()

        self.stepper: ea.typing.StepperProtocol = ea.PositionVerlet()
        self.time = np.float64(0.0)

    def apply_command_state(self, state: FrameState) -> None:
        """
        Overwrite frame state at command update time.
        RateOnlyFrameBC then keeps velocity/omega enforced between command updates.
        """
        self.frame_state.update(state)
        np.copyto(self.frame.position_collection[:, 0], self.frame_state.position)
        np.copyto(self.frame.director_collection[..., 0], self.frame_state.director)
        np.copyto(self.frame.velocity_collection[:, 0], self.frame_state.velocity)
        np.copyto(self.frame.omega_collection[:, 0], self.frame_state.omega_local)
        np.copyto(self.frame.acceleration_collection[:, 0], self.frame_state.acceleration)
        np.copyto(self.frame.alpha_collection[:, 0], self.frame_state.alpha_local)
        np.copyto(self.frame.external_forces, 0.0)
        np.copyto(self.frame.external_torques, 0.0)

    def reset_command_reaction(self) -> None:
        self.tick_impulse.reset()

    def step(self) -> float:
        self.time = self.stepper.step(self.sim, self.time, np.float64(self.py_dt))
        return float(self.time)


def sine_frame_state(t: float, amp: float = 0.1, freq: float = 1.0) -> FrameState:
    omega = 2.0 * np.pi * freq
    pos = np.array([0.0, amp * np.sin(omega * t), 0.0])
    vel = np.array([0.0, amp * omega * np.cos(omega * t), 0.0])
    acc = np.array([0.0, -amp * omega * omega * np.sin(omega * t), 0.0])
    default_director = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return FrameState(
        position=pos,
        director=default_director,
        velocity=vel,
        acceleration=acc,
        omega=np.zeros(3),
        alpha=np.zeros(3),
    )


def _plot_force_vector_with_magnitude(
    time: np.ndarray,
    mean_force: np.ndarray,
    force_mag: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time, mean_force[:, 0], label="Fx")
    ax.plot(time, mean_force[:, 1], label="Fy")
    ax.plot(time, mean_force[:, 2], label="Fz")
    ax.plot(time, force_mag, "k--", linewidth=2.0, label="|F|")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean force [N]")
    ax.set_title("Mean Rod-on-Frame Force Vector (Real Simulation Time)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_force_vs_time(
    time: np.ndarray,
    force_mag: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, force_mag, color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|F| [N]")
    ax.set_title("Rod-on-Frame Force Magnitude vs Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_demo(
    final_time: float = 3.0,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "cosim_test",
    output_interval: float = 1.0 / 100.0,
    render: bool = False,
    render_speed: float = 1.0,
    render_fps: int | None = 100,
    sine_amp: float = 0.1,
    sine_freq: float = 1.0,
    force_vector_scale: float = 1.0,
) -> dict[str, object]:
    cfg = CoSimConfig()
    engine = CoSimEngine(cfg)
    param_tag = f"_k{cfg.joint_k:g}_nu{cfg.joint_nu:g}_t{cfg.isaac_dt:g}"
    tagged_output_name = f"{output_name}{param_tag}"
    if final_time <= 0.0:
        raise ValueError(f"final_time must be positive, got {final_time}.")
    if output_interval <= 0.0:
        raise ValueError(f"output_interval must be positive, got {output_interval}.")

    total_steps = int(np.ceil(final_time / cfg.py_dt))
    command_step = engine.command_step
    sample_step = max(1, int(np.round(output_interval / cfg.py_dt)))

    sampled_time: list[float] = [0.0]
    sampled_rod_position: list[np.ndarray] = [engine.rod.position_collection.copy()]
    sampled_rod_director: list[np.ndarray] = [engine.rod.director_collection.copy()]
    sampled_frame_position: list[np.ndarray] = [engine.frame.position_collection.copy()]
    sampled_frame_director: list[np.ndarray] = [engine.frame.director_collection.copy()]
    sampled_mean_force: list[np.ndarray] = [np.zeros(3)]

    next_command_step = 0
    command_start_time = 0.0
    mean_force_now = np.zeros(3)
    print_step = max(1, int(np.round(0.1 / cfg.py_dt)))

    for step in range(total_steps):
        if step >= next_command_step:
            command_start_time = step * cfg.py_dt
            engine.apply_command_state(
                sine_frame_state(command_start_time, amp=sine_amp, freq=sine_freq)
            )
            engine.reset_command_reaction()
            next_command_step += command_step

        sim_time = engine.step()
        command_elapsed = max(sim_time - command_start_time, cfg.py_dt)
        mean_force_now = engine.tick_impulse.linear_impulse / command_elapsed

        if (step + 1) % sample_step == 0 or step == total_steps - 1:
            sampled_time.append(sim_time)
            sampled_rod_position.append(engine.rod.position_collection.copy())
            sampled_rod_director.append(engine.rod.director_collection.copy())
            sampled_frame_position.append(engine.frame.position_collection.copy())
            sampled_frame_director.append(engine.frame.director_collection.copy())
            sampled_mean_force.append(mean_force_now.copy())

        if step == 0 or (step + 1) % print_step == 0:
            print(
                f"step={step + 1:07d}/{total_steps:07d} "
                f"time={sim_time:8.5f} "
                f"Fmean={mean_force_now}"
            )

    sampled_time_arr = np.asarray(sampled_time)
    sampled_rod_pos_arr = np.asarray(sampled_rod_position)
    sampled_rod_dir_arr = np.asarray(sampled_rod_director)
    sampled_frame_pos_arr = np.asarray(sampled_frame_position)
    sampled_frame_dir_arr = np.asarray(sampled_frame_director)
    sampled_mean_force_arr = np.asarray(sampled_mean_force)
    sampled_force_mag_arr = np.linalg.norm(sampled_mean_force_arr, axis=1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / f"{tagged_output_name}_state.npz"
    np.savez(
        state_path,
        time=sampled_time_arr,
        rod_position=sampled_rod_pos_arr,
        rod_director=sampled_rod_dir_arr,
        frame_position=sampled_frame_pos_arr,
        frame_director=sampled_frame_dir_arr,
        mean_force=sampled_mean_force_arr,
        mean_force_magnitude=sampled_force_mag_arr,
        py_dt=cfg.py_dt,
        isaac_dt=cfg.isaac_dt,
        final_time=final_time,
        sine_amp=sine_amp,
        sine_freq=sine_freq,
        output_interval=output_interval,
        command_step=command_step,
        sample_step=sample_step,
    )

    force_vec_plot_path = output_dir / f"{tagged_output_name}_mean_force_vector.png"
    _plot_force_vector_with_magnitude(
        time=sampled_time_arr,
        mean_force=sampled_mean_force_arr,
        force_mag=sampled_force_mag_arr,
        output_path=force_vec_plot_path,
    )

    force_mag_plot_path = output_dir / f"{tagged_output_name}_force_vs_time.png"
    _plot_force_vs_time(
        time=sampled_time_arr,
        force_mag=sampled_force_mag_arr,
        output_path=force_mag_plot_path,
    )

    video_path = output_dir / f"{tagged_output_name}_4view.mp4"
    if render:
        frame_span = 0.05
        frame_line = sampled_frame_pos_arr + frame_span * np.stack(
            [-sampled_frame_dir_arr[:, 2, :, 0], sampled_frame_dir_arr[:, 2, :, 0]],
            axis=-1,
        )
        n_nodes = sampled_rod_pos_arr.shape[2]
        pad_len = max(0, n_nodes - frame_line.shape[2])
        frame_line = np.pad(
            frame_line, ((0, 0), (0, 0), (0, pad_len)), constant_values=np.nan
        )

        rod_for_plot = np.concatenate(
            [sampled_rod_pos_arr[:, None, ...], frame_line[:, None, ...]], axis=1
        )

        joint_center = sampled_rod_pos_arr[:, :, 0].mean(axis=0)
        x_window = max(0.25 * cfg.base_length, 0.2)
        y_window = 0.25
        z_window = 0.2
        bounds = (
            (
                float(joint_center[0] - x_window / 2),
                float(joint_center[0] + x_window / 2),
            ),
            (
                float(joint_center[1] - y_window / 2),
                float(joint_center[1] + y_window / 2),
            ),
            (
                float(joint_center[2] - z_window / 2),
                float(joint_center[2] + z_window / 2),
            ),
        )

        pp.plot_rods_multiview(
            rod_for_plot,
            video_path=video_path,
            times=sampled_time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=None,
            colors=["#ff7f0e", "#1f77b4"],
            bounds=bounds,
            force_origins=sampled_frame_pos_arr[:, :, 0],
            force_vectors=sampled_mean_force_arr,
            force_scale=force_vector_scale,
            force_color="#d62728",
            force_label="Mean Rod->Frame Force",
            show_force_magnitude=True,
        )
    else:
        video_path = None

    return {
        "state_path": state_path,
        "video_path": video_path,
        "force_vector_plot_path": force_vec_plot_path,
        "force_magnitude_plot_path": force_mag_plot_path,
        "time": sampled_time_arr,
        "mean_force": sampled_mean_force_arr,
        "mean_force_magnitude": sampled_force_mag_arr,
    }


if __name__ == "__main__":
    results = run_demo(final_time=5.0, render=True)
    print(
        f"Saved npz to {results['state_path']} "
        f"(render={bool(results['video_path'])})."
    )
