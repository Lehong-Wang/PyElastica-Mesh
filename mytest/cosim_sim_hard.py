"""
Hard-driven co-sim: directly overwrite rod base pose/velocity/acceleration each step
from a sine trajectory (no intermediate frame or joint). Useful when you want
purely kinematic driving of the rod end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import elastica as ea
import matplotlib.pyplot as plt
from render_scripts import post_processing as pp
from tqdm import trange


def _sine_kinematics(time: float, amp: float, freq: float):
    omega = 2.0 * np.pi * freq
    pos = np.array([0.0, amp * np.sin(omega * time), 0.0])
    vel = np.array([0.0, amp * omega * np.cos(omega * time), 0.0])
    acc = np.array([0.0, -amp * omega * omega * np.sin(omega * time), 0.0])
    return pos, vel, acc


def _apply_drive(system, time: float, amp: float, freq: float, base_R: np.ndarray):
    pos, vel, acc = _sine_kinematics(time, amp, freq)
    system.position_collection[..., 0] = pos
    system.director_collection[..., 0] = base_R
    system.velocity_collection[..., 0] = vel
    system.omega_collection[..., 0] = 0.0
    system.acceleration_collection[..., 0] = acc
    system.alpha_collection[..., 0] = 0.0


def _plot_force_vs_time(
    time: np.ndarray,
    force: np.ndarray,
    output_path: Path,
) -> None:
    force_mag = np.linalg.norm(force, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time, force[:, 0], label="Fx")
    ax.plot(time, force[:, 1], label="Fy")
    ax.plot(time, force[:, 2], label="Fz")
    ax.plot(time, force_mag, "k--", linewidth=2.0, label="|F|")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force [N]")
    ax.set_title("Base Internal Force Proxy vs Time (Hard Drive)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def cosim_sim_hard(
    final_time: float = 3.0,
    dt: float = 1.0e-5,
    update_interval: float = 2e-3,
    n_elem: int = 40,
    base_length: float = 1.0,
    base_radius: float = 2.5e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 1e6,
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 1e-2,
    sine_amp: float = 0.1,
    sine_freq: float = 1.0,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "cosim_sim_hard",
    output_interval: float = 1.0 / 100.0,
    render: bool = False,
    render_speed: float = 1.0,
    render_fps: int | None = 100,
) -> dict[str, object]:
    """Kinematically clamp node 0 to sine motion (pos/vel/acc), integrate the rod."""

    class CosimSim(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.Constraints,
        ea.CallBacks,
        ea.Damping,
    ):
        pass

    simulator = CosimSim()

    # Rod setup
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

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

    # Optional damping
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )

    # Boundary condition: keep orientation fixed at node 0; translation applied discretely.
    class FixedOrientationBC(ea.ConstraintBase):
        def __init__(self, pos_ref: np.ndarray, director_ref: np.ndarray, **kwargs):
            super().__init__(**kwargs)
            self.base_R = np.asarray(director_ref, dtype=float)

        def constrain_values(self, system, time: np.float64) -> None:
            system.director_collection[..., 0] = self.base_R

        def constrain_rates(self, system, time: np.float64) -> None:
            system.omega_collection[..., 0] = 0.0

    simulator.constrain(rod).using(
        FixedOrientationBC,
        constrained_director_idx=(0,),
        constrained_position_idx=(0,),
    )

    collector: dict[str, object] = {}

    class CosimCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.rod_position: list[np.ndarray] = []
            self.rod_director: list[np.ndarray] = []
            self.base_internal_force: list[np.ndarray] = []
            collector["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.rod_position.append(system.position_collection.copy())
            self.rod_director.append(system.director_collection.copy())
            self.base_internal_force.append(system.internal_forces[:, 0].copy())

    step_skip = max(1, int(np.round(output_interval / dt)))
    simulator.collect_diagnostics(rod).using(CosimCallback, step_skip=step_skip)

    simulator.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()

    total_steps = int(np.ceil(final_time / dt))
    update_steps = max(1, int(np.round(update_interval / dt)))
    next_update_step = 0
    time = np.float64(0.0)

    for step in trange(total_steps, desc="Integrating", unit="step"):
        if step >= next_update_step:
            _apply_drive(rod, float(time), sine_amp, sine_freq, rod.director_collection[..., 0])
            next_update_step += update_steps
        time = timestepper.step(simulator, time, dt)

    cb = collector["cb"]
    time_arr = np.asarray(cb.time)
    rod_pos_arr = np.asarray(cb.rod_position)
    rod_dir_arr = np.asarray(cb.rod_director)
    base_force_arr = np.asarray(cb.base_internal_force)
    base_force_mag_arr = np.linalg.norm(base_force_arr, axis=1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{output_name}_state.npz"
    np.savez(
        state_path,
        time=time_arr,
        rod_position=rod_pos_arr,
        rod_director=rod_dir_arr,
        base_internal_force=base_force_arr,
        base_internal_force_magnitude=base_force_mag_arr,
        dt=dt,
        final_time=final_time,
        sine_amp=sine_amp,
        sine_freq=sine_freq,
        update_interval=update_interval,
    )

    force_plot_path = output_dir / f"{output_name}_force_vs_time.png"
    _plot_force_vs_time(
        time=time_arr,
        force=base_force_arr,
        output_path=force_plot_path,
    )

    joint_center = rod_pos_arr[:, :, 0].mean(axis=0)
    x_window = max(0.25 * base_length, 0.2)
    y_window = max(2.5 * sine_amp, 0.1)
    z_window = 0.2
    bounds = (
        (float(joint_center[0] - x_window / 2), float(joint_center[0] + x_window / 2)),
        (float(joint_center[1] - y_window / 2), float(joint_center[1] + y_window / 2)),
        (float(joint_center[2] - z_window / 2), float(joint_center[2] + z_window / 2)),
    )
    
    video_path = output_dir / f"{output_name}_4view.mp4"
    if render:
        force_norm_max = float(np.max(base_force_mag_arr)) if base_force_mag_arr.size else 0.0
        force_scale = 0.0
        if force_norm_max > 1.0e-12:
            force_scale = 0.1 * base_length / force_norm_max

        pp.plot_rods_multiview(
            rod_pos_arr,
            video_path=video_path,
            times=time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=None,
            colors=["#ff7f0e"],
            bounds=bounds,
            force_origins=rod_pos_arr[:, :, 0],
            force_vectors=base_force_arr,
            force_scale=force_scale,
            force_color="#d62728",
            force_label="Base Force",
            show_force_magnitude=True,
        )
    else:
        video_path = None

    return {
        "state_path": state_path,
        "video_path": video_path,
        "force_plot_path": force_plot_path,
        "time": time_arr,
        "rod_position": rod_pos_arr,
        "rod_director": rod_dir_arr,
        "base_internal_force": base_force_arr,
        "base_internal_force_magnitude": base_force_mag_arr,
    }


if __name__ == "__main__":
    results = cosim_sim_hard(render=True)
    print(f"Saved npz to {results['state_path']} (render={bool(results['video_path'])}).")
