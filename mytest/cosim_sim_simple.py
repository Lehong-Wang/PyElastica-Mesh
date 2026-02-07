"""
Minimal co-sim example: single Cosserat rod with its clamped base driven by
pure sinusoidal translation along the global Y-axis (no rotation).
Exports rod node positions plus the driving frame pose; optional 4-view render.
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


class SineYFrameBC(ea.ConstraintBase):
    """Translate a rigid body sinusoidally along Y while keeping orientation fixed."""

    def __init__(
        self,
        fixed_position: np.ndarray,
        fixed_directors: np.ndarray,
        amp: float,
        freq: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p0 = np.asarray(fixed_position, dtype=float)
        self.R0 = np.asarray(fixed_directors, dtype=float)
        self.amp = float(amp)
        self.freq = float(freq)

    def constrain_values(self, system, time: np.float64) -> None:
        disp = self.amp * np.sin(2.0 * np.pi * self.freq * float(time))
        system.position_collection[..., 0] = self.p0 + np.array([0.0, disp, 0.0])
        system.director_collection[..., 0] = self.R0

    def constrain_rates(self, system, time: np.float64) -> None:
        omega = 2.0 * np.pi * self.freq
        vel_y = omega * self.amp * np.cos(2.0 * np.pi * self.freq * float(time))
        system.velocity_collection[..., 0] = np.array([0.0, vel_y, 0.0])
        system.omega_collection[..., 0] = 0.0


class StepYFrameBC(ea.ConstraintBase):
    """
    Translate a rigid body along Y with a square-wave (step) displacement and zero angular motion.
    Displacement toggles between +amp and -amp at the requested frequency.
    """

    def __init__(
        self,
        fixed_position: np.ndarray,
        fixed_directors: np.ndarray,
        amp: float,
        freq: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p0 = np.asarray(fixed_position, dtype=float)
        self.R0 = np.asarray(fixed_directors, dtype=float)
        self.amp = float(amp)
        self.freq = float(freq)

    def _square(self, t: float) -> float:
        phase = np.sin(2.0 * np.pi * self.freq * t)
        return 1.0 if phase >= 0.0 else -1.0

    def constrain_values(self, system, time: np.float64) -> None:
        disp = self.amp * self._square(float(time))
        system.position_collection[..., 0] = self.p0 + np.array([0.0, disp, 0.0])
        system.director_collection[..., 0] = self.R0

    def constrain_rates(self, system, time: np.float64) -> None:
        # Ideal step has impulses at edges; we keep velocity/omega zero for a kinematic clamp.
        system.velocity_collection[..., 0] = 0.0
        system.omega_collection[..., 0] = 0.0


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


def cosim_sim_simple(
    final_time: float = 5.0,
    dt: float = 1.0e-5,
    n_elem: int = 40,
    base_length: float = 1.0,
    base_radius: float = 2.5e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 1e6,
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 1e-2,
    joint_k: float = 5e2,
    joint_nu: float = 20.0,
    joint_kt: float = 1e1,
    joint_nut: float = 0.0,
    external_dt_tag: float | None = None,
    sine_amp: float = 0.1,
    sine_freq: float = 1.0,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "cosim_sim_simple",
    output_interval: float = 1.0 / 100.0,
    render: bool = False,
    render_speed: float = 1.0,
    render_fps: int | None = 100,
    force_vector_scale: float | None = None,
) -> dict[str, object]:
    """Drive rod base with pure Y-sine and export rod + frame trajectory."""

    class CosimSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Connections,
        ea.CallBacks,
        ea.Damping,
    ):
        """Rod attached to a driven frame via FixedJoint."""

    simulator = CosimSim()

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

    # Small proxy rigid body that the rod is jointed to; its motion is prescribed.
    frame = ea.Cylinder(
        start=np.array([-0.05, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        base_length=0.1,
        base_radius=0.01,
        density=1_000.0,
    )
    simulator.append(frame)

    # simulator.add_forcing_to(rod).using(
    #     ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
    # )

    # Optional gentle damping to suppress high-frequency noise during base shaking.
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )

    # Prescribe frame motion; rod is attached via a FixedJoint for positional + rotational coupling.
    simulator.constrain(frame).using(
        SineYFrameBC,
        # StepYFrameBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        amp=sine_amp,
        freq=sine_freq,
    )

    simulator.connect(frame, rod, first_connect_idx=0, second_connect_idx=0).using(
        ea.FixedJoint, k=joint_k, nu=joint_nu, kt=joint_kt, nut=joint_nut)
    #     ea.FixedJoint,
    #     k=1e3,
    #     nu=1,
    #     kt=1e3,
    #     nut=1e-2,
    # )

    collector: dict[str, object] = {}

    class CosimCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.rod_position: list[np.ndarray] = []
            self.frame_position: list[np.ndarray] = []
            self.frame_director: list[np.ndarray] = []
            self.frame_force: list[np.ndarray] = []
            collector["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.rod_position.append(system.position_collection.copy())
            self.frame_position.append(frame.position_collection.copy())
            self.frame_director.append(frame.director_collection.copy())
            self.frame_force.append(frame.external_forces[:, 0].copy())

    step_skip = max(1, int(np.round(output_interval / dt)))
    simulator.collect_diagnostics(rod).using(CosimCallback, step_skip=step_skip)

    simulator.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    cb = collector["cb"]
    time_arr = np.asarray(cb.time)
    rod_pos_arr = np.asarray(cb.rod_position)
    frame_pos_arr = np.asarray(cb.frame_position)
    frame_dir_arr = np.asarray(cb.frame_director)
    mean_force_arr = np.asarray(cb.frame_force)
    force_mag_arr = np.linalg.norm(mean_force_arr, axis=1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_tag = dt if external_dt_tag is None else float(external_dt_tag)
    tagged_output_name = (
        f"{output_name}_k{joint_k:g}_nu{joint_nu:g}_t{t_tag:g}"
    )

    state_path = output_dir / f"{tagged_output_name}_state.npz"
    np.savez(
        state_path,
        time=time_arr,
        rod_position=rod_pos_arr,
        frame_position=frame_pos_arr,
        frame_director=frame_dir_arr,
        mean_force=mean_force_arr,
        mean_force_magnitude=force_mag_arr,
        dt=dt,
        final_time=final_time,
        frame_amp=sine_amp,
        frame_freq=sine_freq,
        joint_k=joint_k,
        joint_nu=joint_nu,
        joint_kt=joint_kt,
        joint_nut=joint_nut,
        external_dt_tag=t_tag,
    )

    force_vec_plot_path = output_dir / f"{tagged_output_name}_mean_force_vector.png"
    _plot_force_vector_with_magnitude(
        time=time_arr,
        mean_force=mean_force_arr,
        force_mag=force_mag_arr,
        output_path=force_vec_plot_path,
    )

    force_mag_plot_path = output_dir / f"{tagged_output_name}_force_vs_time.png"
    _plot_force_vs_time(
        time=time_arr,
        force_mag=force_mag_arr,
        output_path=force_mag_plot_path,
    )

    video_path = output_dir / f"{tagged_output_name}_4view.mp4"
    if render:
        # Build a tiny 2-node segment representing the frame so it can be drawn alongside the rod.
        # Use the frame's third material axis as its span direction.
        frame_span = 0.05
        frame_line = frame_pos_arr + frame_span * np.stack(
            [-frame_dir_arr[:, 2, :, 0], frame_dir_arr[:, 2, :, 0]], axis=-1
        )  # (n_frames, 3, 2)
        # Pad frame polyline to match rod node count for concatenation (NaNs create a plotting break).
        n_nodes = rod_pos_arr.shape[2]
        pad_len = max(0, n_nodes - frame_line.shape[2])
        frame_line = np.pad(frame_line, ((0, 0), (0, 0), (0, pad_len)), constant_values=np.nan)

        # Combine rod and frame trajectories for multibody plotting.
        rod_for_plot = np.concatenate([rod_pos_arr[:, None, ...], frame_line[:, None, ...]], axis=1)

        # Joint-centric bounds to zoom in near the connection.
        joint_center = rod_pos_arr[:, :, 0].mean(axis=0)
        x_window = max(0.25 * base_length, 0.2)
        y_window = max(2.5 * sine_amp, 0.1)
        z_window = 0.2
        bounds = (
            (float(joint_center[0] - x_window / 2), float(joint_center[0] + x_window / 2)),
            (float(joint_center[1] - y_window / 2), float(joint_center[1] + y_window / 2)),
            (float(joint_center[2] - z_window / 2), float(joint_center[2] + z_window / 2)),
        )

        if force_vector_scale is None:
            force_norm_max = float(np.max(force_mag_arr)) if force_mag_arr.size else 0.0
            force_scale = 0.0
            if force_norm_max > 1.0e-12:
                force_scale = 0.1 * base_length / force_norm_max
        else:
            force_scale = float(force_vector_scale)

        pp.plot_rods_multiview(
            rod_for_plot,
            video_path=video_path,
            times=time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=None,
            colors=["#ff7f0e", "#1f77b4"],
            bounds=bounds,
            force_origins=frame_pos_arr[:, :, 0],
            force_vectors=mean_force_arr,
            force_scale=force_scale,
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
        "time": time_arr,
        "rod_position": rod_pos_arr,
        "frame_position": frame_pos_arr,
        "frame_director": frame_dir_arr,
        "mean_force": mean_force_arr,
        "mean_force_magnitude": force_mag_arr,
    }


if __name__ == "__main__":
    results = cosim_sim_simple(render=True)
    print(f"Saved npz to {results['state_path']} (render={bool(results['video_path'])}).")
