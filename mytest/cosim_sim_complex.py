"""
Co-sim example with discrete pose updates: rod attached to a driven frame (FixedJoint).
Frame pose/velocity/acc are resampled from a sine trajectory every `update_interval`
and applied to the frame state; integration proceeds in between updates.
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
from render_scripts import post_processing as pp
from tqdm import trange


class FixedOrientationBC(ea.ConstraintBase):
    """Keep a rigid body's orientation fixed; zero angular velocity."""

    def __init__(self, director_ref: np.ndarray, fixed_directors: np.ndarray | None = None, **kwargs):
        super().__init__(**kwargs)
        if fixed_directors is None:
            fixed_directors = director_ref
        self.fixed_directors = np.asarray(fixed_directors, dtype=float)

    def constrain_values(self, system, time: np.float64) -> None:
        system.director_collection[..., 0] = self.fixed_directors

    def constrain_rates(self, system, time: np.float64) -> None:
        system.omega_collection[..., 0] = 0.0


def _sine_kinematics(time: float, amp: float, freq: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, vel, acc) for sine motion along +Y."""
    omega = 2.0 * np.pi * freq
    pos = np.array([0.0, amp * np.sin(omega * time), 0.0])
    vel = np.array([0.0, amp * omega * np.cos(omega * time), 0.0])
    acc = np.array([0.0, -amp * omega * omega * np.sin(omega * time), 0.0])
    return pos, vel, acc


def _set_frame_state(frame, base_pos: np.ndarray, base_R: np.ndarray, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray):
    """Overwrite frame state with provided pose/velocity; keep orientation fixed to base_R."""
    base_pos_vec = np.asarray(base_pos).reshape(3,)
    np.copyto(frame.position_collection, (base_pos_vec + pos).reshape(3, 1))
    np.copyto(frame.director_collection, base_R)
    np.copyto(frame.velocity_collection, vel.reshape(3, 1))
    np.copyto(frame.omega_collection, 0.0)
    np.copyto(frame.acceleration_collection, acc.reshape(3, 1))
    np.copyto(frame.alpha_collection, 0.0)
    np.copyto(frame.external_forces, 0.0)
    np.copyto(frame.external_torques, 0.0)


def cosim_sim_complex(
    final_time: float = 3.0,
    dt: float = 1.0e-5,
    update_interval: float = 0.05,
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
    output_name: str = "cosim_sim_complex",
    output_interval: float = 1.0 / 100.0,
    render: bool = False,
    render_speed: float = 1.0,
    render_fps: int | None = 100,
) -> dict[str, object]:
    """Chunked drive: refresh frame kinematics every `update_interval` using sine motion."""

    class CosimSim(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.Connections,
        ea.Constraints,
        ea.CallBacks,
        ea.Damping,
    ):
        """Rod + driven frame with joint coupling."""

    simulator = CosimSim()

    # Rod
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

    # Driven frame (rigid body), orientation fixed.
    frame = ea.Cylinder(
        start=np.array([-0.05, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        base_length=0.1,
        base_radius=0.01,
        density=5_000.0,
    )
    simulator.append(frame)

    # Optional damping on rod.
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )

    # Joint coupling.
    simulator.connect(frame, rod, first_connect_idx=0, second_connect_idx=0).using(
        # ea.FixedJoint, k=1e3, nu=1, kt=1e2, nut=0)
        ea.FixedJoint, k=1e5, nu=1, kt=1e1, nut=0)


    collector: dict[str, object] = {}

    class CosimCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.rod_position: list[np.ndarray] = []
            self.rod_director: list[np.ndarray] = []
            self.frame_position: list[np.ndarray] = []
            self.frame_director: list[np.ndarray] = []
            collector["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.rod_position.append(rod.position_collection.copy())
            self.rod_director.append(rod.director_collection.copy())
            self.frame_position.append(frame.position_collection.copy())
            self.frame_director.append(frame.director_collection.copy())

    step_skip = max(1, int(np.round(output_interval / dt)))
    simulator.collect_diagnostics(rod).using(CosimCallback, step_skip=step_skip)

    # Initial frame state from sine at t=0.
    base_pos0 = frame.position_collection[:, 0].copy()
    base_R0 = frame.director_collection.copy()
    pos, vel, acc = _sine_kinematics(0.0, sine_amp, sine_freq)
    _set_frame_state(frame, base_pos0, base_R0, pos, vel, acc)

    # Lock frame orientation so joint torques cannot rotate it.
    simulator.constrain(frame).using(
        FixedOrientationBC,
        constrained_director_idx=(0,),
    )

    simulator.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()

    total_steps = int(np.ceil(final_time / dt))
    update_steps = max(1, int(np.round(update_interval / dt)))
    next_update_step = 0
    time = np.float64(0.0)

    for step in trange(total_steps, desc="Integrating", unit="step"):
        if step >= next_update_step:
            pos, vel, acc = _sine_kinematics(float(time), sine_amp, sine_freq)
            _set_frame_state(frame, base_pos0, base_R0, pos, vel, acc)
            next_update_step += update_steps

        # Ensure driven frame carries no accumulated loads between steps.
        frame.external_forces[...] = 0.0
        frame.external_torques[...] = 0.0

        time = timestepper.step(simulator, time, dt)

    cb = collector["cb"]
    time_arr = np.asarray(cb.time)
    rod_pos_arr = np.asarray(cb.rod_position)
    rod_dir_arr = np.asarray(cb.rod_director)
    frame_pos_arr = np.asarray(cb.frame_position)
    frame_dir_arr = np.asarray(cb.frame_director)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{output_name}_state.npz"
    np.savez(
        state_path,
        time=time_arr,
        rod_position=rod_pos_arr,
        rod_director=rod_dir_arr,
        frame_position=frame_pos_arr,
        frame_director=frame_dir_arr,
        dt=dt,
        final_time=final_time,
        sine_amp=sine_amp,
        sine_freq=sine_freq,
        update_interval=update_interval,
    )

    video_path = output_dir / f"{output_name}_4view.mp4"
    if render:
        frame_span = 0.05
        frame_line = frame_pos_arr + frame_span * np.stack(
            [-frame_dir_arr[:, 2, :, 0], frame_dir_arr[:, 2, :, 0]], axis=-1
        )  # (n_frames, 3, 2)
        n_nodes = rod_pos_arr.shape[2]
        pad_len = max(0, n_nodes - frame_line.shape[2])
        frame_line = np.pad(frame_line, ((0, 0), (0, 0), (0, pad_len)), constant_values=np.nan)

        rod_for_plot = np.concatenate([rod_pos_arr[:, None, ...], frame_line[:, None, ...]], axis=1)

        joint_center = rod_pos_arr[:, :, 0].mean(axis=0)
        x_window = max(0.25 * base_length, 0.2)
        y_window = max(2.5 * sine_amp, 0.1)
        z_window = 0.2
        bounds = (
            (float(joint_center[0] - x_window / 2), float(joint_center[0] + x_window / 2)),
            (float(joint_center[1] - y_window / 2), float(joint_center[1] + y_window / 2)),
            (float(joint_center[2] - z_window / 2), float(joint_center[2] + z_window / 2)),
        )

        pp.plot_rods_multiview(
            rod_for_plot,
            video_path=video_path,
            times=time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=None,
            colors=["#ff7f0e", "#1f77b4"],
            bounds=bounds,
        )
    else:
        video_path = None

    return {
        "state_path": state_path,
        "video_path": video_path,
        "time": time_arr,
        "rod_position": rod_pos_arr,
        "rod_director": rod_dir_arr,
        "frame_position": frame_pos_arr,
        "frame_director": frame_dir_arr,
    }


if __name__ == "__main__":
    results = cosim_sim_complex(render=True)
    print(f"Saved npz to {results['state_path']} (render={bool(results['video_path'])}).")
