"""
Simulate a Cosserat rod with:
- Base node position fixed at origin (0, 0, 0) only.
- Tip node pose (position + director) driven by `motion.npz`.

The motion file must contain:
- `time`: shape (T,)
- `pos`: shape (T, 3)
- `director`: shape (T, 3, 3) (already in PyElastica director format)

Tip motion is interpolated between input samples:
- position: linear interpolation
- director: quaternion slerp

Outputs:
- `<output_name>_state.npz`
- `<output_name>_4view.mp4` (via render_scripts.post_processing)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure matplotlib can write its cache in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

# Make repository importable when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import elastica as ea
from render_scripts import post_processing as pp


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0 or np.isnan(n):
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    m00, m01, m02 = rotation[0, 0], rotation[0, 1], rotation[0, 2]
    m10, m11, m12 = rotation[1, 0], rotation[1, 1], rotation[1, 2]
    m20, m21, m22 = rotation[2, 0], rotation[2, 1], rotation[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0 or np.isnan(n):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation of unit quaternions in xyzw convention."""
    qa = q0 / np.linalg.norm(q0)
    qb = q1 / np.linalg.norm(q1)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = (1.0 - alpha) * qa + alpha * qb
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * qa + s1 * qb


def _rotvec_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Rotation vector from a 3x3 rotation matrix."""
    tr = np.trace(rotation)
    cos_theta = np.clip(0.5 * (tr - 1.0), -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1.0e-12:
        return np.zeros(3)

    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1.0e-12:
        # Near pi: use diagonal fallback.
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        axis = axis / max(np.linalg.norm(axis), 1.0e-12)
        return axis * theta

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sin_theta)
    return axis * theta


def _fmt_param(value: float | int) -> str:
    """Format a numeric parameter for filename tokens."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return np.format_float_positional(float(value), trim="-")


def _build_output_suffix(
    *,
    n_elem: int,
    ground_z: float,
    density: float,
    youngs_modulus: float,
    damping_ratio: float,
    ground_contact_k: float,
    ground_contact_nu: float,
) -> str:
    return (
        f"_n{_fmt_param(n_elem)}"
        f"_z{_fmt_param(ground_z)}"
        f"_d{_fmt_param(density)}"
        f"_yn{_fmt_param(youngs_modulus)}"
        f"_dp{_fmt_param(damping_ratio)}"
        f"_k{_fmt_param(ground_contact_k)}"
        f"_nu{_fmt_param(ground_contact_nu)}"
    )


def _resolve_unique_output_paths(
    output_dir: Path,
    output_tag: str,
    render: bool,
) -> tuple[Path, Path | None]:
    """Return non-conflicting state/video paths by appending _{index} if needed."""
    state_path = output_dir / f"{output_tag}_state.npz"
    video_path = output_dir / f"{output_tag}_4view.mp4"

    if not state_path.exists() and (not render or not video_path.exists()):
        return state_path, (video_path if render else None)

    idx = 1
    while True:
        state_path = output_dir / f"{output_tag}_{idx}_state.npz"
        video_path = output_dir / f"{output_tag}_{idx}_4view.mp4"
        if not state_path.exists() and (not render or not video_path.exists()):
            return state_path, (video_path if render else None)
        idx += 1


def load_motion_npz(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load trajectory arrays from `motion.npz` directly.
    Returns (time, pos, director) with shapes (T,), (T,3), (T,3,3) and time shifted to start at 0.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {path}")

    data = np.load(path, allow_pickle=True)
    missing = [k for k in ("time", "pos", "director") if k not in data]
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")

    time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    pos = np.asarray(data["pos"], dtype=np.float64)
    director = np.asarray(data["director"], dtype=np.float64)

    if time.ndim != 1:
        raise ValueError(f"`time` must be 1D, got shape {time.shape}.")
    if pos.shape != (time.size, 3):
        raise ValueError(
            f"`pos` must have shape (T,3) matching time; got {pos.shape} with T={time.size}."
        )
    if director.shape != (time.size, 3, 3):
        raise ValueError(
            "`director` must have shape (T,3,3) matching time; "
            f"got {director.shape} with T={time.size}."
        )

    if time.size < 2:
        raise ValueError("`time` must contain at least 2 samples.")

    time = time - time[0]
    if time[-1] <= 0.0:
        raise ValueError("Motion duration must be positive.")
    n = 1000

    # T = -np.array([[0,0,1], [0,-1,0], [1,0,0]])
    T = np.array([[0,0,1], [0,1,0],[-1,0,0]])
    # print(director)
    # print(np.tile(T, (n,1,1)))
    # return time[:n], pos[:n], np.tile(T, (n,1,1))
    # return time[:n], pos[:n], T @ director[:n]
    return time, pos, T@director


class TipMotionInterpolator:
    """Interpolate tip position and director across trajectory timestamps."""

    def __init__(self, time: np.ndarray, pos: np.ndarray, director: np.ndarray):
        self.time = np.asarray(time, dtype=np.float64)
        self.pos = np.asarray(pos, dtype=np.float64)
        self.director = np.asarray(director, dtype=np.float64)
        if self.time.ndim != 1:
            raise ValueError("time must be 1D.")
        if self.pos.shape != (self.time.size, 3):
            raise ValueError(f"pos must have shape (T,3), got {self.pos.shape}")
        if self.director.shape != (self.time.size, 3, 3):
            raise ValueError(
                f"director must have shape (T,3,3), got {self.director.shape}"
            )
        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError("time must be strictly increasing.")

        self.quat = np.stack([_matrix_to_quat_xyzw(r) for r in self.director], axis=0)
        for i in range(1, self.quat.shape[0]):
            if np.dot(self.quat[i - 1], self.quat[i]) < 0.0:
                self.quat[i] *= -1.0

        dt = np.diff(self.time)
        self.segment_velocity = np.diff(self.pos, axis=0) / dt[:, None]
        self.segment_omega_world = np.zeros((self.time.size - 1, 3), dtype=np.float64)
        for i in range(self.time.size - 1):
            # director is world->local; convert to local->world for relative world rotation.
            rel_rotation_world = self.director[i + 1].T @ self.director[i]
            rotvec = _rotvec_from_rotation_matrix(rel_rotation_world)
            self.segment_omega_world[i] = rotvec / dt[i]

        self.t_start = float(self.time[0])
        self.t_end = float(self.time[-1])

    def _segment(self, t: float) -> tuple[int, float]:
        if t <= self.t_start:
            return 0, 0.0
        if t >= self.t_end:
            return self.time.size - 2, 1.0
        idx = int(np.searchsorted(self.time, t, side="right") - 1)
        idx = int(np.clip(idx, 0, self.time.size - 2))
        t0 = self.time[idx]
        t1 = self.time[idx + 1]
        alpha = float((t - t0) / (t1 - t0))
        return idx, alpha

    def sample_pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        idx, alpha = self._segment(t)
        pos = (1.0 - alpha) * self.pos[idx] + alpha * self.pos[idx + 1]
        quat = _slerp_xyzw(self.quat[idx], self.quat[idx + 1], alpha)
        director = _quat_xyzw_to_matrix(quat)
        return pos, director

    def sample_rates(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t <= self.t_start or t >= self.t_end:
            return np.zeros(3), np.zeros(3)

        idx, alpha = self._segment(t)
        vel_world = self.segment_velocity[idx]
        quat = _slerp_xyzw(self.quat[idx], self.quat[idx + 1], alpha)
        director = _quat_xyzw_to_matrix(quat)

        # PyElastica stores omega in local coordinates.
        omega_world = self.segment_omega_world[idx]
        omega_local = director @ omega_world
        return vel_world, omega_local


def _choose_normal(direction: np.ndarray) -> np.ndarray:
    """Pick a stable vector orthogonal to direction."""
    direction = direction / np.linalg.norm(direction)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    normal = trial - np.dot(trial, direction) * direction
    n = np.linalg.norm(normal)
    if n < 1.0e-12:
        return np.array([0.0, 1.0, 0.0])
    return normal / n


class TipDrivenBC(ea.ConstraintBase):
    """Prescribe tip node position and tip element director from interpolated motion."""

    def __init__(
        self,
        fixed_tip_position: np.ndarray,
        fixed_tip_director: np.ndarray,
        motion: TipMotionInterpolator,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.fixed_tip_position = np.asarray(fixed_tip_position, dtype=np.float64)
        self.fixed_tip_director = np.asarray(fixed_tip_director, dtype=np.float64)
        self.motion = motion

    def constrain_values(self, system, time: np.float64) -> None:
        pos, director = self.motion.sample_pose(float(time))
        system.position_collection[..., -1] = pos
        system.director_collection[..., -1] = director

    def constrain_rates(self, system, time: np.float64) -> None:
        vel, omega_local = self.motion.sample_rates(float(time))
        system.velocity_collection[..., -1] = vel
        system.omega_collection[..., -1] = omega_local


class SineYFixedEndBC(ea.ConstraintBase):
    """Constrain one end with sinusoidal translation along global +Y and fixed orientation."""

    def __init__(
        self,
        fixed_position: np.ndarray,
        fixed_directors: np.ndarray,
        amp: float,
        freq: float,
        phase: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.p0 = np.asarray(fixed_position, dtype=np.float64).reshape(3)
        self.d0 = np.asarray(fixed_directors, dtype=np.float64).reshape(3, 3)
        self.amp = float(amp)
        self.freq = float(freq)
        self.phase = float(phase)
        if self.constrained_position_idx.size == 0:
            raise ValueError("SineYFixedEndBC requires one constrained position index.")
        self.pos_idx = int(self.constrained_position_idx[0])
        self.dir_idx = (
            int(self.constrained_director_idx[0])
            if self.constrained_director_idx.size
            else self.pos_idx
        )

    def _theta(self, t: float) -> float:
        return 2.0 * np.pi * self.freq * t + self.phase

    def constrain_values(self, system, time: np.float64) -> None:
        theta = self._theta(float(time))
        disp_y = self.amp * np.sin(theta)
        system.position_collection[..., self.pos_idx] = self.p0 + np.array([0.0, disp_y, 0.0])
        if self.constrained_director_idx.size:
            system.director_collection[..., self.dir_idx] = self.d0

    def constrain_rates(self, system, time: np.float64) -> None:
        theta = self._theta(float(time))
        vel_y = 2.0 * np.pi * self.freq * self.amp * np.cos(theta)
        system.velocity_collection[..., self.pos_idx] = np.array([0.0, vel_y, 0.0])
        if self.constrained_director_idx.size:
            system.omega_collection[..., self.dir_idx] = np.zeros(3)


def _add_ground_contact_with_friction(
    simulator: Any,
    rod: Any,
    *,
    ground_z: float,
    ground_contact_k: float,
    ground_contact_nu: float,
    ground_static_mu: tuple[float, float, float],
    ground_kinetic_mu: tuple[float, float, float],
    ground_slip_velocity_tol: float,
) -> None:
    """Attach plane contact + anisotropic friction against a horizontal ground."""
    plane_origin = np.array([0.0, 0.0, float(ground_z)], dtype=np.float64)
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    plane = ea.Plane(plane_origin=plane_origin, plane_normal=plane_normal)
    simulator.append(plane)
    simulator.add_forcing_to(rod).using(
        ea.AnisotropicFrictionalPlane,
        k=float(ground_contact_k),
        nu=float(ground_contact_nu),
        plane_origin=plane_origin,
        plane_normal=plane_normal,
        slip_velocity_tol=float(ground_slip_velocity_tol),
        static_mu_array=np.asarray(ground_static_mu, dtype=np.float64),
        kinetic_mu_array=np.asarray(ground_kinetic_mu, dtype=np.float64),
    )
    simulator.detect_contact_between(rod, plane).using(
        ea.RodPlaneContact,
        k=float(ground_contact_k),
        nu=float(ground_contact_nu),
    )


def run_swing_sine_y(
    final_time: float = 10.0,
    dt: float = 1.0e-5,
    n_elem: int = 40,
    base_length: float = 1.0,
    base_radius: float = 3.0e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 1.0e6,
    shear_modulus_ratio: float = 1.5,
    axial_stretch_stiffening: float = 1.0e5,
    damping_constant: float = 2.0e-2,
    sine_amp: float = 0.1,
    sine_freq: float = 1.0,
    sine_phase: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ground_z: float = -0.01,
    ground_contact_k: float = 1.0e4,
    ground_contact_nu: float = 5.0,
    ground_static_mu: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ground_kinetic_mu: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ground_slip_velocity_tol: float = 1.0e-6,
    output_interval: float = 0.01,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "swing_sine_y",
    render: bool = True,
    render_speed: float = 1.0,
    render_fps: int | None = None,
) -> dict[str, object]:
    """
    Simulate a rod where:
    - node 0 is fixed at origin,
    - node -1 moves sinusoidally in Y,
    - node -1 director stays fixed in the +X-aligned frame.
    """

    class SwingSineSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        """Simulation container for sine-driven base rod."""

    simulator = SwingSineSim()

    # Requested setup: root fixed at origin, rod aligned with +X.
    start_arr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dir_arr = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    normal_arr = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=start_arr,
        direction=dir_arr,
        normal=normal_arr,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=youngs_modulus / (2.0 * shear_modulus_ratio),
    )
    if axial_stretch_stiffening <= 0.0:
        raise ValueError(
            f"axial_stretch_stiffening must be > 0, got {axial_stretch_stiffening}"
        )
    if not np.isclose(axial_stretch_stiffening, 1.0):
        # Following mytest/vary_wire_property: penalize axial stretch (d3 direction).
        rod.shear_matrix[2, 2, :] *= float(axial_stretch_stiffening)
    simulator.append(rod)

    # Root fixed at origin.
    simulator.constrain(rod).using(
        ea.FixedConstraint,
        constrained_position_idx=(0,),
    )

    # Tip moves sinusoidally along Y while keeping +X-aligned director frame.
    simulator.constrain(rod).using(
        SineYFixedEndBC,
        constrained_position_idx=(-1,),
        constrained_director_idx=(-1,),
        amp=sine_amp,
        freq=sine_freq,
        phase=sine_phase,
    )

    if gravity is not None:
        simulator.add_forcing_to(rod).using(
            ea.GravityForces, acc_gravity=np.asarray(gravity, dtype=np.float64)
        )

    _add_ground_contact_with_friction(
        simulator,
        rod,
        ground_z=ground_z,
        ground_contact_k=ground_contact_k,
        ground_contact_nu=ground_contact_nu,
        ground_static_mu=ground_static_mu,
        ground_kinetic_mu=ground_kinetic_mu,
        ground_slip_velocity_tol=ground_slip_velocity_tol,
    )

    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )

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
            self.time.append(float(time))
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    step_skip = max(1, int(np.round(output_interval / dt)))
    simulator.collect_diagnostics(rod).using(RodCallback, step_skip=step_skip)

    simulator.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    cb = collector["cb"]
    time_arr = np.asarray(cb.time, dtype=np.float64)
    pos_arr = np.asarray(cb.position, dtype=np.float64)
    dir_arr_hist = np.asarray(cb.director, dtype=np.float64)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_suffix = _build_output_suffix(
        n_elem=n_elem,
        ground_z=ground_z,
        youngs_modulus=youngs_modulus,
        damping_ratio=damping_constant,
        ground_contact_k=ground_contact_k,
        ground_contact_nu=ground_contact_nu,
    )
    output_tag = f"{output_name}{output_suffix}"

    state_path, video_path_four = _resolve_unique_output_paths(
        output_dir=output_dir,
        output_tag=output_tag,
        render=render,
    )
    np.savez(
        state_path,
        time=time_arr,
        position=pos_arr,
        director=dir_arr_hist,
        dt=dt,
        final_time=final_time,
        sine_amp=sine_amp,
        sine_freq=sine_freq,
        sine_phase=sine_phase,
        axial_stretch_stiffening=float(axial_stretch_stiffening),
        ground_z=float(ground_z),
        ground_contact_k=float(ground_contact_k),
        ground_contact_nu=float(ground_contact_nu),
        ground_static_mu=np.asarray(ground_static_mu, dtype=np.float64),
        ground_kinetic_mu=np.asarray(ground_kinetic_mu, dtype=np.float64),
    )

    if render:
        assert video_path_four is not None
        pp.plot_rods_multiview(
            pos_arr,
            video_path=video_path_four,
            times=time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=float(ground_z),
            colors=["#1f77b4"],
        )
    else:
        video_path_four = None

    return {
        "state_path": state_path,
        "video_path_four": video_path_four,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr_hist,
    }


def run_swing(
    motion_path: Path | str = Path(__file__).resolve().parent / "motion.npz",
    dt: float = 1.0e-5,
    n_elem: int = 20,
    base_length: float | None = None,
    base_radius: float = 3.0e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 5.0e6,
    shear_modulus_ratio: float = 1.5,
    axial_stretch_stiffening: float = 1.0e5,
    damping_constant: float = 2.0e-2,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ground_z: float = -0.01,
    ground_contact_k: float = 1.0e4,
    ground_contact_nu: float = 5.0,
    ground_static_mu: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ground_kinetic_mu: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ground_slip_velocity_tol: float = 1.0e-6,
    final_time: float | None = None,
    output_interval: float = 0.01,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "swing",
    render: bool = True,
    render_speed: float = 1.0,
    render_fps: int | None = None,
) -> dict[str, object]:
    """
    Run rod simulation with:
    - node 0 position fixed at origin,
    - node -1 + director -1 driven by interpolated motion.
    """

    motion_time, motion_pos, motion_director = load_motion_npz(motion_path)
    motion = TipMotionInterpolator(motion_time, motion_pos, motion_director)

    class SwingSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        """Simulation container for endpoint-driven rod."""

    simulator = SwingSim()

    origin = np.zeros(3)
    tip0 = motion_pos[0]
    tip_dist = float(np.linalg.norm(tip0 - origin))
    if base_length is None:
        base_length = max(tip_dist, 1.0e-3)
    else:
        base_length = float(base_length)
        if base_length <= 0.0:
            raise ValueError(f"base_length must be > 0, got {base_length}")

    if tip_dist > 1.0e-12:
        direction = (tip0 - origin) / tip_dist
    else:
        direction = np.array([1.0, 0.0, 0.0])
    normal = _choose_normal(direction)

    print("direction", direction)
    print("normal", normal)

    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=origin,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=youngs_modulus / (2.0 * shear_modulus_ratio),
    )
    if axial_stretch_stiffening <= 0.0:
        raise ValueError(
            f"axial_stretch_stiffening must be > 0, got {axial_stretch_stiffening}"
        )
    if not np.isclose(axial_stretch_stiffening, 1.0):
        # Following mytest/vary_wire_property: penalize axial stretch (d3 direction).
        rod.shear_matrix[2, 2, :] *= float(axial_stretch_stiffening)
    simulator.append(rod)

    # Base: position fixed only (orientation unconstrained).
    simulator.constrain(rod).using(
        ea.FixedConstraint,
        constrained_position_idx=(0,),
    )

    # Tip: position + orientation prescribed from motion trajectory.
    simulator.constrain(rod).using(
        TipDrivenBC,
        constrained_position_idx=(-1,),
        constrained_director_idx=(-1,),
        motion=motion,
    )

    if gravity is not None:
        simulator.add_forcing_to(rod).using(
            ea.GravityForces, acc_gravity=np.asarray(gravity, dtype=np.float64)
        )

    _add_ground_contact_with_friction(
        simulator,
        rod,
        ground_z=ground_z,
        ground_contact_k=ground_contact_k,
        ground_contact_nu=ground_contact_nu,
        ground_static_mu=ground_static_mu,
        ground_kinetic_mu=ground_kinetic_mu,
        ground_slip_velocity_tol=ground_slip_velocity_tol,
    )

    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )

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
            self.time.append(float(time))
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    step_skip = max(1, int(np.round(output_interval / dt)))
    simulator.collect_diagnostics(rod).using(RodCallback, step_skip=step_skip)

    simulator.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()

    if final_time is None:
        final_time = motion.t_end
    final_time = float(final_time)
    if final_time <= 0.0:
        raise ValueError(f"final_time must be > 0, got {final_time}")

    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    cb = collector["cb"]
    time_arr = np.asarray(cb.time, dtype=np.float64)
    pos_arr = np.asarray(cb.position, dtype=np.float64)
    dir_arr = np.asarray(cb.director, dtype=np.float64)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_suffix = _build_output_suffix(
        n_elem=n_elem,
        ground_z=ground_z,
        density=density,
        youngs_modulus=youngs_modulus,
        damping_ratio=damping_constant,
        ground_contact_k=ground_contact_k,
        ground_contact_nu=ground_contact_nu,
    )
    output_tag = f"{output_name}{output_suffix}"

    state_path, video_path_four = _resolve_unique_output_paths(
        output_dir=output_dir,
        output_tag=output_tag,
        render=render,
    )
    np.savez(
        state_path,
        time=time_arr,
        position=pos_arr,
        director=dir_arr,
        tip_time=motion_time,
        tip_pos=motion_pos,
        tip_director=motion_director,
        dt=dt,
        final_time=final_time,
        base_length=base_length,
        axial_stretch_stiffening=float(axial_stretch_stiffening),
        ground_z=float(ground_z),
        ground_contact_k=float(ground_contact_k),
        ground_contact_nu=float(ground_contact_nu),
        ground_static_mu=np.asarray(ground_static_mu, dtype=np.float64),
        ground_kinetic_mu=np.asarray(ground_kinetic_mu, dtype=np.float64),
        motion_path=str(Path(motion_path)),
    )

    if render:
        assert video_path_four is not None
        pp.plot_rods_multiview(
            pos_arr,
            video_path=video_path_four,
            times=time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=float(ground_z),
            colors=["#1f77b4"],
            # bounds=
        )
    else:
        video_path_four = None

    return {
        "state_path": state_path,
        "video_path_four": video_path_four,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr,
        "tip_time": motion_time,
        "tip_pos": motion_pos,
        "tip_director": motion_director,
    }


if __name__ == "__main__":

    result = run_swing(
        motion_path=Path(__file__).resolve().parent / "motion.npz",
        dt=1.0e-5,
        n_elem=20,
        base_length=None,
        base_radius=0.0127,
        density=1200.0,
        youngs_modulus=5.0e5,
        shear_modulus_ratio=1.5,
        axial_stretch_stiffening=1.0e2,
        damping_constant=2.0e-2,
        ground_z=-0.013,
        ground_contact_k=1.0e2,
        ground_contact_nu=1.0,
        ground_static_mu=(0.5, 0.5, 0.5),
        ground_kinetic_mu=(0.2, 0.2, 0.2),
        ground_slip_velocity_tol=1.0e-6,
        final_time=None,
        output_interval=0.01,
        output_dir=Path(__file__).resolve().parent,
        output_name="swing",
        render=True,
        render_speed=1.0,
        render_fps=None,
    )

    # result = run_swing_sine_y(
    #     dt=1.0e-5,
    #     n_elem=20,
    #     base_length=2.0,
    #     base_radius=3.0e-3,
    #     density=1_000.0,
    #     youngs_modulus=5.0e6,
    #     shear_modulus_ratio=1.5,
    #     damping_constant=5.0e-2,
    #     sine_amp = 0.5,
    #     sine_freq = 4.0,
    #     sine_phase = 0.0,
    #     ground_z=-0.05,
    #     ground_contact_k=1.0e3,
    #     ground_contact_nu=5.0,
    #     ground_static_mu=(1.0, 1.0, 1.0),
    #     ground_kinetic_mu=(0.5, 0.5, 0.5),
    #     ground_slip_velocity_tol=1.0e-6,
    #     final_time=2.0,
    #     output_interval=0.05,
    #     output_dir=Path(__file__).resolve().parent,
    #     output_name="swing",
    #     render=True,
    #     render_speed=1.0,
    #     render_fps=None,
    # )
    print(
        f"Saved state to {result['state_path']} and 4-view video to "
        f"{result['video_path_four']}."
    )
