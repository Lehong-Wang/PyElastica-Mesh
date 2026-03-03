"""Fixed bunny + gravity rod demo with a position-driven rod endpoint trajectory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import numpy as np

import elastica as ea
from examples.MeshCase import post_processing


DEFAULT_WAYPOINTS = np.array(
    [
        [0.0, -0.1, 0.1],
        [0.0, 0.1, 0.1],
    ],
    dtype=np.float64,
)


def _format_tag_value(value: float | int) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    text = text.replace("+", "")
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text


def _build_output_paths(
    output: str,
    n_elem: int,
    youngs_modulus: float,
    damping_constant: float,
    waypoint_segment_duration: float,
    contact_friction: float,
) -> tuple[Path, Path]:
    output_path = Path(output)
    suffix = output_path.suffix if output_path.suffix else ".mp4"
    tag = (
        f"n{n_elem}"
        f"_E{_format_tag_value(float(youngs_modulus))}"
        f"_d{_format_tag_value(float(damping_constant))}"
        f"_sd{_format_tag_value(float(waypoint_segment_duration))}"
        f"_cf{_format_tag_value(float(contact_friction))}"
    )
    tagged_video = output_path.with_name(f"{output_path.stem}_{tag}{suffix}")
    tagged_npz = output_path.with_name(f"{output_path.stem}_{tag}_state.npz")
    return tagged_video, tagged_npz


def _load_waypoints_from_npz(npz_path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Waypoint file not found: {path}")

    with np.load(path) as data:
        if key is not None:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.files)}")
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
                raise ValueError(
                    f"Array '{key}' in {path} must have shape (N, 3) with N >= 2, got {arr.shape}."
                )
            return arr

        for name in data.files:
            arr = np.asarray(data[name], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 2:
                return arr

    raise ValueError(
        f"No array with shape (N, 3), N>=2 found in {path}. "
        "Provide a key via waypoints_npz_key if needed."
    )


def _resolve_unique_output_paths(video_path: Path, npz_path: Path) -> tuple[Path, Path]:
    if not video_path.exists() and not npz_path.exists():
        return video_path, npz_path

    idx = 1
    while True:
        candidate_video = video_path.with_name(f"{video_path.stem}_{idx}{video_path.suffix}")
        candidate_npz = npz_path.with_name(f"{npz_path.stem}_{idx}{npz_path.suffix}")
        if not candidate_video.exists() and not candidate_npz.exists():
            return candidate_video, candidate_npz
        idx += 1


def _build_interpolated_trajectory(
    waypoints: np.ndarray,
    waypoint_segment_duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] < 2:
        raise ValueError("waypoints must have shape (N, 3) with N >= 2.")
    if waypoint_segment_duration <= 0.0:
        raise ValueError("waypoint_segment_duration must be > 0.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")

    n_segments = waypoints.shape[0] - 1
    total_time = float(n_segments * waypoint_segment_duration)

    time_samples = np.arange(0.0, total_time, dt, dtype=np.float64)
    if time_samples.size == 0 or not np.isclose(time_samples[-1], total_time):
        time_samples = np.append(time_samples, total_time)

    segment_coordinate = time_samples / waypoint_segment_duration
    segment_idx = np.floor(segment_coordinate).astype(np.int64)
    segment_idx = np.clip(segment_idx, 0, n_segments - 1)
    local_alpha = segment_coordinate - segment_idx

    p0 = waypoints[segment_idx]
    p1 = waypoints[segment_idx + 1]
    position_samples = (1.0 - local_alpha[:, None]) * p0 + local_alpha[:, None] * p1

    return time_samples, position_samples


class MovingPositionBC(ea.ConstraintBase):
    """Constrain only one rod node position (and its translational rate) along a trajectory."""

    def __init__(
        self,
        fixed_position: np.ndarray,
        trajectory_time: np.ndarray,
        trajectory_position: np.ndarray,
        hold_last: bool = True,
        release_time: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.constrained_position_idx.size != 1:
            raise ValueError("MovingPositionBC expects exactly one constrained position index.")
        self.node_idx = int(self.constrained_position_idx[0])

        self.fixed_position = np.asarray(fixed_position, dtype=np.float64).reshape(3)

        self.trajectory_time = np.asarray(trajectory_time, dtype=np.float64).reshape(-1)
        self.trajectory_position = np.asarray(trajectory_position, dtype=np.float64)
        if self.trajectory_time.ndim != 1 or self.trajectory_position.ndim != 2:
            raise ValueError("Invalid trajectory dimensions.")
        if self.trajectory_position.shape != (self.trajectory_time.size, 3):
            raise ValueError("trajectory_position must have shape (len(trajectory_time), 3).")
        if self.trajectory_time.size < 2:
            raise ValueError("Trajectory must contain at least 2 samples.")
        if not np.all(np.diff(self.trajectory_time) > 0.0):
            raise ValueError("trajectory_time must be strictly increasing.")

        self.hold_last = bool(hold_last)
        if release_time is not None and release_time < 0.0:
            raise ValueError("release_time must be >= 0 when provided.")
        self.release_time = None if release_time is None else float(release_time)

    def _trajectory_state(self, time_value: float) -> tuple[np.ndarray, np.ndarray]:
        if time_value <= self.trajectory_time[0]:
            p0 = self.trajectory_position[0]
            p1 = self.trajectory_position[1]
            dt = self.trajectory_time[1] - self.trajectory_time[0]
            return p0, (p1 - p0) / dt

        t_end = self.trajectory_time[-1]
        if time_value >= t_end:
            if self.hold_last:
                return self.trajectory_position[-1], np.zeros(3, dtype=np.float64)
            p0 = self.trajectory_position[-2]
            p1 = self.trajectory_position[-1]
            dt = self.trajectory_time[-1] - self.trajectory_time[-2]
            return p1, (p1 - p0) / dt

        idx = np.searchsorted(self.trajectory_time, time_value, side="right") - 1
        idx = int(np.clip(idx, 0, self.trajectory_time.size - 2))
        t0 = self.trajectory_time[idx]
        t1 = self.trajectory_time[idx + 1]
        p0 = self.trajectory_position[idx]
        p1 = self.trajectory_position[idx + 1]
        alpha = (time_value - t0) / (t1 - t0)
        position = (1.0 - alpha) * p0 + alpha * p1
        velocity = (p1 - p0) / (t1 - t0)
        return position, velocity

    def constrain_values(self, system, time: np.float64) -> None:
        t = float(time)
        if self.release_time is not None and t >= self.release_time:
            return
        position, _ = self._trajectory_state(t)
        system.position_collection[:, self.node_idx] = position

    def constrain_rates(self, system, time: np.float64) -> None:
        t = float(time)
        if self.release_time is not None and t >= self.release_time:
            return
        _, velocity = self._trajectory_state(t)
        system.velocity_collection[:, self.node_idx] = velocity


def _prepare_video_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    available_writers = set(animation.writers.list())
    if "ffmpeg" in available_writers:
        return output
    if "pillow" in available_writers:
        animation.writers._registered["ffmpeg"] = animation.writers["pillow"]  # type: ignore[attr-defined]
        gif_path = output.with_suffix(".gif")
        print(f"[bunny_fixed_waypoint_rod] ffmpeg unavailable; falling back to GIF: {gif_path}")
        return gif_path
    raise RuntimeError("No supported writer found. Install ffmpeg or pillow.")


def run_bunny_fixed_waypoint_rod(
    mesh_path: str = "demo_2/bunny_small.stl",
    output: str = "demo_2/bunny_fixed_waypoint_rod_4view.mp4",
    waypoints: np.ndarray | None = None,
    waypoints_npz_path: str | Path | None = None,
    waypoints_npz_key: str | None = None,
    waypoint_segment_duration: float = 0.04,
    final_hold_time: float = 0.5,
    no_end_constraint_time: float = 1.0,
    dt: float = 1.0e-5,
    n_elem: int = 100,
    rod_length: float = 0.6,
    rod_radius: float = 0.003175,
    rod_density: float = 1000.0,
    rod_youngs_modulus: float = 2.0e5,
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 0.4,
    mesh_density: float = 800.0,
    contact_k: float = 5.0e3,
    contact_nu: float = 2.0,
    contact_velocity_damping: float = 0.2,
    contact_friction: float = 5,
    self_contact_k: float = 5.0e3,
    self_contact_nu: float = 2.0,
    render_around_bunny_only: bool = True,
    bunny_render_margin: float = 0.05,
    render_speed: float = 1.0,
    render_fps: int | None = 5,
) -> tuple[Path, Path]:
    class BunnyRodWaypointSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        pass

    if dt <= 0.0:
        raise ValueError("dt must be > 0.")
    if final_hold_time < 0.0:
        raise ValueError("final_hold_time must be >= 0.")
    if no_end_constraint_time < 0.0:
        raise ValueError("no_end_constraint_time must be >= 0.")
    if self_contact_k < 0.0:
        raise ValueError("self_contact_k must be >= 0.")
    if self_contact_nu < 0.0:
        raise ValueError("self_contact_nu must be >= 0.")
    if bunny_render_margin < 0.0:
        raise ValueError("bunny_render_margin must be >= 0.")

    if waypoints_npz_path is not None:
        waypoint_array = _load_waypoints_from_npz(
            npz_path=waypoints_npz_path,
            key=waypoints_npz_key,
        )
    elif waypoints is not None:
        waypoint_array = np.asarray(waypoints, dtype=np.float64)
    else:
        waypoint_array = DEFAULT_WAYPOINTS.copy()
    
    # waypoint_array = waypoint_array[:30]

    trajectory_time, trajectory_position = _build_interpolated_trajectory(
        waypoints=waypoint_array,
        waypoint_segment_duration=waypoint_segment_duration,
        dt=dt,
    )

    simulator = BunnyRodWaypointSim()
    release_time = float(trajectory_time[-1] + final_hold_time)

    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=trajectory_position[0],
        direction=np.array([0.0, 0.0, -1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        base_length=rod_length,
        base_radius=rod_radius,
        density=rod_density,
        youngs_modulus=rod_youngs_modulus,
        shear_modulus=rod_youngs_modulus / (2.0 * shear_modulus_ratio),
    )
    simulator.append(rod)

    simulator.constrain(rod).using(
        MovingPositionBC,
        constrained_position_idx=(0,),
        trajectory_time=trajectory_time,
        trajectory_position=trajectory_position,
        hold_last=True,
        release_time=release_time,
    )

    simulator.add_forcing_to(rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81], dtype=np.float64)
    )
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

    mesh = ea.Mesh(mesh_path)
    volume = mesh.compute_volume()
    inertia = mesh.compute_inertia_tensor(density=mesh_density)
    bunny = ea.MeshRigidBody(
        mesh=mesh,
        center_of_mass=np.zeros(3, dtype=np.float64),
        mass_second_moment_of_inertia=inertia,
        density=mesh_density,
        volume=volume,
    )
    simulator.append(bunny)

    simulator.detect_contact_between(rod, bunny).using(
        ea.RodMeshContact,
        k=contact_k,
        nu=contact_nu,
        velocity_damping_coefficient=contact_velocity_damping,
        friction_coefficient=contact_friction,
        mesh_frozen=True,
    )
    simulator.detect_contact_between(rod, rod).using(
        ea.RodSelfContact,
        k=self_contact_k,
        nu=self_contact_nu,
    )

    collector_store: dict[str, object] = {}

    class RodMeshCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.rod_position: list[np.ndarray] = []
            self.rod_director: list[np.ndarray] = []
            self.mesh_position: list[np.ndarray] = []
            self.mesh_director: list[np.ndarray] = []
            collector_store["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(float(time))
            self.rod_position.append(system.position_collection.copy())
            self.rod_director.append(system.director_collection.copy())
            self.mesh_position.append(bunny.position_collection[:, 0].copy())
            self.mesh_director.append(bunny.director_collection[:, :, 0].copy())

    fps_for_skip = render_fps if render_fps is not None else 30
    step_skip = max(1, int(1.0 / (fps_for_skip * dt)))
    simulator.collect_diagnostics(rod).using(RodMeshCallBack, step_skip=step_skip)

    final_time = float(release_time + no_end_constraint_time)
    total_steps = int(np.ceil(final_time / dt))
    tagged_video_path, npz_path = _build_output_paths(
        output=output,
        n_elem=n_elem,
        youngs_modulus=rod_youngs_modulus,
        damping_constant=damping_constant,
        waypoint_segment_duration=waypoint_segment_duration,
        contact_friction=contact_friction,
    )
    output_path = _prepare_video_path(tagged_video_path)
    output_path, npz_path = _resolve_unique_output_paths(output_path, npz_path)

    def _construct_state_snapshot() -> dict[str, np.ndarray] | None:
        cb = collector_store.get("cb")
        if cb is None or len(cb.time) == 0:
            return None
        state_snapshot = {
            "time": np.asarray(cb.time),
            "mesh_position": np.asarray(cb.mesh_position),
            "mesh_director": np.asarray(cb.mesh_director),
            "rod_position": np.asarray(cb.rod_position),
            "rod_director": np.asarray(cb.rod_director),
        }
        print(f"[bunny_fixed_waypoint_rod] Exported state snapshot at {state_snapshot['time'].shape[0]} time steps.")
        print(f"mesh_position shape: {state_snapshot['mesh_position'].shape}")
        print(f"mesh_director shape: {state_snapshot['mesh_director'].shape}")
        print(f"rod_position shape: {state_snapshot['rod_position'].shape}")
        print(f"rod_director shape: {state_snapshot['rod_director'].shape}")
        return state_snapshot

    def _save_npz_from_callback() -> None:
        state_snapshot = _construct_state_snapshot()
        if state_snapshot is None:
            return
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            time=state_snapshot["time"],
            mesh_position=state_snapshot["mesh_position"],
            mesh_director=state_snapshot["mesh_director"],
            rod_position=state_snapshot["rod_position"],
            rod_director=state_snapshot["rod_director"],
            waypoints=waypoint_array,
            trajectory_time=trajectory_time,
            trajectory_position=trajectory_position,
            dt=np.float64(dt),
            final_time=np.float64(final_time),
            release_time=np.float64(release_time),
            n_elem=np.int64(n_elem),
            youngs_modulus=np.float64(rod_youngs_modulus),
            damping_constant=np.float64(damping_constant),
            waypoint_segment_duration=np.float64(waypoint_segment_duration),
            final_hold_time=np.float64(final_hold_time),
            no_end_constraint_time=np.float64(no_end_constraint_time),
            self_contact_k=np.float64(self_contact_k),
            self_contact_nu=np.float64(self_contact_nu),
        )

    def _render_from_callback() -> None:
        cb = collector_store.get("cb")
        if cb is None or len(cb.time) == 0:
            return

        rod_pos = np.asarray(cb.rod_position)
        bunny_pos = np.asarray(cb.mesh_position)
        if render_around_bunny_only:
            bunny_center = bunny_pos[0]
            extent = float(bunny.radius + bunny_render_margin)
            x_min = float(bunny_center[0] - extent)
            x_max = float(bunny_center[0] + extent)
            y_min = float(bunny_center[1] - extent)
            y_max = float(bunny_center[1] + extent)
            z_min = float(bunny_center[2] - extent)
            z_max = float(bunny_center[2] + extent)
        else:
            margin = max(0.08, 2.0 * float(bunny.radius))
            x_min = float(min(np.min(rod_pos[:, 0, :]), np.min(bunny_pos[:, 0]) - bunny.radius) - margin)
            x_max = float(max(np.max(rod_pos[:, 0, :]), np.max(bunny_pos[:, 0]) + bunny.radius) + margin)
            y_min = float(min(np.min(rod_pos[:, 1, :]), np.min(bunny_pos[:, 1]) - bunny.radius) - margin)
            y_max = float(max(np.max(rod_pos[:, 1, :]), np.max(bunny_pos[:, 1]) + bunny.radius) + margin)
            z_min = float(min(np.min(rod_pos[:, 2, :]), np.min(bunny_pos[:, 2]) - bunny.radius) - margin)
            z_max = float(max(np.max(rod_pos[:, 2, :]), np.max(bunny_pos[:, 2]) + bunny.radius) + margin)

        mesh_data = {
            "time": cb.time,
            "position": cb.mesh_position,
            "director": cb.mesh_director,
            "mesh": mesh.mesh,
        }
        post_processing.plot_mesh_multiview_animation(
            mesh_data,
            video_name=str(output_path),
            fps=render_fps,
            speed_factor=render_speed,
            bounds=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
            rod_positions=cb.rod_position,
        )

    try:
        simulator.finalize()
        timestepper = ea.PositionVerlet()
        ea.integrate(timestepper, simulator, final_time, total_steps)
        _save_npz_from_callback()
        _render_from_callback()
    except Exception:
        try:
            _save_npz_from_callback()
        except Exception as save_exc:  # noqa: BLE001
            print(f"[bunny_fixed_waypoint_rod] NPZ save failed after exception: {save_exc}")
        try:
            _render_from_callback()
        except Exception as render_exc:  # noqa: BLE001
            print(f"[bunny_fixed_waypoint_rod] Rendering failed after exception: {render_exc}")
        raise

    return output_path, npz_path


if __name__ == "__main__":
    saved_video_path, saved_npz_path = run_bunny_fixed_waypoint_rod(
        waypoints_npz_path="demo_2/traj.npz",
        waypoints_npz_key = "positions",
        final_hold_time=1.0,
        no_end_constraint_time=2.0,
    )
    print(f"Saved video to: {saved_video_path}")
    print(f"Saved npz to: {saved_npz_path}")
