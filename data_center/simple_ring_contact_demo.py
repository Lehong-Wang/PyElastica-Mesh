"""
Simple rod-mesh contact demo for data_center/ring_1_3.stl.

Requested setup:
- Rod length: 1.0
- Elements: 40
- Radius: 0.002
- Start: (-0.05, -0.6, 1.32)
- Direction: -z
- Young's modulus: 1e6
- dt: 3e-5
- One endpoint moves from y=-0.6 to y=-0.4, other end remains free.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.animation as animation
import numpy as np

import elastica as ea
from render_scripts import post_processing as pp


class MoveEndpointYBC(ea.ConstraintBase):
    """Move one constrained rod node along y and set endpoint director at end."""

    def __init__(
        self,
        fixed_position: np.ndarray,
        fixed_director: np.ndarray,
        end_y: float,
        move_duration: float,
        target_director: np.ndarray,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.constrained_position_idx.size != 1:
            raise ValueError("MoveEndpointYBC expects exactly one constrained position index.")
        if self.constrained_director_idx.size != 1:
            raise ValueError("MoveEndpointYBC expects exactly one constrained director index.")
        if move_duration <= 0.0:
            raise ValueError("move_duration must be > 0.")

        self.node_idx = int(self.constrained_position_idx[0])
        self.elem_idx = int(self.constrained_director_idx[0])
        self.start_position = np.asarray(fixed_position, dtype=np.float64).reshape(3)
        self.target_position = self.start_position.copy()
        self.target_position[1] = float(end_y)
        self.move_duration = float(move_duration)
        self.velocity = (self.target_position - self.start_position) / self.move_duration
        self.start_director = np.asarray(fixed_director, dtype=np.float64).reshape(3, 3)
        self.target_director = np.asarray(target_director, dtype=np.float64).reshape(3, 3)
        self._end_tol = 1.0e-12

    def _is_finished(self, t: float) -> bool:
        return t >= (self.move_duration - self._end_tol)

    def _position_and_velocity(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t <= 0.0:
            return self.start_position, self.velocity
        if self._is_finished(t):
            return self.target_position, np.zeros(3, dtype=np.float64)

        alpha = t / self.move_duration
        position = (1.0 - alpha) * self.start_position + alpha * self.target_position
        return position, self.velocity

    def constrain_values(self, system, time: np.float64) -> None:
        t = float(time)
        position, _ = self._position_and_velocity(t)
        system.position_collection[:, self.node_idx] = position
        system.director_collection[:, :, self.elem_idx] = (
            self.target_director if self._is_finished(t) else self.start_director
        )

    def constrain_rates(self, system, time: np.float64) -> None:
        _, velocity = self._position_and_velocity(float(time))
        system.velocity_collection[:, self.node_idx] = velocity
        system.omega_collection[:, self.elem_idx] = 0.0


def _prepare_video_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _fmt_param(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return np.format_float_positional(float(value), trim="-")


def _build_output_suffix(
    *,
    dt: float,
    contact_k: float,
    contact_nu: float,
    damping_constant: float,
    n_elements: int,
    density: float,
    base_length: float,
    base_radius: float,
    youngs_modulus: float,
) -> str:
    return (
        f"_dt{_fmt_param(dt)}"
        f"_k{_fmt_param(contact_k)}"
        f"_nu{_fmt_param(contact_nu)}"
        f"_dmp{_fmt_param(damping_constant)}"
        f"_ne{_fmt_param(n_elements)}"
        f"_rho{_fmt_param(density)}"
        f"_L{_fmt_param(base_length)}"
        f"_r{_fmt_param(base_radius)}"
        f"_E{_fmt_param(youngs_modulus)}"
    )


def _strip_output_stem(stem: str) -> str:
    out = stem
    for suffix in ("_4view", "_state"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out


def _preferred_video_suffix() -> str:
    available_writers = set(animation.writers.list())
    if "ffmpeg" in available_writers:
        ffmpeg_writer = animation.writers["ffmpeg"]
        if "pillow" not in ffmpeg_writer.__name__.lower():
            return ".mp4"
    if "pillow" in available_writers:
        animation.writers._registered["ffmpeg"] = animation.writers["pillow"]  # type: ignore[attr-defined]
        return ".gif"
    raise RuntimeError("No supported writer found. Install ffmpeg or pillow.")


def _resolve_unique_output_paths(
    *,
    video_dir: Path,
    state_dir: Path,
    output_tag: str,
    video_suffix: str,
) -> tuple[Path, Path]:
    state_path = state_dir / f"{output_tag}_state.npz"
    video_path = video_dir / f"{output_tag}_4view{video_suffix}"

    if not state_path.exists() and not video_path.exists():
        return video_path, state_path

    idx = 1
    while True:
        state_path = state_dir / f"{output_tag}_{idx}_state.npz"
        video_path = video_dir / f"{output_tag}_{idx}_4view{video_suffix}"
        if not state_path.exists() and not video_path.exists():
            return video_path, state_path
        idx += 1


def run_simple_ring_contact(
    mesh_path: str | Path = Path(__file__).resolve().parent / "ring_1_2.stl",
    n_elements: int = 100,
    density: float = 1000.0,
    base_length: float = 0.5,
    base_radius: float = 0.002,
    youngs_modulus: float = 1.0e6,
    dt: float = 2.0e-5,
    move_duration: float = 0.5,
    settle_time: float = 0.5,
    end_y: float = -0.4,
    contact_k: float = 1.0e4,
    contact_nu: float = 1.0,
    plane_x: float = -0.075,
    plane_k: float = 5.0e2,
    plane_nu: float = 5.0,
    damping_constant: float = 5.0e-2,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    output_video: str | Path = Path(__file__).resolve().parent / "simple_ring_contact_4view.mp4",
    render_fps: int | None = 30,
    render_speed: float = 1.0,
    output_state: str | Path = Path(__file__).resolve().parent / "simple_ring_contact_state.npz",
) -> tuple[Path, Path]:
    class SimpleRingContactSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        pass

    simulator = SimpleRingContactSim()

    start = np.array([-0.06, -0.6, 1.32], dtype=np.float64)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elements,
        start=start,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=youngs_modulus / (2.0 * 1.5),
    )
    simulator.append(rod)

    simulator.constrain(rod).using(
        MoveEndpointYBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        end_y=end_y,
        move_duration=move_duration,
        target_director=np.array(
            # rows are [d1, d2, d3]; here d1=-z and d3=-x
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
            # [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
    )

    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )
    simulator.add_forcing_to(rod).using(
        ea.GravityForces,
        acc_gravity=np.asarray(gravity, dtype=np.float64),
    )



    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = ea.Mesh(str(mesh_path))
    volume = mesh.compute_volume()
    density_mesh = 1000.0
    inertia = mesh.compute_inertia_tensor(density=density_mesh)
    ring = ea.MeshRigidBody(
        mesh=mesh,
        density=density_mesh,
        volume=volume,
        mass_second_moment_of_inertia=inertia,
    )
    simulator.append(ring)

    simulator.detect_contact_between(rod, ring).using(
        ea.RodMeshContact,
        k=contact_k,
        nu=contact_nu,
        mesh_frozen=True,
    )

    wall_plane = ea.Plane(
        plane_origin=np.array([plane_x, 0.0, 0.0], dtype=np.float64),
        plane_normal=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    simulator.append(wall_plane)
    simulator.detect_contact_between(rod, wall_plane).using(
        ea.RodPlaneContact,
        k=plane_k,
        nu=plane_nu,
    )

    collector_store: dict[str, object] = {}

    class SimpleCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = int(step_skip)
            self.time: list[float] = []
            self.rod_position: list[np.ndarray] = []
            self.rod_director: list[np.ndarray] = []
            self.mesh_position: list[np.ndarray] = []
            self.mesh_director: list[np.ndarray] = []
            collector_store["cb"] = self

        def add_snapshot(self, system, time_value: float) -> None:
            self.time.append(float(time_value))
            self.rod_position.append(system.position_collection.copy())
            self.rod_director.append(system.director_collection.copy())
            self.mesh_position.append(ring.position_collection[:, 0].copy())
            self.mesh_director.append(ring.director_collection[:, :, 0].copy())

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.add_snapshot(system, float(time))

    simulator.collect_diagnostics(rod).using(
        SimpleCallback,
        step_skip=max(1, int(0.005 / dt)),
    )

    final_time = float(move_duration + settle_time)
    total_steps = int(np.ceil(final_time / dt))

    output_video_path = Path(output_video)
    output_state_path = Path(output_state)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_state_path.parent.mkdir(parents=True, exist_ok=True)

    output_base = _strip_output_stem(output_video_path.stem)
    if output_base == "":
        output_base = _strip_output_stem(output_state_path.stem)
    if output_base == "":
        output_base = "simple_ring_contact"
    output_suffix = _build_output_suffix(
        dt=dt,
        contact_k=contact_k,
        contact_nu=contact_nu,
        damping_constant=damping_constant,
        n_elements=n_elements,
        density=density,
        base_length=base_length,
        base_radius=base_radius,
        youngs_modulus=youngs_modulus,
    )
    output_tag = f"{output_base}{output_suffix}"
    video_suffix = _preferred_video_suffix()
    output_video_path, output_state_path = _resolve_unique_output_paths(
        video_dir=output_video_path.parent,
        state_dir=output_state_path.parent,
        output_tag=output_tag,
        video_suffix=video_suffix,
    )
    output_video_path = _prepare_video_path(output_video_path)
    if output_video_path.suffix == ".gif":
        print(
            f"[simple_ring_contact_demo] ffmpeg unavailable; falling back to GIF: {output_video_path}"
        )

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    ea.integrate(timestepper, simulator, final_time, total_steps)

    callback = collector_store.get("cb")
    if callback is None:
        raise RuntimeError("Callback did not initialize; no state to save.")
    if len(callback.time) == 0 or callback.time[-1] < (final_time - 0.5 * dt):
        callback.add_snapshot(rod, final_time)

    np.savez_compressed(
        output_state_path,
        time=np.asarray(callback.time),
        rod_position=np.asarray(callback.rod_position),
        rod_director=np.asarray(callback.rod_director),
        mesh_position=np.asarray(callback.mesh_position),
        mesh_director=np.asarray(callback.mesh_director),
        dt=np.float64(dt),
        final_time=np.float64(final_time),
        start_position=start,
        end_y=np.float64(end_y),
        plane_x=np.float64(plane_x),
        n_elements=np.int64(n_elements),
        density=np.float64(density),
        base_length=np.float64(base_length),
        base_radius=np.float64(base_radius),
        youngs_modulus=np.float64(youngs_modulus),
    )

    mesh_dict = {
        "mesh": mesh.mesh,
        "position": np.asarray(callback.mesh_position),
        "director": np.asarray(callback.mesh_director),
        "time": np.asarray(callback.time),
    }
    rod_pos_arr = np.asarray(callback.rod_position)
    mesh_pos_arr = np.asarray(callback.mesh_position)
    mesh_radius = float(ring.radius)
    margin_xy = 0.05
    x_min = float(
        min(np.min(rod_pos_arr[:, 0, :]), np.min(mesh_pos_arr[:, 0] - mesh_radius))
        - margin_xy
    )
    x_max = float(
        max(np.max(rod_pos_arr[:, 0, :]), np.max(mesh_pos_arr[:, 0] + mesh_radius))
        + margin_xy
    )
    y_min = float(
        min(np.min(rod_pos_arr[:, 1, :]), np.min(mesh_pos_arr[:, 1] - mesh_radius))
        - margin_xy
    )
    y_max = float(
        max(np.max(rod_pos_arr[:, 1, :]), np.max(mesh_pos_arr[:, 1] + mesh_radius))
        + margin_xy
    )
    render_bounds = ((x_min, x_max), (y_min, y_max), (1.2, 1.4))
    pp.plot_rods_with_mesh_multiview(
        mesh_dict=mesh_dict,
        rod_positions=rod_pos_arr,
        video_path=output_video_path,
        times=np.asarray(callback.time),
        fps=render_fps,
        speed=render_speed,
        bounds=render_bounds,
        plane_z=None,
    )
    return output_video_path, output_state_path


if __name__ == "__main__":
    video_path, state_path = run_simple_ring_contact()
    print(f"Saved video to: {video_path}")
    print(f"Saved state to: {state_path}")
