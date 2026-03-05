"""
Generate multiple cable-like rods guided toward sampled target poses.

Behavior:
- Load `1_pos` and `1_director` from `data_center/data_center_points.npz`.
- Randomly sample 16 targets.
- For each sampled target, create a rod from a left/right start side based on target y
  versus mean y (80% left if y < mean_y, otherwise 20% left).
- Endpoint motion is staged:
  1) Move only in y to target y over 1.0 s.
  2) Interpolate to target position and target director over 0.5 s.
- Add contact against:
  - fixed ring meshes (`data_center/ring_1_1.stl`, `data_center/ring_1_2.stl`)
    via RodMeshContact,
  - all other rods via RodRodContact,
  - plane x = -0.075 with normal +x via RodPlaneContact.
- Render a four-view video with rods + ring mesh + visualized x-plane.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import numpy as np
import open3d as o3d

# Keep matplotlib writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))
matplotlib.use("Agg")

# Make repository importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import elastica as ea
from render_scripts import post_processing as pp


def _orthonormalize_rotation(mat: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto SO(3) using SVD."""
    u, _, vt = np.linalg.svd(mat)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


class StagedEndpointBC(ea.ConstraintBase):
    """Constrain one node/element with an optional delayed two-stage trajectory."""

    def __init__(
        self,
        fixed_position: np.ndarray,
        fixed_director: np.ndarray,
        target_position: np.ndarray,
        target_director: np.ndarray,
        stage_one_duration: float = 1.0,
        stage_two_duration: float = 0.5,
        start_time: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.constrained_position_idx.size != 1:
            raise ValueError("StagedEndpointBC expects exactly one constrained position index.")
        if self.constrained_director_idx.size != 1:
            raise ValueError("StagedEndpointBC expects exactly one constrained director index.")
        if stage_one_duration <= 0.0 or stage_two_duration <= 0.0:
            raise ValueError("Stage durations must be positive.")
        if start_time < 0.0:
            raise ValueError("start_time must be >= 0.")

        self.node_idx = int(self.constrained_position_idx[0])
        self.elem_idx = int(self.constrained_director_idx[0])

        self.start_position = np.asarray(fixed_position, dtype=np.float64).reshape(3)
        self.start_director = _orthonormalize_rotation(
            np.asarray(fixed_director, dtype=np.float64).reshape(3, 3)
        )

        self.target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
        self.target_director = _orthonormalize_rotation(
            np.asarray(target_director, dtype=np.float64).reshape(3, 3)
        )

        self.stage_one_duration = float(stage_one_duration)
        self.stage_two_duration = float(stage_two_duration)
        self.start_time = float(start_time)
        self.stage_switch_time = self.stage_one_duration
        self.stage_end_time = self.stage_one_duration + self.stage_two_duration

        self.intermediate_position = self.start_position.copy()
        self.intermediate_position[1] = self.target_position[1]

    def _state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_t = t - self.start_time
        if local_t <= 0.0:
            return (
                self.start_position,
                self.start_director,
                np.zeros(3, dtype=np.float64),
            )

        if local_t < self.stage_switch_time:
            alpha = local_t / self.stage_one_duration
            pos = (1.0 - alpha) * self.start_position + alpha * self.intermediate_position
            vel = (self.intermediate_position - self.start_position) / self.stage_one_duration
            return pos, self.start_director, vel

        if local_t < self.stage_end_time:
            beta = (local_t - self.stage_switch_time) / self.stage_two_duration
            pos = (1.0 - beta) * self.intermediate_position + beta * self.target_position
            vel = (self.target_position - self.intermediate_position) / self.stage_two_duration
            dir_lerp = (1.0 - beta) * self.start_director + beta * self.target_director
            director = _orthonormalize_rotation(dir_lerp)
            return pos, director, vel

        return (
            self.target_position,
            self.target_director,
            np.zeros(3, dtype=np.float64),
        )

    def constrain_values(self, system, time: np.float64) -> None:
        pos, director, _ = self._state(float(time))
        system.position_collection[:, self.node_idx] = pos
        system.director_collection[:, :, self.elem_idx] = director

    def constrain_rates(self, system, time: np.float64) -> None:
        _, _, vel = self._state(float(time))
        system.velocity_collection[:, self.node_idx] = vel
        # Director is directly constrained, so keep angular rate fixed at zero.
        system.omega_collection[:, self.elem_idx] = 0.0


def _prepare_video_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    available_writers = set(animation.writers.list())
    if "ffmpeg" in available_writers:
        ffmpeg_writer_cls = animation.writers["ffmpeg"]
        ffmpeg_writer_name = getattr(ffmpeg_writer_cls, "__name__", "").lower()
        # If ffmpeg has previously been aliased to Pillow, keep using GIF.
        if "pillow" not in ffmpeg_writer_name:
            return output
    if "pillow" in available_writers:
        animation.writers._registered["ffmpeg"] = animation.writers["pillow"]  # type: ignore[attr-defined]
        gif_path = output.with_suffix(".gif")
        print(f"[generate_cable] ffmpeg unavailable; falling back to GIF: {gif_path}")
        return gif_path
    raise RuntimeError("No supported writer found. Install ffmpeg or pillow.")


def _format_tag_value(value: float | int) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    text = text.replace("+", "")
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text


def _build_tagged_output_paths(
    output: Path,
    state_output: Path,
    dt: float,
    contact_k: float,
    contact_nu: float,
    damping_constant: float,
    n_elements: int,
    density: float,
    base_length: float,
    base_radius: float,
    youngs_modulus: float,
    activation_interval: float,
) -> tuple[Path, Path]:
    output_suffix = output.suffix if output.suffix else ".mp4"
    state_suffix = state_output.suffix if state_output.suffix else ".npz"

    tag = (
        f"dt{_format_tag_value(float(dt))}"
        f"_contact_k{_format_tag_value(float(contact_k))}"
        f"_contact_nu{_format_tag_value(float(contact_nu))}"
        f"_damping_constant{_format_tag_value(float(damping_constant))}"
        f"_n_elements{_format_tag_value(int(n_elements))}"
        f"_density{_format_tag_value(float(density))}"
        f"_base_length{_format_tag_value(float(base_length))}"
        f"_base_radius{_format_tag_value(float(base_radius))}"
        f"_youngs_modulus{_format_tag_value(float(youngs_modulus))}"
        f"_activation_interval{_format_tag_value(float(activation_interval))}"
    )

    tagged_output = output.with_name(f"{output.stem}_{tag}{output_suffix}")
    tagged_state = state_output.with_name(f"{state_output.stem}_{tag}{state_suffix}")
    return tagged_output, tagged_state


def _resolve_unique_output_paths(video_path: Path, state_path: Path) -> tuple[Path, Path]:
    if not video_path.exists() and not state_path.exists():
        return video_path, state_path

    index = 1
    while True:
        candidate_video = video_path.with_name(
            f"{video_path.stem}_{index}{video_path.suffix}"
        )
        candidate_state = state_path.with_name(
            f"{state_path.stem}_{index}{state_path.suffix}"
        )
        if not candidate_video.exists() and not candidate_state.exists():
            return candidate_video, candidate_state
        index += 1


def _compute_bounds_with_margin(
    rod_positions: np.ndarray,
    mesh_vertices: np.ndarray,
    margin: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    rod_min = np.nanmin(rod_positions, axis=(0, 1, 3))
    rod_max = np.nanmax(rod_positions, axis=(0, 1, 3))
    mesh_min = np.nanmin(mesh_vertices, axis=0)
    mesh_max = np.nanmax(mesh_vertices, axis=0)
    low = np.minimum(rod_min, mesh_min) - margin
    high = np.maximum(rod_max, mesh_max) + margin
    return (
        (float(low[0]), float(high[0])),
        (float(low[1]), float(high[1])),
        (float(low[2]), float(high[2])),
    )


def _build_render_mesh_with_plane(
    ring_mesh_legacy_list: list[o3d.geometry.TriangleMesh],
    plane_x: float,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    wall_thickness: float = 0.002,
) -> o3d.geometry.TriangleMesh:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    span_y = max(0.05, ymax - ymin)
    span_z = max(0.05, zmax - zmin)

    wall = o3d.geometry.TriangleMesh.create_box(
        width=float(wall_thickness),
        height=float(span_y),
        depth=float(span_z),
    )
    wall.translate(
        np.array(
            [
                plane_x - 0.5 * wall_thickness,
                ymin,
                zmin,
            ],
            dtype=np.float64,
        )
    )
    wall.compute_vertex_normals()
    wall.compute_triangle_normals()

    if len(ring_mesh_legacy_list) == 0:
        raise ValueError("ring_mesh_legacy_list must contain at least one mesh.")

    combined = o3d.geometry.TriangleMesh(ring_mesh_legacy_list[0])
    for mesh_i in ring_mesh_legacy_list[1:]:
        combined += o3d.geometry.TriangleMesh(mesh_i)
    combined += wall
    combined.compute_vertex_normals()
    combined.compute_triangle_normals()
    return combined


def generate_cable(
    data_npz_path: str | Path = "data_center/data_center_points.npz",
    mesh_path_one: str | Path = "data_center/ring_1_1.stl",
    mesh_path_two: str | Path = "data_center/ring_1_2.stl",
    output: str | Path = "data_center/generate_cable_4view.mp4",
    state_output: str | Path = "data_center/generate_cable_state.npz",
    n_sample: int = 6,
    seed: int | None = None,
    dt: float = 2.0e-5,
    contact_k: float = 2.0e3,
    contact_nu: float = 2.0,
    damping_constant: float = 2e-1,
    n_elements: int = 100,
    density: float = 1000.0,
    base_length: float = 0.6,
    base_radius: float = 0.002,
    youngs_modulus: float = 1.0e6,
    stage_one_duration: float = 0.5,
    stage_two_duration: float = 0.25,
    final_hold_time: float = 0.5,
    activation_interval: float = 0.3,
    outward_y_step: float = 0.01,
    render_fps: int | None = 30,
    render_speed: float = 1.0,
) -> tuple[Path, Path]:
    class CableSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        pass

    data_npz_path = Path(data_npz_path)
    mesh_path_one = Path(mesh_path_one)
    mesh_path_two = Path(mesh_path_two)
    tagged_output_path, tagged_state_output_path = _build_tagged_output_paths(
        output=Path(output),
        state_output=Path(state_output),
        dt=dt,
        contact_k=contact_k,
        contact_nu=contact_nu,
        damping_constant=damping_constant,
        n_elements=n_elements,
        density=density,
        base_length=base_length,
        base_radius=base_radius,
        youngs_modulus=youngs_modulus,
        activation_interval=activation_interval,
    )
    output_path = _prepare_video_path(tagged_output_path)
    output_path, state_output_path = _resolve_unique_output_paths(
        output_path, tagged_state_output_path
    )
    state_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_npz_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_npz_path}")
    if not mesh_path_one.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path_one}")
    if not mesh_path_two.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path_two}")

    with np.load(data_npz_path) as data:
        if "1_pos" not in data or "1_director" not in data:
            raise KeyError(
                f"{data_npz_path} must contain keys '1_pos' and '1_director'. Found: {list(data.files)}"
            )
        all_positions = np.asarray(data["1_pos"], dtype=np.float64)
        all_directors = np.asarray(data["1_director"], dtype=np.float64)

    if all_positions.ndim != 2 or all_positions.shape[1] != 3:
        raise ValueError(f"'1_pos' must have shape (N,3), got {all_positions.shape}")
    if all_directors.shape != (all_positions.shape[0], 3, 3):
        raise ValueError(
            f"'1_director' must have shape (N,3,3) matching N in '1_pos', got {all_directors.shape}"
        )
    if n_sample <= 0 or n_sample > all_positions.shape[0]:
        raise ValueError(f"n_sample must be in [1, {all_positions.shape[0]}], got {n_sample}")

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng(seed)

    # Ensure random sampling never returns duplicate connector poses, even if
    # source npz contains duplicated rows.
    pose_director_flat = np.ascontiguousarray(
        np.concatenate([all_positions, all_directors.reshape(all_directors.shape[0], -1)], axis=1)
    )
    pose_director_key = pose_director_flat.view(
        np.dtype((np.void, pose_director_flat.dtype.itemsize * pose_director_flat.shape[1]))
    ).ravel()
    unique_first_indices = np.unique(pose_director_key, return_index=True)[1]
    unique_source_indices = np.sort(unique_first_indices.astype(np.int64))

    if n_sample > unique_source_indices.size:
        raise ValueError(
            "n_sample must be <= number of unique (position,director) entries "
            f"({unique_source_indices.size}), got {n_sample}"
        )

    sampled_indices = rng.choice(unique_source_indices, size=n_sample, replace=False)
    sampled_positions = all_positions[sampled_indices]
    sampled_directors = np.asarray(
        [_orthonormalize_rotation(all_directors[idx]) for idx in sampled_indices],
        dtype=np.float64,
    )

    mean_pos = sampled_positions.mean(axis=0)
    mean_x, mean_y, mean_z = (float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2]))

    left_start = np.array([mean_x - 0.02, mean_y - 0.3, mean_z + 0.0], dtype=np.float64)
    right_start = np.array([mean_x - 0.02, mean_y + 0.3, mean_z + 0.0], dtype=np.float64)

    simulator = CableSim()

    rods: list[ea.CosseratRod] = []
    rod_start_positions: list[np.ndarray] = []

    n_elem = int(n_elements)
    rod_length = float(base_length)
    rod_radius = float(base_radius)
    rod_density = float(density)
    youngs_modulus_val = float(youngs_modulus)
    shear_modulus = youngs_modulus_val / (2.0 * 1.5)
    rod_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    rod_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Avoid exact rod-rod centerline overlap at t=0 (rod-rod kernel divides by distance).
    side_spacing = 3.0 * rod_radius
    left_count = 0
    right_count = 0

    if activation_interval < 0.0:
        raise ValueError("activation_interval must be >= 0.")
    if outward_y_step < 0.0:
        raise ValueError("outward_y_step must be >= 0.")

    for rod_global_idx, (target_pos, target_dir) in enumerate(
        zip(sampled_positions, sampled_directors)
    ):
        target_y = float(target_pos[1])
        if target_y < mean_y - 0.1:
            create_left = True
        elif abs(target_y - mean_y) <= 0.1:
            create_left = bool(rng.random() < 0.5)
        else:
            create_left = False
        start = left_start.copy() if create_left else right_start.copy()

        if create_left:
            local_idx = left_count
            left_count += 1
        else:
            local_idx = right_count
            right_count += 1

        # Place rods on a tiny x-y lattice around the requested side start.
        # This preserves the side logic while preventing zero-distance rod pairs.
        grid_cols = 4
        gx = local_idx % grid_cols
        gy = local_idx // grid_cols
        if create_left:
            # start[0] += (gx - 0.5 * (grid_cols - 1)) * side_spacing
            start[1] -= outward_y_step * local_idx
        else:
            # Right starts are already near/behind x=-0.075 wall; bias offsets
            # toward +x to reduce deep initial wall penetration.
            # start[0] += gx * side_spacing
            start[1] += outward_y_step * local_idx
        start[1] += (gy - 0.5) * side_spacing

        rod = ea.CosseratRod.straight_rod(
            n_elements=n_elem,
            start=start,
            direction=rod_direction,
            normal=rod_normal,
            base_length=rod_length,
            base_radius=rod_radius,
            density=rod_density,
            youngs_modulus=youngs_modulus_val,
            shear_modulus=shear_modulus,
        )
        simulator.append(rod)
        rods.append(rod)
        rod_start_positions.append(start)

        simulator.constrain(rod).using(
            StagedEndpointBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
            target_position=target_pos,
            target_director=target_dir,
            stage_one_duration=stage_one_duration,
            stage_two_duration=stage_two_duration,
            start_time=rod_global_idx * activation_interval,
        )

        simulator.dampen(rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=float(damping_constant),
            time_step=dt,
        )

    ring_mesh_one = ea.Mesh(str(mesh_path_one))
    ring_mesh_two = ea.Mesh(str(mesh_path_two))
    ring_density = 1000.0
    ring_volume_one = ring_mesh_one.compute_volume()
    ring_inertia_one = ring_mesh_one.compute_inertia_tensor(density=ring_density)
    ring_body_one = ea.MeshRigidBody(
        mesh=ring_mesh_one,
        density=ring_density,
        volume=ring_volume_one,
        mass_second_moment_of_inertia=ring_inertia_one,
    )
    simulator.append(ring_body_one)

    ring_volume_two = ring_mesh_two.compute_volume()
    ring_inertia_two = ring_mesh_two.compute_inertia_tensor(density=ring_density)
    ring_body_two = ea.MeshRigidBody(
        mesh=ring_mesh_two,
        density=ring_density,
        volume=ring_volume_two,
        mass_second_moment_of_inertia=ring_inertia_two,
    )
    simulator.append(ring_body_two)

    plane_x = -0.075
    wall_plane = ea.Plane(
        plane_origin=np.array([plane_x, 0.0, 0.0], dtype=np.float64),
        plane_normal=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    simulator.append(wall_plane)

    # Conservative contact parameters for multi-rod stability.
    rod_mesh_k = float(contact_k)
    rod_mesh_nu = float(contact_nu)
    rod_rod_k = float(contact_k)
    rod_rod_nu = float(contact_nu)
    rod_plane_k = 0.25 * float(contact_k)
    rod_plane_nu = float(contact_nu)

    for rod in rods:
        simulator.detect_contact_between(rod, ring_body_one).using(
            ea.RodMeshContact,
            k=rod_mesh_k,
            nu=rod_mesh_nu,
            mesh_frozen=True,
        )
        simulator.detect_contact_between(rod, ring_body_two).using(
            ea.RodMeshContact,
            k=rod_mesh_k,
            nu=rod_mesh_nu,
            mesh_frozen=True,
        )
        simulator.detect_contact_between(rod, wall_plane).using(
            ea.RodPlaneContact,
            k=rod_plane_k,
            nu=rod_plane_nu,
        )

    for i in range(len(rods)):
        for j in range(i + 1, len(rods)):
            simulator.detect_contact_between(rods[i], rods[j]).using(
                ea.RodRodContact,
                k=rod_rod_k,
                nu=rod_rod_nu,
            )

    collector_store: dict[str, object] = {}

    class MultiRodCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = int(step_skip)
            self.time: list[float] = []
            self.rod_position: list[np.ndarray] = []
            self.rod_director: list[np.ndarray] = []
            self.mesh_position: list[np.ndarray] = []
            self.mesh_director: list[np.ndarray] = []
            collector_store["cb"] = self

        def add_snapshot(self, time_value: float) -> None:
            # Avoid duplicate timestamps when we manually seed t=0.
            if self.time and abs(float(time_value) - self.time[-1]) < 1.0e-14:
                return
            self.time.append(float(time_value))
            self.rod_position.append(
                np.stack([rod_i.position_collection.copy() for rod_i in rods], axis=0)
            )
            self.rod_director.append(
                np.stack([rod_i.director_collection.copy() for rod_i in rods], axis=0)
            )
            self.mesh_position.append(ring_body_one.position_collection[:, 0].copy())
            self.mesh_director.append(ring_body_one.director_collection[:, :, 0].copy())

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.add_snapshot(float(time))

    fps_for_skip = render_fps if render_fps is not None else 30
    step_skip = max(1, int(1.0 / (fps_for_skip * dt)))
    simulator.collect_diagnostics(rods[0]).using(MultiRodCallback, step_skip=step_skip)

    last_start_time = (len(rods) - 1) * activation_interval if len(rods) > 0 else 0.0
    final_time = last_start_time + stage_one_duration + stage_two_duration + final_hold_time
    total_steps = int(np.ceil(final_time / dt))

    try:
        simulator.finalize()
        cb = collector_store.get("cb")
        if cb is None:
            raise RuntimeError("Callback did not initialize before integration.")
        cb.add_snapshot(0.0)
        timestepper = ea.PositionVerlet()
        ea.integrate(timestepper, simulator, final_time, total_steps)
    finally:
        cb = collector_store.get("cb")
        if cb is None or len(cb.time) == 0:
            raise RuntimeError("No callback data collected; cannot render outputs.")

    time_arr = np.asarray(cb.time, dtype=np.float64)
    rod_pos_arr = np.asarray(cb.rod_position, dtype=np.float64)
    rod_dir_arr = np.asarray(cb.rod_director, dtype=np.float64)
    mesh_pos_arr = np.asarray(cb.mesh_position, dtype=np.float64)
    mesh_dir_arr = np.asarray(cb.mesh_director, dtype=np.float64)

    ring_vertices_one = np.asarray(ring_mesh_one.mesh.vertices, dtype=np.float64)
    ring_vertices_two = np.asarray(ring_mesh_two.mesh.vertices, dtype=np.float64)
    ring_vertices = np.vstack([ring_vertices_one, ring_vertices_two])
    base_bounds = _compute_bounds_with_margin(rod_pos_arr, ring_vertices, margin=0.08)
    render_mesh = _build_render_mesh_with_plane(
        [ring_mesh_one.mesh, ring_mesh_two.mesh],
        plane_x,
        base_bounds,
    )
    render_vertices = np.asarray(render_mesh.vertices, dtype=np.float64)
    render_bounds_auto = _compute_bounds_with_margin(
        rod_pos_arr, render_vertices, margin=0.05
    )
    render_bounds = (
        render_bounds_auto[0],
        render_bounds_auto[1],
        (1.2, 1.4),
    )

    mesh_dict = {
        "mesh": render_mesh,
        "position": mesh_pos_arr,
        "director": mesh_dir_arr,
        "time": time_arr,
    }
    pp.plot_rods_with_mesh_multiview(
        mesh_dict=mesh_dict,
        rod_positions=rod_pos_arr,
        video_path=output_path,
        times=time_arr,
        fps=render_fps,
        speed=render_speed,
        bounds=render_bounds,
        plane_z=None,
    )

    state_payload: dict[str, object] = {
        "sampled_indices": sampled_indices.astype(np.int64),
        "seed": np.int64(seed),
        "sampled_positions": sampled_positions,
        "sampled_directors": sampled_directors,
        "rod_start_positions": np.asarray(rod_start_positions),
        "time": time_arr,
        "rod_position": rod_pos_arr,
        "rod_director": rod_dir_arr,
        "mesh_position": mesh_pos_arr,
        "mesh_director": mesh_dir_arr,
        "dt": np.float64(dt),
        "final_time": np.float64(final_time),
        "stage_one_duration": np.float64(stage_one_duration),
        "stage_two_duration": np.float64(stage_two_duration),
        "activation_interval": np.float64(activation_interval),
        "outward_y_step": np.float64(outward_y_step),
        "contact_k": np.float64(contact_k),
        "contact_nu": np.float64(contact_nu),
        "damping_constant": np.float64(damping_constant),
        "n_elements": np.int64(n_elem),
        "density": np.float64(rod_density),
        "base_length": np.float64(rod_length),
        "base_radius": np.float64(rod_radius),
        "youngs_modulus": np.float64(youngs_modulus_val),
        "plane_x": np.float64(plane_x),
    }
    for rod_idx in range(len(rods)):
        key_id = rod_idx + 1
        state_payload[f"{key_id}_time"] = time_arr
        state_payload[f"{key_id}_pos"] = rod_pos_arr[:, rod_idx, :, :]
        state_payload[f"{key_id}_director"] = rod_dir_arr[:, rod_idx, :, :, :]

    np.savez_compressed(state_output_path, **state_payload)

    duplicate_count = int(all_positions.shape[0] - unique_source_indices.size)
    print(f"[generate_cable] seed: {seed}")
    print(f"[generate_cable] sampled indices: {sampled_indices.tolist()}")
    if duplicate_count > 0:
        print(
            f"[generate_cable] skipped {duplicate_count} duplicate source rows "
            "(same position and director)."
        )
    print(f"[generate_cable] saved video: {output_path}")
    print(f"[generate_cable] saved state: {state_output_path}")
    return output_path, state_output_path


if __name__ == "__main__":
    generate_cable()
