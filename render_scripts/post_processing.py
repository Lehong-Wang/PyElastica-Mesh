"""
Lightweight plotting utilities for render scripts.

Features
- Rod-only video with optional ground planes; nodes drawn in black, elements colored.
- Rod-only four-view option (3D + orthographic front/right/top).
- Rod + mesh animations (single- or four-view) following the MeshCase style.
- Supports up to ~10 rods with distinct colors (tab10 palette).

Usage
    from render_scripts import post_processing as pp
    pp.plot_rods_video(rod_positions, video_path="rods.mp4")
    pp.plot_rods_with_mesh_multiview(mesh_dict, rod_positions, video_path="mv.mp4")

Expected shapes
- rod_positions: (n_frames, n_rods, 3, n_nodes) or (n_frames, 3, n_nodes) for one rod.
- mesh_dict: {"mesh": open3d TriangleMesh, "position": (n_frames,3),
              "director": (n_frames,3,3), "time": (n_frames,) optional}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

# Headless rendering must be configured before importing pyplot.
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

# Ensure matplotlib can write its cache in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_FPS = 30


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _compute_render_indices(
    times: np.ndarray | None,
    n_frames: int,
    fps: int | None,
    speed: float,
) -> tuple[int, list[int]]:
    if n_frames == 0:
        return DEFAULT_FPS, []
    if speed <= 0.0:
        raise ValueError(f"speed must be > 0.0, got {speed}.")
    times_arr = None if times is None or len(times) == 0 else np.asarray(times)
    default_fps = DEFAULT_FPS if fps is None else fps
    if times_arr is None or len(times_arr) < 2:
        return default_fps, list(range(n_frames))
    dt = float(np.median(np.diff(times_arr)))
    fps_out = default_fps if fps is not None else max(1, int(round(1.0 / (dt * speed))))
    total_time = float(times_arr[-1] - times_arr[0])
    if total_time <= 0.0:
        return fps_out, list(range(n_frames))

    # If fps is explicitly set, generate exactly that playback cadence and
    # map each render timestamp to the nearest simulation frame (duplicates allowed).
    # This preserves the requested fps and video duration ~= total_time / speed.
    if fps is not None:
        render_duration = total_time / speed
        n_out = max(1, int(round(render_duration * fps_out)))
        t0 = float(times_arr[0])
        t_render = t0 + (np.arange(n_out, dtype=float) / fps_out) * speed
        t_render = np.clip(t_render, t0, float(times_arr[-1]))

        idx_right = np.searchsorted(times_arr, t_render, side="left")
        idx_right = np.clip(idx_right, 0, n_frames - 1)
        idx_left = np.clip(idx_right - 1, 0, n_frames - 1)

        dt_left = np.abs(t_render - times_arr[idx_left])
        dt_right = np.abs(times_arr[idx_right] - t_render)
        indices = np.where(dt_right < dt_left, idx_right, idx_left).astype(int)
        return fps_out, indices.tolist()

    target = max(2, int(round((total_time / speed) * fps_out)))
    target = min(target, n_frames)
    indices = np.linspace(0, n_frames - 1, num=target, dtype=int)
    indices = np.unique(indices)
    if indices[-1] != n_frames - 1:
        indices = np.append(indices, n_frames - 1)
    return fps_out, indices.tolist()


def _color_cycle(n: int) -> list[str]:
    cmap = plt.get_cmap("tab10")
    return [matplotlib.colors.to_hex(cmap(i % cmap.N)) for i in range(n)]


def _frame_iter(frame_indices: Sequence[int], desc: str):
    """Wrap frame indices with a tqdm progress bar when useful."""
    if len(frame_indices) <= 1:
        return frame_indices
    return tqdm(frame_indices, total=len(frame_indices), desc=desc)


def _auto_bounds(arrays: Sequence[np.ndarray], margin: float = 0.1):
    """
    Compute global xyz bounds over any array that has one axis of length 3 (xyz).
    Works with shapes like (T, R, 3, N), (T, 3, N), or (N, 3).
    """
    global_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    global_max = -global_min
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        coord_axes = [i for i, s in enumerate(arr.shape) if s == 3]
        if not coord_axes:
            continue
        coord_axis = coord_axes[-1]
        reduce_axes = tuple(i for i in range(arr.ndim) if i != coord_axis)
        arr_min = np.nanmin(arr, axis=reduce_axes)
        arr_max = np.nanmax(arr, axis=reduce_axes)
        if arr_min.shape != (3,) or arr_max.shape != (3,):
            continue
        global_min = np.minimum(global_min, arr_min)
        global_max = np.maximum(global_max, arr_max)
    global_min -= margin
    global_max += margin
    return ((global_min[0], global_max[0]), (global_min[1], global_max[1]), (global_min[2], global_max[2]))


def _add_plane(ax, bounds, z=0.0, color="lightgray", alpha=0.25):
    (xmin, xmax), (ymin, ymax), _ = bounds
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
    zz = np.full_like(xx, z)
    return ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, linewidth=0)


def _mesh_to_poly(mesh_o3d, transform=None):
    import open3d as o3d

    mesh_copy = o3d.geometry.TriangleMesh(mesh_o3d)
    if transform is not None:
        mesh_copy.transform(transform)
    vertices = np.asarray(mesh_copy.vertices)
    triangles = np.asarray(mesh_copy.triangles)
    triangle_verts = [vertices[tri] for tri in triangles]
    return Poly3DCollection(triangle_verts, alpha=0.6, facecolor="tab:blue", edgecolor="k"), vertices, triangles


def _project_triangles(vertices, triangles, R, t, view: str):
    tri_vertices = (R @ vertices.T).T + t
    polys = []
    for tri in triangles:
        v = tri_vertices[tri]
        if view == "front":
            polys.append(v[:, [1, 2]])  # y-z
        elif view == "right":
            polys.append(v[:, [0, 2]])  # x-z
        elif view == "top":
            polys.append(v[:, [0, 1]])  # x-y
    return polys


# -----------------------------------------------------------------------------
# Rod-only rendering
# -----------------------------------------------------------------------------


def plot_rods_video(
    rod_positions: np.ndarray,
    video_path: str | Path = "rods.mp4",
    times: np.ndarray | None = None,
    fps: int | None = None,
    speed: float = 1.0,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    plane_z: float | Sequence[float] | None = 0.0,
    linewidth: float = 2.0,
    node_size: float = 8.0,
    colors: Sequence[str] | None = None,
):
    """Render rods (up to ~10) in 3D with optional ground plane.

    rod_positions shape: (n_frames, n_rods, 3, n_nodes) or (n_frames, 3, n_nodes).
    """

    rod_positions = np.asarray(rod_positions)
    if rod_positions.ndim == 3:
        rod_positions = rod_positions[:, None, ...]

    n_frames, n_rods = rod_positions.shape[:2]
    fps_used, frame_indices = _compute_render_indices(times, n_frames, fps, speed)
    if len(frame_indices) == 0:
        return

    if bounds is None:
        bounds = _auto_bounds([rod_positions])

    palette = _color_cycle(n_rods) if colors is None else list(colors)
    plane_zs: list[float] = []
    if plane_z is None:
        plane_zs = []
    elif isinstance(plane_z, (float, int)):
        plane_zs = [float(plane_z)]
    else:
        plane_zs = [float(p) for p in plane_z]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    plane_surfaces = [ _add_plane(ax, bounds, z=z) for z in plane_zs ]

    lines, scatters = [], []
    first = rod_positions[frame_indices[0]]
    for r in range(n_rods):
        (line,) = ax.plot(
            first[r, 0], first[r, 1], first[r, 2], "-", lw=linewidth, color=palette[r % len(palette)],
            label=f"Rod {r+1}",
        )
        scat = ax.scatter(first[r, 0], first[r, 1], first[r, 2], color="k", s=node_size)
        lines.append(line)
        scatters.append(scat)
    ax.legend(loc="upper right")

    writer = animation.writers["ffmpeg"](fps=fps_used)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(video_path), dpi=200):
        for idx in _frame_iter(frame_indices, "Rendering rods"):
            xyz_all = rod_positions[idx]
            for r, (line, scat) in enumerate(zip(lines, scatters)):
                xyz = xyz_all[r]
                line.set_data(xyz[0], xyz[1])
                line.set_3d_properties(xyz[2])
                scat._offsets3d = (xyz[0], xyz[1], xyz[2])
            writer.grab_frame()
    plt.close(fig)


def plot_rods_multiview(
    rod_positions: np.ndarray,
    video_path: str | Path = "rods_multiview.mp4",
    times: np.ndarray | None = None,
    fps: int | None = None,
    speed: float = 1.0,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    plane_z: float | Sequence[float] | None = 0.0,
    colors: Sequence[str] | None = None,
    linewidth: float = 2.0,
    node_size: float = 6.0,
    force_origins: np.ndarray | None = None,
    force_vectors: np.ndarray | None = None,
    force_scale: float = 1.0,
    force_color: str = "crimson",
    force_label: str = "Force",
    show_force_magnitude: bool = True,
):
    """Four-view (3D + front/right/top) animation for rods.

    Optional force overlay:
    - force_origins: (n_frames, 3)
    - force_vectors: (n_frames, 3)
    """

    rod_positions = np.asarray(rod_positions)
    if rod_positions.ndim == 3:
        rod_positions = rod_positions[:, None, ...]
    n_frames, n_rods = rod_positions.shape[:2]

    if (force_origins is None) != (force_vectors is None):
        raise ValueError("force_origins and force_vectors must be provided together.")
    force_overlay = force_origins is not None and force_vectors is not None
    if force_overlay:
        force_origins = np.asarray(force_origins, dtype=float)
        force_vectors = np.asarray(force_vectors, dtype=float)
        if force_origins.ndim == 3 and force_origins.shape[-1] == 1:
            force_origins = force_origins[..., 0]
        if force_vectors.ndim == 3 and force_vectors.shape[-1] == 1:
            force_vectors = force_vectors[..., 0]
        if force_origins.shape != (n_frames, 3):
            raise ValueError(
                f"force_origins must have shape (n_frames, 3), got {force_origins.shape}"
            )
        if force_vectors.shape != (n_frames, 3):
            raise ValueError(
                f"force_vectors must have shape (n_frames, 3), got {force_vectors.shape}"
            )

    fps_used, frame_indices = _compute_render_indices(times, n_frames, fps, speed)
    if len(frame_indices) == 0:
        return

    if bounds is None:
        bounds = _auto_bounds([rod_positions])

    palette = _color_cycle(n_rods) if colors is None else list(colors)
    plane_zs: list[float]
    if plane_z is None:
        plane_zs = []
    elif isinstance(plane_z, (float, int)):
        plane_zs = [float(plane_z)]
    else:
        plane_zs = [float(p) for p in plane_z]

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_front = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[1, 1])

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    def _setup_axes():
        ax3d.set_xlim(xmin, xmax); ax3d.set_ylim(ymin, ymax); ax3d.set_zlim(zmin, zmax)
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        for ax, lbls, xl, yl in [
            (ax_front, ("Y", "Z"), (ymin, ymax), (zmin, zmax)),
            (ax_right, ("X", "Z"), (xmin, xmax), (zmin, zmax)),
            (ax_top, ("X", "Y"), (xmin, xmax), (ymin, ymax)),
        ]:
            ax.set_xlabel(lbls[0]); ax.set_ylabel(lbls[1]); ax.set_aspect("equal")
            ax.set_xlim(*xl); ax.set_ylim(*yl)

    _setup_axes()

    plane_surfaces = [_add_plane(ax3d, bounds, z=z) for z in plane_zs]
    plane_front = [ax_front.axhline(z, color="lightgray", linestyle="--", linewidth=1) for z in plane_zs]
    plane_right = [ax_right.axhline(z, color="lightgray", linestyle="--", linewidth=1) for z in plane_zs]
    # Top view does not show horizontal planes.

    rod3d_lines, rod3d_scats = [], []
    rod_front_lines, rod_right_lines, rod_top_lines = [], [], []
    first = rod_positions[frame_indices[0]]
    for r in range(n_rods):
        color = palette[r % len(palette)]
        (l3d,) = ax3d.plot(first[r, 0], first[r, 1], first[r, 2], "-", lw=linewidth, color=color, label=f"Rod {r+1}")
        scat3d = ax3d.scatter(first[r, 0], first[r, 1], first[r, 2], color="k", s=node_size)
        (lf,) = ax_front.plot(first[r, 1], first[r, 2], "-", color=color)
        (lr,) = ax_right.plot(first[r, 0], first[r, 2], "-", color=color)
        (lt,) = ax_top.plot(first[r, 0], first[r, 1], "-", color=color)
        rod3d_lines.append(l3d); rod3d_scats.append(scat3d)
        rod_front_lines.append(lf); rod_right_lines.append(lr); rod_top_lines.append(lt)

    force_line_3d = None
    force_line_front = None
    force_line_right = None
    force_line_top = None
    force_text = None
    if force_overlay:
        assert force_origins is not None
        assert force_vectors is not None
        f0 = frame_indices[0]
        origin0 = force_origins[f0]
        tip0 = origin0 + force_scale * force_vectors[f0]
        (force_line_3d,) = ax3d.plot(
            [origin0[0], tip0[0]],
            [origin0[1], tip0[1]],
            [origin0[2], tip0[2]],
            color=force_color,
            linewidth=2.5,
            label=force_label,
        )
        (force_line_front,) = ax_front.plot(
            [origin0[1], tip0[1]],
            [origin0[2], tip0[2]],
            color=force_color,
            linewidth=2.2,
        )
        (force_line_right,) = ax_right.plot(
            [origin0[0], tip0[0]],
            [origin0[2], tip0[2]],
            color=force_color,
            linewidth=2.2,
        )
        (force_line_top,) = ax_top.plot(
            [origin0[0], tip0[0]],
            [origin0[1], tip0[1]],
            color=force_color,
            linewidth=2.2,
        )
        if show_force_magnitude:
            force_text = fig.text(
                0.5,
                0.985,
                f"{force_label} |F| = {np.linalg.norm(force_vectors[f0]):.3e}",
                ha="center",
                va="top",
                color=force_color,
            )

    ax3d.legend(loc="upper right")

    writer = animation.writers["ffmpeg"](fps=fps_used)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(video_path), dpi=180):
        for idx in _frame_iter(frame_indices, "Rendering rod 4-view"):
            xyz_all = rod_positions[idx]
            for r in range(n_rods):
                xyz = xyz_all[r]
                rod3d_lines[r].set_data(xyz[0], xyz[1]); rod3d_lines[r].set_3d_properties(xyz[2])
                rod3d_scats[r]._offsets3d = (xyz[0], xyz[1], xyz[2])
                rod_front_lines[r].set_xdata(xyz[1]); rod_front_lines[r].set_ydata(xyz[2])
                rod_right_lines[r].set_xdata(xyz[0]); rod_right_lines[r].set_ydata(xyz[2])
                rod_top_lines[r].set_xdata(xyz[0]); rod_top_lines[r].set_ydata(xyz[1])
            if force_overlay:
                assert force_origins is not None
                assert force_vectors is not None
                assert force_line_3d is not None
                assert force_line_front is not None
                assert force_line_right is not None
                assert force_line_top is not None
                origin = force_origins[idx]
                tip = origin + force_scale * force_vectors[idx]
                force_line_3d.set_data([origin[0], tip[0]], [origin[1], tip[1]])
                force_line_3d.set_3d_properties([origin[2], tip[2]])
                force_line_front.set_xdata([origin[1], tip[1]])
                force_line_front.set_ydata([origin[2], tip[2]])
                force_line_right.set_xdata([origin[0], tip[0]])
                force_line_right.set_ydata([origin[2], tip[2]])
                force_line_top.set_xdata([origin[0], tip[0]])
                force_line_top.set_ydata([origin[1], tip[1]])
                if force_text is not None:
                    force_text.set_text(
                        f"{force_label} |F| = {np.linalg.norm(force_vectors[idx]):.3e}"
                    )
            writer.grab_frame()
    plt.close(fig)


# -----------------------------------------------------------------------------
# Rod + mesh rendering (single and multi-view)
# -----------------------------------------------------------------------------


def plot_rods_with_mesh(
    mesh_dict: dict,
    rod_positions: np.ndarray,
    video_path: str | Path = "rods_mesh.mp4",
    times: np.ndarray | None = None,
    fps: int | None = None,
    speed: float = 1.0,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    plane_z: float | Sequence[float] | None = 0.0,
    colors: Sequence[str] | None = None,
    linewidth: float = 2.0,
    node_size: float = 8.0,
):
    """Single 3D view rods + moving mesh (TriangleMesh).

    mesh_dict fields: mesh (o3d TriangleMesh), position (T,3), director (T,3,3), time optional.
    rod_positions: (T, n_rods, 3, n_nodes) or (T, 3, n_nodes).
    """

    rod_positions = np.asarray(rod_positions)
    if rod_positions.ndim == 3:
        rod_positions = rod_positions[:, None, ...]
    n_frames, n_rods = rod_positions.shape[:2]

    times = mesh_dict.get("time") if times is None else times
    fps_used, frame_indices = _compute_render_indices(times, n_frames, fps, speed)
    if len(frame_indices) == 0:
        return

    mesh_o3d = mesh_dict["mesh"]
    mesh_pos = np.asarray(mesh_dict["position"])[:n_frames]
    mesh_dir = np.asarray(mesh_dict["director"])[:n_frames]

    base_poly, vertices_body, triangles = _mesh_to_poly(mesh_o3d)

    if bounds is None:
        mesh_min = vertices_body.min(0) + mesh_pos.min(0)
        mesh_max = vertices_body.max(0) + mesh_pos.max(0)
        bounds = _auto_bounds([rod_positions, np.stack([mesh_min, mesh_max], axis=0)])

    palette = _color_cycle(n_rods) if colors is None else list(colors)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    plane_zs: list[float] = []
    if plane_z is None:
        plane_zs = []
    elif isinstance(plane_z, (float, int)):
        plane_zs = [float(plane_z)]
    else:
        plane_zs = [float(p) for p in plane_z]
    plane_surfaces = [_add_plane(ax, bounds, z=z) for z in plane_zs]

    lines, scatters = [], []
    first = rod_positions[frame_indices[0]]
    for r in range(n_rods):
        (line,) = ax.plot(
            first[r, 0], first[r, 1], first[r, 2], "-", lw=linewidth, color=palette[r % len(palette)],
            label=f"Rod {r+1}",
        )
        scat = ax.scatter(first[r, 0], first[r, 1], first[r, 2], color="k", s=node_size)
        lines.append(line)
        scatters.append(scat)
    ax.legend(loc="upper right")

    T = np.eye(4)
    T[:3, :3] = mesh_dir[frame_indices[0]]
    T[:3, 3] = mesh_pos[frame_indices[0]]
    poly, _, _ = _mesh_to_poly(mesh_o3d, T)
    ax.add_collection3d(poly)

    writer = animation.writers["ffmpeg"](fps=fps_used)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(video_path), dpi=200):
        for idx in _frame_iter(frame_indices, "Rendering rods+mesh"):
            # Update mesh
            poly.remove()
            T[:3, :3] = mesh_dir[idx]
            T[:3, 3] = mesh_pos[idx]
            poly, _, _ = _mesh_to_poly(mesh_o3d, T)
            ax.add_collection3d(poly)

            # Update rods
            xyz_all = rod_positions[idx]
            for line, scat, xyz in zip(lines, scatters, xyz_all):
                line.set_data(xyz[0], xyz[1])
                line.set_3d_properties(xyz[2])
                scat._offsets3d = (xyz[0], xyz[1], xyz[2])

            writer.grab_frame()
    plt.close(fig)


def plot_rods_with_mesh_multiview(
    mesh_dict: dict,
    rod_positions: np.ndarray,
    video_path: str | Path = "rods_mesh_multiview.mp4",
    times: np.ndarray | None = None,
    fps: int | None = None,
    speed: float = 1.0,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    plane_z: float | Sequence[float] | None = 0.0,
    colors: Sequence[str] | None = None,
    linewidth: float = 2.0,
    node_size: float = 6.0,
):
    """Four-view (3D + front/right/top) animation with rods and mesh.

    Defaults mirror examples/MeshCase/multiview rendering.
    """

    rod_positions = np.asarray(rod_positions)
    if rod_positions.ndim == 3:
        rod_positions = rod_positions[:, None, ...]
    n_frames, n_rods = rod_positions.shape[:2]

    mesh_o3d = mesh_dict["mesh"]
    mesh_pos = np.asarray(mesh_dict["position"])[:n_frames]
    mesh_dir = np.asarray(mesh_dict["director"])[:n_frames]
    times = mesh_dict.get("time") if times is None else times

    base_poly, vertices_body, triangles = _mesh_to_poly(mesh_o3d)

    if bounds is None:
        mesh_min = vertices_body.min(0) + mesh_pos.min(0)
        mesh_max = vertices_body.max(0) + mesh_pos.max(0)
        bounds = _auto_bounds([rod_positions, np.stack([mesh_min, mesh_max], axis=0)])

    fps_used, frame_indices = _compute_render_indices(times, n_frames, fps, speed)
    if len(frame_indices) == 0:
        return

    palette = _color_cycle(n_rods) if colors is None else list(colors)
    plane_zs: list[float]
    if plane_z is None:
        plane_zs = []
    elif isinstance(plane_z, (float, int)):
        plane_zs = [float(plane_z)]
    else:
        plane_zs = [float(p) for p in plane_z]

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_front = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[1, 1])

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    def _setup_axes():
        ax3d.set_xlim(xmin, xmax); ax3d.set_ylim(ymin, ymax); ax3d.set_zlim(zmin, zmax)
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        for ax, lbls, xl, yl in [
            (ax_front, ("Y", "Z"), (ymin, ymax), (zmin, zmax)),
            (ax_right, ("X", "Z"), (xmin, xmax), (zmin, zmax)),
            (ax_top, ("X", "Y"), (xmin, xmax), (ymin, ymax)),
        ]:
            ax.set_xlabel(lbls[0]); ax.set_ylabel(lbls[1]); ax.set_aspect("equal")
            ax.set_xlim(*xl); ax.set_ylim(*yl)

    _setup_axes()

    # Initial mesh
    T = np.eye(4)
    T[:3, :3] = mesh_dir[frame_indices[0]]
    T[:3, 3] = mesh_pos[frame_indices[0]]
    poly3d, _, _ = _mesh_to_poly(mesh_o3d, T)
    ax3d.add_collection3d(poly3d)

    pf = pr = pt = None
    def _draw_projections(R, t):
        polys_front = _project_triangles(vertices_body, triangles, R, t, "front")
        polys_right = _project_triangles(vertices_body, triangles, R, t, "right")
        polys_top = _project_triangles(vertices_body, triangles, R, t, "top")
        return (
            PolyCollection(polys_front, facecolor="tab:blue", alpha=0.5, edgecolor="k"),
            PolyCollection(polys_right, facecolor="tab:blue", alpha=0.5, edgecolor="k"),
            PolyCollection(polys_top, facecolor="tab:blue", alpha=0.5, edgecolor="k"),
        )

    R0, t0 = mesh_dir[frame_indices[0]], mesh_pos[frame_indices[0]]
    pf, pr, pt = _draw_projections(R0, t0)
    ax_front.add_collection(pf); ax_right.add_collection(pr); ax_top.add_collection(pt)

    # Rod artists
    rod3d, rod_front, rod_right, rod_top = [], [], [], []
    first = rod_positions[frame_indices[0]]
    for r in range(n_rods):
        color = palette[r % len(palette)]
        (l3d,) = ax3d.plot(first[r, 0], first[r, 1], first[r, 2], "-", lw=linewidth, color=color, label=f"Rod {r+1}")
        scat3d = ax3d.scatter(first[r, 0], first[r, 1], first[r, 2], color="k", s=node_size)
        (lf,) = ax_front.plot(first[r, 1], first[r, 2], "-", color=color)
        (lr,) = ax_right.plot(first[r, 0], first[r, 2], "-", color=color)
        (lt,) = ax_top.plot(first[r, 0], first[r, 1], "-", color=color)
        rod3d.append((l3d, scat3d)); rod_front.append(lf); rod_right.append(lr); rod_top.append(lt)
    ax3d.legend(loc="upper right")

    writer = animation.writers["ffmpeg"](fps=fps_used)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(video_path), dpi=180):
        for idx in _frame_iter(frame_indices, "Rendering 4-view"):
            R = mesh_dir[idx]; t = mesh_pos[idx]
            # Mesh update
            poly3d.remove(); pf.remove(); pr.remove(); pt.remove()
            T[:3, :3] = R; T[:3, 3] = t
            poly3d, _, _ = _mesh_to_poly(mesh_o3d, T)
            ax3d.add_collection3d(poly3d)
            pf, pr, pt = _draw_projections(R, t)
            ax_front.add_collection(pf); ax_right.add_collection(pr); ax_top.add_collection(pt)

            # Rods update
            xyz_all = rod_positions[idx]
            for r in range(n_rods):
                xyz = xyz_all[r]
                line3d, scat3d = rod3d[r]
                line3d.set_data(xyz[0], xyz[1]); line3d.set_3d_properties(xyz[2])
                scat3d._offsets3d = (xyz[0], xyz[1], xyz[2])
                rod_front[r].set_xdata(xyz[1]); rod_front[r].set_ydata(xyz[2])
                rod_right[r].set_xdata(xyz[0]); rod_right[r].set_ydata(xyz[2])
                rod_top[r].set_xdata(xyz[0]); rod_top[r].set_ydata(xyz[1])

            writer.grab_frame()
    plt.close(fig)
