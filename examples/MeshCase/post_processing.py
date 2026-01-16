"""
Post-processing utilities for mesh visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import matplotlib.animation as animation
from tqdm import tqdm

DEFAULT_FPS = 30

def _compute_render_indices(times, n_frames, fps, speed_factor):
    times_arr = np.asarray(times) if times is not None else None
    default_fps = DEFAULT_FPS if fps is None else fps
    if n_frames == 0:
        return default_fps, []

    if times_arr is None or len(times_arr) < 2:
        return default_fps, list(range(n_frames))

    times_arr = times_arr[:n_frames]
    total_time = float(times_arr[-1] - times_arr[0])
    dt = np.median(np.diff(times_arr))
    fps_out = fps if fps is not None else max(
        1, min(30, int(round(1.0 / (dt * speed_factor))))
    )
    if total_time <= 0.0:
        return fps_out, list(range(n_frames))

    target_frames = max(2, int(round((total_time / speed_factor) * fps_out)))
    target_frames = min(target_frames, n_frames)
    indices = np.linspace(0, n_frames - 1, num=target_frames, dtype=int)
    indices = np.unique(indices)
    if indices[-1] != n_frames - 1:
        indices = np.append(indices, n_frames - 1)
    return fps_out, indices.tolist()


def mesh_to_poly3d_collection(mesh_o3d, transform=None):
    """
    Convert Open3D mesh to matplotlib Poly3DCollection.
    """
    import open3d as o3d

    mesh_copy = o3d.geometry.TriangleMesh(mesh_o3d)
    if transform is not None:
        mesh_copy.transform(transform)

    vertices = np.asarray(mesh_copy.vertices)
    triangles = np.asarray(mesh_copy.triangles)

    triangle_verts = [vertices[tri] for tri in triangles]
    return Poly3DCollection(
        triangle_verts, alpha=0.6, facecolor="tab:blue", edgecolor="k"
    )


def plot_mesh_animation(
    mesh_data_dict,
    video_name="mesh_animation.mp4",
    fps=None,
    speed_factor: float = 1.0,
    xlim=(-2, 2),
    ylim=(-2, 2),
    zlim=(-2, 2),
    rod_positions=None,
):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    mesh_o3d = mesh_data_dict["mesh"]
    positions = mesh_data_dict["position"]
    directors = mesh_data_dict["director"]
    times = mesh_data_dict.get("time")

    fps_render, frame_indices = _compute_render_indices(
        times, len(positions), fps, speed_factor
    )
    if len(frame_indices) == 0:
        return
    start_idx = frame_indices[0]

    T = np.eye(4)
    T[:3, :3] = directors[start_idx]
    T[:3, 3] = positions[start_idx]
    poly = mesh_to_poly3d_collection(mesh_o3d, T)
    ax.add_collection3d(poly)

    rod_line = None
    if rod_positions is not None:
        rod_positions = [np.asarray(rp) for rp in rod_positions]
        rod_line, = ax.plot(
            rod_positions[start_idx][0],
            rod_positions[start_idx][1],
            rod_positions[start_idx][2],
            "r-",
        )

    writer = animation.writers["ffmpeg"](fps=fps_render)
    with writer.saving(fig, video_name, dpi=150):
        for idx in tqdm(frame_indices, total=len(frame_indices), desc="Rendering 3D"):
            pos = positions[idx]
            director = directors[idx]
            poly.remove()
            T[:3, :3] = director
            T[:3, 3] = pos
            poly = mesh_to_poly3d_collection(mesh_o3d, T)
            ax.add_collection3d(poly)
            if rod_line is not None:
                rod_line.set_xdata(rod_positions[idx][0])
                rod_line.set_ydata(rod_positions[idx][1])
                rod_line.set_3d_properties(rod_positions[idx][2])
            writer.grab_frame()

    plt.close(fig)


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


def plot_mesh_multiview_animation(
    mesh_data_dict,
    video_name="mesh_multiview.mp4",
    fps=None,
    speed_factor: float = 1.0,
    rod_positions=None,
    bounds=((-2, 2), (-2, 2), (-2, 2)),
):
    mesh_o3d = mesh_data_dict["mesh"]
    positions = mesh_data_dict["position"]
    directors = mesh_data_dict["director"]
    times = mesh_data_dict.get("time")
    vertices = np.asarray(mesh_o3d.vertices)
    triangles = np.asarray(mesh_o3d.triangles, dtype=np.int64)

    rod_positions = [np.asarray(rp) for rp in rod_positions] if rod_positions else None

    fps_render, frame_indices = _compute_render_indices(
        times, len(positions), fps, speed_factor
    )
    if len(frame_indices) == 0:
        return
    start_idx = frame_indices[0]

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_front = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[1, 1])

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    def _setup_axes():
        ax3d.set_xlim(xmin, xmax)
        ax3d.set_ylim(ymin, ymax)
        ax3d.set_zlim(zmin, zmax)
        for ax, lbls in [
            (ax_front, ("Y", "Z")),
            (ax_right, ("X", "Z")),
            (ax_top, ("X", "Y")),
        ]:
            ax.set_xlabel(lbls[0])
            ax.set_ylabel(lbls[1])
            ax.set_aspect("equal")
            ax.set_xlim(
                ymin if lbls[0] == "Y" else xmin,
                ymax if lbls[0] == "Y" else xmax,
            )
            ax.set_ylim(
                zmin if lbls[1] == "Z" else ymin,
                zmax if lbls[1] == "Z" else ymax,
            )

    _setup_axes()

    T = np.eye(4)
    T[:3, :3] = directors[start_idx]
    T[:3, 3] = positions[start_idx]
    poly3d = mesh_to_poly3d_collection(mesh_o3d, T)
    ax3d.add_collection3d(poly3d)

    def _draw_projections(R, t, rod_pts):
        polys_front = _project_triangles(vertices, triangles, R, t, "front")
        polys_right = _project_triangles(vertices, triangles, R, t, "right")
        polys_top = _project_triangles(vertices, triangles, R, t, "top")
        return (
            PolyCollection(polys_front, facecolor="tab:blue", alpha=0.5, edgecolor="k"),
            PolyCollection(polys_right, facecolor="tab:blue", alpha=0.5, edgecolor="k"),
            PolyCollection(polys_top, facecolor="tab:blue", alpha=0.5, edgecolor="k"),
        )

    R0 = directors[start_idx]
    t0 = positions[start_idx]
    pf0, pr0, pt0 = _draw_projections(
        R0, t0, rod_positions[start_idx] if rod_positions else None
    )
    ax_front.add_collection(pf0)
    ax_right.add_collection(pr0)
    ax_top.add_collection(pt0)

    if rod_positions is not None:
        rod3d, = ax3d.plot(
            rod_positions[start_idx][0],
            rod_positions[start_idx][1],
            rod_positions[start_idx][2],
            "r-",
        )
        rod_front, = ax_front.plot(
            rod_positions[start_idx][1], rod_positions[start_idx][2], "r-"
        )
        rod_right, = ax_right.plot(
            rod_positions[start_idx][0], rod_positions[start_idx][2], "r-"
        )
        rod_top, = ax_top.plot(
            rod_positions[start_idx][0], rod_positions[start_idx][1], "r-"
        )
    else:
        rod3d = rod_front = rod_right = rod_top = None

    writer = animation.writers["ffmpeg"](fps=fps_render)
    with writer.saving(fig, video_name, dpi=150):
        for idx in tqdm(
            frame_indices, total=len(frame_indices), desc="Rendering multi-view"
        ):
            pos = positions[idx]
            director = directors[idx]
            poly3d.remove()
            T[:3, :3] = director
            T[:3, 3] = pos
            poly3d = mesh_to_poly3d_collection(mesh_o3d, T)
            ax3d.add_collection3d(poly3d)

            for coll in [pf0, pr0, pt0]:
                coll.remove()
            pf0, pr0, pt0 = _draw_projections(
                director, pos, rod_positions[idx] if rod_positions else None
            )
            ax_front.add_collection(pf0)
            ax_right.add_collection(pr0)
            ax_top.add_collection(pt0)

            if rod_positions is not None:
                rp = rod_positions[idx]
                rod3d.set_xdata(rp[0]); rod3d.set_ydata(rp[1]); rod3d.set_3d_properties(rp[2])
                rod_front.set_xdata(rp[1]); rod_front.set_ydata(rp[2])
                rod_right.set_xdata(rp[0]); rod_right.set_ydata(rp[2])
                rod_top.set_xdata(rp[0]); rod_top.set_ydata(rp[1])

            writer.grab_frame()

    plt.close(fig)
