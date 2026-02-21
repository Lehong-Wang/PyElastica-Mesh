"""Custom post-processing helpers for mytest experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Headless rendering mode.
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _pick_writer(video_path: Path, fps: int):
    available = set(animation.writers.list())
    if "ffmpeg" in available:
        return animation.writers["ffmpeg"](fps=fps), video_path
    if "pillow" in available:
        fallback = video_path.with_suffix(".gif")
        print(f"ffmpeg writer is unavailable; writing GIF instead: {fallback}")
        return animation.writers["pillow"](fps=fps), fallback
    raise RuntimeError("No supported movie writer found. Install ffmpeg or pillow.")


def _segment_line_data(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = starts.shape[1]
    x = np.empty(3 * n, dtype=np.float64)
    y = np.empty(3 * n, dtype=np.float64)
    z = np.empty(3 * n, dtype=np.float64)
    x[0::3] = starts[0]
    x[1::3] = ends[0]
    x[2::3] = np.nan
    y[0::3] = starts[1]
    y[1::3] = ends[1]
    y[2::3] = np.nan
    z[0::3] = starts[2]
    z[1::3] = ends[2]
    z[2::3] = np.nan
    return x, y, z


def plot_rod_four_view_with_directors(
    position_history: np.ndarray,
    director_history: np.ndarray,
    time_history: np.ndarray,
    video_path: str | Path = "rod_4view_directors.mp4",
    *,
    fps: int = 60,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    director_scale: float | None = None,
    director_stride: int = 1,
    frame_stride: int = 1,
    show_time_text: bool = False,
) -> Path:
    """
    Render one rod in four-view (3D + YZ/XZ/XY) with per-element directors.

    Parameters
    ----------
    position_history : ndarray
        Shape (n_frames, 3, n_nodes).
    director_history : ndarray
        Shape (n_frames, 3, 3, n_elem). director_history[:,i,:,k] is d_i for element k.
    time_history : ndarray
        Shape (n_frames,).
    """

    position_history = np.asarray(position_history)
    director_history = np.asarray(director_history)
    time_history = np.asarray(time_history)

    if position_history.ndim != 3:
        raise ValueError(
            f"position_history must be (n_frames,3,n_nodes), got {position_history.shape}"
        )
    if director_history.ndim != 4:
        raise ValueError(
            f"director_history must be (n_frames,3,3,n_elem), got {director_history.shape}"
        )

    n_frames, _, n_nodes = position_history.shape
    n_elem = n_nodes - 1
    expected_shape = (n_frames, 3, 3, n_elem)
    if director_history.shape != expected_shape:
        raise ValueError(
            f"director_history shape mismatch, expected {expected_shape}, got {director_history.shape}"
        )
    if time_history.shape[0] != n_frames:
        raise ValueError(
            f"time_history length mismatch, expected {n_frames}, got {time_history.shape[0]}"
        )

    if bounds is None:
        xyz_min = np.min(position_history, axis=(0, 2))
        xyz_max = np.max(position_history, axis=(0, 2))
        span = xyz_max - xyz_min
        pad = np.maximum(0.08 * span, 1.0e-4)
        bounds = tuple(
            (float(xyz_min[i] - pad[i]), float(xyz_max[i] + pad[i])) for i in range(3)
        )
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    if director_scale is None:
        elem_lengths = np.linalg.norm(
            position_history[0, :, 1:] - position_history[0, :, :-1], axis=0
        )
        director_scale = 0.5 * float(np.mean(elem_lengths))

    director_stride = max(1, int(director_stride))
    frame_stride = max(1, int(frame_stride))

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_front = fig.add_subplot(gs[0, 1])  # y-z
    ax_right = fig.add_subplot(gs[1, 0])  # x-z
    ax_top = fig.add_subplot(gs[1, 1])  # x-y

    ax3d.set_xlim(xmin, xmax)
    ax3d.set_ylim(ymin, ymax)
    ax3d.set_zlim(zmin, zmax)
    ax3d.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    for ax, labels, xlim, ylim in [
        (ax_front, ("Y", "Z"), (ymin, ymax), (zmin, zmax)),
        (ax_right, ("X", "Z"), (xmin, xmax), (zmin, zmax)),
        (ax_top, ("X", "Y"), (xmin, xmax), (ymin, ymax)),
    ]:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

    (rod3d,) = ax3d.plot([], [], [], color="k", lw=2.0, label="rod")
    (rod_front,) = ax_front.plot([], [], color="k", lw=2.0)
    (rod_right,) = ax_right.plot([], [], color="k", lw=2.0)
    (rod_top,) = ax_top.plot([], [], color="k", lw=2.0)

    colors = ("tab:red", "tab:green", "tab:blue")
    labels = ("d1", "d2", "d3")
    dir3d_lines, dir_front_lines, dir_right_lines, dir_top_lines = [], [], [], []
    for color, label in zip(colors, labels):
        (l3d,) = ax3d.plot([], [], [], color=color, lw=1.2, label=label)
        (lf,) = ax_front.plot([], [], color=color, lw=1.0)
        (lr,) = ax_right.plot([], [], color=color, lw=1.0)
        (lt,) = ax_top.plot([], [], color=color, lw=1.0)
        dir3d_lines.append(l3d)
        dir_front_lines.append(lf)
        dir_right_lines.append(lr)
        dir_top_lines.append(lt)

    title = fig.suptitle("") if show_time_text else None
    ax3d.legend(loc="upper right")

    out_video_path = Path(video_path)
    writer, out_video_path = _pick_writer(out_video_path, fps)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = list(range(0, n_frames, frame_stride))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    interrupted = False
    writer.setup(fig, str(out_video_path), dpi=180)
    try:
        for frame_idx in tqdm(
            frame_indices,
            total=len(frame_indices),
            desc="Rendering rod 4-view (directors)",
        ):
            xyz = position_history[frame_idx]
            rod3d.set_data(xyz[0], xyz[1])
            rod3d.set_3d_properties(xyz[2])
            rod_front.set_data(xyz[1], xyz[2])
            rod_right.set_data(xyz[0], xyz[2])
            rod_top.set_data(xyz[0], xyz[1])

            centers = 0.5 * (xyz[:, :-1] + xyz[:, 1:])
            centers = centers[:, ::director_stride]

            for axis in range(3):
                dirs = director_history[frame_idx, axis, :, ::director_stride]
                ends = centers + director_scale * dirs
                xs, ys, zs = _segment_line_data(centers, ends)
                dir3d_lines[axis].set_data(xs, ys)
                dir3d_lines[axis].set_3d_properties(zs)
                dir_front_lines[axis].set_data(ys, zs)
                dir_right_lines[axis].set_data(xs, zs)
                dir_top_lines[axis].set_data(xs, ys)

            if title is not None:
                title.set_text(f"t = {time_history[frame_idx]:.3f} s")
            writer.grab_frame()
    except KeyboardInterrupt:
        interrupted = True
        print("Rendering interrupted by user. Finalizing partial output...")
    finally:
        try:
            writer.finish()
        except Exception as exc:
            if interrupted:
                print(f"Skipping writer finalize error after interrupt: {exc}")
            else:
                raise
        plt.close(fig)

    return out_video_path
