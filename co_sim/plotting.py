"""Plotting and rendering helpers for co-simulation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from render_scripts import post_processing as pp

Bounds3D = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


def _finalize_and_save(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_force_vector_with_magnitude(
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
    _finalize_and_save(fig, output_path)


def plot_force_vs_time(
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
    _finalize_and_save(fig, output_path)


def _sanitize_bounds(bounds: Bounds3D) -> Bounds3D:
    x_bounds = [float(bounds[0][0]), float(bounds[0][1])]
    y_bounds = [float(bounds[1][0]), float(bounds[1][1])]
    z_bounds = [float(bounds[2][0]), float(bounds[2][1])]
    for axis_bounds in (x_bounds, y_bounds, z_bounds):
        if not np.isfinite(axis_bounds[0]) or not np.isfinite(axis_bounds[1]):
            raise ValueError(f"plot_bounds must be finite, got {bounds}.")
        if axis_bounds[0] > axis_bounds[1]:
            axis_bounds[0], axis_bounds[1] = axis_bounds[1], axis_bounds[0]
        if axis_bounds[0] == axis_bounds[1]:
            axis_bounds[0] -= 0.5
            axis_bounds[1] += 0.5
    return (
        (x_bounds[0], x_bounds[1]),
        (y_bounds[0], y_bounds[1]),
        (z_bounds[0], z_bounds[1]),
    )


def _equalized_bounds(bounds: Bounds3D) -> Bounds3D:
    """Expand bounds to a cube around the same center for equal axis unit length."""
    centers = np.array(
        [0.5 * (bounds[0][0] + bounds[0][1]),
         0.5 * (bounds[1][0] + bounds[1][1]),
         0.5 * (bounds[2][0] + bounds[2][1])],
        dtype=float,
    )
    spans = np.array(
        [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], bounds[2][1] - bounds[2][0]],
        dtype=float,
    )
    max_span = float(np.max(spans))
    if not np.isfinite(max_span) or max_span <= 0.0:
        max_span = 1.0
    half = 0.5 * max_span
    return (
        (float(centers[0] - half), float(centers[0] + half)),
        (float(centers[1] - half), float(centers[1] + half)),
        (float(centers[2] - half), float(centers[2] + half)),
    )


def _motion_bounds_from_all_rods(rod_positions: np.ndarray) -> Bounds3D:
    """
    Build xyz bounds from motion range across all rods and all frames.

    Expected shape: (n_frames, n_rods, 3, n_nodes).
    """
    mins = np.nanmin(rod_positions, axis=(0, 1, 3))
    maxs = np.nanmax(rod_positions, axis=(0, 1, 3))
    spans = maxs - mins
    max_span = float(np.max(spans))
    pad = max(0.05 * max_span, 1.0e-4)
    bounds = (
        (float(mins[0] - pad), float(maxs[0] + pad)),
        (float(mins[1] - pad), float(maxs[1] + pad)),
        (float(mins[2] - pad), float(maxs[2] + pad)),
    )
    return _equalized_bounds(bounds)


def render_multiview_video(
    sampled_rod_pos: np.ndarray,
    sampled_frame_pos: np.ndarray,
    sampled_frame_dir: np.ndarray,
    sampled_time: np.ndarray,
    sampled_mean_force: np.ndarray,
    video_path: Path,
    render_fps: int | None,
    render_speed: float,
    force_vector_scale: float,
    plot_bounds: Bounds3D | None = None,
) -> None:
    frame_span = 0.05
    frame_line = sampled_frame_pos + frame_span * np.stack(
        [-sampled_frame_dir[:, 2, :, 0], sampled_frame_dir[:, 2, :, 0]],
        axis=-1,
    )
    n_nodes = sampled_rod_pos.shape[2]
    pad_len = max(0, n_nodes - frame_line.shape[2])
    frame_line = np.pad(frame_line, ((0, 0), (0, 0), (0, pad_len)), constant_values=np.nan)

    rod_for_plot = np.concatenate([sampled_rod_pos[:, None, ...], frame_line[:, None, ...]], axis=1)
    bounds = (
        _motion_bounds_from_all_rods(rod_for_plot)
        if plot_bounds is None
        else _equalized_bounds(_sanitize_bounds(plot_bounds))
    )

    pp.plot_rods_multiview(
        rod_for_plot,
        video_path=video_path,
        times=sampled_time,
        fps=render_fps,
        speed=render_speed,
        plane_z=None,
        colors=["#ff7f0e", "#1f77b4"],
        bounds=bounds,
        force_origins=sampled_frame_pos[:, :, 0],
        force_vectors=sampled_mean_force,
        force_scale=force_vector_scale,
        force_color="#d62728",
        force_label="Mean Rod->Frame Force",
        show_force_magnitude=True,
    )
