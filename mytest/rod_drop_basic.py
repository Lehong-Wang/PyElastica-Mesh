"""
Drop a 1 m Cosserat rod from z=1 m onto the plane z=0 under gravity,
record the state, and render a video.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure matplotlib can write its cache in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

# Make the repository importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

# Headless rendering must be configured before importing pyplot.
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import elastica as ea


def _compute_render_indices(
    times: np.ndarray | None, n_frames: int, fps: int | None, speed: float
) -> tuple[int, list[int]]:
    if n_frames == 0:
        return 30, []
    times_arr = None if times is None or len(times) == 0 else np.asarray(times)
    default_fps = 30 if fps is None else fps
    if times_arr is None or len(times_arr) < 2:
        return default_fps, list(range(n_frames))
    dt = float(np.median(np.diff(times_arr)))
    fps_out = fps if fps is not None else max(1, int(round(1.0 / (dt * speed))))
    total_time = float(times_arr[-1] - times_arr[0])
    if total_time <= 0.0:
        return fps_out, list(range(n_frames))
    target = max(2, int(round((total_time / speed) * fps_out)))
    target = min(target, n_frames)
    indices = np.linspace(0, n_frames - 1, num=target, dtype=int)
    indices = np.unique(indices)
    if indices[-1] != n_frames - 1:
        indices = np.append(indices, n_frames - 1)
    return fps_out, indices.tolist()


def _render_rod_video(
    positions: np.ndarray,
    times: np.ndarray,
    video_path: Path,
    speed: float = 1.0,
    fps: int | None = None,
) -> None:
    """Render rod centerline history to an mp4."""
    n_frames = positions.shape[0]
    fps_used, frame_indices = _compute_render_indices(times, n_frames, fps, speed)
    if len(frame_indices) == 0:
        return

    pos_subset = positions[frame_indices]
    finite_mask = np.isfinite(pos_subset)
    if not finite_mask.any():
        return
    xyz_min = np.nanmin(pos_subset, axis=(0, 2))
    xyz_max = np.nanmax(pos_subset, axis=(0, 2))
    for i in range(3):
        if not np.isfinite(xyz_min[i]) or not np.isfinite(xyz_max[i]):
            xyz_min[i], xyz_max[i] = -0.5, 0.5
        if abs(xyz_max[i] - xyz_min[i]) < 1e-6:
            xyz_min[i] -= 0.5
            xyz_max[i] += 0.5
    margin = 0.1
    xmin, ymin, zmin = xyz_min - margin
    xmax, ymax, zmax = xyz_max + margin

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 2),
        np.linspace(ymin, ymax, 2),
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color="lightgray", alpha=0.25, linewidth=0)

    line, = ax.plot(
        pos_subset[0, 0],
        pos_subset[0, 1],
        pos_subset[0, 2],
        "b-",
        lw=2,
    )

    writer = animation.writers["ffmpeg"](fps=fps_used)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(video_path), dpi=200):
        for idx in frame_indices:
            xyz = positions[idx]
            line.set_data(xyz[0], xyz[1])
            line.set_3d_properties(xyz[2])
            writer.grab_frame()

    plt.close(fig)


def run_rod_drop(
    final_time: float = 3.0,
    dt: float = 1.0e-5,
    n_elem: int = 20,
    base_length: float = 1.0,
    base_radius: float = 5.0e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 1e6,
    shear_modulus_ratio: float = 1.5,
    contact_k: float = 1e2,
    contact_nu: float = 4.0,
    damping_constant: float = 1e-2,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_drop",
    output_interval: float = 0.01,
    seed: int | None = None,
    render_speed: float = 1.0,
    render_fps: int | None = None,
) -> dict[str, object]:
    """
    Simulate a free-falling Cosserat rod hitting a plane.

    Returns a dictionary with recorded arrays and file paths.
    """
    # rng = np.random.default_rng(seed)
    # direction, normal = _random_orientation(rng)

    center = np.array([0.0, 0.0, 1.0])
    # start = center - 0.5 * base_length * direction
    start = np.array([0.0, 0.0, 0.1])
    direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    normal = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)

    class RodDropSim(ea.BaseSystemCollection, ea.Forcing, ea.Contact, ea.CallBacks, ea.Damping):
        """Simulation container for the rod drop."""

    simulator = RodDropSim()

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

    plane = ea.Plane(plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]))
    simulator.append(plane)



    simulator.add_forcing_to(rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
    )

    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

    simulator.detect_contact_between(rod, plane).using(
        ea.RodPlaneContact, k=contact_k, nu=contact_nu
    )

    collector: dict[str, object] = {}

    class RodDropCallback(ea.CallBackBaseClass):
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
            self.time.append(time)
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    step_skip = max(1, int(output_interval / dt))
    simulator.collect_diagnostics(rod).using(RodDropCallback, step_skip=step_skip)

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    cb: RodDropCallback = collector["cb"]  # type: ignore[assignment]
    time_arr = np.asarray(cb.time)
    pos_arr = np.asarray(cb.position)
    dir_arr = np.asarray(cb.director)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / f"{output_name}_state.npz"
    seed_val = np.int64(-1 if seed is None else int(seed))

    np.savez(
        state_path,
        time=time_arr,
        position=pos_arr,
        director=dir_arr,
        radius=rod.radius.copy(),
        lengths=rod.lengths.copy(),
        direction_init=direction,
        normal_init=normal,
        start=start,
        seed=seed_val,
        dt=dt,
        final_time=final_time,
    )

    video_path = output_dir / f"{output_name}.mp4"
    _render_rod_video(pos_arr, time_arr, video_path, speed=render_speed, fps=render_fps)

    return {
        "state_path": state_path,
        "video_path": video_path,
        "direction": direction,
        "normal": normal,
        "start": start,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr,
    }


if __name__ == "__main__":
    results = run_rod_drop(
        final_time=2.0, dt=1.0e-5,
        damping_constant=1e-2, 
        seed=45, 
        youngs_modulus=1e8, 
        contact_k=1e2, 
        contact_nu=4.0, 
        )
    print(
        f"Saved npz to {results['state_path']} and video to {results['video_path']}.\n"
        f"Initial tangent={results['direction']}, normal={results['normal']}."
    )
