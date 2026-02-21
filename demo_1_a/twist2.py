"""Helical-buckling simulation with four-view rendering (twist2)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Ensure matplotlib can write cache files in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import elastica as ea
from mytest.postprocessing import plot_rod_four_view_with_directors


class HelicalBucklingFourViewSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Damping, ea.Forcing, ea.CallBacks
):
    pass


class RodHistoryCallback(ea.CallBackBaseClass):
    """Collect rod states for four-view rendering."""

    def __init__(self, step_skip: int, callback_params: dict[str, list]):
        super().__init__()
        self.step_skip = int(step_skip)
        self.callback_params = callback_params

    def make_callback(self, system, time: np.float64, current_step: int) -> None:
        if current_step % self.step_skip:
            return
        self.callback_params["time"].append(float(time))
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["director"].append(system.director_collection.copy())


def _fmt_tag(value: float) -> str:
    """Compact numeric formatter for filename tags."""
    return format(float(value), ".6g")


def simulate_buckling(
    *,
    n_elem: int = 1000,
    final_time: float = 1000.0,
    twisting_time: float = 80.0,
    slack: float = 0.3,
    angular_rate: float = 0.5,
    output_interval: float = 1,
    render_fps: int | None = 1,
    frame_stride: int = 1,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "twist2",
) -> dict[str, object]:
    """
    Simulate helical buckling, following examples/HelicalBucklingCase/helicalbuckling.py.
    """
    simulator = HelicalBucklingFourViewSimulator()

    # Setup parameters (same as examples/HelicalBucklingCase/helicalbuckling.py)
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 10.0
    base_radius = 0.35
    base_area = np.pi * base_radius**2
    density = 1.0 / base_area
    nu = 0.01 / density / base_area
    youngs_modulus = 1.0e5
    poisson_ratio = 9.0
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    shear_matrix = np.repeat(
        shear_modulus * np.identity((3))[:, :, np.newaxis], n_elem, axis=2
    )
    temp_bend_matrix = np.zeros((3, 3))
    np.fill_diagonal(temp_bend_matrix, [1.345, 1.345, 0.789])
    bend_matrix = np.repeat(temp_bend_matrix[:, :, np.newaxis], n_elem - 1, axis=2)

    rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    rod.shear_matrix = shear_matrix
    rod.bend_matrix = bend_matrix
    simulator.append(rod)

    dl = base_length / n_elem
    # dt = 1.0e-3 * dl
    dt = 1.0e-2 * dl
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=nu,
        time_step=dt,
    )
    # HelicalBucklingBC is parameterized by number_of_rotations.
    number_of_rotations = angular_rate * twisting_time / np.pi
    simulator.constrain(rod).using(
        ea.HelicalBucklingBC,
        constrained_position_idx=(0, -1),
        constrained_director_idx=(0, -1),
        twisting_time=twisting_time,
        slack=slack,
        number_of_rotations=number_of_rotations,
    )

    history: dict[str, list] = ea.defaultdict(list)
    step_skip = max(1, int(np.round(output_interval / dt)))
    simulator.collect_diagnostics(rod).using(
        RodHistoryCallback,
        step_skip=step_skip,
        callback_params=history,
    )

    simulator.finalize()
    rod.velocity_collection[..., int(n_elem / 2)] += np.array([0.0, 1.0e-6, 0.0])

    total_steps = int(final_time / dt)
    print("Total steps:", total_steps)
    ea.integrate(ea.PositionVerlet(), simulator, final_time, total_steps)

    time_arr = np.asarray(history["time"])
    position_arr = np.asarray(history["position"])
    director_arr = np.asarray(history["director"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tagged_name = (
        f"{output_name}"
        f"_t{_fmt_tag(final_time)}"
        f"_dt{_fmt_tag(dt)}"
        f"_n{n_elem}"
        f"_v{_fmt_tag(angular_rate)}"
        f"_r{_fmt_tag(base_radius)}"
        f"_y{_fmt_tag(youngs_modulus)}"
    )
    state_path = output_dir / f"{tagged_name}_state.npz"
    video_path = output_dir / f"{tagged_name}_4view.mp4"

    np.savez(
        state_path,
        time=time_arr,
        position=position_arr,
        director=director_arr,
        dt=dt,
        final_time=final_time,
        n_elem=n_elem,
        twisting_time=twisting_time,
        slack=slack,
        angular_rate=angular_rate,
        rod_radius=base_radius,
        youngs_modulus=youngs_modulus,
        number_of_rotations=number_of_rotations,
    )

    xyz_min = np.min(position_arr, axis=(0, 2))
    xyz_max = np.max(position_arr, axis=(0, 2))
    span = xyz_max - xyz_min
    pad = np.maximum(0.15 * span, 6.0 * base_radius)
    bounds = tuple(
        (float(xyz_min[i] - pad[i]), float(xyz_max[i] + pad[i])) for i in range(3)
    )

    video_path = plot_rod_four_view_with_directors(
        position_arr,
        director_arr,
        time_arr,
        video_path,
        fps=render_fps if render_fps is not None else 60,
        bounds=bounds,
        frame_stride=frame_stride,
        director_stride=1,
    )

    return {
        "rod": rod,
        "time": time_arr,
        "position": position_arr,
        "director": director_arr,
        "state_path": state_path,
        "video_path": video_path,
    }


def simulate_buckling_reference(
    *,
    n_elem: int = 100,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "twist2_reference",
) -> dict[str, object]:
    """
    Full-scale parameters from examples/HelicalBucklingCase/helicalbuckling.py.
    """
    return simulate_buckling(
        n_elem=n_elem,
        final_time=10500.0,
        twisting_time=500.0,
        slack=3.0,
        angular_rate=27.0 * np.pi / 500.0,
        output_interval=10.0,
        render_fps=60,
        frame_stride=1,
        output_dir=output_dir,
        output_name=output_name,
    )


if __name__ == "__main__":
    results = simulate_buckling()
    print(f"Saved state to: {results['state_path']}")
    print(f"Saved four-view video to: {results['video_path']}")
