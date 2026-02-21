"""Static knot relaxation from exported pose with self-contact, gravity, and 4-view rendering."""

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


class StaticKnotSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Contact,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass


class RodHistoryCallback(ea.CallBackBaseClass):
    """Collect rod states for rendering."""

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


class FixedEndPositionsBC(ea.ConstraintBase):
    """Fix both end node positions and end-element directors."""

    def __init__(
        self,
        position_start: np.ndarray,
        position_end: np.ndarray,
        director_start: np.ndarray,
        director_end: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.position_start = np.asarray(position_start, dtype=np.float64).reshape(3)
        self.position_end = np.asarray(position_end, dtype=np.float64).reshape(3)
        self.director_start = np.asarray(director_start, dtype=np.float64).reshape(3, 3)
        self.director_end = np.asarray(director_end, dtype=np.float64).reshape(3, 3)
        if self.constrained_position_idx.size != 2:
            raise ValueError(
                "FixedEndPositionsBC expects constrained_position_idx=(0, -1)."
            )
        if self.constrained_director_idx.size != 2:
            raise ValueError(
                "FixedEndPositionsBC expects constrained_director_idx=(0, -1)."
            )
        self.start_idx = int(self.constrained_position_idx[0])
        self.end_idx = int(self.constrained_position_idx[1])
        self.start_dir_idx = int(self.constrained_director_idx[0])
        self.end_dir_idx = int(self.constrained_director_idx[1])

    def constrain_values(self, system, time: np.float64) -> None:
        system.position_collection[..., self.start_idx] = self.position_start
        system.position_collection[..., self.end_idx] = self.position_end
        system.director_collection[..., self.start_dir_idx] = self.director_start
        system.director_collection[..., self.end_dir_idx] = self.director_end

    def constrain_rates(self, system, time: np.float64) -> None:
        system.velocity_collection[..., self.start_idx] = 0.0
        system.velocity_collection[..., self.end_idx] = 0.0
        system.omega_collection[..., self.start_dir_idx] = 0.0
        system.omega_collection[..., self.end_dir_idx] = 0.0


def _fmt_tag(value: float) -> str:
    return format(float(value), ".6g")


def _make_unique_tag(output_dir: Path, base_tag: str) -> str:
    """
    Return unique tag by appending _i when output files already exist.

    Checks state npz and both possible video outputs (mp4/gif).
    """
    candidate = base_tag
    i = 1
    while True:
        state_path = output_dir / f"{candidate}_state.npz"
        video_mp4 = output_dir / f"{candidate}_4view.mp4"
        video_gif = output_dir / f"{candidate}_4view.gif"
        if not (state_path.exists() or video_mp4.exists() or video_gif.exists()):
            return candidate
        candidate = f"{base_tag}_{i}"
        i += 1


def _upsample_pose(
    position: np.ndarray, directors: np.ndarray, target_n_elem: int
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample 20-element pose to a higher multiple via linear pos + blockwise directors."""

    source_n_elem = position.shape[1] - 1
    if target_n_elem == source_n_elem:
        return position, directors
    if target_n_elem < source_n_elem:
        raise ValueError(
            f"target_n_elem={target_n_elem} must be >= source_n_elem={source_n_elem}"
        )
    if target_n_elem % source_n_elem != 0:
        raise ValueError(
            "target_n_elem must be an integer multiple of source n_elem "
            f"({source_n_elem}), got {target_n_elem}"
        )

    factor = target_n_elem // source_n_elem

    up_pos = np.zeros((3, target_n_elem + 1), dtype=np.float64)
    node_id = 0
    for e in range(source_n_elem):
        p0 = position[:, e]
        p1 = position[:, e + 1]
        for j in range(factor):
            alpha = j / factor
            up_pos[:, node_id] = (1.0 - alpha) * p0 + alpha * p1
            node_id += 1
    up_pos[:, -1] = position[:, -1]

    up_dir = np.repeat(directors, factor, axis=2)
    return up_pos, up_dir


def _load_exported_pose(
    npz_path: Path, expected_length: float, target_n_elem: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    if "pos" not in data or "director" not in data:
        raise KeyError(f"{npz_path} must contain `pos` and `director` arrays.")

    pos = np.asarray(data["pos"], dtype=np.float64)
    director = np.asarray(data["director"], dtype=np.float64)

    if pos.ndim != 2:
        raise ValueError(f"`pos` must be 2D, got shape {pos.shape}")
    if pos.shape[0] == 3:
        position = pos
    elif pos.shape[1] == 3:
        position = pos.T
    else:
        raise ValueError(f"`pos` must have 3 coordinates, got shape {pos.shape}")

    source_n_elem = position.shape[1] - 1
    if source_n_elem < 2:
        raise ValueError(f"`pos` must define at least 2 elements, got {source_n_elem}")

    if director.shape == (source_n_elem, 3, 3):
        directors = np.transpose(director, (1, 2, 0))
    elif director.shape == (3, 3, source_n_elem):
        directors = director
    else:
        raise ValueError(
            "`director` must be (n_elem,3,3) or (3,3,n_elem), "
            f"got {director.shape} for n_elem={source_n_elem}"
        )

    # Keep user-requested rod origin at [0, 0, 0].
    position = position - position[:, [0]]

    if target_n_elem is not None and target_n_elem > source_n_elem:
        position, directors = _upsample_pose(position, directors, target_n_elem)

    n_elem = position.shape[1] - 1

    # Re-orthonormalize to satisfy strict CosseratRod factory checks and
    # enforce d3 == centerline tangent at each element.
    tangents = position[:, 1:] - position[:, :-1]
    tangents /= np.linalg.norm(tangents, axis=0, keepdims=True)
    fixed_directors = np.zeros_like(directors)
    for k in range(n_elem):
        d3 = tangents[:, k]

        d1_raw = directors[0, :, k]
        d1 = d1_raw - np.dot(d1_raw, d3) * d3
        d1_norm = np.linalg.norm(d1)
        if d1_norm < 1.0e-12:
            d1_raw = directors[1, :, k]
            d1 = d1_raw - np.dot(d1_raw, d3) * d3
            d1_norm = np.linalg.norm(d1)
        if d1_norm < 1.0e-12:
            # Fallback if both in-plane axes are degenerate.
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(ref, d3)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            d1 = ref - np.dot(ref, d3) * d3
            d1_norm = np.linalg.norm(d1)
        d1 /= d1_norm

        d2 = np.cross(d3, d1)
        d2 /= np.linalg.norm(d2)

        fixed_directors[0, :, k] = d1
        fixed_directors[1, :, k] = d2
        fixed_directors[2, :, k] = d3
    directors = fixed_directors

    centerline_length = float(
        np.sum(np.linalg.norm(position[:, 1:] - position[:, :-1], axis=0))
    )
    if not np.isclose(centerline_length, expected_length, rtol=1.0e-2, atol=1.0e-6):
        print(
            f"Warning: loaded centerline length={centerline_length:.6g} m, "
            f"expected {expected_length:.6g} m"
        )

    return position, directors


def simulate_static_knot(
    *,
    input_npz: Path | str = Path(__file__).resolve().parent / "rope_chain_export_2.npz",
    final_time: float = 1.0,
    dt: float = 1.0e-5,
    output_interval: float = 1.0 / 100.0,
    render_fps: int = 100,
    frame_stride: int = 1,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "static_knot",
    n_elem: int = 60,
    gravity: np.ndarray = np.array([0.0, 0.0, -9.80665]),
    self_contact_k: float = 2.0e2,
    self_contact_nu: float = 1.0,
    bounds_pad_factor: float = 0.1,
    bounds_pad_radius_mult: float = 3.0,
) -> dict[str, object]:
    sim = StaticKnotSimulator()

    base_length = 0.1
    # base_radius = 0.00875
    base_radius = 0.0005
    density = 1000.0
    youngs_modulus = 1.0e4
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])

    input_npz = Path(input_npz)
    position, directors = _load_exported_pose(
        input_npz, expected_length=base_length, target_n_elem=n_elem
    )
    n_elem = position.shape[1] - 1

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
    # Apply imported full-shape pose only at t=0 (initial condition).
    # During simulation, only endpoint BCs are enforced.
    rod.position_collection[...] = position
    rod.director_collection[...] = directors
    sim.append(rod)

    sim.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=0.5,
        time_step=dt,
    )
    sim.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=np.asarray(gravity))
    sim.detect_contact_between(rod, rod).using(
        ea.RodSelfContact, k=self_contact_k, nu=self_contact_nu
    )
    sim.constrain(rod).using(
        FixedEndPositionsBC,
        constrained_position_idx=(0, -1),
        constrained_director_idx=(0, -1),
    )

    history: dict[str, list] = ea.defaultdict(list)
    step_skip = max(1, int(np.round(output_interval / dt)))
    sim.collect_diagnostics(rod).using(
        RodHistoryCallback,
        step_skip=step_skip,
        callback_params=history,
    )

    sim.finalize()
    total_steps = int(np.ceil(final_time / dt))
    print("Total steps:", total_steps)
    ea.integrate(ea.PositionVerlet(), sim, final_time, total_steps)

    time_arr = np.asarray(history["time"])
    position_arr = np.asarray(history["position"])
    director_arr = np.asarray(history["director"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tagged_name = (
        f"{output_name}"
        f"_t{_fmt_tag(final_time)}"
        f"_dt{_fmt_tag(dt)}"
        f"_n{n_elem}"
        f"_r{_fmt_tag(base_radius)}"
        f"_y{_fmt_tag(youngs_modulus)}"
    )
    tagged_name = _make_unique_tag(output_dir, base_tagged_name)
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
        rod_radius=base_radius,
        rod_length=base_length,
        youngs_modulus=youngs_modulus,
        gravity=np.asarray(gravity),
        self_contact_k=self_contact_k,
        self_contact_nu=self_contact_nu,
        source_npz=str(input_npz),
    )

    xyz_min = np.min(position_arr, axis=(0, 2))
    xyz_max = np.max(position_arr, axis=(0, 2))
    span = xyz_max - xyz_min
    pad = np.maximum(bounds_pad_factor * span, bounds_pad_radius_mult * base_radius)
    bounds = tuple(
        (float(xyz_min[i] - pad[i]), float(xyz_max[i] + pad[i])) for i in range(3)
    )

    video_path = plot_rod_four_view_with_directors(
        position_arr,
        director_arr,
        time_arr,
        video_path=video_path,
        fps=render_fps,
        bounds=bounds,
        frame_stride=frame_stride,
        director_scale=0.8 * base_length / n_elem,
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


if __name__ == "__main__":
    results = simulate_static_knot()
    print(f"Saved state to: {results['state_path']}")
    print(f"Saved four-view video to: {results['video_path']}")
