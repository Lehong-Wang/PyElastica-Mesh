"""
Drop multiple Cosserat rods from staggered heights onto the plane z=0 under gravity,
record the state, and render a color-coded video.

Batch behavior (as requested):
- Run 50 randomized simulations (random seed per run).
- For the FIRST THREE runs only:
    * save full state npz (time/position/director/dt/final_time + per-rod radius & YM)
    * render full 4-view video
- For ALL 50 runs:
    * save ONLY the last recorded callback frame into ONE single npz file, with requested metadata
    * ALSO record per-rod randomized Young's modulus and radius in that single npz.

Added randomness:
- Per run, per rod:
    * Young's modulus sampled log-uniform in [1e5, 1e10]
    * Radius sampled from [0.001, 0.003, 0.005]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

# Ensure matplotlib can write its cache in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

# Make the repository importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

# Headless rendering must be configured before importing pyplot.
matplotlib.use("Agg")

import numpy as np

import elastica as ea
from elastica.external_forces import NoForces
from render_scripts import post_processing as pp


def _as_per_rod_array(x: float | Sequence[float] | np.ndarray, num_rods: int, name: str) -> np.ndarray:
    """
    Convert scalar or sequence to shape (num_rods,) float array.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full((num_rods,), float(arr), dtype=float)
    if arr.shape == (num_rods,):
        return arr.astype(float, copy=False)
    raise ValueError(f"{name} must be a scalar or shape ({num_rods},), got shape={arr.shape}")


def _sample_log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    """
    Sample log-uniform between [low, high].
    """
    if low <= 0 or high <= 0 or high < low:
        raise ValueError("log-uniform requires 0 < low <= high")
    u = rng.uniform(np.log10(low), np.log10(high), size=size)
    return (10.0 ** u).astype(float)


def run_multi_rod_drop(
    final_time: float = 3.0,
    dt: float = 1.0e-5,
    n_elem: int = 20,
    base_length: float = 1.0,
    base_radius: float | Sequence[float] | np.ndarray = 5.0e-3,  # scalar or per-rod
    density: float = 1_000.0,
    youngs_modulus: float | Sequence[float] | np.ndarray = 1e6,  # scalar or per-rod
    shear_modulus_ratio: float = 1.5,
    contact_k: float = 1e4,
    contact_nu: float = 4.0,
    friction_coefficient: float = 1.0,
    damping_constant: float = 1e-2,
    num_rods: int = 8,
    height_gap: float = 0.05,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_drop_multi",
    output_interval: float = 0.01,
    seed: int | None = None,
    render_speed: float = 1.0,
    render_fps: int | None = None,
    render: bool = True,
    save_state: bool = True,
    initial_impulse_mag: float = 0.05,
    initial_impulse_duration: float = 0.0,
) -> dict[str, object]:
    """
    Simulate multiple free-falling Cosserat rods hitting a plane.

    Returns a dictionary with recorded arrays and file paths.
    """

    class MultiRodDropSim(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        """Simulation container for multiple rod drops."""

    simulator = MultiRodDropSim()
    rng = np.random.default_rng(seed)

    # Expand per-rod params
    radius_per_rod = _as_per_rod_array(base_radius, num_rods, "base_radius")
    ym_per_rod = _as_per_rod_array(youngs_modulus, num_rods, "youngs_modulus")

    base_center = np.array([0.0, 0.0, 0.5])
    base_direction = np.array([1.0, 0.0, 0.2])

    rods: list[ea.CosseratRod] = []
    impulse_nodes: list[int] = []
    impulse_vectors: list[np.ndarray] = []
    impulse_steps: list[int] = []
    for i in range(num_rods):
        # Small random jitter in center position and rotation about z
        center_i = base_center + np.array([0.0, 0.0, i * height_gap])
        jitter_pos = (rng.random(3) - 0.5) * 0.5  # ±0.03 m
        jitter_pos[2] = (rng.random() - 0.5) * 0.04  # keep vertical jitter within ±0.02
        center_i = center_i + jitter_pos

        rot_angle = (rng.random() - 0.5) * 3  # ~ ±0.05 rad about z
        rot_mat = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle), 0.0],
                [np.sin(rot_angle), np.cos(rot_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dir_i = rot_mat @ base_direction
        dir_i = dir_i / (np.linalg.norm(dir_i) + 1e-12)
        # Build an orthonormal normal vector robustly (avoid near-parallel temp).
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(dir_i, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        norm_i = np.cross(dir_i, tmp)
        norm_i /= np.linalg.norm(norm_i) + 1e-12

        ym_i = float(ym_per_rod[i])
        r_i = float(radius_per_rod[i])

        start_i = center_i - 0.5 * base_length * dir_i

        rod = ea.CosseratRod.straight_rod(
            n_elements=n_elem,
            start=start_i,
            direction=dir_i,
            normal=norm_i,
            base_length=base_length,
            base_radius=r_i,
            density=density,
            youngs_modulus=ym_i,
            shear_modulus=ym_i / (2.0 * shear_modulus_ratio),
        )
        simulator.append(rod)
        rods.append(rod)

        # Choose a random node and impulse direction in the xy-plane (applied over duration).
        node_idx = int(rng.integers(0, rod.n_nodes))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
        impulse_vec = initial_impulse_mag * direction
        steps = max(0, int(np.ceil(initial_impulse_duration / dt)))
        impulse_nodes.append(node_idx)
        impulse_vectors.append(impulse_vec)
        impulse_steps.append(steps)

    plane = ea.Plane(plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]))
    simulator.append(plane)

    static_mu = np.array([friction_coefficient * 2.0] * 3)
    kinetic_mu = np.array([friction_coefficient] * 3)

    class InitialImpulseForce(NoForces):
        def __init__(self, node_index: int, impulse: np.ndarray, steps: int):
            super().__init__()
            self.node_index = node_index
            self.impulse = np.asarray(impulse, dtype=float)
            self.steps_remaining = steps

        def apply_forces(self, system, time=np.float64(0.0)):
            if self.steps_remaining <= 0:
                return
            system.external_forces[..., self.node_index] += self.impulse
            self.steps_remaining -= 1

    for rod, node_idx, impulse_vec, steps in zip(rods, impulse_nodes, impulse_vectors, impulse_steps):
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

        simulator.detect_contact_between(rod, rod).using(
            ea.RodSelfContact, k=1e4, nu=10
        )   

        simulator.add_forcing_to(rod).using(
            ea.AnisotropicFrictionalPlane,
            k=contact_k,
            nu=contact_nu,
            plane_origin=np.zeros(3),
            plane_normal=np.array([0.0, 0.0, 1.0]),
            slip_velocity_tol=1e-6,
            static_mu_array=static_mu,
            kinetic_mu_array=kinetic_mu,
        )

        if initial_impulse_mag > 0.0 and steps > 0:
            simulator.add_forcing_to(rod).using(
                InitialImpulseForce, node_index=node_idx, impulse=impulse_vec, steps=steps
            )

    # Pairwise rod-rod contact (remove this block if you truly want rod-rod disabled)
    for i in range(len(rods)):
        for j in range(i + 1, len(rods)):
            simulator.detect_contact_between(rods[i], rods[j]).using(
                ea.RodRodContact, k=contact_k, nu=contact_nu
            )

    collector: dict[str, ea.CallBackBaseClass] = {}

    class RodDropCallback(ea.CallBackBaseClass):
        def __init__(self, name: str, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.position: list[np.ndarray] = []
            self.director: list[np.ndarray] = []
            collector[name] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    step_skip = max(1, int(output_interval / dt))
    callback_names: list[str] = []
    for idx, rod in enumerate(rods):
        name = f"rod_{idx}"
        simulator.collect_diagnostics(rod).using(
            RodDropCallback, name=name, step_skip=step_skip
        )
        callback_names.append(name)

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    callbacks: list[RodDropCallback] = [collector[name] for name in callback_names]

    time_arr = np.asarray(callbacks[0].time)
    pos_arr = np.stack([np.asarray(cb.position) for cb in callbacks], axis=1)
    dir_arr = np.stack([np.asarray(cb.director) for cb in callbacks], axis=1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / f"{output_name}_state.npz"
    if save_state:
        np.savez(
            state_path,
            time=time_arr,
            position=pos_arr,
            director=dir_arr,
            dt=dt,
            final_time=final_time,
            radius_per_rod=radius_per_rod,
            youngs_modulus_per_rod=ym_per_rod,
            impulse_nodes=np.asarray(impulse_nodes, dtype=np.int64),
            impulse_vectors=np.asarray(impulse_vectors, dtype=float),
        )
    else:
        state_path = None

    video_path_four = output_dir / f"{output_name}_4view.mp4"
    colors = pp._color_cycle(num_rods)

    if render:
        pp.plot_rods_multiview(
            pos_arr,
            video_path=video_path_four,
            times=time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=0.0,
            colors=colors,
        )
    else:
        video_path_four = None

    return {
        "state_path": state_path,
        "video_path_four": video_path_four,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr,
        "colors": colors,
        "radius_per_rod": radius_per_rod,
        "youngs_modulus_per_rod": ym_per_rod,
        "impulse_nodes": np.asarray(impulse_nodes, dtype=np.int64),
        "impulse_vectors": np.asarray(impulse_vectors, dtype=float),
        "impulse_steps": np.asarray(impulse_steps, dtype=np.int64),
    }


def run_multi_rod_drop_batch(
    n_runs: int = 50,
    master_seed: int | None = None,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_drop_multi",
    # base sim params
    final_time: float = 3.0,
    dt: float = 2.0e-5,
    n_elem: int = 20,
    base_length: float = 1.0,
    density: float = 1_000.0,
    shear_modulus_ratio: float = 1.5,
    contact_k: float = 1e6,
    contact_nu: float = 20.0,
    friction_coefficient: float = 1.5,
    damping_constant: float = 1e-2,
    num_rods: int = 4,
    height_gap: float = 0.2,
    output_interval: float = 0.01,
    render_speed: float = 1.0,
    render_fps: int | None = None,
    save_full_for_first_k: int = 3,
    # randomness controls
    radius_choices: Sequence[float] = (0.001, 0.003, 0.005),
    youngs_modulus_low: float = 1e9,
    youngs_modulus_high: float = 1e10,
    initial_impulse_mag: float = 0.05,
    initial_impulse_duration: float = 0.1,
) -> dict[str, object]:
    """
    Batch runner:

    - Runs n_runs simulations with random seeds (seed_arr).
    - For first save_full_for_first_k runs:
        saves full state npz + renders full mp4 video.
    - For ALL runs:
        saves only LAST recorded callback frame into one single npz:
            time_arr (shared), pos_arr (last frames for all runs),
            dir_arr (last frames for all runs), plus requested metadata.
    - Additionally records per-run, per-rod:
        radius_arr (chosen from radius_choices)
        youngs_modulus_arr (log-uniform in [youngs_modulus_low, youngs_modulus_high])
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(master_seed)

    # seeds for positional/orientation jitter per run
    seed_arr = rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=n_runs,
        dtype=np.uint32,
    ).astype(np.int64)

    radius_choices_arr = np.asarray(radius_choices, dtype=float)
    if radius_choices_arr.ndim != 1 or radius_choices_arr.size == 0:
        raise ValueError("radius_choices must be a non-empty 1D sequence of floats")

    # Per-run, per-rod randomized material/geometry
    # radius: categorical
    radius_arr = rng.choice(radius_choices_arr, size=(n_runs, num_rods), replace=True).astype(float)
    # Young's modulus: log-uniform
    youngs_modulus_arr = np.stack(
        [_sample_log_uniform(rng, youngs_modulus_low, youngs_modulus_high, size=num_rods) for _ in range(n_runs)],
        axis=0,
    )  # (n_runs, num_rods)
    print(youngs_modulus_arr)

    pos_last_list: list[np.ndarray] = []
    dir_last_list: list[np.ndarray] = []
    impulse_nodes_list: list[np.ndarray] = []
    impulse_vectors_list: list[np.ndarray] = []
    impulse_steps_list: list[np.ndarray] = []
    time_arr_ref: np.ndarray | None = None

    saved_full_states: list[Path] = []
    saved_full_videos: list[Path] = []

    for run_idx in range(n_runs):
        seed = int(seed_arr[run_idx])
        do_full = run_idx < save_full_for_first_k
        per_run_name = f"{output_name}_run{run_idx:03d}"

        res = run_multi_rod_drop(
            final_time=final_time,
            dt=dt,
            n_elem=n_elem,
            base_length=base_length,
            base_radius=radius_arr[run_idx],           # per-rod radius
            density=density,
            youngs_modulus=youngs_modulus_arr[run_idx],# per-rod YM
            shear_modulus_ratio=shear_modulus_ratio,
            contact_k=contact_k,
            contact_nu=contact_nu,
            friction_coefficient=friction_coefficient,
            damping_constant=damping_constant,
            initial_impulse_mag=initial_impulse_mag,
            initial_impulse_duration=initial_impulse_duration,
            num_rods=num_rods,
            height_gap=height_gap,
            output_dir=output_dir,
            output_name=per_run_name,
            output_interval=output_interval,
            seed=seed,
            render_speed=render_speed,
            render_fps=render_fps,
            render=do_full,        # ONLY first 3
            save_state=do_full,    # ONLY first 3
        )

        time_arr = res["time"]
        pos_arr = res["position"]
        dir_arr = res["director"]
        impulse_nodes = res["impulse_nodes"]
        impulse_vectors = res["impulse_vectors"]
        impulse_steps = res["impulse_steps"]

        if time_arr_ref is None:
            time_arr_ref = time_arr
        else:
            if len(time_arr_ref) != len(time_arr) or not np.allclose(time_arr_ref, time_arr):
                raise RuntimeError(
                    "time_arr differs across runs; check dt/final_time/output_interval consistency."
                )

        # Save the final RECORDED callback frame for this run (always)
        pos_last_list.append(pos_arr[-1])  # (num_rods, 3, n_nodes)
        dir_last_list.append(dir_arr[-1])  # (num_rods, 3, 3, n_elems)
        impulse_nodes_list.append(impulse_nodes)
        impulse_vectors_list.append(impulse_vectors)
        impulse_steps_list.append(impulse_steps)

        if do_full:
            if res["state_path"] is not None:
                saved_full_states.append(Path(res["state_path"]))
            if res["video_path_four"] is not None:
                saved_full_videos.append(Path(res["video_path_four"]))

    pos_last_arr = np.stack(pos_last_list, axis=0)  # (n_runs, num_rods, 3, n_nodes)
    dir_last_arr = np.stack(dir_last_list, axis=0)  # (n_runs, num_rods, 3, 3, n_elems)
    impulse_nodes_arr = np.stack(impulse_nodes_list, axis=0)  # (n_runs, num_rods)
    impulse_vectors_arr = np.stack(impulse_vectors_list, axis=0)  # (n_runs, num_rods, 3)
    impulse_steps_arr = np.stack(impulse_steps_list, axis=0)  # (n_runs, num_rods)

    # Single file containing last-frame state for all runs (all 50)
    batch_npz_path = output_dir / f"{output_name}_all_runs_last_frames.npz"
    assert time_arr_ref is not None

    # Save exactly the fields you asked for originally, plus the new per-rod randomized arrays.
    np.savez(
        batch_npz_path,
        # requested names
        time_arr=time_arr_ref,
        pos_arr=pos_last_arr,
        dir_arr=dir_last_arr,
        dt=dt,
        final_time=final_time,
        damping_constant=damping_constant,
        seed_arr=seed_arr,
        contact_k=contact_k,
        contact_nu=contact_nu,
        friction_coefficient=friction_coefficient,
        # updated material/geometry recording
        youngs_modulus_arr=youngs_modulus_arr,   # (n_runs, num_rods)
        radius_arr=radius_arr,                   # (n_runs, num_rods)
        radius_choices=radius_choices_arr,
        youngs_modulus_range=np.array([youngs_modulus_low, youngs_modulus_high], dtype=float),
        impulse_nodes_arr=impulse_nodes_arr,
        impulse_vectors_arr=impulse_vectors_arr,
        impulse_steps_arr=impulse_steps_arr,
        initial_impulse_duration=initial_impulse_duration,
    )

    return {
        "batch_npz_path": batch_npz_path,
        "seed_arr": seed_arr,
        "pos_last_arr_shape": pos_last_arr.shape,
        "dir_last_arr_shape": dir_last_arr.shape,
        "youngs_modulus_arr_shape": youngs_modulus_arr.shape,
        "radius_arr_shape": radius_arr.shape,
        "saved_full_states": saved_full_states,
        "saved_full_videos": saved_full_videos,
    }


if __name__ == "__main__":
    results = run_multi_rod_drop_batch(
        n_runs=50,
        master_seed=123,  # controls seed_arr + YM/radius sampling
        output_name="rod_drop_multi_8",
        final_time=1.5,
        dt=1.0e-5,
        damping_constant=5e-2,
        contact_k=1e6,
        contact_nu=20.0,
        friction_coefficient=1.5,
        height_gap=0.2,
        output_interval=0.05,
        num_rods=8,
        save_full_for_first_k=3,  # first 3 => full npz + full video
        radius_choices=(0.001, 0.003, 0.005),
        youngs_modulus_low=5e5,
        youngs_modulus_high=2e7,
        initial_impulse_mag=0.05,
        initial_impulse_duration=0.02
    )

    print(f"[OK] Saved LAST-FRAMES npz (all runs) to: {results['batch_npz_path']}")
    print(f"[OK] Seeds (first 5): {results['seed_arr'][:5]} ... (total {len(results['seed_arr'])})")
    print(f"[OK] pos_arr(last frames) shape: {results['pos_last_arr_shape']}")
    print(f"[OK] dir_arr(last frames) shape: {results['dir_last_arr_shape']}")
    print(f"[OK] youngs_modulus_arr shape: {results['youngs_modulus_arr_shape']}")
    print(f"[OK] radius_arr shape: {results['radius_arr_shape']}")

    if results["saved_full_states"]:
        print("[OK] Full state npz saved for first runs:")
        for p in results["saved_full_states"]:
            print("  -", p)

    if results["saved_full_videos"]:
        print("[OK] Full 4-view videos saved for first runs:")
        for p in results["saved_full_videos"]:
            print("  -", p)
