"""
Batch runner for randomized hanging/sweeping rods.

Behavior (mirrors rod_drop_multi_batch_force.py style):
- Run 50 randomized simulations (seeded per-run from a master seed).
- For the FIRST THREE runs only:
    * save full state npz (time/position/director/dt/final_time)
    * render full 4-view video
- For ALL runs:
    * save ONLY the last recorded callback frame into ONE single npz file
    * record per-rod randomized Young's modulus (log-uniform [1e5, 1e10])
      and radii (choices {0.001, 0.003, 0.005})
    * record impulse metadata

The single batch npz matches the field naming used in rod_drop_multi_batch_force.py.
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

from render_scripts.rod_hang_sweep import run_hang_with_sweep  # type: ignore


def run_hang_with_sweep_batch(
    n_runs: int = 25,
    save_full_for_first_k: int = 3,
    master_seed: int = 123,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_hang_sweep",
    final_time: float = 2.0,
    dt: float = 2.0e-5,
    n_elem: int = 20,
    base_length: float = 1.0,
    density: float = 1_000.0,
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 5e-2,
    contact_k: float = 2e4,
    contact_nu: float = 5.0,
    total_rods: int = 4,
    output_interval: float = 0.01,
    render_speed: float = 1.0,
    render_fps: int | None = None,
    available_radii: Sequence[float] = (0.001, 0.003, 0.005),
    ym_low: float = 5e5,
    ym_high: float = 2e7,
) -> dict[str, object]:
    """
    Batch 50 randomized hanging/sweeping simulations; aggregate last frames and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(master_seed)

    seed_arr = rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=n_runs,
        dtype=np.uint32,
    ).astype(np.int64)

    radius_choices_arr = np.asarray(available_radii, dtype=float)
    if radius_choices_arr.ndim != 1 or radius_choices_arr.size == 0:
        raise ValueError("available_radii must be a non-empty 1D sequence of floats")

    # Pre-sample per-run, per-rod radii and YM to keep deterministic, even though run_hang_with_sweep also samples.
    # We pass per-run seed; we also store what was actually realized from the run output for fidelity.
    pos_last_list: list[np.ndarray] = []
    dir_last_list: list[np.ndarray] = []
    impulse_nodes_list: list[np.ndarray] = []
    impulse_vectors_list: list[np.ndarray] = []
    impulse_steps_list: list[np.ndarray] = []
    radius_list: list[np.ndarray] = []
    ym_list: list[np.ndarray] = []
    hanging_counts: list[int] = []
    sweep_counts: list[int] = []

    time_arr_ref: np.ndarray | None = None
    saved_full_states: list[Path] = []
    saved_full_videos: list[Path] = []

    for run_idx in range(n_runs):
        seed = int(seed_arr[run_idx])
        do_full = run_idx < save_full_for_first_k
        per_run_name = f"{output_name}_run{run_idx:03d}"

        res = run_hang_with_sweep(
            final_time=final_time,
            dt=dt,
            n_elem=n_elem,
            base_length=base_length,
            density=density,
            shear_modulus_ratio=shear_modulus_ratio,
            damping_constant=damping_constant,
            contact_k=contact_k,
            contact_nu=contact_nu,
            total_rods=total_rods,
            available_radii=tuple(available_radii),
            ym_low=ym_low,
            ym_high=ym_high,
            output_dir=output_dir,
            output_name=per_run_name,
            output_interval=output_interval,
            seed=seed,
            render_speed=render_speed,
            render_fps=render_fps,
            render=do_full,
            save_state=do_full,
        )

        time_arr = res["time"]
        pos_arr = res["position"]
        dir_arr = res["director"]
        radii = res["radii"]
        yms = res["youngs_modulus"]
        impulse_nodes = res["impulse_nodes"]
        impulse_vectors = res["impulse_vectors"]
        impulse_steps = res["impulse_steps"]

        if time_arr_ref is None:
            time_arr_ref = time_arr
        else:
            if len(time_arr_ref) != len(time_arr) or not np.allclose(time_arr_ref, time_arr):
                raise RuntimeError("time_arr differs across runs; check dt/final_time/output_interval consistency.")

        mid_idx = len(pos_arr) // 2
        pos_last_list.append(pos_arr[mid_idx])
        pos_last_list.append(pos_arr[-1])
        dir_last_list.append(dir_arr[mid_idx])
        dir_last_list.append(dir_arr[-1])
        impulse_nodes_list.append(impulse_nodes)
        impulse_vectors_list.append(impulse_vectors)
        impulse_steps_list.append(impulse_steps)
        radius_list.append(radii)
        ym_list.append(yms)
        hanging_counts.append(int(res["hanging_count"]))
        sweep_counts.append(int(res["sweep_count"]))

        if do_full:
            if res["state_path"] is not None:
                saved_full_states.append(Path(res["state_path"]))
            if res["video_path_four"] is not None:
                saved_full_videos.append(Path(res["video_path_four"]))

    pos_last_arr = np.stack(pos_last_list, axis=0)  # (n_runs*2, num_rods, 3, n_nodes); even=mid, odd=final
    dir_last_arr = np.stack(dir_last_list, axis=0)  # (n_runs*2, num_rods, 3, 3, n_elems)
    # impulse_nodes_arr = np.stack(impulse_nodes_list, axis=0)
    # impulse_vectors_arr = np.stack(impulse_vectors_list, axis=0)
    # impulse_steps_arr = np.stack(impulse_steps_list, axis=0)
    # radius_arr = np.stack(radius_list, axis=0)
    # ym_arr = np.stack(ym_list, axis=0)
    # hanging_counts_arr = np.asarray(hanging_counts, dtype=int)
    # sweep_counts_arr = np.asarray(sweep_counts, dtype=int)
    rep2 = lambda x: np.repeat(np.asarray(x), 2, axis=0)

    impulse_nodes_arr   = rep2(np.stack(impulse_nodes_list,   0))
    impulse_vectors_arr = rep2(np.stack(impulse_vectors_list, 0))
    impulse_steps_arr   = rep2(np.stack(impulse_steps_list,   0))
    radius_arr          = rep2(np.stack(radius_list,          0))
    ym_arr              = rep2(np.stack(ym_list,              0))
    hanging_counts_arr  = rep2(np.asarray(hanging_counts, int))
    sweep_counts_arr    = rep2(np.asarray(sweep_counts,   int))

    batch_npz_path = output_dir / f"{output_name}_all_runs_last_frames.npz"
    assert time_arr_ref is not None

    np.savez(
        batch_npz_path,
        time_arr=time_arr_ref,
        pos_arr=pos_last_arr,
        dir_arr=dir_last_arr,
        dt=dt,
        final_time=final_time,
        damping_constant=damping_constant,
        seed_arr=seed_arr,
        contact_k=contact_k,
        contact_nu=contact_nu,
        youngs_modulus_arr=ym_arr,
        radius_arr=radius_arr,
        radius_choices=radius_choices_arr,
        youngs_modulus_range=np.array([ym_low, ym_high], dtype=float),
        impulse_nodes_arr=impulse_nodes_arr,
        impulse_vectors_arr=impulse_vectors_arr,
        impulse_steps_arr=impulse_steps_arr,
        hanging_count_arr=hanging_counts_arr,
        sweep_count_arr=sweep_counts_arr,
    )

    return {
        "batch_npz_path": batch_npz_path,
        "seed_arr": seed_arr,
        "pos_last_arr_shape": pos_last_arr.shape,
        "dir_last_arr_shape": dir_last_arr.shape,
        "youngs_modulus_arr_shape": ym_arr.shape,
        "radius_arr_shape": radius_arr.shape,
        "saved_full_states": saved_full_states,
        "saved_full_videos": saved_full_videos,
    }


if __name__ == "__main__":
    results = run_hang_with_sweep_batch(
        n_runs=25,
        save_full_for_first_k=3,
        master_seed=123,
        output_name="rod_hang_sweep",
        final_time=2.0,
        dt=2.0e-5,
        total_rods=8,
    )
    print(
        f"Saved batch npz to {results['batch_npz_path']} "
        f"(pos_last_arr_shape={results['pos_last_arr_shape']}, dir_last_arr_shape={results['dir_last_arr_shape']})."
    )
