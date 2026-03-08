"""
Batch runner for data_center/generate_data.py.

This script mirrors the seed-driven batch pattern used by
render_scripts/rod_drop_multi_batch.py, but keeps each run's state output
format unchanged by calling generate_cable() directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_center.generate_data import generate_cable


def run_generate_data_batch(
    n_runs: int = 10,
    master_seed: int | None = None,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "generate_data_batch",
    connector_ids: tuple[int, ...] = (2, 3, 4),
    **generate_kwargs: Any,
) -> dict[str, object]:
    """
    Run generate_cable() batch for connector IDs.

    Notes
    -----
    - Loops connector_id in connector_ids (default: 2,3,4).
    - Uses corresponding mesh path `data_center/ring_{connector_id}.stl`.
    - Uses corresponding keys `{connector_id}_pos` and `{connector_id}_director`.
    - Each run writes its own state NPZ through generate_cable().
    - The state NPZ schema is unchanged (same keys/format as generate_data.py).
    - No extra metadata NPZ is written.
    """
    if n_runs <= 0:
        raise ValueError(f"n_runs must be > 0, got {n_runs}")
    connector_ids = tuple(int(c) for c in connector_ids)
    if len(connector_ids) == 0:
        raise ValueError("connector_ids must be non-empty.")
    if any(c <= 0 for c in connector_ids):
        raise ValueError(f"connector_ids must be >= 1, got {connector_ids}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(master_seed)
    seed_arr = rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=(len(connector_ids), n_runs),
        dtype=np.uint32,
    ).astype(np.int64)

    defaults: dict[str, Any] = {
        "data_npz_path": Path(__file__).resolve().parent / "data_center_points.npz",
    }
    for key, value in defaults.items():
        generate_kwargs.setdefault(key, value)

    state_paths: dict[int, list[str]] = {cid: [] for cid in connector_ids}
    video_paths: dict[int, list[str]] = {cid: [] for cid in connector_ids}

    for cid_idx, cid in enumerate(connector_ids):
        mesh_path = Path(__file__).resolve().parent / f"ring_{cid}.stl"
        for run_idx in range(n_runs):
            seed = int(seed_arr[cid_idx, run_idx])
            run_tag = f"{output_name}_c{cid}_run{run_idx:03d}"
            run_output = output_dir / f"{run_tag}_4view.mp4"
            run_state = output_dir / f"{run_tag}_state.npz"
            local_kwargs = dict(generate_kwargs)
            local_kwargs["connector_id"] = cid
            local_kwargs["mesh_path"] = mesh_path

            video_path, state_path = generate_cable(
                output=run_output,
                state_output=run_state,
                seed=seed,
                **local_kwargs,
            )
            video_paths[cid].append(str(video_path))
            state_paths[cid].append(str(state_path))
            print(
                f"[generate_data_batch] connector={cid} run={run_idx:03d} seed={seed} "
                f"state={Path(state_path).name}"
            )

    return {
        "connector_ids": np.asarray(connector_ids, dtype=np.int64),
        "seed_arr": seed_arr,
        "state_paths": state_paths,
        "video_paths": video_paths,
    }


if __name__ == "__main__":
    result = run_generate_data_batch(
        n_runs=1,
        master_seed=123,
        output_name="generate_data_batch",
        connector_ids=(2, 3, 4),
    )
    print(
        "[generate_data_batch] finished connectors: "
        f"{result['connector_ids'].tolist()}"
    )
