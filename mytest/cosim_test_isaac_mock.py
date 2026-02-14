"""
Dummy Isaac-process script that drives the refactored CoSimEngine.

Timing rules:
- PyElastica integrates with `py_dt`.
- Frame commands update every `isaac_dt`.
- NPZ sampling uses `output_interval` (independent).
- Render sampling uses `1 / render_fps` (independent).
All schedules are based on real simulation time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from co_sim.engine import CoSimEngine
from co_sim.isaac_process import sine_frame_state
from co_sim.models import CoSimConfig, FrameState
from co_sim.plotting import (
    plot_force_vector_with_magnitude,
    plot_force_vs_time,
    render_multiview_video,
)

_EPS = 1.0e-12


def _empty_sample_buffer(initial_snapshot, initial_force: np.ndarray) -> dict[str, list[np.ndarray] | list[float]]:
    return {
        "time": [initial_snapshot.sim_time],
        "rod_position": [initial_snapshot.rod_position],
        "rod_director": [initial_snapshot.rod_director],
        "frame_position": [initial_snapshot.frame_position],
        "frame_director": [initial_snapshot.frame_director],
        "mean_force": [initial_force.copy()],
    }


def _append_sample(
    sample_buffer: dict[str, list[np.ndarray] | list[float]],
    snapshot,
    mean_force: np.ndarray,
) -> None:
    sample_buffer["time"].append(snapshot.sim_time)
    sample_buffer["rod_position"].append(snapshot.rod_position)
    sample_buffer["rod_director"].append(snapshot.rod_director)
    sample_buffer["frame_position"].append(snapshot.frame_position)
    sample_buffer["frame_director"].append(snapshot.frame_director)
    sample_buffer["mean_force"].append(mean_force.copy())


def _buffer_to_arrays(
    sample_buffer: dict[str, list[np.ndarray] | list[float]]
) -> dict[str, np.ndarray]:
    out = {
        "time": np.asarray(sample_buffer["time"]),
        "rod_position": np.asarray(sample_buffer["rod_position"]),
        "rod_director": np.asarray(sample_buffer["rod_director"]),
        "frame_position": np.asarray(sample_buffer["frame_position"]),
        "frame_director": np.asarray(sample_buffer["frame_director"]),
        "mean_force": np.asarray(sample_buffer["mean_force"]),
    }
    out["mean_force_magnitude"] = np.linalg.norm(out["mean_force"], axis=1)
    return out


def _run_isaac_loop(
    engine: CoSimEngine,
    cfg: CoSimConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    initial_snapshot = engine.snapshot()
    zero_force = np.zeros(3)

    npz_buffer = _empty_sample_buffer(initial_snapshot, zero_force)
    render_buffer = _empty_sample_buffer(initial_snapshot, zero_force) if cfg.render else None

    next_npz_sample_time = cfg.output_interval
    render_interval = (1.0 / float(cfg.render_fps)) if cfg.render and cfg.render_fps else None
    next_render_sample_time = render_interval if render_interval is not None else np.inf
    next_print_time = 0.1
    last_mean_force = zero_force.copy()

    def _observer(step_time: float, mean_force: np.ndarray) -> None:
        nonlocal next_npz_sample_time, next_render_sample_time, last_mean_force
        last_mean_force = mean_force
        snapshot_cache = None

        if step_time + _EPS >= next_npz_sample_time:
            if snapshot_cache is None:
                snapshot_cache = engine.snapshot()
            _append_sample(npz_buffer, snapshot_cache, mean_force)
            while step_time + _EPS >= next_npz_sample_time:
                next_npz_sample_time += cfg.output_interval

        if render_buffer is not None and step_time + _EPS >= next_render_sample_time:
            if snapshot_cache is None:
                snapshot_cache = engine.snapshot()
            _append_sample(render_buffer, snapshot_cache, mean_force)
            while step_time + _EPS >= next_render_sample_time:
                next_render_sample_time += render_interval  # type: ignore[operator]

    while float(engine.time) + _EPS < cfg.final_time:
        command_time = float(engine.time)
        frame_command = sine_frame_state(
            command_time,
            amp=cfg.command_sine_amp,
            freq=cfg.command_sine_freq,
        )

        remaining_time = cfg.final_time - command_time
        advance_time = min(cfg.isaac_dt, remaining_time)
        impulse_result = engine.update_frame_state(
            frame_command,
            duration=advance_time,
            observer=_observer,
        )
        if impulse_result.elapsed_time > 0.0:
            last_mean_force = impulse_result.linear_impulse / impulse_result.elapsed_time

        if cfg.print_progress and impulse_result.sim_time + _EPS >= next_print_time:
            print(f"time={impulse_result.sim_time:8.5f} Fmean={last_mean_force}")
            while impulse_result.sim_time + _EPS >= next_print_time:
                next_print_time += 0.1

    final_snapshot = engine.snapshot()
    if npz_buffer["time"][-1] < final_snapshot.sim_time - _EPS:
        _append_sample(npz_buffer, final_snapshot, last_mean_force)
    if render_buffer is not None and render_buffer["time"][-1] < final_snapshot.sim_time - _EPS:
        _append_sample(render_buffer, final_snapshot, last_mean_force)

    npz_sampled = _buffer_to_arrays(npz_buffer)
    render_sampled = _buffer_to_arrays(render_buffer) if render_buffer is not None else None
    return npz_sampled, render_sampled


def _save_state_and_plots(
    cfg: CoSimConfig,
    output_dir: Path,
    tagged_output_name: str,
    sampled: dict[str, np.ndarray],
) -> tuple[Path, Path, Path]:
    state_path = output_dir / f"{tagged_output_name}_state.npz"
    np.savez(
        state_path,
        time=sampled["time"],
        rod_position=sampled["rod_position"],
        rod_director=sampled["rod_director"],
        frame_position=sampled["frame_position"],
        frame_director=sampled["frame_director"],
        mean_force=sampled["mean_force"],
        mean_force_magnitude=sampled["mean_force_magnitude"],
        py_dt=cfg.py_dt,
        isaac_dt=cfg.isaac_dt,
        final_time=cfg.final_time,
        sine_amp=cfg.command_sine_amp,
        sine_freq=cfg.command_sine_freq,
        output_interval=cfg.output_interval,
        npz_sample_interval=cfg.output_interval,
        render_sample_interval=(1.0 / float(cfg.render_fps))
        if cfg.render and cfg.render_fps
        else np.nan,
    )

    force_vec_plot_path = output_dir / f"{tagged_output_name}_mean_force_vector.png"
    plot_force_vector_with_magnitude(
        time=sampled["time"],
        mean_force=sampled["mean_force"],
        force_mag=sampled["mean_force_magnitude"],
        output_path=force_vec_plot_path,
    )

    force_mag_plot_path = output_dir / f"{tagged_output_name}_force_vs_time.png"
    plot_force_vs_time(
        time=sampled["time"],
        force_mag=sampled["mean_force_magnitude"],
        output_path=force_mag_plot_path,
    )
    return state_path, force_vec_plot_path, force_mag_plot_path


def run_demo(config: CoSimConfig) -> dict[str, object]:
    cfg = config
    if cfg.final_time <= 0.0:
        raise ValueError(f"final_time must be positive, got {cfg.final_time}.")
    if cfg.py_dt <= 0.0:
        raise ValueError(f"py_dt must be positive, got {cfg.py_dt}.")
    if cfg.isaac_dt <= 0.0:
        raise ValueError(f"isaac_dt must be positive, got {cfg.isaac_dt}.")
    if cfg.output_interval <= 0.0:
        raise ValueError(f"output_interval must be positive, got {cfg.output_interval}.")
    if cfg.render and (cfg.render_fps is None or cfg.render_fps <= 0):
        raise ValueError(f"render_fps must be positive when render=True, got {cfg.render_fps}.")

    frame_init: FrameState = sine_frame_state(
        0.0,
        amp=cfg.command_sine_amp,
        freq=cfg.command_sine_freq,
    )
    engine = CoSimEngine(config=cfg, frame_initial_state=frame_init)
    npz_sampled, render_sampled = _run_isaac_loop(engine=engine, cfg=cfg)

    output_dir = (
        Path(__file__).resolve().parent
        if cfg.output_dir is None
        else Path(cfg.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    param_tag = f"_k{cfg.joint_k:g}_nu{cfg.joint_nu:g}_t{cfg.isaac_dt:g}"
    tagged_output_name = f"{cfg.output_name}{param_tag}"
    state_path, force_vec_plot_path, force_mag_plot_path = _save_state_and_plots(
        cfg=cfg,
        output_dir=output_dir,
        tagged_output_name=tagged_output_name,
        sampled=npz_sampled,
    )

    video_path: Path | None = output_dir / f"{tagged_output_name}_4view.mp4"
    if cfg.render:
        assert render_sampled is not None
        render_multiview_video(
            sampled_rod_pos=render_sampled["rod_position"],
            sampled_frame_pos=render_sampled["frame_position"],
            sampled_frame_dir=render_sampled["frame_director"],
            sampled_time=render_sampled["time"],
            sampled_mean_force=render_sampled["mean_force"],
            cfg=cfg,
            video_path=video_path,
            render_fps=cfg.render_fps,
            render_speed=cfg.render_speed,
            force_vector_scale=cfg.force_vector_scale,
        )
    else:
        video_path = None

    return {
        "state_path": state_path,
        "video_path": video_path,
        "force_vector_plot_path": force_vec_plot_path,
        "force_magnitude_plot_path": force_mag_plot_path,
        "time": npz_sampled["time"],
        "mean_force": npz_sampled["mean_force"],
        "mean_force_magnitude": npz_sampled["mean_force_magnitude"],
    }


if __name__ == "__main__":
    demo_cfg = CoSimConfig(
        py_dt = 1.0e-5,
        isaac_dt = 1.0e-1,
        final_time = 3.0,
        joint_k = 5.0e2,
        joint_nu = 20.0,
        render=True)

    
    results = run_demo(config=demo_cfg)
    print(
        f"Saved npz to {results['state_path']} "
        f"(render={bool(results['video_path'])})."
    )
