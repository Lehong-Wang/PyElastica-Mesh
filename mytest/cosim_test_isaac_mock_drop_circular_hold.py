"""
Staged dummy Isaac-process script:
1) hold frame fixed for 1s while rod drops under gravity, with:
   - damping x5
   - fixed-joint k and nu scaled to 1/5
2) run the circular command trajectory
3) hold at the trajectory endpoint for 2s
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
from co_sim.isaac_process import circular_yz_frame_state
from co_sim.models import CoSimConfig, FrameState
from co_sim.plotting import (
    Bounds3D,
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


def run_demo(
    config: CoSimConfig,
    drop_duration: float = 1.0,
    circular_duration: float = 4.0,
    hold_duration: float = 2.0,
    plot_bounds: Bounds3D | None = None,
) -> dict[str, object]:
    cfg = config
    if cfg.py_dt <= 0.0:
        raise ValueError(f"py_dt must be positive, got {cfg.py_dt}.")
    if cfg.isaac_dt <= 0.0:
        raise ValueError(f"isaac_dt must be positive, got {cfg.isaac_dt}.")
    if cfg.output_interval <= 0.0:
        raise ValueError(f"output_interval must be positive, got {cfg.output_interval}.")
    if cfg.render and (cfg.render_fps is None or cfg.render_fps <= 0):
        raise ValueError(f"render_fps must be positive when render=True, got {cfg.render_fps}.")
    if drop_duration < 0.0:
        raise ValueError(f"drop_duration must be >= 0, got {drop_duration}.")
    if circular_duration <= 0.0:
        raise ValueError(f"circular_duration must be > 0, got {circular_duration}.")
    if hold_duration < 0.0:
        raise ValueError(f"hold_duration must be >= 0, got {hold_duration}.")

    frame_init = circular_yz_frame_state(0.0, motion_duration=circular_duration)
    engine = CoSimEngine(config=cfg, frame_initial_state=frame_init)

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

    def _advance_phase(
        duration: float,
        command_fn,
        phase_name: str,
    ) -> None:
        nonlocal last_mean_force, next_print_time
        elapsed = 0.0
        while elapsed + _EPS < duration:
            dt = min(cfg.isaac_dt, duration - elapsed)
            frame_state: FrameState = command_fn(elapsed)
            impulse_result = engine.update_frame_state(
                frame_state,
                duration=dt,
                observer=_observer,
            )
            if impulse_result.elapsed_time > 0.0:
                last_mean_force = impulse_result.linear_impulse / impulse_result.elapsed_time
                elapsed += impulse_result.elapsed_time
            else:
                elapsed += dt

            if cfg.print_progress and impulse_result.sim_time + _EPS >= next_print_time:
                print(f"[{phase_name}] time={impulse_result.sim_time:8.5f} Fmean={last_mean_force}")
                while impulse_result.sim_time + _EPS >= next_print_time:
                    next_print_time += 0.1

    runtime_damping_constant = float(cfg.damping_constant)
    runtime_joint_k = float(cfg.joint_k)
    runtime_joint_nu = float(cfg.joint_nu)

    drop_damping_constant = 20.0 * runtime_damping_constant
    drop_joint_k = runtime_joint_k / 10.0
    drop_joint_nu = runtime_joint_nu / 5.0

    if drop_duration > 0.0:
        engine._set_rod_damping_constant(drop_damping_constant)
        engine._set_fixed_joint_k(drop_joint_k)
        engine._fixed_joint.nu = np.float64(drop_joint_nu)
        try:
            _advance_phase(
                duration=drop_duration,
                command_fn=lambda _t: frame_init,
                phase_name="drop",
            )
        finally:
            engine._set_rod_damping_constant(runtime_damping_constant)
            engine._set_fixed_joint_k(runtime_joint_k)
            engine._fixed_joint.nu = np.float64(runtime_joint_nu)

    _advance_phase(
        duration=circular_duration,
        command_fn=lambda phase_t: circular_yz_frame_state(
            phase_t,
            motion_duration=circular_duration,
        ),
        phase_name="circular",
    )

    end_state = circular_yz_frame_state(circular_duration, motion_duration=circular_duration)
    if hold_duration > 0.0:
        _advance_phase(
            duration=hold_duration,
            command_fn=lambda _t: end_state,
            phase_name="hold",
        )

    final_snapshot = engine.snapshot()
    if npz_buffer["time"][-1] < final_snapshot.sim_time - _EPS:
        _append_sample(npz_buffer, final_snapshot, last_mean_force)
    if render_buffer is not None and render_buffer["time"][-1] < final_snapshot.sim_time - _EPS:
        _append_sample(render_buffer, final_snapshot, last_mean_force)

    npz_sampled = _buffer_to_arrays(npz_buffer)
    render_sampled = _buffer_to_arrays(render_buffer) if render_buffer is not None else None

    output_dir = (
        Path(__file__).resolve().parent
        if cfg.output_dir is None
        else Path(cfg.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_tag = f"_drop{drop_duration:g}_circ{circular_duration:g}_hold{hold_duration:g}"
    param_tag = f"_k{cfg.joint_k:g}_nu{cfg.joint_nu:g}_t{cfg.isaac_dt:g}"
    tagged_output_name = f"{cfg.output_name}{phase_tag}{param_tag}"

    state_path = output_dir / f"{tagged_output_name}_state.npz"
    np.savez(
        state_path,
        time=npz_sampled["time"],
        rod_position=npz_sampled["rod_position"],
        rod_director=npz_sampled["rod_director"],
        frame_position=npz_sampled["frame_position"],
        frame_director=npz_sampled["frame_director"],
        mean_force=npz_sampled["mean_force"],
        mean_force_magnitude=npz_sampled["mean_force_magnitude"],
        py_dt=cfg.py_dt,
        isaac_dt=cfg.isaac_dt,
        total_time=drop_duration + circular_duration + hold_duration,
        drop_duration=drop_duration,
        circular_duration=circular_duration,
        hold_duration=hold_duration,
        drop_damping_scale=5.0,
        drop_joint_k_scale=0.2,
        drop_joint_nu_scale=0.2,
        output_interval=cfg.output_interval,
        npz_sample_interval=cfg.output_interval,
        render_sample_interval=(1.0 / float(cfg.render_fps))
        if cfg.render and cfg.render_fps
        else np.nan,
    )

    force_vec_plot_path = output_dir / f"{tagged_output_name}_mean_force_vector.png"
    plot_force_vector_with_magnitude(
        time=npz_sampled["time"],
        mean_force=npz_sampled["mean_force"],
        force_mag=npz_sampled["mean_force_magnitude"],
        output_path=force_vec_plot_path,
    )

    force_mag_plot_path = output_dir / f"{tagged_output_name}_force_vs_time.png"
    plot_force_vs_time(
        time=npz_sampled["time"],
        force_mag=npz_sampled["mean_force_magnitude"],
        output_path=force_mag_plot_path,
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
            video_path=video_path,
            render_fps=cfg.render_fps,
            render_speed=cfg.render_speed,
            force_vector_scale=cfg.force_vector_scale,
            plot_bounds=plot_bounds,
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
    # Optional explicit 3D bounds:
    # plot_bounds = ((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))
    plot_bounds = None

    drop_duration = 1.5
    circular_duration = 1.0
    hold_duration = 2.0

    demo_cfg = CoSimConfig(
        py_dt=1.0e-5,
        isaac_dt=1.0e-2,
        final_time=drop_duration + circular_duration + hold_duration,
        output_name="cosim_drop_circular_hold",
        axial_stretch_stiffening=1.0e2,
        use_ground_contact=True,
        ground_z=-0.5,
        ground_contact_k=1.0e2,
        ground_contact_nu=1.0,
        ground_static_mu=(0.8, 0.8, 0.8),
        ground_kinetic_mu=(0.6, 0.6, 0.6),
        ground_slip_velocity_tol=1.0e-6,
        render_fps = 15,
        render=True,
    )
    results = run_demo(
        config=demo_cfg,
        drop_duration=drop_duration,
        circular_duration=circular_duration,
        hold_duration=hold_duration,
        plot_bounds=plot_bounds,
    )
    print(
        f"Saved npz to {results['state_path']} "
        f"(render={bool(results['video_path'])})."
    )
