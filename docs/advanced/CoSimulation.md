# Co-Simulation Engine

This page summarizes the `co_sim` implementation used by `mytest/cosim_test_isaac_mock.py`.

## Overview

Model assumptions:

- one Cosserat rod (PyElastica),
- one attachment frame (Isaac side),
- frame is kinematic (commanded), rod is dynamic,
- rod and frame are coupled by a fixed joint.

Per Isaac update, co-sim takes one `FrameState` and returns one `ImpulseResult`.

## Isaac <-> PyElastica Dataflow

For each update window `[t_k, t_{k+1}]`:

1. Isaac computes frame command and sends `FrameState`.
2. Call `engine.update_frame_state(frame_state, duration=delta_t)`.
3. Engine applies frame state and advances PyElastica internal steps (`py_dt`) until `delta_t` is consumed.
4. During internal stepping, joint reaction loads on frame are accumulated as impulse.
5. Frame loads are zeroed each step so frame stays kinematic.
6. Engine returns `ImpulseResult` (`linear_impulse`, `angular_impulse`, `elapsed_time`, `sim_time`).

Common conversion:

- mean force over update window: `linear_impulse / elapsed_time`.

## Core API

Primary class: `CoSimEngine` (`co_sim/engine.py`).

- `CoSimEngine(config: CoSimConfig, rod_initial_state=None, frame_initial_state=None)`
  - builds scene, damping, and fixed joint,
  - initializes kinematic frame command state.
- `update_frame_state(frame_state, duration=None, observer=None) -> ImpulseResult`
  - main co-sim call per external update.
- `snapshot() -> SceneSnapshot`
  - returns rod/frame pose for logging or rendering.

Related models (`co_sim/models.py`):

- `CoSimConfig`, `FrameState`, `ImpulseResult`, `SceneSnapshot`.

## Quick Start

```bash
python mytest/cosim_test_isaac_mock.py
```

Programmatic:

```python
from co_sim.models import CoSimConfig
from mytest.cosim_test_isaac_mock import run_demo

cfg = CoSimConfig(
    py_dt=1.0e-5,
    isaac_dt=1.0e-1,
    final_time=3.0,
    output_interval=1.0e-2,
    render=False,
)
results = run_demo(cfg)
```

Direct engine loop:

```python
import numpy as np
from co_sim.engine import CoSimEngine
from co_sim.models import CoSimConfig, FrameState

cfg = CoSimConfig()
engine = CoSimEngine(cfg)

while float(engine.time) < cfg.final_time:
    t = float(engine.time)
    cmd = FrameState(
        position=np.array([0.0, 0.1 * np.sin(2.0 * np.pi * t), 0.0]),
        director=cfg.frame_initial_director,
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        omega=np.zeros(3),
        alpha=np.zeros(3),
    )
    dt = min(cfg.isaac_dt, cfg.final_time - t)
    impulse = engine.update_frame_state(cmd, duration=dt)
```

## Timing Rules

All schedules use real simulation time and are independent:

- `py_dt`: internal integration step,
- `isaac_dt`: external command/update period,
- `output_interval`: NPZ sampling period,
- `1 / render_fps`: render sampling period.

No integer multiple relationship is required.

## Main Config Knobs

All runtime parameters are in `CoSimConfig`.

- timing:
  - `py_dt`, `isaac_dt`, `final_time`, `output_interval`.
- rod:
  - `n_elem`, `base_length`, `base_radius`, `density`, `youngs_modulus`, `shear_modulus_ratio`, `damping_constant`.
- joint:
  - `joint_k`, `joint_nu`, `joint_kt`, `joint_nut`.
- initial states:
  - `rod_start`, `rod_direction`, `rod_normal`, `frame_initial_*`.
- command model (mock script):
  - `command_sine_amp`, `command_sine_freq`.
- output and render:
  - `output_name`, `output_dir`, `render`, `render_fps`, `render_speed`, `force_vector_scale`, `print_progress`.

## Outputs

Demo outputs:

- `{output_name}_..._state.npz`,
- `{output_name}_..._mean_force_vector.png`,
- `{output_name}_..._force_vs_time.png`,
- optional `{output_name}_..._4view.mp4` when `render=True`.

Main NPZ arrays:

- `time`,
- `rod_position`, `rod_director`,
- `frame_position`, `frame_director`,
- `mean_force`, `mean_force_magnitude`.

## Notes and Troubleshooting

- The frame is intentionally kinematic; impulse is the reaction output for co-sim coupling.
- A previous stall near `2.1s` was a floating-point non-progress issue at update boundaries, fixed in `CoSimEngine.update_frame_state(...)` by snapping to target time when needed.
