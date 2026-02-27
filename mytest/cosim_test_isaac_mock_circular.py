"""
Dummy Isaac-process script with circular frame motion on world XZ plane.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from co_sim.isaac_process import circular_yz_frame_state
from co_sim.models import CoSimConfig, FrameState
from mytest.cosim_test_isaac_mock import run_demo


def circular_command(command_time: float, cfg: CoSimConfig) -> FrameState:
    return circular_yz_frame_state(
        command_time,
        motion_duration=cfg.final_time,
    )


if __name__ == "__main__":
    # Optional explicit 3D bounds:
    # plot_bounds = ((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))
    plot_bounds = None

    demo_cfg = CoSimConfig(
        py_dt=1.0e-5,
        isaac_dt=1.0e-2,
        final_time=0.5,
        base_length = 1.0,
        base_radius = 5e-3,
        density = 1_000.0,
        youngs_modulus = 5.0e6,
        output_name="cosim_circular_xz",
        use_ground_contact=True,
        ground_z=0.0,
        ground_contact_k=1.0e4,
        ground_contact_nu=5.0,
        ground_static_mu=(0.8, 0.8, 0.8),
        ground_kinetic_mu=(0.6, 0.6, 0.6),
        ground_slip_velocity_tol=1.0e-6,
        render=True,
    )
    results = run_demo(
        config=demo_cfg,
        command_generator=circular_command,
        plot_bounds=plot_bounds,
    )
    print(
        f"Saved npz to {results['state_path']} "
        f"(render={bool(results['video_path'])})."
    )
