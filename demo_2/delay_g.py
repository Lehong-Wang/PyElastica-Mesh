"""Delayed-gravity mesh demo with 4-view rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.animation as animation

import elastica as ea
from examples.MeshCase import post_processing


class DelayedGravity(ea.NoForces):
    """Apply gravity only after a chosen simulation time."""

    def __init__(self, acc_gravity: np.ndarray, switch_on_time: float = 1.0) -> None:
        super().__init__()
        if switch_on_time < 0.0:
            raise ValueError("switch_on_time must be >= 0.0")
        self.acc_gravity = np.asarray(acc_gravity, dtype=np.float64).reshape(3)
        self.switch_on_time = float(switch_on_time)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        if float(time) < self.switch_on_time:
            return
        ea.GravityForces.compute_gravity_forces(
            self.acc_gravity, system.mass, system.external_forces
        )


def run_delayed_gravity_mesh(
    mesh_path: str = "mytest/bunny_low_10.stl",
    switch_on_time: float = 1.0,
    final_time: float = 2.0,
    dt: float = 5.0e-4,
    output: str = "demo_2/delay_g_4view.mp4",
    density: float = 1000.0,
    com_start: tuple[float, float, float] = (0.0, 0.0, 1.5),
    render_speed: float = 1.0,
    render_fps: int | None = None,
) -> Path:
    class DelayGravityMeshSim(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.CallBacks,
    ):
        pass

    simulator = DelayGravityMeshSim()

    mesh = ea.Mesh(mesh_path)
    volume = mesh.compute_volume()
    mesh_body = ea.MeshRigidBody(
        mesh=mesh,
        center_of_mass=np.asarray(com_start, dtype=np.float64),
        density=density,
        volume=volume,
    )
    simulator.append(mesh_body)

    simulator.add_forcing_to(mesh_body).using(
        DelayedGravity,
        acc_gravity=np.array([0.0, 0.0, -9.81]),
        switch_on_time=switch_on_time,
    )

    collector_store: dict[str, object] = {}

    class MeshCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.position: list[np.ndarray] = []
            self.director: list[np.ndarray] = []
            collector_store["mesh_cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.position.append(system.position_collection[:, 0].copy())
            self.director.append(system.director_collection[:, :, 0].copy())

    simulator.collect_diagnostics(mesh_body).using(
        MeshCallBack, step_skip=max(1, int(0.01 / dt))
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_output_path: dict[str, Path] = {"path": output_path}

    def _render_from_callback() -> None:
        cb = collector_store.get("mesh_cb")
        if cb is None or len(cb.time) == 0:
            return

        positions = np.asarray(cb.position)
        x_min = float(np.min(positions[:, 0]) - 0.6)
        x_max = float(np.max(positions[:, 0]) + 0.6)
        y_min = float(np.min(positions[:, 1]) - 0.6)
        y_max = float(np.max(positions[:, 1]) + 0.6)
        z_min = float(min(0.0, np.min(positions[:, 2]) - 0.3))
        z_max = float(np.max(positions[:, 2]) + 0.6)

        mesh_data = {
            "time": cb.time,
            "position": cb.position,
            "director": cb.director,
            "mesh": mesh.mesh,
        }

        render_path = output_path
        available_writers = set(animation.writers.list())
        if "ffmpeg" not in available_writers:
            if "pillow" not in available_writers:
                raise RuntimeError(
                    "No supported video writer found. Install ffmpeg or pillow."
                )
            # Keep four-view rendering path and fallback to GIF when ffmpeg is absent.
            animation.writers._registered["ffmpeg"] = animation.writers["pillow"]  # type: ignore[attr-defined]
            render_path = output_path.with_suffix(".gif")
            print(f"[delay_g] ffmpeg unavailable; falling back to GIF: {render_path}")

        post_processing.plot_mesh_multiview_animation(
            mesh_data,
            video_name=str(render_path),
            fps=render_fps,
            speed_factor=render_speed,
            bounds=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        )
        rendered_output_path["path"] = render_path

    try:
        simulator.finalize()
        timestepper = ea.PositionVerlet()
        total_steps = int(np.ceil(final_time / dt))
        ea.integrate(timestepper, simulator, final_time, total_steps)
        _render_from_callback()
    except Exception:
        try:
            _render_from_callback()
        except Exception as render_exc:  # noqa: BLE001
            print(f"[delay_g] Rendering failed after exception: {render_exc}")
        raise

    return rendered_output_path["path"]


if __name__ == "__main__":
    saved_path = run_delayed_gravity_mesh(
        mesh_path="mytest/bunny_low_10.stl",
        switch_on_time=1.0,
        final_time=2.0,
        dt=5.0e-4,
        output="demo_2/delay_g_4view.mp4",
        render_speed=1.0,
        render_fps=None,
    )
    print(f"Saved 4-view video to: {saved_path}")
