"""
Mesh freefall example using MeshRigidBody.
"""

import numpy as np
import elastica as ea
from examples.MeshCase import post_processing


def mesh_freefall_simulation(
    final_time: float = 0.1,
    dt: float = 5.0e-4,
    output: str = "mesh_freefall.mp4",
    render_speed: float = 1.0,
    render_fps: int | None = None,
):
    class MeshFreefall(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
    ):
        pass

    simulator = MeshFreefall()

    mesh = ea.Mesh("mytest/bunny.stl")
    volume = mesh.compute_volume()
    density = 1000.0
    com_start = np.array([0.0, 0.0, 1.5])

    mesh_body = ea.MeshRigidBody(
        mesh=mesh,
        center_of_mass=com_start,
        density=density,
        volume=volume,
    )
    simulator.append(mesh_body)

    gravity = np.array([0.0, 0.0, -9.81])
    simulator.add_forcing_to(mesh_body).using(ea.GravityForces, acc_gravity=gravity)

    collector_store: dict[str, object] = {}

    class MeshCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time = []
            self.position = []
            self.director = []
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

    def _render_from_callback():
        cb = collector_store.get("mesh_cb")
        if cb is None or len(cb.time) == 0:
            return
        mesh_data = {
            "time": cb.time,
            "position": cb.position,
            "director": cb.director,
            "mesh": mesh.mesh,
        }
        post_processing.plot_mesh_multiview_animation(
            mesh_data,
            video_name=output,
            fps=render_fps,
            speed_factor=render_speed,
            bounds=((-2, 2), (-2, 2), (-0.5, 2.5)),
        )

    try:
        simulator.finalize()
        timestepper = ea.PositionVerlet()
        total_steps = int(final_time / dt)
        ea.integrate(timestepper, simulator, final_time, total_steps)
        _render_from_callback()
    except Exception as exc:  # noqa: BLE001
        try:
            _render_from_callback()
        except Exception as render_exc:  # noqa: BLE001
            print(f"[mesh_freefall] Rendering failed after exception: {render_exc}")
        raise


if __name__ == "__main__":
    mesh_freefall_simulation(render_speed=1.0, render_fps=None)
