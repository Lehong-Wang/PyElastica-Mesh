"""
Rod drops onto a frozen (immovable) mesh.
"""

import numpy as np
import elastica as ea
from examples.MeshCase import post_processing


def frozen_mesh_contact(
    final_time: float = 0.08,
    dt: float = 5.0e-4,
    output: str = "mesh_frozen_contact.mp4",
    render_speed: float = 1.0,
    render_fps: int | None = None,
):
    class FrozenMeshSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
    ):
        pass

    simulator = FrozenMeshSim()

    rod = ea.CosseratRod.straight_rod(
        n_elements=10,
        start=np.array([0.0, 0.0, 0.8]),
        direction=np.array([0.0, 0.01, -1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        base_length=0.4,
        base_radius=0.02,
        density=1000.0,
        youngs_modulus=1e6,
        shear_modulus=1e6 / (2.0 * 1.5),
    )
    simulator.append(rod)

    mesh = ea.Mesh("mytest/cube_tight.stl")
    volume = mesh.compute_volume()
    inertia = mesh.compute_inertia_tensor(density=1000.0)
    mesh_body = ea.MeshRigidBody(
        mesh=mesh,
        center_of_mass=np.array([0.0, 0.0, 0.2]),
        mass_second_moment_of_inertia=inertia,
        density=1000.0,
        volume=volume,
    )
    mesh_body.mass = np.float64(1e20)
    mesh_body.inv_mass_second_moment_of_inertia[:] = 0.0
    simulator.append(mesh_body)

    simulator.detect_contact_between(rod, mesh_body).using(
        ea.RodMeshContact, k=1e4, nu=5.0, mesh_frozen=True
    )

    simulator.add_forcing_to(rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
    )

    collector_store: dict[str, object] = {}

    class RodMeshCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time = []
            self.mesh_position = []
            self.mesh_director = []
            self.rod_position = []
            collector_store["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.mesh_position.append(mesh_body.position_collection[:, 0].copy())
            self.mesh_director.append(mesh_body.director_collection[:, :, 0].copy())
            self.rod_position.append(rod.position_collection.copy())

    simulator.collect_diagnostics(rod).using(
        RodMeshCallBack, step_skip=max(1, int(0.005 / dt))
    )

    def _render_from_callback():
        cb = collector_store.get("cb")
        if cb is None or len(cb.time) == 0:
            return
        mesh_data = {
            "time": cb.time,
            "position": cb.mesh_position,
            "director": cb.mesh_director,
            "mesh": mesh.mesh,
        }
        post_processing.plot_mesh_multiview_animation(
            mesh_data,
            video_name=output,
            fps=render_fps,
            speed_factor=render_speed,
            bounds=((-1.0, 1.0), (-1.0, 1.0), (-0.2, 1.0)),
            rod_positions=cb.rod_position,
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
            print(f"[mesh_frozen_contact] Rendering failed after exception: {render_exc}")
        raise


if __name__ == "__main__":
    frozen_mesh_contact(
        final_time=3.0,
        dt=5.0e-4,
        output="mesh_frozen_contact.mp4",
        render_speed=1.0,
        render_fps=None,
    )
