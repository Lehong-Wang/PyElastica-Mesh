"""
Rod colliding with a mesh obstacle.
"""

import numpy as np
import elastica as ea
from examples.MeshCase import post_processing


def rod_mesh_collision(
    final_time: float = 0.08,
    dt: float = 5.0e-4,
    output: str = "mesh_rod_collision.mp4",
    render_speed: float = 1.0,
    render_fps: int | None = None,
):
    class RodMeshSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
    ):
        pass

    simulator = RodMeshSim()

    n_elem = 5
    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=np.array([-0.0, 0.0, 0.8]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.5,
        base_radius=0.005,
        density=5000.0,
        youngs_modulus=1e6,
        shear_modulus=1e6 / (2.0 * 1.5),
    )
    simulator.append(rod)

    simulator.constrain(rod).using(
        ea.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    mesh = ea.Mesh("mytest/bunny_low_10.stl")
    # mesh = ea.Mesh("mytest/cube_tight.stl")
    density_mesh = 10.0
    volume = mesh.compute_volume()
    inertia = mesh.compute_inertia_tensor(density=density_mesh)
    mesh_body = ea.MeshRigidBody(
        mesh=mesh,
        # center_of_mass=np.array([0.0, 0.0, 0.5]),
        mass_second_moment_of_inertia=inertia,
        density=density_mesh,
        volume=volume,
    )
    simulator.append(mesh_body)

    simulator.detect_contact_between(rod, mesh_body).using(
        ea.RodMeshContact, k=1e4, nu=5.0
    )

    gravity = np.array([0.0, 0.0, -9.81])
    simulator.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=gravity)

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
        # RodMeshCallBack, step_skip=max(1, int(0.005 / dt))
        RodMeshCallBack, step_skip=max(1, int(1 / 100 / dt))
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
            bounds=((-1.5, 2.0), (-1.0, 1.0), (-0.5, 1.5)),
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
            print(f"[mesh_rod_collision] Rendering failed after exception: {render_exc}")
        raise


if __name__ == "__main__":
    rod_mesh_collision(
        final_time=5.0,
        dt=1.0e-3,
        output="mesh_rod_collision.mp4",
        render_speed=1.0,
        render_fps=None,
    )
