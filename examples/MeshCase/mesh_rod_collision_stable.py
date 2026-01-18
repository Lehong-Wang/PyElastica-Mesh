"""
Rod colliding with a mesh obstacle.
"""

import numpy as np
import elastica as ea
from examples.MeshCase import post_processing

# NEW: for plotting energy
import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt


def rod_mesh_collision(
    final_time: float = 0.08,
    dt: float = 5.0e-4,
    nu: float = 5e-2,
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
        ea.Damping
    ):
        pass

    simulator = RodMeshSim()

    n_elem = 20
    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=np.array([-0.0, 0.0, 0.8]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.5,
        base_radius=0.005,
        density=2000.0,
        youngs_modulus=1e8,
        shear_modulus=1e8 / (2.0 * 1.5),
    )
    simulator.append(rod)

    simulator.constrain(rod).using(
        ea.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    from elastica.dissipation import AnalyticalLinearDamper

    simulator.dampen(rod).using(
        AnalyticalLinearDamper,
        damping_constant=nu,
        time_step=dt,
    )

    mesh = ea.Mesh("mytest/bunny_low_10.stl")
    density_mesh = 10.0
    volume = mesh.compute_volume()
    inertia = mesh.compute_inertia_tensor(density=density_mesh)
    mesh_body = ea.MeshRigidBody(
        mesh=mesh,
        mass_second_moment_of_inertia=inertia,
        density=density_mesh,
        volume=volume,
    )
    simulator.append(mesh_body)

    gravity = np.array([0.0, 0.0, -9.81])
    simulator.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=gravity)

    simulator.detect_contact_between(rod, mesh_body).using(
        ea.RodMeshContact, k=1e4, nu=5.0
    )

    collector_store: dict[str, object] = {}

    class RodMeshCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip

            # existing
            self.time = []
            self.mesh_position = []
            self.mesh_director = []
            self.rod_position = []

            # NEW: energies
            self.translational_energy = []
            self.rotational_energy = []
            self.bending_energy = []
            self.shear_energy = []
            self.potential_energy = []
            self.total_energy = []

            collector_store["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return

            self.time.append(time)
            self.mesh_position.append(mesh_body.position_collection[:, 0].copy())
            self.mesh_director.append(mesh_body.director_collection[:, :, 0].copy())
            self.rod_position.append(rod.position_collection.copy())

            # --- NEW: compute energies from rod API ---
            T_trans = float(rod.compute_translational_energy())
            T_rot = float(rod.compute_rotational_energy())
            U_bend = float(rod.compute_bending_energy())
            U_shear = float(rod.compute_shear_energy())

            # gravitational potential: U = - Σ m_i (g · r_i)
            g_dot_r = np.einsum("i,ij->j", gravity, rod.position_collection)
            U_grav = float(-(rod.mass * g_dot_r).sum())

            self.translational_energy.append(T_trans)
            self.rotational_energy.append(T_rot)
            self.bending_energy.append(U_bend)
            self.shear_energy.append(U_shear)
            self.potential_energy.append(U_grav)

            self.total_energy.append(T_trans + T_rot + U_bend + U_shear + U_grav)

    simulator.collect_diagnostics(rod).using(
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

    # NEW: energy plotting helper
    def _plot_energy_from_callback():
        cb = collector_store.get("cb")
        if cb is None or len(cb.time) == 0:
            return

        t = np.asarray(cb.time)

        fig = plt.figure()
        plt.plot(t, cb.translational_energy, label="Translational KE")
        plt.plot(t, cb.rotational_energy, label="Rotational KE")
        plt.plot(t, cb.bending_energy, label="Bending energy")
        plt.plot(t, cb.shear_energy, label="Shear energy")
        plt.plot(t, cb.potential_energy, label="Gravity PE")
        plt.plot(t, cb.total_energy, label="Total", linewidth=2.0)

        plt.xlabel("Time [s]")
        plt.ylabel("Energy [J]")
        plt.legend()
        plt.tight_layout()

        energy_png = output.replace(".mp4", "_graph.png")
        fig.savefig(energy_png, dpi=200)
        plt.close(fig)

        print(f"[mesh_rod_collision] Saved energy plot -> {energy_png}")

    try:
        simulator.finalize()
        timestepper = ea.PositionVerlet()
        total_steps = int(final_time / dt)
        ea.integrate(timestepper, simulator, final_time, total_steps)

        _render_from_callback()
        _plot_energy_from_callback()  # NEW

    except Exception as exc:  # noqa: BLE001
        try:
            _render_from_callback()
            _plot_energy_from_callback()
        except Exception as render_exc:  # noqa: BLE001
            print(f"[mesh_rod_collision] Post-processing failed after exception: {render_exc}")
        raise


if __name__ == "__main__":
    dt = 3.0e-5
    nu = 1e-2
    rod_mesh_collision(
        final_time=5.0,
        dt=dt,
        nu=nu,
        output=f"mesh_rod_collision_dt{dt}_nu{nu}.mp4",
        render_speed=1.0,
        render_fps=None,
    )
