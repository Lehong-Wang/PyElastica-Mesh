import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import elastica as ea
from examples.RigidBodyCases.RodRigidBodyContact.post_processing import (
    plot_video_with_surface,
)


def simulate_mesh_drop(
    plot_com: bool = True,
    plot_com_3d: bool = True,
    plot_video_points: bool = True,
    plot_video_surface: bool = True,
) -> None:
    class MeshDropSimulator(ea.BaseSystemCollection, ea.Forcing, ea.CallBacks):
        pass

    simulator = MeshDropSimulator()

    # Load mesh and create rigid body
    mesh = ea.Mesh("tests/cube.stl")
    center_of_mass = np.array([0.0, 0.0, 1.0])

    cube_edge = 2.0
    volume = cube_edge**3
    density = 1.0
    mass = density * volume

    inertia = np.zeros((3, 3), dtype=np.float64)
    np.fill_diagonal(inertia, (mass * cube_edge**2) / 6.0)

    rigid_body = ea.MeshRigidBody(
        mesh,
        center_of_mass,
        inertia,
        density,
        volume,
    )

    rigid_body.velocity_collection[:, 0] = np.array([5, 0.0, 0.0])
    rigid_body.omega_collection[:, 0] = np.array([2.0, 0.0, 0.0])

    simulator.append(rigid_body)

    # gravity = np.array([0.0, 0.0, -9.81])
    # simulator.add_forcing_to(rigid_body).using(ea.GravityForces, acc_gravity=gravity)

    class RigidBodyCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict, store_faces: bool):
            super().__init__()
            self.every = step_skip
            self.callback_params = callback_params
            self.store_faces = store_faces

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                system.update_faces()
                self.callback_params["time"].append(time)
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
                )
                self.callback_params["position"].append(system.face_centers.copy())
                self.callback_params["radius"].append(
                    np.full(system.n_faces, 0.02, dtype=np.float64)
                )
                if self.store_faces:
                    self.callback_params["faces"].append(system.faces.copy())
                    self.callback_params["director"].append(
                        system.director_collection.copy()
                    )

    dt = 1e-3
    final_time = 3.0
    total_steps = int(final_time / dt)
    step_skip = 10

    data = ea.defaultdict(list)
    simulator.collect_diagnostics(rigid_body).using(
        RigidBodyCallback,
        step_skip=step_skip,
        callback_params=data,
        store_faces=plot_video_surface,
    )

    simulator.finalize()
    ea.integrate(ea.PositionVerlet(), simulator, final_time, total_steps)

    times = np.array(data["time"])
    com = np.array(data["com"])

    if plot_com:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot(times, com[:, 0], label="x")
        ax.plot(times, com[:, 2], label="z")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("center of mass [m]")
        ax.legend()
        fig.tight_layout()
        fig.savefig("mesh_cube_fall_com.png")

    if plot_com_3d:
        fig = plt.figure(figsize=(8, 6), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(com[:, 0], com[:, 1], com[:, 2], label="COM path")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        fig.tight_layout()
        fig.savefig("mesh_cube_fall_com_3d.png")

    if plot_video_points:
        plot_video_with_surface(
            [data],
            video_name="mesh_cube_fall.mp4",
            fps=30,
            step=1,
            x_limits=(-3.0, 3.0),
            y_limits=(-3.0, 3.0),
            z_limits=(-3.0, 3.0),
            dpi=100,
            vis3D=True,
            vis2D=False,
        )

    if plot_video_surface and data["faces"]:
        faces_history = data["faces"]
        director_history = data["director"]
        com_history = data["com"]
        fig_3d = plt.figure(figsize=(8, 6), dpi=150)
        ax_3d = fig_3d.add_subplot(111, projection="3d")
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("z")

        max_extent = 2.5
        ax_3d.set_xlim(-max_extent, max_extent)
        ax_3d.set_ylim(-max_extent, max_extent)
        ax_3d.set_zlim(-max_extent, max_extent)

        def faces_to_triangles(faces):
            triangles = []
            for i in range(faces.shape[-1]):
                triangles.append(faces[:, :, i].T)
            return triangles

        poly = Poly3DCollection(
            faces_to_triangles(faces_history[0]), alpha=0.6, facecolor="tab:blue"
        )
        ax_3d.add_collection3d(poly)

        writer = animation.writers["ffmpeg"](fps=30, metadata={"artist": "pyelastica"})
        axis_len = 0.6
        com0 = com_history[0].reshape(3)
        d0 = director_history[0].reshape(3, 3)
        x_line, = ax_3d.plot(
            [com0[0], com0[0] + axis_len * d0[0, 0]],
            [com0[1], com0[1] + axis_len * d0[0, 1]],
            [com0[2], com0[2] + axis_len * d0[0, 2]],
            color="r",
        )
        y_line, = ax_3d.plot(
            [com0[0], com0[0] + axis_len * d0[1, 0]],
            [com0[1], com0[1] + axis_len * d0[1, 1]],
            [com0[2], com0[2] + axis_len * d0[1, 2]],
            color="g",
        )
        z_line, = ax_3d.plot(
            [com0[0], com0[0] + axis_len * d0[2, 0]],
            [com0[1], com0[1] + axis_len * d0[2, 1]],
            [com0[2], com0[2] + axis_len * d0[2, 2]],
            color="b",
        )
        with writer.saving(fig_3d, "mesh_cube_fall_surface.mp4", dpi=150):
            for faces, director, com in zip(
                faces_history, director_history, com_history
            ):
                poly.remove()
                poly = Poly3DCollection(
                    faces_to_triangles(faces), alpha=0.6, facecolor="tab:blue"
                )
                ax_3d.add_collection3d(poly)
                com = com.reshape(3)
                director = director.reshape(3, 3)
                x_line.set_data(
                    [com[0], com[0] + axis_len * director[0, 0]],
                    [com[1], com[1] + axis_len * director[0, 1]],
                )
                x_line.set_3d_properties(
                    [com[2], com[2] + axis_len * director[0, 2]]
                )
                y_line.set_data(
                    [com[0], com[0] + axis_len * director[1, 0]],
                    [com[1], com[1] + axis_len * director[1, 1]],
                )
                y_line.set_3d_properties(
                    [com[2], com[2] + axis_len * director[1, 2]]
                )
                z_line.set_data(
                    [com[0], com[0] + axis_len * director[2, 0]],
                    [com[1], com[1] + axis_len * director[2, 1]],
                )
                z_line.set_3d_properties(
                    [com[2], com[2] + axis_len * director[2, 2]]
                )
                writer.grab_frame()


if __name__ == "__main__":
    simulate_mesh_drop(
        plot_com=True,
        plot_com_3d=True,
        plot_video_points=True,
        plot_video_surface=True,
    )
