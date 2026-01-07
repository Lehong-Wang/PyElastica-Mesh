import numpy as np
import elastica as ea
from examples.RigidBodyCases.RodRigidBodyContact.post_processing import (
    plot_velocity,
    plot_video_with_surface,
)


def cylinder_move_towards_rod(inclination_angle: float = 0.0) -> None:
    class RodCylinderMoveContact(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Contact,
        ea.CallBacks,
        ea.Forcing,
        ea.Damping,
    ):
        pass

    simulator = RodCylinderMoveContact()

    final_time = 6.0
    time_step = 5e-4
    total_steps = int(final_time / time_step) + 1
    rendering_fps = 30
    step_skip = int(1.0 / (rendering_fps * time_step))

    rod_length = 0.8
    rod_radius = 0.01
    cylinder_height = 0.2
    cylinder_radius = 0.1
    density = 1750
    youngs_modulus = 3e5
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))
    n_elem = 50

    start = np.array([0.2, 0.0, 0.5])
    direction = np.array([0.0, 0.0, -1.0])
    normal = np.array([0.0, 1.0, 0.0])

    rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        rod_length,
        rod_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )

    simulator.append(rod)

    cylinder_start = np.array([0.0, 0.0, 0.0])
    cylinder_direction = np.array([0.0, 0.0, 1.0])
    cylinder_normal = np.array([0.0, 1.0, 0.0])

    rigid_body = ea.Cylinder(
        start=cylinder_start,
        direction=cylinder_direction,
        normal=cylinder_normal,
        base_length=cylinder_height,
        base_radius=cylinder_radius,
        density=density,
    )
    rigid_body.velocity_collection[:, 0] = np.array([0.1, 0.0, 0.0])

    simulator.append(rigid_body)

    simulator.detect_contact_between(rod, rigid_body).using(
        ea.RodCylinderContact,
        k=5e4,
        nu=0.1,
    )

    simulator.constrain(rod).using(
        ea.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    damping_constant = 1e-2
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    post_processing_dict_list = []

    class StraightRodCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict):
            super().__init__()
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["radius"].append(system.radius.copy())
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
                )
                if current_step == 0:
                    self.callback_params["lengths"].append(system.rest_lengths.copy())
                else:
                    self.callback_params["lengths"].append(system.lengths.copy())
                self.callback_params["com_velocity"].append(
                    system.compute_velocity_center_of_mass()
                )
                total_energy = (
                    system.compute_translational_energy()
                    + system.compute_rotational_energy()
                )
                self.callback_params["total_energy"].append(total_energy.copy())

    class RigidCylinderCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict, resize_cylinder_elems):
            super().__init__()
            self.every = step_skip
            self.callback_params = callback_params
            self.n_elem_cylinder = resize_cylinder_elems
            self.n_node_cylinder = resize_cylinder_elems + 1

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)

                cylinder_center_position = system.position_collection[..., 0].reshape(
                    3, 1
                )
                cylinder_length = system.length
                cylinder_direction = system.director_collection[2, :, :].reshape(3, 1)
                cylinder_radius = system.radius

                start_position = (
                    cylinder_center_position - cylinder_length / 2 * cylinder_direction
                )

                cylinder_position_collection = (
                    start_position
                    + np.linspace(0, cylinder_length, self.n_node_cylinder)
                    * cylinder_direction
                )
                cylinder_radius_collection = (
                    np.ones((self.n_elem_cylinder)) * cylinder_radius
                )
                cylinder_length_collection = (
                    np.ones((self.n_elem_cylinder)) * cylinder_length
                )
                cylinder_velocity_collection = (
                    np.ones((self.n_node_cylinder)) * system.velocity_collection
                )

                self.callback_params["position"].append(
                    cylinder_position_collection.copy()
                )
                self.callback_params["velocity"].append(
                    cylinder_velocity_collection.copy()
                )
                self.callback_params["radius"].append(cylinder_radius_collection.copy())
                self.callback_params["com"].append(
                    system.compute_position_center_of_mass()
                )
                self.callback_params["lengths"].append(
                    cylinder_length_collection.copy()
                )
                self.callback_params["com_velocity"].append(
                    system.velocity_collection[..., 0].copy()
                )
                total_energy = (
                    system.compute_translational_energy()
                    + system.compute_rotational_energy()
                )
                self.callback_params["total_energy"].append(total_energy[..., 0].copy())

    post_processing_dict_list.append(ea.defaultdict(list))
    simulator.collect_diagnostics(rod).using(
        StraightRodCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[0],
    )

    post_processing_dict_list.append(ea.defaultdict(list))
    simulator.collect_diagnostics(rigid_body).using(
        RigidCylinderCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[1],
        resize_cylinder_elems=n_elem,
    )

    simulator.finalize()
    timestepper = ea.PositionVerlet()

    ea.integrate(timestepper, simulator, final_time, total_steps)

    plot_video_with_surface(
        post_processing_dict_list,
        video_name="cylinder_move_rod.mp4",
        fps=rendering_fps,
        step=1,
        x_limits=(-0.5, 1.0),
        y_limits=(-0.5, 0.5),
        z_limits=(-1.0, 1.0),
        dpi=100,
        vis3D=True,
        vis2D=True,
    )

    plot_velocity(
        post_processing_dict_list[0],
        post_processing_dict_list[1],
        filename="cylinder_move_rod_velocity.png",
        SAVE_FIGURE=True,
    )


if __name__ == "__main__":
    cylinder_move_towards_rod()
