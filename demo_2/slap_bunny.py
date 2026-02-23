"""Rod slaps a bunny mesh with delayed gravity on the bunny."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import numpy as np

import elastica as ea
from examples.MeshCase import post_processing

INCH_TO_METER = 0.0254


class DelayedGravity(ea.NoForces):
    """Apply gravity only after a selected simulation time."""

    def __init__(self, acc_gravity: np.ndarray, switch_on_time: float) -> None:
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


def _prepare_video_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    available_writers = set(animation.writers.list())
    if "ffmpeg" in available_writers:
        return output
    if "pillow" in available_writers:
        # Reuse the multiview pipeline while writing GIF if ffmpeg is missing.
        animation.writers._registered["ffmpeg"] = animation.writers["pillow"]  # type: ignore[attr-defined]
        gif_path = output.with_suffix(".gif")
        print(f"[slap_bunny] ffmpeg unavailable; falling back to GIF: {gif_path}")
        return gif_path
    raise RuntimeError("No supported writer found. Install ffmpeg or pillow.")


def run_slap_bunny(
    mesh_path: str = "demo_2/bunny_low_small.stl",
    output: str = "demo_2/slap_bunny_4view.mp4",
    final_time: float = .7,
    dt: float = 2.0e-5,
    n_elem: int = 20,
    rod_density: float = 1000.0,
    bunny_density: float = 800.0,
    rod_youngs_modulus: float = 1.0e6,
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 0.05,
    contact_k: float = 1.0e3,
    contact_nu: float = 2.0,
    contact_velocity_damping: float = 0.0,
    contact_friction: float = 0.0,
    bunny_gravity_switch_on: float = 0.35,
    render_speed: float = 1.0,
    render_fps: int | None = 50,
) -> Path:
    class SlapBunnySim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        pass

    simulator = SlapBunnySim()

    rod_length = 0.3
    rod_diameter = 0.5 * INCH_TO_METER
    rod_radius = 0.5 * rod_diameter

    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=rod_length,
        base_radius=rod_radius,
        density=rod_density,
        youngs_modulus=rod_youngs_modulus,
        shear_modulus=rod_youngs_modulus / (2.0 * shear_modulus_ratio),
    )
    simulator.append(rod)
    simulator.constrain(rod).using(
        ea.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    mesh = ea.Mesh(mesh_path)
    volume = mesh.compute_volume()
    inertia = mesh.compute_inertia_tensor(density=bunny_density)
    bunny = ea.MeshRigidBody(
        mesh=mesh,
        center_of_mass=np.array([0.1, 0.0, -0.2]),
        mass_second_moment_of_inertia=inertia,
        density=bunny_density,
        volume=volume,
    )
    simulator.append(bunny)

    gravity = np.array([0.0, 0.0, -9.81])
    simulator.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=gravity)
    simulator.add_forcing_to(bunny).using(
        DelayedGravity, acc_gravity=gravity, switch_on_time=bunny_gravity_switch_on
    )

    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

    simulator.detect_contact_between(rod, bunny).using(
        ea.RodMeshContact,
        k=contact_k,
        nu=contact_nu,
        velocity_damping_coefficient=contact_velocity_damping,
        friction_coefficient=contact_friction,
        mesh_frozen=False,
    )

    collector_store: dict[str, object] = {}

    class SlapCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.rod_position: list[np.ndarray] = []
            self.bunny_position: list[np.ndarray] = []
            self.bunny_director: list[np.ndarray] = []
            collector_store["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(float(time))
            self.rod_position.append(system.position_collection.copy())
            self.bunny_position.append(bunny.position_collection[:, 0].copy())
            self.bunny_director.append(bunny.director_collection[:, :, 0].copy())

    simulator.collect_diagnostics(rod).using(
        SlapCallBack, step_skip=max(1, int(0.01 / dt))
    )

    output_path = _prepare_video_path(Path(output))

    def _render_from_callback() -> None:
        cb = collector_store.get("cb")
        if cb is None or len(cb.time) == 0:
            return

        rod_pos = np.asarray(cb.rod_position)
        bunny_pos = np.asarray(cb.bunny_position)
        margin = max(0.15, 2.0 * float(bunny.radius))

        x_min = float(min(np.min(rod_pos[:, 0, :]), np.min(bunny_pos[:, 0]) - bunny.radius) - margin)
        x_max = float(max(np.max(rod_pos[:, 0, :]), np.max(bunny_pos[:, 0]) + bunny.radius) + margin)
        y_min = float(min(np.min(rod_pos[:, 1, :]), np.min(bunny_pos[:, 1]) - bunny.radius) - margin)
        y_max = float(max(np.max(rod_pos[:, 1, :]), np.max(bunny_pos[:, 1]) + bunny.radius) + margin)
        z_min = float(min(np.min(rod_pos[:, 2, :]), np.min(bunny_pos[:, 2]) - bunny.radius) - margin)
        z_max = float(max(np.max(rod_pos[:, 2, :]), np.max(bunny_pos[:, 2]) + bunny.radius) + margin)

        mesh_data = {
            "time": cb.time,
            "position": cb.bunny_position,
            "director": cb.bunny_director,
            "mesh": mesh.mesh,
        }
        post_processing.plot_mesh_multiview_animation(
            mesh_data,
            video_name=str(output_path),
            fps=render_fps,
            speed_factor=render_speed,
            rod_positions=cb.rod_position,
            bounds=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        )

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
            print(f"[slap_bunny] Rendering failed after exception: {render_exc}")
        raise

    return output_path


if __name__ == "__main__":
    video_path = run_slap_bunny()
    print(f"Saved four-view output to: {video_path}")
