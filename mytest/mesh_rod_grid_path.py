"""
Rod-mesh collision with a moving clamped end following a prescribed path.

Setup (per request):
- Mesh: fixed `mytest/grid.stl`.
- Rod: start (0, 0, 0.3), points +Y, length 2.0 m, 40 elements.
- Young's modulus: 1e6 Pa (shear modulus inferred with nu=0.5).
- dt: 2e-5 s.
- End motion: constant 0.5 m/s through given waypoints; orientation held fixed.

Run: `.venv/bin/python mytest/mesh_rod_grid_path.py`
"""

from __future__ import annotations

import numpy as np
import elastica as ea
from examples.MeshCase import post_processing

# Waypoints for the clamped end (world frame)
WAYPOINTS = np.array(
    [
        (0.0, 0.0, 0.3),
        (1.0, 0.1, 0.3),
        (1.0, -0.4, 0.3),
        (0.0, -0.5, 0.3),
        (0.0, -1.0, 0.5),
        (-1.0, -1.0, 0.5),
        (0.0, 0.0, 0.5),
        (-1.0, 0.5, 0.2),
        (-1.0, 1.0, 0.2),
    ],
    dtype=np.float64,
)


def _segment_data(waypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit directions and lengths for each segment of the polyline."""
    seg_vecs = np.diff(waypoints, axis=0)
    seg_len = np.linalg.norm(seg_vecs, axis=1)
    seg_dir = seg_vecs / seg_len[:, None]
    return seg_dir, seg_len


class MovingEndBC(ea.ConstraintBase):
    """Translate node 0 along the polyline at constant speed; keep orientation fixed."""

    def __init__(self, fixed_position, fixed_director, waypoints, speed, **kwargs):
        super().__init__(**kwargs)
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.speed = float(speed)
        self.seg_dir, self.seg_len = _segment_data(self.waypoints)
        self.cum_len = np.concatenate(([0.0], np.cumsum(self.seg_len)))
        self.total_len = self.cum_len[-1]
        self.fixed_director = np.asarray(fixed_director, dtype=np.float64)

    def _path_position(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and segment direction at arc-length s along the path."""
        s_clamped = np.clip(s, 0.0, self.total_len)
        idx = np.searchsorted(self.cum_len, s_clamped, side="right") - 1
        idx = min(idx, len(self.seg_len) - 1)
        local_s = s_clamped - self.cum_len[idx]
        pos = self.waypoints[idx] + local_s * self.seg_dir[idx]
        return pos, self.seg_dir[idx]

    def constrain_values(self, system, time: np.float64) -> None:
        s = self.speed * float(time)
        pos, _ = self._path_position(s)
        system.position_collection[..., 0] = pos
        system.director_collection[..., 0] = self.fixed_director

    def constrain_rates(self, system, time: np.float64) -> None:
        s = self.speed * float(time)
        _, seg_dir = self._path_position(s)
        velocity = self.speed * seg_dir
        system.velocity_collection[..., 0] = velocity
        system.omega_collection[..., 0] = 0.0


def mesh_rod_grid_path_sim(
    dt: float = 1.0e-5,
    rod_length: float = 2.0,
    n_elem: int = 40,
    speed: float = 1.0,
    youngs_modulus: float = 1.0e7,
    output: str = "mesh_rod_grid_path.mp4",
    render_fps: int | None = 50,
    render_speed: float = 1.0,
):
    class RodMeshSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.CallBacks,
        ea.Damping,
    ):
        pass

    simulator = RodMeshSim()

    # --- Rod ---
    rod_radius = 0.01
    density = 1200.0
    shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))  # nu=0.5
    start = np.array([0.0, 0.0, 0.3])
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])

    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=start,
        direction=direction,
        normal=normal,
        base_length=rod_length,
        base_radius=rod_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    simulator.append(rod)

    simulator.constrain(rod).using(
        MovingEndBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        waypoints=WAYPOINTS,
        speed=speed,
    )

    gravity = np.array([0.0, 0.0, -9.81])
    simulator.add_forcing_to(rod).using(ea.GravityForces, acc_gravity=gravity)

    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=5e-2,
        time_step=dt,
    )

    # --- Mesh (fixed) ---
    mesh = ea.Mesh(
        "mytest/grid_tight.stl",
    )
    density_mesh = 1.0
    volume = 1.0
    inertia = mesh.compute_inertia_tensor(density=density_mesh)
    mesh_body = ea.MeshRigidBody(
        mesh=mesh,
        mass_second_moment_of_inertia=inertia,
        density=density_mesh,
        volume=volume,
    )
    simulator.append(mesh_body)

    simulator.detect_contact_between(rod, mesh_body).using(
        ea.RodMeshContact,
        k=1e4,
        nu=5.0,
        mesh_frozen=True,
    )

    collector: dict[str, object] = {}

    class RodMeshCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time = []
            self.mesh_position = []
            self.mesh_director = []
            self.rod_position = []
            collector["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.mesh_position.append(mesh_body.position_collection[:, 0].copy())
            self.mesh_director.append(mesh_body.director_collection[:, :, 0].copy())
            self.rod_position.append(rod.position_collection.copy())

    fps_for_skip = render_fps if render_fps is not None else 30
    step_skip = max(1, int(1.0 / (fps_for_skip * dt)))
    simulator.collect_diagnostics(rod).using(RodMeshCallback, step_skip=step_skip)

    _, seg_len = _segment_data(WAYPOINTS)
    total_path_time = float(np.sum(seg_len) / speed)
    final_time = total_path_time + 1.0  # buffer after motion
    # final_time = 1.0
    total_steps = int(final_time / dt)

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    ea.integrate(timestepper, simulator, final_time, total_steps)

    cb = collector.get("cb")
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
        bounds=((-2.0, 2.0), (-2.0, 2.0), (-0.5, 2.5)),
        rod_positions=cb.rod_position,
    )


if __name__ == "__main__":
    mesh_rod_grid_path_sim()
