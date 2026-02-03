"""
Drop five Cosserat rods onto the xy-plane with a cube mesh obstacle and render
four synchronized views (3D + front/right/top).

The rod setup mirrors ``rod_drop_multi.py`` while adding rod–mesh contact using
``mytest/cube_tight.stl``. State is saved to NPZ and a four-view MP4 is
generated for quick inspection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure matplotlib can write its cache in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

# Make the repository importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

# Headless rendering must be configured before importing pyplot.
matplotlib.use("Agg")
import numpy as np

import elastica as ea
from render_scripts import post_processing as pp


def run_rod_drop_cube(
    final_time: float = 1.5,
    dt: float = 1.0e-5,
    n_elem: int = 20,
    base_length: float = 0.7,
    base_radius: float = 3.0e-3,
    density: float = 1_200.0,
    youngs_modulus: float = 5e6,
    shear_modulus_ratio: float = 1.5,
    contact_k: float = 2.0e5,
    contact_nu: float = 10.0,
    friction_coefficient: float = 1.0,
    damping_constant: float = 1e-2,
    num_rods: int = 5,
    height_gap: float = 0.05,
    lateral_jitter: float = 0.04,
    cube_path: str | Path = Path("mytest/cube_tight.stl"),
    cube_density: float = 5_000.0,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_drop_cube",
    output_interval: float = 0.01,
    seed: int | None = None,
    render_speed: float = 1.0,
    render_fps: int | None = None,
) -> dict[str, object]:
    """
    Simulate five free-falling rods impacting a plane and a cube mesh obstacle.

    Returns a dictionary with recorded arrays and output file paths.
    """

    class RodCubeSim(
        ea.BaseSystemCollection, ea.Forcing, ea.Contact, ea.CallBacks, ea.Damping
    ):
        """Simulation container for rod + cube drop."""

    simulator = RodCubeSim()

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Create rods
    # ------------------------------------------------------------------
    base_start = np.array([0.0, 0.0, 0.18])
    direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    normal = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)

    rods: list[ea.CosseratRod] = []
    for i in range(num_rods):
        start_i = base_start + np.array([0.0, 0.0, i * height_gap])
        jitter_pos = (rng.random(3) - 0.5) * lateral_jitter  # ±lateral_jitter/2
        jitter_pos[2] = (rng.random() - 0.5) * 0.02
        start_i = start_i + jitter_pos

        rot_angle = (rng.random() - 0.5) * 0.2  # ±0.1 rad about z
        rot_mat = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle), 0.0],
                [np.sin(rot_angle), np.cos(rot_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dir_i = rot_mat @ direction
        norm_i = rot_mat @ normal

        rod = ea.CosseratRod.straight_rod(
            n_elements=n_elem,
            start=start_i,
            direction=dir_i,
            normal=norm_i,
            base_length=base_length,
            base_radius=base_radius,
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=youngs_modulus / (2.0 * shear_modulus_ratio),
        )
        simulator.append(rod)
        rods.append(rod)

    # Plane at z = 0
    plane = ea.Plane(plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]))
    simulator.append(plane)

    # ------------------------------------------------------------------
    # Cube mesh obstacle (per-rod cube, separately loaded per rod)
    # ------------------------------------------------------------------
    cubes: list[ea.MeshRigidBody] = []
    primary_cube_mesh: ea.Mesh | None = None

    # ------------------------------------------------------------------
    # Forces, contact, damping
    # ------------------------------------------------------------------
    static_mu = np.array([
        friction_coefficient * 2.0,
        friction_coefficient * 2.0,
        friction_coefficient * 2.0,
    ])
    kinetic_mu = np.array([
        friction_coefficient,
        friction_coefficient,
        friction_coefficient,
    ])

    for ridx, rod in enumerate(rods):
        simulator.add_forcing_to(rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
        )

        simulator.dampen(rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=dt,
        )

        simulator.detect_contact_between(rod, plane).using(
            ea.RodPlaneContactWithAnisotropicFriction,
            k=10.0,
            nu=1e-4,
            slip_velocity_tol=1e-6,
            static_mu_array=static_mu,
            kinetic_mu_array=kinetic_mu,
        )

        # simulator.detect_contact_between(rod, plane).using(
        #     ea.RodPlaneContact, k=contact_k, nu=contact_nu,
        # )

        # simulator.add_forcing_to(rod).using(
        #     ea.AnisotropicFrictionalPlane,
        #     k=contact_k,
        #     nu=contact_nu,
        #     plane_origin=np.zeros(3),
        #     plane_normal=np.array([0.0, 0.0, 1.0]),
        #     slip_velocity_tol=1e-6,
        #     static_mu_array=static_mu,
        #     kinetic_mu_array=kinetic_mu,
        # )

        # Dedicated cube for this rod; separately load the mesh to give each its own ray scene.
        cube_mesh = ea.Mesh(str(cube_path))
        cube_volume = cube_mesh.compute_volume()
        cube_inertia = cube_mesh.compute_inertia_tensor(density=cube_density)
        cube_height = float(cube_mesh.obb.extent[2])

        cube_body = ea.MeshRigidBody(
            mesh=cube_mesh,
            density=cube_density,
            volume=cube_volume,
            mass_second_moment_of_inertia=cube_inertia,
        )
        if primary_cube_mesh is None:
            primary_cube_mesh = cube_mesh
        cube_body.position_collection[:, 0] = np.array(
            [
                rod.position_collection[0, 0] + 0.3,
                rod.position_collection[1, 0],
                0.5 * cube_height,
            ]
        )
        simulator.append(cube_body)
        cubes.append(cube_body)

        simulator.detect_contact_between(rod, cube_body).using(
            ea.RodMeshContact, k=contact_k, nu=contact_nu, mesh_frozen=True
        )

        # simulator.detect_contact_between(rod, rod).using(
        #     ea.RodSelfContact, k=1e4, nu=10)

    # Pairwise rod-rod contact to prevent interpenetration.
    for i in range(len(rods)):
        for j in range(i + 1, len(rods)):
            simulator.detect_contact_between(rods[i], rods[j]).using(
                ea.RodRodContact, k=contact_k, nu=contact_nu
            )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    collector: dict[str, ea.CallBackBaseClass] = {}

    class RodCallback(ea.CallBackBaseClass):
        def __init__(self, name: str, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.position: list[np.ndarray] = []
            self.director: list[np.ndarray] = []
            collector[name] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    class CubeCallback(ea.CallBackBaseClass):
        def __init__(self, name: str, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.position: list[np.ndarray] = []
            self.director: list[np.ndarray] = []
            collector[name] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    step_skip = max(1, int(output_interval / dt))
    rod_names: list[str] = []
    for idx, rod in enumerate(rods):
        name = f"rod_{idx}"
        simulator.collect_diagnostics(rod).using(
            RodCallback, name=name, step_skip=step_skip
        )
        rod_names.append(name)

    cube_names: list[str] = []
    if cubes:
        simulator.collect_diagnostics(cubes[0]).using(
            CubeCallback, name="cube_primary", step_skip=step_skip
        )
        cube_names.append("cube_primary")

    # ------------------------------------------------------------------
    # Integrate
    # ------------------------------------------------------------------
    simulator.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))

    def _assemble_outputs():
        rod_callbacks: list[RodCallback] = [collector[name] for name in rod_names]
        cube_callbacks: list[CubeCallback] = [collector[name] for name in cube_names] if cube_names else []

        if len(rod_callbacks) == 0:
            return None

        n_rods = len(rod_callbacks)
        n_elems = rods[0].director_collection.shape[2]
        n_nodes = n_elems + 1

        if len(rod_callbacks[0].time) > 0:
            time_arr = np.asarray(rod_callbacks[0].time)
            pos_arr = np.stack([np.asarray(cb.position) for cb in rod_callbacks], axis=1)
            dir_arr = np.stack([np.asarray(cb.director) for cb in rod_callbacks], axis=1)
        else:
            time_arr = np.zeros((0,), dtype=float)
            pos_arr = np.zeros((0, n_rods, 3, n_nodes), dtype=float)
            dir_arr = np.zeros((0, n_rods, 3, 3, n_elems), dtype=float)

        if cube_callbacks and len(cube_callbacks[0].time) > 0:
            cube_pos = np.asarray(cube_callbacks[0].position)  # (T,3,1)
            cube_dir = np.asarray(cube_callbacks[0].director)  # (T,3,3,1)
        else:
            cube_pos = np.zeros((0, 3, 1), dtype=float)
            cube_dir = np.zeros((0, 3, 3, 1), dtype=float)

        return time_arr, pos_arr, dir_arr, cube_pos, cube_dir

    try:
        ea.integrate(timestepper, simulator, final_time, total_steps)
    except Exception as exc:  # noqa: BLE001
        print(f"[rod_drop_cube] Simulation failed early: {exc}")
    finally:
        outputs = _assemble_outputs()

    if outputs is None:
        raise RuntimeError("No callbacks collected; cannot assemble outputs.")

    time_arr, pos_arr, dir_arr, cube_pos, cube_dir = outputs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / f"{output_name}_state.npz"
    seed_val = np.int64(-1 if seed is None else int(seed))

    np.savez(
        state_path,
        time=time_arr,
        position=pos_arr,
        director=dir_arr,
        cube_position=cube_pos,
        cube_director=cube_dir,
        dt=dt,
        final_time=final_time,
        seed=seed_val,
    )

    colors = pp._color_cycle(num_rods)
    video_path_four = output_dir / f"{output_name}_4view.mp4"

    if time_arr.shape[0] > 0:
        if primary_cube_mesh is not None and cube_pos.shape[0] > 0:
            cube_dict = {
                "mesh": primary_cube_mesh.mesh,
                "position": np.squeeze(cube_pos, axis=-1),  # (T,3)
                "director": np.squeeze(cube_dir, axis=-1),  # (T,3,3)
                "time": time_arr,
            }
            pp.plot_rods_with_mesh_multiview(
                cube_dict,
                pos_arr,
                video_path=video_path_four,
                times=time_arr,
                fps=render_fps,
                speed=render_speed,
                plane_z=0.0,
                colors=colors,
            )
        else:
            pp.plot_rods_multiview(
                pos_arr,
                video_path=video_path_four,
                times=time_arr,
                fps=render_fps,
                speed=render_speed,
                plane_z=0.0,
                colors=colors,
            )
    else:
        print("[rod_drop_cube] No frames collected; skipping video render.")

    return {
        "state_path": state_path,
        "video_path_four": video_path_four,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr,
        "cube_position": cube_pos,
        "cube_director": cube_dir,
        "colors": colors,
    }


if __name__ == "__main__":
    results = run_rod_drop_cube(
        final_time=1.0,
        dt=1.0e-5,
        base_radius=2.5e-3,
        damping_constant=1e-1,
        youngs_modulus=5e6,
        contact_k=1e4,
        contact_nu=10.0,
        friction_coefficient=1.2,
        height_gap=0.04,
        seed=123,
        num_rods=5,
    )
    print(
        f"Saved npz to {results['state_path']} and four-view video to "
        f"{results['video_path_four']} (num_rods={len(results['colors'])})."
    )
