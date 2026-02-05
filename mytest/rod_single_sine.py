"""
Single Cosserat rod with one end fixed, base at (0, 0, 0.5), length 1 m, 10 elements.
The fixed end oscillates along +Y with a sinusoidal displacement. Gravity and damping applied.
Saves state to npz; rendering optional via post_processing.
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

import numpy as np
import elastica as ea
from render_scripts import post_processing as pp


class SineBaseBC(ea.ConstraintBase):
    """
    One-end boundary condition with sinusoidal z-translation of the base node
    and sinusoidal rotation about the global Y-axis (swing +X ↔ ±Z).
    """

    def __init__(
        self,
        fixed_position: np.ndarray,
        fixed_directors: np.ndarray,
        amp: float,
        freq: float,
        rot_amp: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p0 = np.asarray(fixed_position, dtype=float)
        self.R0 = np.asarray(fixed_directors, dtype=float)
        self.amp = float(amp)
        self.freq = float(freq)
        self.rot_amp = float(rot_amp)

    @staticmethod
    def _rot_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def constrain_values(self, system, time: np.float64) -> None:
        disp = self.amp * np.sin(2.0 * np.pi * self.freq * float(time))
        theta = self.rot_amp * np.sin(2.0 * np.pi * self.freq * float(time))
        R = self._rot_y(theta) @ self.R0
        system.position_collection[..., 0] = self.p0 + np.array([0.0, 0.0, disp])
        system.director_collection[..., 0] = R

    def constrain_rates(self, system, time: np.float64) -> None:
        omega = 2.0 * np.pi * self.freq
        vel_z = omega * self.amp * np.cos(2.0 * np.pi * self.freq * float(time))
        theta_dot = omega * self.rot_amp * np.cos(2.0 * np.pi * self.freq * float(time))
        system.velocity_collection[..., 0] = np.array([0.0, 0.0, vel_z])
        system.omega_collection[..., 0] = np.array([0.0, theta_dot, 0.0])


def run_single_sine(
    final_time: float = 10.0,
    dt: float = 2.0e-5,
    n_elem: int = 20,
    base_length: float = 1.0,
    base_radius: float = 2.5e-3,
    density: float = 1_000.0,
    youngs_modulus: float = 1e6,
    shear_modulus_ratio: float = 1.5,
    damping_constant: float = 1e-2,
    sine_amp: float = 0.05,
    sine_freq: float = 1.0,
    rot_amp: float = np.pi / 2,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "rod_single_sine",
    output_interval: float = 0.01,
    render: bool = False,
    render_speed: float = 1.0,
    render_fps: int | None = 30,
) -> dict[str, object]:
    """
    Simulate a single rod with sinusoidally translating base.
    """

    class SingleSineSim(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Contact, ea.CallBacks, ea.Damping
    ):
        """Simulation container for single rod with moving base and plane contact."""

    simulator = SingleSineSim()

    start = np.array([0.0, 0.0, 0.2])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])

    rod = ea.CosseratRod.straight_rod(
        n_elements=n_elem,
        start=start,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=youngs_modulus / (2.0 * shear_modulus_ratio),
    )
    simulator.append(rod)

    simulator.add_forcing_to(rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
    )
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )
    simulator.constrain(rod).using(
        SineBaseBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        amp=sine_amp,
        freq=sine_freq,
        rot_amp=rot_amp,
    )

    # Plane contact at z=0
    plane = ea.Plane(plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]))
    simulator.append(plane)
    simulator.detect_contact_between(rod, plane).using(
        ea.RodPlaneContact, k=1e4, nu=5.0
    )
    simulator.add_forcing_to(rod).using(
        ea.AnisotropicFrictionalPlane,
        k=1e4,
        nu=5.0,
        plane_origin=np.zeros(3),
        plane_normal=np.array([0.0, 0.0, 1.0]),
        slip_velocity_tol=1e-6,
        static_mu_array=np.array([1.0, 1.0, 1.0]),
        kinetic_mu_array=np.array([0.5, 0.5, 0.5]),
    )

    collector: dict[str, RodCallback] = {}

    class RodCallback(ea.CallBackBaseClass):
        def __init__(self, step_skip: int):
            super().__init__()
            self.step_skip = step_skip
            self.time: list[float] = []
            self.position: list[np.ndarray] = []
            self.director: list[np.ndarray] = []
            collector["cb"] = self

        def make_callback(self, system, time, current_step):
            if current_step % self.step_skip:
                return
            self.time.append(time)
            self.position.append(system.position_collection.copy())
            self.director.append(system.director_collection.copy())

    step_skip = max(1, int(np.round(output_interval / dt)))
    simulator.collect_diagnostics(rod).using(RodCallback, step_skip=step_skip)

    simulator.finalize()
    timestepper: ea.typing.StepperProtocol = ea.PositionVerlet()
    total_steps = int(np.ceil(final_time / dt))
    ea.integrate(timestepper, simulator, final_time, total_steps)

    cb = collector["cb"]
    time_arr = np.asarray(cb.time)
    pos_arr = np.asarray(cb.position)
    dir_arr = np.asarray(cb.director)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{output_name}_state.npz"
    np.savez(
        state_path,
        time=time_arr,
        position=pos_arr,
        director=dir_arr,
        dt=dt,
        final_time=final_time,
        sine_amp=sine_amp,
        sine_freq=sine_freq,
        rot_amp=rot_amp,
    )

    video_path = output_dir / f"{output_name}_4view.mp4"
    if render:
        pp.plot_rods_multiview(
            pos_arr,
            video_path=video_path,
            times=time_arr,
            fps=render_fps,
            speed=render_speed,
            plane_z=0.0,
            colors=["#1f77b4"],
        )
    else:
        video_path = None

    return {
        "state_path": state_path,
        "video_path": video_path,
        "time": time_arr,
        "position": pos_arr,
        "director": dir_arr,
    }


if __name__ == "__main__":
    results = run_single_sine(render=True)
    print(f"Saved npz to {results['state_path']} (render={bool(results['video_path'])}).")
