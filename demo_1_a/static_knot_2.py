"""KnotCase reproduction with four-view rendering and tagged output naming."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Ensure matplotlib can write cache files in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KNOT_CASE_DIR = PROJECT_ROOT / "examples" / "KnotCase"
if str(KNOT_CASE_DIR) not in sys.path:
    sys.path.insert(0, str(KNOT_CASE_DIR))

import elastica as ea
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from knot_forcing import TargetPoseProportionalControl
from mytest.postprocessing import plot_rod_four_view_with_directors


class SoftRodSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


class KnotHistoryCallback(ea.CallBackBaseClass):
    """Record rod diagnostics exactly like KnotCase callback."""

    def __init__(self, callback_params: dict[str, list[Any]], every: int = 200) -> None:
        super().__init__()
        self.every = int(every)
        self.callback_params = callback_params

    def make_callback(self, system, time: float, current_step: int) -> None:
        if current_step % self.every != 0:
            return
        self.callback_params["time"].append(float(time))
        self.callback_params["step"].append(int(current_step))
        self.callback_params["radius"].append(system.radius.copy())
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["orientation"].append(system.director_collection.copy())


class MovingOtherEndBC(ea.ConstraintBase):
    """
    Keep one node anchored, and move the opposite end in x over a time window,
    while prescribing the moved end director (fixed or rotating by schedule).
    """

    def __init__(
        self,
        moving_end_position: np.ndarray,
        anchored_position: np.ndarray,
        moving_end_director: np.ndarray,
        move_start_time: float,
        move_end_time: float,
        move_dx: float,
        rotate_start_time: float,
        rotate_end_time: float,
        rotate_angular_rate: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.moving_end_position_0 = np.asarray(moving_end_position, dtype=np.float64).reshape(
            3
        )
        self.anchored_position_0 = np.asarray(anchored_position, dtype=np.float64).reshape(3)
        self.moving_end_director_0 = np.asarray(
            moving_end_director, dtype=np.float64
        ).reshape(3, 3)
        self.move_start_time = float(move_start_time)
        self.move_end_time = float(move_end_time)
        self.move_dx = float(move_dx)
        self.rotate_start_time = float(rotate_start_time)
        self.rotate_end_time = float(rotate_end_time)
        self.rotate_angular_rate = float(rotate_angular_rate)

        if self.constrained_position_idx.size != 2:
            raise ValueError(
                "MovingOtherEndBC expects constrained_position_idx=(-1, -20)."
            )
        if self.constrained_director_idx.size != 1:
            raise ValueError(
                "MovingOtherEndBC expects constrained_director_idx=(-1,)."
            )

        if self.move_end_time < self.move_start_time:
            raise ValueError("move_end_time must be >= move_start_time.")
        self.move_duration = self.move_end_time - self.move_start_time
        if self.rotate_end_time < self.rotate_start_time:
            raise ValueError("rotate_end_time must be >= rotate_start_time.")
        self.rotate_duration = self.rotate_end_time - self.rotate_start_time

        self.moving_end_idx = int(self.constrained_position_idx[0])
        self.anchored_idx = int(self.constrained_position_idx[1])
        self.moving_end_dir_idx = int(self.constrained_director_idx[0])

    def _motion_ratio(self, t: float) -> float:
        if t <= self.move_start_time:
            return 0.0
        if t >= self.move_end_time:
            return 1.0
        if self.move_duration <= 0.0:
            return 1.0
        return (t - self.move_start_time) / self.move_duration

    def _rotation_theta(self, t: float) -> float:
        if t <= self.rotate_start_time:
            return 0.0
        if t >= self.rotate_end_time:
            return self.rotate_angular_rate * self.rotate_duration
        return self.rotate_angular_rate * (t - self.rotate_start_time)

    def _director_with_rotation(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        d1 = self.moving_end_director_0[0]
        d2 = self.moving_end_director_0[1]
        d3 = self.moving_end_director_0[2]
        return np.array(
            [
                c * d1 + s * d2,
                -s * d1 + c * d2,
                d3,
            ],
            dtype=np.float64,
        )

    def constrain_values(self, system, time: np.float64) -> None:
        t = float(time)
        ratio = self._motion_ratio(t)
        target = self.moving_end_position_0.copy()
        target[0] += self.move_dx * ratio
        theta = self._rotation_theta(t)
        moving_end_director = self._director_with_rotation(theta)

        system.position_collection[..., self.moving_end_idx] = target
        system.position_collection[..., self.anchored_idx] = self.anchored_position_0
        system.director_collection[..., self.moving_end_dir_idx] = moving_end_director

    def constrain_rates(self, system, time: np.float64) -> None:
        t = float(time)
        vx = 0.0
        if self.move_start_time < t < self.move_end_time and self.move_duration > 0.0:
            vx = self.move_dx / self.move_duration

        system.velocity_collection[..., self.moving_end_idx] = 0.0
        system.velocity_collection[0, self.moving_end_idx] = vx
        system.velocity_collection[..., self.anchored_idx] = 0.0
        if self.rotate_start_time < t < self.rotate_end_time:
            current_director = system.director_collection[..., self.moving_end_dir_idx]
            d3_world = current_director[2]
            omega_world = self.rotate_angular_rate * d3_world
            system.omega_collection[..., self.moving_end_dir_idx] = (
                current_director @ omega_world
            )
        else:
            system.omega_collection[..., self.moving_end_dir_idx] = 0.0


def _fmt_tag(value: float) -> str:
    return format(float(value), ".6g")


def _make_unique_tag(output_dir: Path, base_tag: str) -> str:
    """Return unique tag by appending _i when output files already exist."""
    candidate = base_tag
    i = 1
    while True:
        state_path = output_dir / f"{candidate}_state.npz"
        video_mp4 = output_dir / f"{candidate}_4view.mp4"
        video_gif = output_dir / f"{candidate}_4view.gif"
        lwt_png = output_dir / f"{candidate}_LWT.png"
        if not (
            state_path.exists() or video_mp4.exists() or video_gif.exists() or lwt_png.exists()
        ):
            return candidate
        candidate = f"{base_tag}_{i}"
        i += 1


def simulate_static_knot_2(
    *,
    final_time: float = 50.0,
    dt: float = 2e-5,
    n_elem: int = 50,
    output_dir: Path | str = Path(__file__).resolve().parent,
    output_name: str = "static_knot_2",
    render_fps: int = 100,
    frame_stride: int = 1,
    callback_every: int = 500,
) -> dict[str, object]:
    """
    Same core setup/logic as examples/KnotCase/knot_simulation.py.
    Only output/rendering is adapted to project style.
    """
    simulator = SoftRodSimulator()
    recorded_history: dict[str, list[Any]] = defaultdict(list)

    # KnotCase parameters (exact defaults from the example).
    start = np.zeros((3,))
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 2.0
    base_radius = 0.0175
    density = 1070
    youngs_modulus = 4e6
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2.0 * (poisson_ratio + 1.0))

    stretchable_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    simulator.append(stretchable_rod)

    post_twist_angular_rate = 2.0
    motion_time = [2.0, 4.0, 6.0, 9.0, 45.0]
    other_end_move_dx = -0.1
    other_end_move_start = motion_time[1]
    other_end_move_end = motion_time[2]
    this_end_move_dx = 0.0
    this_end_move_start = motion_time[1]
    this_end_move_end = motion_time[2]
    both_end_rotate_start = motion_time[3]
    both_end_rotate_end = motion_time[4]
    post_twist_duration = both_end_rotate_end - both_end_rotate_start

    if post_twist_duration < 0.0:
        raise ValueError("motion_time must satisfy motion_time[4] >= motion_time[3].")

    base_orientation = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    def orientation_rotating_about_d3(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        d1 = base_orientation[0]
        d2 = base_orientation[1]
        d3 = base_orientation[2]
        return np.array(
            [
                c * d1 + s * d2,
                -s * d1 + c * d2,
                d3,
            ],
            dtype=np.float64,
        )

    def base_target(t: float, rod) -> tuple[np.ndarray, np.ndarray]:
        del rod  # Match example signature; rod is unused in this target trajectory.
        target_position = direction * base_length - 5.0 * base_radius * normal + 0.3
        R = 8.0
        final_phase2_position = np.array(
            [0.7, 0.0, 0.0],
            dtype=np.float64,
        )
        post_move_position = final_phase2_position.copy()
        post_move_position[0] += this_end_move_dx
        if t <= motion_time[0]:
            ratio = min(t / motion_time[0], 1.0)
            angular_ratio = ratio * np.pi * 2.0
            position = target_position * ratio
            orientation_twist = np.array(
                [
                    [0.0, np.cos(angular_ratio), np.sin(angular_ratio)],
                    [0.0, -np.sin(angular_ratio), np.cos(angular_ratio)],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        elif t <= motion_time[1]:
            ratio = min((t - motion_time[0]) / (motion_time[1] - motion_time[0]), 1.0)
            position = np.array(
                [
                    (target_position[0] - 0.5) * (1.0 - ratio) + 0.8,
                    -R * base_radius * np.cos(2.0 * ratio * 8.0) * (1.0 - ratio),
                    -R * base_radius * np.sin(2.0 * ratio * 8.0) * (1.0 - ratio),
                    # -R * base_radius * np.cos(2.0 * ratio * 12.0) * (1.0 - ratio),
                    # -R * base_radius * np.sin(2.0 * ratio * 12.0) * (1.0 - ratio),
                ],
                dtype=np.float64,
            )
            angular_ratio = (1.0 - ratio) * np.pi * 2.0
            orientation_twist = np.array(
                [
                    [0.0, np.cos(angular_ratio), -np.sin(angular_ratio)],
                    [0.0, np.sin(angular_ratio), np.cos(angular_ratio)],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        elif t <= motion_time[2]:
            # 4-5s: this end moves +x by 0.2 while the other end moves -x by 0.5.
            ratio = min((t - this_end_move_start) / (this_end_move_end - this_end_move_start), 1.0)
            position = final_phase2_position.copy()
            position[0] += this_end_move_dx * ratio
            orientation_twist = orientation_rotating_about_d3(0.0)
        elif t <= motion_time[3]:
            # 5-7s: no movement.
            position = post_move_position.copy()
            orientation_twist = orientation_rotating_about_d3(0.0)
        elif t <= motion_time[4]:
            # Rotate this end during [7, 18] so both ends rotate together.
            position = post_move_position.copy()
            twist_time = t - motion_time[3]
            theta = post_twist_angular_rate * twist_time
            orientation_twist = orientation_rotating_about_d3(theta)
        else:
            # For t > 18, keep everything still.
            position = post_move_position.copy()
            final_theta = post_twist_angular_rate * post_twist_duration
            orientation_twist = orientation_rotating_about_d3(final_theta)
        return position, orientation_twist

    # Proportional control gains from KnotCase.
    p_linear = 3.0e3
    p_angular = 5.0e0
    simulator.add_forcing_to(stretchable_rod).using(
        TargetPoseProportionalControl,
        elem_index=0,
        p_linear_value=p_linear,
        p_angular_value=p_angular,
        target=base_target,
        ramp_up_time=1.0e-6,
        target_history=recorded_history["base_pose"],
    )

    # Boundary conditions:
    # - node -20 stays fixed
    # - node -1 moves in x by -0.5 during t in [4, 5]
    # - node -1 director rotates during t in [7, 18]
    simulator.constrain(stretchable_rod).using(
        MovingOtherEndBC,
        constrained_position_idx=(-1, -20),
        constrained_director_idx=(-1,),
        move_start_time=other_end_move_start,
        move_end_time=other_end_move_end,
        move_dx=other_end_move_dx,
        rotate_start_time=both_end_rotate_start,
        rotate_end_time=both_end_rotate_end,
        rotate_angular_rate=-post_twist_angular_rate,
    )

    # Self contact from KnotCase.
    simulator.detect_contact_between(stretchable_rod, stretchable_rod).using(
        ea.RodSelfContact,
        k=1.0e4,
        nu=3.0,
    )

    # Gravity from KnotCase.
    simulator.add_forcing_to(stretchable_rod).using(
        ea.GravityForces,
        acc_gravity=np.array([0.0, 0.0, -9.80665]),
    )

    # Damping from KnotCase.
    damping_constant = 5.0
    simulator.dampen(stretchable_rod).using(
        ea.AnalyticalLinearDamper,
        translational_damping_constant=damping_constant,
        rotational_damping_constant=damping_constant * 0.01,
        time_step=dt,
    )
    simulator.dampen(stretchable_rod).using(ea.LaplaceDissipationFilter, filter_order=5)

    simulator.collect_diagnostics(stretchable_rod).using(
        KnotHistoryCallback,
        callback_params=recorded_history,
        every=callback_every,
    )

    simulator.finalize()
    total_steps = int(final_time / dt)
    print("Total steps:", total_steps)
    ea.integrate(ea.PositionVerlet(), simulator, final_time, total_steps)

    time_arr = np.asarray(recorded_history["time"])
    position_arr = np.asarray(recorded_history["position"])
    director_arr = np.asarray(recorded_history["orientation"])
    radius_arr = np.asarray(recorded_history["radius"])

    if recorded_history["base_pose"]:
        target_position_arr = np.asarray([p for p, _ in recorded_history["base_pose"]])
        target_orientation_arr = np.asarray([q for _, q in recorded_history["base_pose"]])
    else:
        target_position_arr = np.zeros((0, 3), dtype=np.float64)
        target_orientation_arr = np.zeros((0, 3, 3), dtype=np.float64)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tagged_name = (
        f"{output_name}"
        f"_t{_fmt_tag(final_time)}"
        f"_dt{_fmt_tag(dt)}"
        f"_n{n_elem}"
        f"_r{_fmt_tag(base_radius)}"
        f"_y{_fmt_tag(youngs_modulus)}"
    )
    tagged_name = _make_unique_tag(output_dir, base_tagged_name)
    state_path = output_dir / f"{tagged_name}_state.npz"
    video_path = output_dir / f"{tagged_name}_4view.mp4"
    lwt_path = output_dir / f"{tagged_name}_LWT.png"

    np.savez(
        state_path,
        time=time_arr,
        step=np.asarray(recorded_history["step"]),
        position=position_arr,
        director=director_arr,
        radius=radius_arr,
        target_position=target_position_arr,
        target_orientation=target_orientation_arr,
        dt=dt,
        final_time=final_time,
        n_elem=n_elem,
        rod_radius=base_radius,
        rod_length=base_length,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        shear_modulus=shear_modulus,
        p_linear=p_linear,
        p_angular=p_angular,
        motion_time=np.asarray(motion_time, dtype=np.float64),
        post_twist_duration=post_twist_duration,
        post_twist_angular_rate=post_twist_angular_rate,
        other_end_move_start=other_end_move_start,
        other_end_move_end=other_end_move_end,
        other_end_move_dx=other_end_move_dx,
        this_end_move_start=this_end_move_start,
        this_end_move_end=this_end_move_end,
        this_end_move_dx=this_end_move_dx,
        both_end_rotate_start=both_end_rotate_start,
        both_end_rotate_end=both_end_rotate_end,
        callback_every=callback_every,
    )

    xyz_min = np.min(position_arr, axis=(0, 2))
    xyz_max = np.max(position_arr, axis=(0, 2))
    span = xyz_max - xyz_min
    pad = np.maximum(0.12 * span, 4.0 * base_radius)
    bounds = tuple(
        (float(xyz_min[i] - pad[i]), float(xyz_max[i] + pad[i])) for i in range(3)
    )

    video_path = plot_rod_four_view_with_directors(
        position_arr,
        director_arr,
        time_arr,
        video_path=video_path,
        fps=render_fps,
        bounds=bounds,
        frame_stride=frame_stride,
        director_scale=0.8 * base_length / n_elem,
        director_stride=1,
    )

    # Same topological quantities as KnotCase.
    total_twist, _ = ea.compute_twist(position_arr, director_arr[:, 0, ...])
    total_writhe = ea.compute_writhe(position_arr, np.float64(base_length), "next_tangent")
    total_link = ea.compute_link(
        position_arr,
        director_arr[:, 0, ...],
        radius_arr,
        np.float64(base_length),
        "next_tangent",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_arr, total_twist, label="twist")
    ax.plot(time_arr, total_writhe, label="writhe")
    ax.plot(time_arr, total_link, label="link")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("link-writhe-twist quantity")
    fig.tight_layout()
    fig.savefig(lwt_path, dpi=300)
    plt.close(fig)

    return {
        "rod": stretchable_rod,
        "time": time_arr,
        "position": position_arr,
        "director": director_arr,
        "radius": radius_arr,
        "state_path": state_path,
        "video_path": video_path,
        "lwt_path": lwt_path,
    }


if __name__ == "__main__":
    results = simulate_static_knot_2()
    print(f"Saved state to: {results['state_path']}")
    print(f"Saved four-view video to: {results['video_path']}")
    print(f"Saved LWT plot to: {results['lwt_path']}")
