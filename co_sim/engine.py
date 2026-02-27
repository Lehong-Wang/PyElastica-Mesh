"""Core co-simulation engine: one rod + one kinematic attachment frame."""

from __future__ import annotations

from collections.abc import Callable

import elastica as ea
import numpy as np
from elastica.joint import get_relative_rotation_two_systems

from .models import CoSimConfig, FrameState, ImpulseResult, RodInitialState, SceneSnapshot


def _as_mu_triplet(
    x: np.ndarray | list[float] | tuple[float, float, float] | float,
    name: str,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a scalar or shape (3,), got {arr.shape}.")
    return arr


def default_rod_initial_state(config: CoSimConfig | None = None) -> RodInitialState:
    cfg = CoSimConfig() if config is None else config
    return RodInitialState(
        start=np.asarray(cfg.rod_start, dtype=float),
        direction=np.asarray(cfg.rod_direction, dtype=float),
        normal=np.asarray(cfg.rod_normal, dtype=float),
    )


def default_frame_initial_state(config: CoSimConfig | None = None) -> FrameState:
    cfg = CoSimConfig() if config is None else config
    return FrameState(
        position=np.asarray(cfg.frame_initial_position, dtype=float),
        director=np.asarray(cfg.frame_initial_director, dtype=float),
        velocity=np.asarray(cfg.frame_initial_velocity, dtype=float),
        acceleration=np.asarray(cfg.frame_initial_acceleration, dtype=float),
        omega=np.asarray(cfg.frame_initial_omega, dtype=float),
        alpha=np.asarray(cfg.frame_initial_alpha, dtype=float),
    )


class FrameStateBuffer:
    """Mutable state buffer used by the kinematic frame constraint."""

    def __init__(self) -> None:
        self.position = np.zeros(3)
        self.director = np.eye(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.omega_world = np.zeros(3)
        self.alpha_world = np.zeros(3)

    def update(self, state: FrameState, isaac_t: float = 0.0) -> None:
        s = state.validated()
        self.position[...] = s.position
        self.director[...] = s.director
        self.velocity[...] = s.velocity + s.acceleration * isaac_t
        # self.velocity[...] = s.velocity
        self.acceleration[...] = s.acceleration
        self.omega_world[...] = s.omega + s.alpha * isaac_t
        # self.omega_world[...] = s.omega
        self.alpha_world[...] = s.alpha

    def apply_to_system(self, system) -> None:
        np.copyto(system.position_collection[:, 0], self.position)
        np.copyto(system.director_collection[..., 0], self.director)
        np.copyto(system.velocity_collection[:, 0], self.velocity)
        np.copyto(system.omega_collection[:, 0], self.omega_local)
        np.copyto(system.acceleration_collection[:, 0], self.acceleration)
        np.copyto(system.alpha_collection[:, 0], self.alpha_local)
        np.copyto(system.external_forces, 0.0)
        np.copyto(system.external_torques, 0.0)

    @property
    def omega_local(self) -> np.ndarray:
        # director maps world vectors to local vectors in PyElastica.
        return self.director @ self.omega_world

    @property
    def alpha_local(self) -> np.ndarray:
        return self.director @ self.alpha_world


class RateOnlyFrameBC(ea.ConstraintBase):
    """Constrain frame rates only; position/director are integrated by the stepper."""

    def __init__(self, state: FrameStateBuffer, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def constrain_values(self, system, time: np.float64) -> None:
        # Leave position/director unconstrained so they evolve from rates.
        pass

    def constrain_rates(self, system, time: np.float64) -> None:
        np.copyto(system.velocity_collection[:, 0], self.state.velocity)
        np.copyto(system.omega_collection[:, 0], self.state.omega_local)


class _ImpulseAccumulator:
    def __init__(self) -> None:
        self.linear_impulse = np.zeros(3)
        self.angular_impulse = np.zeros(3)

    def reset(self) -> None:
        self.linear_impulse[...] = 0.0
        self.angular_impulse[...] = 0.0


class _StepSizeBuffer:
    def __init__(self, dt: float) -> None:
        self.dt = float(dt)


class RecordAndZeroFrameLoads(ea.NoForces):
    """
    Record frame loads and then clear them, so the frame remains kinematic.
    """

    def __init__(self, dt_buffer: _StepSizeBuffer, accum: _ImpulseAccumulator):
        super().__init__()
        self.dt_buffer = dt_buffer
        self.accum = accum

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        force_world = system.external_forces[:, 0].copy()
        self.accum.linear_impulse += force_world * self.dt_buffer.dt
        system.external_forces[...] = 0.0

    def apply_torques(self, system, time: np.float64 = np.float64(0.0)) -> None:
        torque_local = system.external_torques[:, 0].copy()
        torque_world = system.director_collection[..., 0].T @ torque_local
        self.accum.angular_impulse += torque_world * self.dt_buffer.dt
        system.external_torques[...] = 0.0


class CoSimEngine:
    """
    Build the co-sim scene and advance one Isaac update at a time.

    One call to `update_frame_state`:
    1) applies the latest frame command state
    2) advances internal PyElastica steps
    3) returns net impulse from the rod/joint on the frame over that interval
    """

    class _Simulator(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.Connections,
        ea.Contact,
        ea.Constraints,
        ea.Damping,
    ):
        pass

    @staticmethod
    def _resolve_rod_initial_state(
        config: CoSimConfig,
        frame_init: FrameState,
        rod_initial_state: RodInitialState | None,
    ) -> RodInitialState:
        theta_raw = config.initial_wire_theta
        if theta_raw is not None:
            theta = float(theta_raw)
            if not np.isfinite(theta):
                raise ValueError(f"initial_wire_theta must be finite, got {theta}.")
            c = float(np.cos(theta))
            s = float(np.sin(theta))
            return RodInitialState(
                start=np.asarray(frame_init.position, dtype=float),
                direction=np.array([c, s, 0.0], dtype=float),
                normal=np.array([0.0, 0.0, 1.0], dtype=float),
            ).validated()

        if rod_initial_state is not None:
            return rod_initial_state.validated()

        # Null theta defaults to frame z-axis, starting at initial frame.
        return RodInitialState(
            start=np.asarray(frame_init.position, dtype=float),
            direction=np.asarray(frame_init.director[2], dtype=float),
            normal=np.asarray(frame_init.director[0], dtype=float),
        ).validated()

    def _resolve_fixed_joint_instance(self):
        for operator_group in self.sim._feature_group_synchronize._operator_collection:
            for operator in operator_group:
                func = getattr(operator, "func", None)
                if func is None:
                    continue
                owner = getattr(func, "__self__", None)
                if isinstance(owner, ea.FixedJoint):
                    return owner
        raise RuntimeError("Unable to locate FixedJoint instance in simulator operators.")

    def _resolve_rod_damper_instance(self):
        for operator_group in self.sim._feature_group_damping._operator_collection:
            for operator in operator_group:
                func = getattr(operator, "func", None)
                if func is None:
                    continue
                owner = getattr(func, "__self__", None)
                if isinstance(owner, ea.AnalyticalLinearDamper):
                    return owner
        raise RuntimeError(
            "Unable to locate AnalyticalLinearDamper instance in simulator operators."
        )

    def _set_rod_damping_constant(self, damping_constant: float) -> None:
        damper = self._rod_damper
        if not hasattr(damper, "_deprecated_damping_protocol"):
            raise TypeError(
                "Configured damper does not support runtime retuning for settle damping."
            )
        damper._dampen_rates_protocol = damper._deprecated_damping_protocol(
            damping_constant=np.float64(damping_constant),
            time_step=np.float64(self.py_dt),
        )

    def _set_fixed_joint_k(self, joint_k: float) -> None:
        self._fixed_joint.k = np.float64(joint_k)

    def _apply_settle_parameters(self, damping_constant: float, joint_k: float) -> None:
        self._set_rod_damping_constant(damping_constant)
        self._set_fixed_joint_k(joint_k)

    def __init__(
        self,
        config: CoSimConfig,
        rod_initial_state: RodInitialState | None = None,
        frame_initial_state: FrameState | None = None,
    ):
        self.config = config
        self.py_dt = float(config.py_dt)
        self.isaac_dt = float(config.isaac_dt)
        if self.py_dt <= 0.0:
            raise ValueError(f"py_dt must be positive, got {self.py_dt}.")
        if self.isaac_dt <= 0.0:
            raise ValueError(f"isaac_dt must be positive, got {self.isaac_dt}.")
        runtime_damping_constant = float(config.damping_constant)
        runtime_joint_k = float(config.joint_k)
        settle_damping_constant = (
            runtime_damping_constant
            if config.settle_damping_constant is None
            else float(config.settle_damping_constant)
        )
        settle_joint_k = (
            runtime_joint_k
            if config.settle_joint_k is None
            else float(config.settle_joint_k)
        )

        frame_init = (
            default_frame_initial_state(config)
            if frame_initial_state is None
            else frame_initial_state
        ).validated()
        rod_init = self._resolve_rod_initial_state(config, frame_init, rod_initial_state)

        self.sim = self._Simulator()
        self.ground_plane = None

        self.rod = ea.CosseratRod.straight_rod(
            n_elements=config.n_elem,
            start=rod_init.start,
            direction=rod_init.direction,
            normal=rod_init.normal,
            base_length=config.base_length,
            base_radius=config.base_radius,
            density=config.density,
            youngs_modulus=config.youngs_modulus,
            shear_modulus=config.youngs_modulus / (2.0 * config.shear_modulus_ratio),
        )
        self.sim.append(self.rod)

        frame_axis = frame_init.director[2]
        frame_normal = frame_init.director[0]
        frame_start = frame_init.position - 0.5 * config.frame_base_length * frame_axis
        self.frame = ea.Cylinder(
            start=frame_start,
            direction=frame_axis,
            normal=frame_normal,
            base_length=config.frame_base_length,
            base_radius=config.frame_base_radius,
            density=config.frame_density,
        )
        self.sim.append(self.frame)

        self.sim.dampen(self.rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=runtime_damping_constant,
            time_step=self.py_dt,
        )

        self.sim.add_forcing_to(self.rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
        )
        if bool(config.use_ground_contact):
            self._add_ground_contact_with_friction(config)

        rest_rot = get_relative_rotation_two_systems(self.frame, 0, self.rod, 0)
        self.sim.connect(self.frame, self.rod, first_connect_idx=0, second_connect_idx=0).using(
            ea.FixedJoint,
            k=runtime_joint_k,
            nu=config.joint_nu,
            kt=config.joint_kt,
            nut=config.joint_nut,
            rest_rotation_matrix=rest_rot,
        )

        self.frame_state = FrameStateBuffer()
        self.frame_state.update(frame_init, isaac_t=0.0)
        self.sim.constrain(self.frame).using(RateOnlyFrameBC, state=self.frame_state)

        # Register after joint so synchronize() sees joint loads first, then records+zeros.
        self._step_size = _StepSizeBuffer(self.py_dt)
        self.tick_impulse = _ImpulseAccumulator()
        self.sim.add_forcing_to(self.frame).using(
            RecordAndZeroFrameLoads,
            dt_buffer=self._step_size,
            accum=self.tick_impulse,
        )

        self.sim.finalize()
        self._fixed_joint = self._resolve_fixed_joint_instance()
        self._rod_damper = self._resolve_rod_damper_instance()
        self.stepper: ea.typing.StepperProtocol = ea.PositionVerlet()
        self.time = np.float64(0.0)

        settle_time = float(config.settle_time)
        if settle_time < 0.0:
            raise ValueError(f"settle_time must be >= 0, got {settle_time}.")
        if settle_time > 0.0:
            self._apply_settle_parameters(
                damping_constant=settle_damping_constant,
                joint_k=settle_joint_k,
            )
            try:
                self._settle_rod(frame_init, settle_time)
            finally:
                self._apply_settle_parameters(
                    damping_constant=runtime_damping_constant,
                    joint_k=runtime_joint_k,
                )

        # Ensure frame state arrays match the supplied initial command exactly.
        self.apply_command_state(frame_init, isaac_t=0.0)

    def _add_ground_contact_with_friction(self, config: CoSimConfig) -> None:
        ground_contact_k = float(config.ground_contact_k)
        ground_contact_nu = float(config.ground_contact_nu)
        ground_slip_velocity_tol = float(config.ground_slip_velocity_tol)
        if ground_contact_k < 0.0:
            raise ValueError(f"ground_contact_k must be >= 0, got {ground_contact_k}.")
        if ground_contact_nu < 0.0:
            raise ValueError(f"ground_contact_nu must be >= 0, got {ground_contact_nu}.")
        if ground_slip_velocity_tol <= 0.0:
            raise ValueError(
                f"ground_slip_velocity_tol must be > 0, got {ground_slip_velocity_tol}."
            )

        plane_origin = np.array([0.0, 0.0, float(config.ground_z)], dtype=np.float64)
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ground_static_mu = _as_mu_triplet(config.ground_static_mu, "ground_static_mu")
        ground_kinetic_mu = _as_mu_triplet(config.ground_kinetic_mu, "ground_kinetic_mu")

        self.ground_plane = ea.Plane(plane_origin=plane_origin, plane_normal=plane_normal)
        self.sim.append(self.ground_plane)
        self.sim.add_forcing_to(self.rod).using(
            ea.AnisotropicFrictionalPlane,
            k=ground_contact_k,
            nu=ground_contact_nu,
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            slip_velocity_tol=ground_slip_velocity_tol,
            static_mu_array=ground_static_mu,
            kinetic_mu_array=ground_kinetic_mu,
        )
        self.sim.detect_contact_between(self.rod, self.ground_plane).using(
            ea.RodPlaneContact,
            k=ground_contact_k,
            nu=ground_contact_nu,
        )

    def _settle_rod(self, frame_state: FrameState, duration: float) -> None:
        # Keep attachment frame fixed and let rod relax under configured forces/contact.
        settle_state = FrameState(
            position=np.asarray(frame_state.position, dtype=np.float64),
            director=np.asarray(frame_state.director, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            acceleration=np.zeros(3, dtype=np.float64),
            omega=np.zeros(3, dtype=np.float64),
            alpha=np.zeros(3, dtype=np.float64),
        )
        settle_duration = float(duration)
        n_steps = int(np.ceil(settle_duration / self.py_dt))
        for step_idx in range(n_steps):
            self.apply_command_state(settle_state, isaac_t=0.0)
            if step_idx < (n_steps - 1):
                dt_step = self.py_dt
            else:
                dt_step = settle_duration - self.py_dt * (n_steps - 1)
            if dt_step <= 0.0:
                break
            self._step_size.dt = float(dt_step)
            self.time = self.stepper.step(self.sim, self.time, np.float64(dt_step))
        self.apply_command_state(settle_state, isaac_t=0.0)

        # Warm-start should not consume external timeline.
        self.time = np.float64(0.0)
        self.tick_impulse.reset()

    def apply_command_state(self, state: FrameState, isaac_t: float = 0.0) -> None:
        self.frame_state.update(state, isaac_t=isaac_t)
        self.frame_state.apply_to_system(self.frame)

    def snapshot(self) -> SceneSnapshot:
        return SceneSnapshot(
            sim_time=float(self.time),
            rod_position=self.rod.position_collection.copy(),
            rod_director=self.rod.director_collection.copy(),
            frame_position=self.frame.position_collection.copy(),
            frame_director=self.frame.director_collection.copy(),
        )

    def update_frame_state(
        self,
        frame_state: FrameState,
        duration: float | None = None,
        observer: Callable[[float, np.ndarray], None] | None = None,
    ) -> ImpulseResult:
        """
        Update frame command and advance the co-sim by one external update interval.

        Parameters
        ----------
        frame_state
            Kinematic frame command from Isaac for this interval.
        duration
            Simulated time to advance for this update.
            Defaults to `isaac_dt`.
        observer
            Optional callback called after each internal step as
            `observer(sim_time, mean_force_window)`.
        """
        update_duration = self.isaac_dt if duration is None else float(duration)
        if update_duration <= 0.0:
            raise ValueError(f"duration must be positive, got {update_duration}.")

        start_time = float(self.time)
        self.apply_command_state(frame_state, isaac_t=self.isaac_dt)
        self.tick_impulse.reset()
        target_time = start_time + update_duration
        while float(self.time) < target_time:
            current_time = float(self.time)
            remaining = target_time - current_time
            time_ulp = np.spacing(current_time)
            if remaining <= max(time_ulp, np.finfo(float).eps):
                # Remaining interval is below time resolution; snap to target.
                self.time = np.float64(target_time)
                break

            dt_step = min(self.py_dt, remaining)
            self._step_size.dt = dt_step
            self.time = self.stepper.step(self.sim, self.time, np.float64(dt_step))
            new_time = float(self.time)
            if new_time <= current_time:
                # Guard against non-progress due floating-point resolution.
                self.time = np.float64(target_time)
                break
            if observer is not None:
                elapsed = max(float(self.time) - start_time, np.finfo(float).eps)
                mean_force = self.tick_impulse.linear_impulse / elapsed
                observer(float(self.time), mean_force.copy())

        elapsed = float(self.time) - start_time
        return ImpulseResult(
            linear_impulse=self.tick_impulse.linear_impulse.copy(),
            angular_impulse=self.tick_impulse.angular_impulse.copy(),
            elapsed_time=elapsed,
            sim_time=float(self.time),
        )
