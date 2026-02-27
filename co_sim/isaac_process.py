"""Dummy Isaac-side command generation helpers."""

from __future__ import annotations

import numpy as np

from .models import FrameState


def _rot_x(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rot_y(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def sine_frame_state(command_time: float, amp: float = 0.1, freq: float = 1.0) -> FrameState:
    """Sine translation in world-Y with fixed orientation and zero angular motion."""
    omega = 2.0 * np.pi * freq
    pos = np.array([0.0, amp * np.sin(omega * command_time), 0.0])
    vel = np.array([0.0, amp * omega * np.cos(omega * command_time), 0.0])
    acc = np.array([0.0, -amp * omega * omega * np.sin(omega * command_time), 0.0])
    default_director = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return FrameState(
        position=pos,
        director=default_director,
        velocity=vel,
        acceleration=acc,
        omega=np.zeros(3),
        alpha=np.zeros(3),
    )


def circular_yz_frame_state(command_time: float, motion_duration: float = 4.0) -> FrameState:
    """
    Legacy-named helper for circular motion in world XZ plane.

    Pose specification:
    - start position: (0, 0, 0.1)
    - initial frame axis (d3): -X
    - rotate around center (0, 0, 0.5) by 180 degrees about world Y.
    The 180-degree sweep is completed over `motion_duration` and then held.
    """
    center_vec = np.array([0.0, 0.0, 0.5], dtype=float)
    start_vec = np.array([0.0, 0.0, 0.1], dtype=float)
    if motion_duration <= 0.0:
        raise ValueError(f"motion_duration must be positive, got {motion_duration}.")

    angle_total = float(np.deg2rad(180.0))
    t = float(command_time)
    if t <= 0.0:
        theta = 0.0
        theta_dot = angle_total / float(motion_duration)
    elif t >= motion_duration:
        theta = angle_total
        theta_dot = 0.0
    else:
        theta = angle_total * (t / float(motion_duration))
        theta_dot = angle_total / float(motion_duration)

    # Keep orientation rotation as-is, but orbit position in the opposite
    # sense around the same center/axis.
    theta_pos = -theta
    theta_dot_pos = -theta_dot

    rel0 = start_vec - center_vec
    rel = _rot_y(theta_pos) @ rel0
    pos = center_vec + rel

    omega_world_pos = np.array([0.0, theta_dot_pos, 0.0], dtype=float)
    vel = np.cross(omega_world_pos, rel)
    acc = np.cross(omega_world_pos, vel)

    director0 = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    director = director0 @ _rot_y(theta)

    return FrameState(
        position=pos,
        director=director,
        velocity=vel,
        acceleration=acc,
        # Angular rate corresponds to orientation rotation (unchanged direction).
        omega=np.array([0.0, theta_dot, 0.0], dtype=float),
        alpha=np.zeros(3, dtype=float),
    )
