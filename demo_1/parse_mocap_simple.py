#!/usr/bin/env python3
"""
parse_mocap_simple.py

Minimal OptiTrack parser + orientation correction.

Core behavior:
1) Parse tracked points and rigid-body poses from CSV.
2) At first valid frame for the target rigid body:
   - origin = rigid body position
   - z_true = world up (+Y)
   - x_true = direction to furthest tracked point, projected to plane orthogonal to z_true
   - y_true = z_true x x_true
   Build B0=[x_true y_true z_true].
3) Compute and store the transform from original frame to corrected frame:
      R_corr = R_meas0^T @ B0
      T_corr = [[R_corr, 0], [0, 1]]
4) Apply to every frame:
      R_true(t) = R_meas(t) @ R_corr

Open3D visualization shows:
- corrected orientation frame (colored axes)
- original measured orientation frame (gray)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps or np.isnan(n):
        return np.zeros_like(v)
    return v / n


def project_to_rotation_matrix(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    rp = u @ vt
    if np.linalg.det(rp) < 0.0:
        u[:, -1] *= -1.0
        rp = u @ vt
    return rp


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0 or np.isnan(n):
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0 or np.isnan(n):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def forward_fill(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-fill NaNs along time axis. Returns (filled, present_raw)."""
    T, M, D = arr.shape
    present = ~np.any(np.isnan(arr), axis=2)
    filled = np.full_like(arr, np.nan)

    last = np.full((M, D), np.nan, dtype=np.float64)
    seen = np.zeros((M,), dtype=bool)

    for t in range(T):
        mask = present[t]
        if np.any(mask):
            last[mask] = arr[t, mask]
            seen[mask] = True
        filled[t, seen] = last[seen]

    return filled, present


def _read_float(row: List[str], col: int) -> float:
    if col >= len(row):
        return np.nan
    s = row[col].strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


@dataclass
class MocapData:
    time_s: np.ndarray

    names: List[str]
    pos_ff: np.ndarray
    pos_present: np.ndarray

    rigid_names: List[str]
    rigid_pos_ff: np.ndarray
    rigid_quat_ff_xyzw: np.ndarray
    rigid_name_to_idx: Dict[str, int]

    T_corr: Optional[np.ndarray]
    corr_frame_idx: Optional[int]


def find_header_index(rows: List[List[str]]) -> int:
    for i, row in enumerate(rows):
        if len(row) >= 2 and row[0].strip() == "Frame" and row[1].strip().startswith("Time"):
            return i
    raise RuntimeError("Could not find OptiTrack header row.")


def load_optitrack_csv(csv_path: str | Path) -> MocapData:
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="") as f:
        rows = list(csv.reader(f))

    h = find_header_index(rows)
    if h < 4:
        raise RuntimeError("Unexpected OptiTrack CSV layout.")

    name_row = rows[h - 3]
    type_row = rows[h - 1]
    axis_row = rows[h]

    pos_cols: Dict[str, Dict[str, int]] = {}
    rot_cols: Dict[str, Dict[str, int]] = {}

    for c in range(2, len(axis_row)):
        name = name_row[c].strip() if c < len(name_row) else ""
        kind = type_row[c].strip() if c < len(type_row) else ""
        axis = axis_row[c].strip()
        if not name:
            continue
        if kind == "Position" and axis in ("X", "Y", "Z"):
            pos_cols.setdefault(name, {})[axis] = c
        elif kind == "Rotation" and axis in ("X", "Y", "Z", "W"):
            rot_cols.setdefault(name, {})[axis] = c

    names = sorted([n for n, d in pos_cols.items() if all(k in d for k in ("X", "Y", "Z"))])
    if not names:
        raise RuntimeError("No Position objects found.")

    rigid_names = sorted(
        [n for n in names if n in rot_cols and all(k in rot_cols[n] for k in ("X", "Y", "Z", "W"))]
    )
    rigid_name_to_idx = {n: i for i, n in enumerate(rigid_names)}

    pos_idx = np.array([[pos_cols[n]["X"], pos_cols[n]["Y"], pos_cols[n]["Z"]] for n in names], dtype=np.int32)
    if rigid_names:
        rpos_idx = np.array([[pos_cols[n]["X"], pos_cols[n]["Y"], pos_cols[n]["Z"]] for n in rigid_names], dtype=np.int32)
        rquat_idx = np.array(
            [[rot_cols[n]["X"], rot_cols[n]["Y"], rot_cols[n]["Z"], rot_cols[n]["W"]] for n in rigid_names],
            dtype=np.int32,
        )
    else:
        rpos_idx = np.zeros((0, 3), dtype=np.int32)
        rquat_idx = np.zeros((0, 4), dtype=np.int32)

    times: List[float] = []
    pos_frames: List[np.ndarray] = []
    rpos_frames: List[np.ndarray] = []
    rquat_frames: List[np.ndarray] = []

    for row in rows[h + 1 :]:
        if len(row) < 2 or not row[0].strip():
            continue
        t = _read_float(row, 1)
        if np.isnan(t):
            continue

        p = np.full((len(names), 3), np.nan, dtype=np.float64)
        for i in range(len(names)):
            vals = np.array([
                _read_float(row, pos_idx[i, 0]),
                _read_float(row, pos_idx[i, 1]),
                _read_float(row, pos_idx[i, 2]),
            ])
            if not np.any(np.isnan(vals)):
                p[i] = vals

        if rigid_names:
            rp = np.full((len(rigid_names), 3), np.nan, dtype=np.float64)
            rq = np.full((len(rigid_names), 4), np.nan, dtype=np.float64)
            for i in range(len(rigid_names)):
                pvals = np.array([
                    _read_float(row, rpos_idx[i, 0]),
                    _read_float(row, rpos_idx[i, 1]),
                    _read_float(row, rpos_idx[i, 2]),
                ])
                qvals = np.array([
                    _read_float(row, rquat_idx[i, 0]),
                    _read_float(row, rquat_idx[i, 1]),
                    _read_float(row, rquat_idx[i, 2]),
                    _read_float(row, rquat_idx[i, 3]),
                ])
                if not np.any(np.isnan(pvals)) and not np.any(np.isnan(qvals)):
                    rp[i] = pvals
                    rq[i] = qvals
            rpos_frames.append(rp)
            rquat_frames.append(rq)

        times.append(float(t))
        pos_frames.append(p)

    if not pos_frames:
        raise RuntimeError("No frames parsed.")

    # OptiTrack exports positions in mm; convert to meters.
    pos_raw = np.stack(pos_frames, axis=0) * 1e-3
    pos_ff, pos_present = forward_fill(pos_raw)

    if rigid_names:
        rpos_raw = np.stack(rpos_frames, axis=0) * 1e-3
        rquat_raw = np.stack(rquat_frames, axis=0)
        rpos_ff, _ = forward_fill(rpos_raw)
        rquat_ff, _ = forward_fill(rquat_raw)
    else:
        T = len(times)
        rpos_ff = np.zeros((T, 0, 3), dtype=np.float64)
        rquat_ff = np.zeros((T, 0, 4), dtype=np.float64)

    return MocapData(
        time_s=np.asarray(times, dtype=np.float64),
        names=names,
        pos_ff=pos_ff,
        pos_present=pos_present,
        rigid_names=rigid_names,
        rigid_pos_ff=rpos_ff,
        rigid_quat_ff_xyzw=rquat_ff,
        rigid_name_to_idx=rigid_name_to_idx,
        T_corr=None,
        corr_frame_idx=None,
    )


def compute_correction_transform(data: MocapData, rigid_name: str) -> Tuple[np.ndarray, int]:
    if rigid_name not in data.rigid_name_to_idx:
        raise KeyError(f"Rigid body '{rigid_name}' not found.")

    ri = data.rigid_name_to_idx[rigid_name]
    T = data.time_s.shape[0]

    frame_idx = None
    for k in range(T):
        p = data.rigid_pos_ff[k, ri]
        q = data.rigid_quat_ff_xyzw[k, ri]
        if not np.any(np.isnan(p)) and not np.any(np.isnan(q)):
            frame_idx = k
            break
    if frame_idx is None:
        raise RuntimeError("No valid rigid-body pose to compute correction.")

    origin = data.rigid_pos_ff[frame_idx, ri]
    pts = data.pos_ff[frame_idx]
    valid = ~np.any(np.isnan(pts), axis=1)
    if np.sum(valid) < 2:
        raise RuntimeError("Not enough valid points at correction frame.")

    diffs = pts[valid] - origin[None, :]
    furthest = diffs[int(np.argmax(np.sum(diffs * diffs, axis=1)))]
    if np.linalg.norm(furthest) < 1e-12:
        raise RuntimeError("Degenerate furthest-point direction.")

    up = WORLD_UP
    x_proj = furthest - np.dot(furthest, up) * up
    if np.linalg.norm(x_proj) < 1e-8:
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if np.linalg.norm(np.cross(up, alt)) < 1e-8:
            alt = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_proj = alt - np.dot(alt, up) * up

    x = normalize(x_proj)
    y = normalize(np.cross(up, x))
    x = normalize(np.cross(y, up))

    B0 = project_to_rotation_matrix(np.stack([x, y, up], axis=1))

    R_meas0 = quat_xyzw_to_R(data.rigid_quat_ff_xyzw[frame_idx, ri])
    R_corr = project_to_rotation_matrix(R_meas0.T @ B0)

    T_corr = np.eye(4, dtype=np.float64)
    T_corr[:3, :3] = R_corr

    err = np.linalg.norm((R_meas0 @ R_corr) - B0)
    if err > 1e-3:
        print(f"[warn] correction sanity error at frame {frame_idx}: {err:.3e}")

    return T_corr, frame_idx


def set_correction_transform(data: MocapData, rigid_name: str) -> None:
    T_corr, frame_idx = compute_correction_transform(data, rigid_name)
    data.T_corr = T_corr
    data.corr_frame_idx = frame_idx


def corrected_rotation_matrix(data: MocapData, quat_xyzw: np.ndarray) -> np.ndarray:
    R_meas = quat_xyzw_to_R(quat_xyzw)
    if data.T_corr is None:
        return project_to_rotation_matrix(R_meas)
    return project_to_rotation_matrix(R_meas @ data.T_corr[:3, :3])


def get_state_at_time(
    data: MocapData,
    t: float,
    rigid_name: str,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if rigid_name not in data.rigid_name_to_idx:
        raise KeyError(f"Rigid body '{rigid_name}' not found.")

    idx = int(np.argmin(np.abs(data.time_s - t)))
    ri = data.rigid_name_to_idx[rigid_name]

    pts = data.pos_ff[idx]
    rpos = data.rigid_pos_ff[idx, ri]
    rquat = data.rigid_quat_ff_xyzw[idx, ri]

    if np.any(np.isnan(rquat)):
        q_true = rquat.copy()
    else:
        q_true = R_to_quat_xyzw(corrected_rotation_matrix(data, rquat))

    return idx, pts.copy(), rpos.copy(), q_true


def animate_open3d(data: MocapData, rigid_name: str) -> None:
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError("open3d is required for --open3d") from e

    if rigid_name not in data.rigid_name_to_idx:
        raise KeyError(f"Rigid body '{rigid_name}' not found.")

    T = data.time_s.shape[0]
    ri = data.rigid_name_to_idx[rigid_name]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Mocap Simple (rigid={rigid_name})", width=1400, height=800)

    pcd = o3d.geometry.PointCloud()

    frame_true = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    frame_true_v0 = np.asarray(frame_true.vertices).copy()

    frame_orig = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.045)
    frame_orig.paint_uniform_color([0.6, 0.6, 0.6])
    frame_orig_v0 = np.asarray(frame_orig.vertices).copy()

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    vis.add_geometry(pcd)
    vis.add_geometry(frame_orig)
    vis.add_geometry(frame_true)

    opt = vis.get_render_option()
    opt.point_size = 14.0

    present_color = np.array([0.2, 0.9, 0.2], dtype=np.float64)
    held_color = np.array([0.6, 0.6, 0.6], dtype=np.float64)

    t = 0
    while vis.poll_events():
        pts = data.pos_ff[t]
        valid = ~np.any(np.isnan(pts), axis=1)
        idxs = np.nonzero(valid)[0]

        draw_pts = pts[valid]
        cols = np.tile(present_color, (draw_pts.shape[0], 1))
        if idxs.size:
            held_mask = ~data.pos_present[t, idxs]
            cols[held_mask] = held_color

        pcd.points = o3d.utility.Vector3dVector(draw_pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        rpos = data.rigid_pos_ff[t, ri]
        rquat = data.rigid_quat_ff_xyzw[t, ri]
        if not np.any(np.isnan(rpos)) and not np.any(np.isnan(rquat)):
            R_orig = project_to_rotation_matrix(quat_xyzw_to_R(rquat))
            R_true = corrected_rotation_matrix(data, rquat)

            V_orig = (R_orig @ frame_orig_v0.T).T + rpos[None, :]
            frame_orig.vertices = o3d.utility.Vector3dVector(V_orig)
            frame_orig.compute_vertex_normals()

            V_true = (R_true @ frame_true_v0.T).T + rpos[None, :]
            frame_true.vertices = o3d.utility.Vector3dVector(V_true)
            frame_true.compute_vertex_normals()

        vis.update_geometry(pcd)
        vis.update_geometry(frame_orig)
        vis.update_geometry(frame_true)
        vis.update_renderer()

        t = (t + 1) % T

    vis.destroy_window()


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal mocap parser and corrected orientation viewer.")
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--rigid", default="rope_rigid", help="Target rigid body name.")
    ap.add_argument("--t", type=float, default=0.0, help="Query time in seconds.")
    ap.add_argument("--open3d", action="store_true", help="Open Open3D viewer.")
    args = ap.parse_args()

    data = load_optitrack_csv(args.csv_path)

    try:
        set_correction_transform(data, args.rigid)
    except Exception as e:
        print(f"[warn] could not compute correction transform: {e}")

    idx, _, rpos, q_true = get_state_at_time(data, args.t, args.rigid)

    print(f"nearest frame idx = {idx}, time = {data.time_s[idx]:.6f}s")
    print(f"num tracked Position objects = {len(data.names)}")
    print(f"rigid '{args.rigid}' position = {rpos}")
    print(f"rigid '{args.rigid}' corrected quaternion (x,y,z,w) = {q_true}")
    print(f"T_corr computed = {data.T_corr is not None}")
    if data.corr_frame_idx is not None:
        print(f"correction frame idx = {data.corr_frame_idx}, time = {data.time_s[data.corr_frame_idx]:.6f}s")

    if args.open3d:
        animate_open3d(data, args.rigid)


if __name__ == "__main__":
    main()
