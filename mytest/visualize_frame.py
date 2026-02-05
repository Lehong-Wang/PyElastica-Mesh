"""
Minimal Open3D viewer for the saved *_state_snapshot.npz.
Plots the rod + mesh at timestep=100.

Usage:
  python view_state_o3d.py \
    --npz mesh_rod_collision_state_snapshot.npz \
    --stl mytest/bunny_low_10_center.stl \
    --t 100
"""

import argparse
import numpy as np
import open3d as o3d


def R_from_director(D: np.ndarray) -> np.ndarray:
    """
    Your callback saves mesh_body.director_collection[:, :, 0] (shape (3,3)).
    In Elastica, director_collection is typically a rotation-like matrix.
    Open3D expects a proper 3x3 rotation.
    We just pass it through; if it looks flipped in your scene, try D.T below.
    """
    return D


def make_rod_lineset(rod_pos: np.ndarray) -> o3d.geometry.LineSet:
    """
    rod_pos: (3, n_nodes)
    """
    pts = rod_pos.T  # (n_nodes, 3)
    n = pts.shape[0]
    lines = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="*_state_snapshot.npz path")
    ap.add_argument("--stl", required=True, help="obstacle mesh STL path used in sim")
    ap.add_argument("--t", type=int, default=100, help="timestep index in snapshot arrays")
    ap.add_argument("--node_spheres", action="store_true", help="draw small spheres at rod nodes")
    ap.add_argument("--sphere_r", type=float, default=0.006, help="rod node sphere radius (meters-ish)")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    rod_position = data["rod_position"]      # (T, 3, n_nodes)
    mesh_position = data["mesh_position"]    # (T, 3)
    mesh_director = data["mesh_director"]    # (T, 3, 3)

    T = rod_position.shape[0]
    t = int(np.clip(args.t, 0, T - 1))

    # --- Rod geometry (lines) ---
    rod_pos_t = rod_position[t]
    rod_ls = make_rod_lineset(rod_pos_t)
    rod_ls.paint_uniform_color([0.1, 0.4, 0.9])  # blue-ish

    geoms = [rod_ls]

    # Optional: spheres at nodes (helps depth perception)
    if args.node_spheres:
        pts = rod_pos_t.T
        for p in pts:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=args.sphere_r)
            s.translate(p)
            s.compute_vertex_normals()
            s.paint_uniform_color([0.9, 0.2, 0.2])
            geoms.append(s)

    # --- Mesh geometry (STL transformed by pose) ---
    mesh = o3d.io.read_triangle_mesh(args.stl)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load STL: {args.stl}")
    mesh.compute_vertex_normals()

    p = mesh_position[t].reshape(3)
    D = mesh_director[t]
    R = R_from_director(D)

    # Apply rigid transform: x_world = R * x_local + p
    # Note: if you see the mesh "rotated wrong", try `R = R.T` instead.
    mesh.rotate(R, center=(0.0, 0.0, 0.0))
    mesh.translate(p)

    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(mesh)

    # Frame + grid for reference
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    # ground plane grid (optional quick hack)
    grid = o3d.geometry.LineSet.create_from_triangle_mesh(
        o3d.geometry.TriangleMesh.create_box(width=3.0, height=0.001, depth=3.0)
    )
    grid.translate([-1.5, -0.0005, -1.5])
    grid.paint_uniform_color([0.85, 0.85, 0.85])
    geoms.append(grid)

    print(f"[open3d] Loaded {args.npz}. Showing timestep t={t} (T={T}).")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Rod+Mesh @ t={t}",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
