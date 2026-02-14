#!/usr/bin/env python3
"""
Quick utility to inspect mesh center of mass and recentering.

Loads a mesh, prints its raw OBB/COM, then uses elastica.mesh.Mesh to
recenter it and visualizes both the original and recentered versions
with Open3D (mesh + COM sphere + OBB lineset).
"""

from __future__ import annotations

import argparse
import numpy as np
import open3d as o3d

from elastica.mesh.mesh_initializer import Mesh


def make_com_sphere(center: np.ndarray, color: tuple[float, float, float], radius: float = 0.01) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center.astype(np.float64))
    sphere.paint_uniform_color(color)
    return sphere


def mesh_to_wireframe(mesh: o3d.geometry.TriangleMesh, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lineset.paint_uniform_color(color)
    return lineset


def visualize_combined(
    raw_mesh: o3d.geometry.TriangleMesh,
    raw_com: np.ndarray,
    raw_obb: o3d.geometry.OrientedBoundingBox,
    centered_mesh: o3d.geometry.TriangleMesh,
    centered_com: np.ndarray,
    centered_obb: o3d.geometry.OrientedBoundingBox,
) -> None:
    raw_color = (1.0, 0.0, 0.0)   # red
    centered_color = (0.0, 0.3, 1.0)  # blue

    geoms: list[o3d.geometry.Geometry] = []
    geoms.append(mesh_to_wireframe(raw_mesh, raw_color))
    geoms.append(mesh_to_wireframe(centered_mesh, centered_color))

    geoms.append(make_com_sphere(raw_com, raw_color, radius=0.008))
    geoms.append(make_com_sphere(centered_com, centered_color, radius=0.008))

    raw_obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(raw_obb)
    raw_obb_lines.paint_uniform_color(raw_color)
    geoms.append(raw_obb_lines)

    centered_obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(centered_obb)
    centered_obb_lines.paint_uniform_color(centered_color)
    geoms.append(centered_obb_lines)

    # Coordinate frame at origin to show original XYZ axes.
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    print("\nOpening viewer (close window to exit)...")
    o3d.visualization.draw(
        geoms,
        title="Mesh COM / OBB (raw vs recentered)",
        bg_color=(1.0, 1.0, 1.0, 1.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect mesh COM/OBB before and after recentering.")
    parser.add_argument(
        "mesh_path",
        nargs="?",
        default="mytest/bunny_low_10.stl",
        help="Path to mesh file (default: mytest/bunny_low_10.stl)",
    )
    args = parser.parse_args()
    mesh_path = args.mesh_path

    print("Open3D version:", o3d.__version__)
    print("Loading mesh from:", mesh_path)

    # Raw mesh (for display and raw COM/OBB stats).
    raw_mesh = o3d.io.read_triangle_mesh(mesh_path)
    if raw_mesh.is_empty():
        raise SystemExit("Loaded mesh is empty.")
    raw_mesh.compute_vertex_normals()
    raw_mesh.compute_triangle_normals()

    raw_vertices = np.asarray(raw_mesh.vertices, dtype=np.float64)
    raw_triangles = np.asarray(raw_mesh.triangles, dtype=np.int32)
    raw_obb = raw_mesh.get_oriented_bounding_box()
    raw_watertight = bool(raw_mesh.is_watertight())
    raw_com = Mesh(raw_mesh, warn_if_not_watertight=False).compute_center_of_mass()

    print("\n=== Raw mesh ===")
    print(f"vertices: {len(raw_vertices)}, triangles: {len(raw_triangles)}")
    print(f"watertight: {raw_watertight}")
    print(f"raw COM: {raw_com}")
    print(f"raw OBB center: {np.asarray(raw_obb.center)}, extent: {np.asarray(raw_obb.extent)}")

    # Recenter using elastica.mesh.Mesh (operates on its own copy).
    mesh_for_elastica = o3d.io.read_triangle_mesh(mesh_path)
    mesh_for_elastica.compute_vertex_normals()
    mesh_for_elastica.compute_triangle_normals()
    mesh_centered = Mesh(mesh_for_elastica)
    centered_mesh = mesh_centered.mesh
    centered_obb = centered_mesh.get_oriented_bounding_box()
    centered_com = mesh_centered.compute_center_of_mass()

    print("\n=== Recentered mesh (elastica.mesh.Mesh) ===")
    print(f"centered COM (should be near zero): {centered_com}")
    print(f"centered OBB center: {np.asarray(centered_obb.center)}, extent: {np.asarray(centered_obb.extent)}")

    visualize_combined(raw_mesh, raw_com, raw_obb, centered_mesh, centered_com, centered_obb)


if __name__ == "__main__":
    main()
