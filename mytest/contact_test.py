#!/usr/bin/env python3
"""
Open3D Contact Probe (Option B) — version-compatible full script

What you asked for:
- Clean mesh first (merge close vertices / remove duplicates / etc.)
- Option B contact for WATERTIGHT:
    signed_distance = unsigned_distance * sign_from_occupancy
    (with near-surface deadband to avoid sign flicker)
- For NON-watertight:
    surface contact only => distance always positive (unsigned)
- Normal:
    use direction (closest->query) normalized;
    if distance is tiny => fall back to surface (triangle) normal
- Visualize interactively in Open3D:
    mesh + optional wireframe overlay + colored penalty vectors along a line of query points

Compatibility goals:
- Avoid TriangleMesh.clone() (uses copy.deepcopy)
- Avoid o3d.core.Tensor.norm() (uses NumPy norm for distance)
- Works with older Open3D where visualization.draw may not exist (falls back to draw_geometries)

Usage example:
python contact_test.py --mesh your.obj --start "0 0 0" --end "0 0 0.2" --num_query 150 --radius 0.01 --show_wireframe
"""

import argparse
import copy
import numpy as np
import open3d as o3d


# ----------------------------
# helpers
# ----------------------------
def parse_xyz(s: str) -> np.ndarray:
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 floats, got: {s}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def viridis_colors(t: np.ndarray) -> np.ndarray:
    """Small viridis-like colormap approximation; no matplotlib dependency."""
    stops = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    cols = np.array(
        [
            [0.267, 0.005, 0.329],
            [0.283, 0.141, 0.458],
            [0.254, 0.265, 0.530],
            [0.207, 0.372, 0.553],
            [0.164, 0.471, 0.558],
            [0.993, 0.906, 0.144],
        ],
        dtype=np.float64,
    )
    t = np.clip(t, 0.0, 1.0)
    out = np.zeros((t.shape[0], 3), dtype=np.float64)

    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        m = (t >= a) & (t <= b)
        if not np.any(m):
            continue
        w = (t[m] - a) / max(b - a, 1e-12)
        out[m] = (1.0 - w)[:, None] * cols[i] + w[:, None] * cols[i + 1]
    return out


# ----------------------------
# mesh cleaning
# ----------------------------
def clean_mesh(mesh: o3d.geometry.TriangleMesh,
               merge_tol: float,
               remove_non_manifold: bool) -> o3d.geometry.TriangleMesh:
    """Best-effort cleaning that works across Open3D versions."""
    mesh = copy.deepcopy(mesh)

    # Merge close vertices (API differs by version; best-effort)
    if merge_tol > 0:
        if hasattr(mesh, "merge_close_vertices"):
            mesh.merge_close_vertices(merge_tol)
        elif hasattr(mesh, "MergeCloseVertices"):
            mesh.MergeCloseVertices(merge_tol)

    if hasattr(mesh, "remove_duplicated_vertices"):
        mesh.remove_duplicated_vertices()
    if hasattr(mesh, "remove_duplicated_triangles"):
        mesh.remove_duplicated_triangles()
    if hasattr(mesh, "remove_degenerate_triangles"):
        mesh.remove_degenerate_triangles()
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()

    if remove_non_manifold and hasattr(mesh, "remove_non_manifold_edges"):
        mesh.remove_non_manifold_edges()

    # Ensure normals exist (needed for small-distance fallback)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def make_scene(mesh_legacy: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)
    return scene


# ----------------------------
# contact probe (Option B)
# ----------------------------
def contact_probe_option_b(
    scene: o3d.t.geometry.RaycastingScene,
    mesh_legacy: o3d.geometry.TriangleMesh,
    points: np.ndarray,
    radius: float,
    trust_occupancy_sign: bool,
    eps_dist: float,
    eps_surface: float,
    occupancy_nsamples: int,
) -> dict:
    """
    Option B:
      - "watertight path": signed distance = unsigned_dist * sign(occupancy), with deadband eps_surface
      - "surface-only path": distance always positive (unsigned)

    Normal:
      - if unsigned_dist > eps_dist: normalize(direction=point-closest)
      - else: triangle normal fallback
    Penalty:
      penetration = max(radius - distance, 0)
      penalty_vec = penetration * normal
    """
    # Query tensor
    pts_f32 = points.astype(np.float32, copy=False)
    q = o3d.core.Tensor(pts_f32, dtype=o3d.core.Dtype.Float32)

    # Closest-point query
    cp = scene.compute_closest_points(q)
    closest_t = cp["points"]  # Tensor (N,3)
    tri_ids = cp["primitive_ids"].numpy().astype(np.int64)

    # Convert closest to numpy once (for compatibility & speed enough for testing)
    closest = closest_t.numpy().astype(np.float64)  # (N,3)

    # Unsigned distance = ||p - closest||
    direction = points - closest
    dist = np.linalg.norm(direction, axis=1)  # (N,)

    # Triangle normals for fallback
    tri_normals = np.asarray(mesh_legacy.triangle_normals, dtype=np.float64)
    valid = (tri_ids >= 0) & (tri_ids < tri_normals.shape[0])
    face_n = np.zeros_like(points, dtype=np.float64)
    face_n[valid] = tri_normals[tri_ids[valid]]

    # Base normal
    normal = np.zeros_like(points, dtype=np.float64)
    stable = dist > eps_dist
    normal[stable] = direction[stable] / dist[stable, None]
    normal[~stable] = face_n[~stable]

    # If still bad, default to +Z
    nrm = np.linalg.norm(normal, axis=1)
    bad = nrm < 1e-12
    normal[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1), 1e-12)[:, None]

    # Signed distance logic
    if trust_occupancy_sign:
        # occupancy in [0,1] (or bool-like). True means inside.
        # Some versions require nsamples arg, some accept it; best-effort.
        try:
            occ = scene.compute_occupancy(q, nsamples=int(occupancy_nsamples)).numpy()
        except TypeError:
            occ = scene.compute_occupancy(q).numpy()

        inside = occ > 0.5

        # Deadband near surface: do NOT flip sign if extremely close
        inside = inside & (dist >= eps_surface)

        sign = np.where(inside, -1.0, 1.0)
        signed_distance = dist * sign

        # Make normal "outward": inside => flip
        normal = normal * sign[:, None]
    else:
        # Surface-only: always positive
        signed_distance = dist

    # Penalty vector
    penetration = np.maximum(radius - signed_distance, 0.0)
    penalty_vec = penetration[:, None] * normal
    penalty_mag = np.linalg.norm(penalty_vec, axis=1)

    return dict(
        points=points,
        closest=closest,
        tri_ids=tri_ids,
        distance=signed_distance,
        penetration=penetration,
        normal=normal,
        penalty_vec=penalty_vec,
        penalty_mag=penalty_mag,
    )


# ----------------------------
# visualization
# ----------------------------
def build_lineset_from_vectors(origins: np.ndarray,
                               vecs: np.ndarray,
                               mags: np.ndarray,
                               min_draw: float = 0.0) -> o3d.geometry.LineSet:
    """
    LineSet: each vector is drawn as line origin -> origin + vec.
    Colors follow magnitude. Optionally skip tiny vectors via min_draw.
    """
    if origins.shape[0] == 0:
        return o3d.geometry.LineSet()

    if min_draw > 0:
        keep = mags >= min_draw
        origins = origins[keep]
        vecs = vecs[keep]
        mags = mags[keep]

    N = origins.shape[0]
    ends = origins + vecs
    pts = np.vstack([origins, ends])  # (2N,3)
    lines = np.array([[i, i + N] for i in range(N)], dtype=np.int32)

    max_mag = float(np.max(mags)) if mags.size else 0.0
    if max_mag > 0:
        rgb = viridis_colors(mags / max_mag)
    else:
        rgb = np.tile(np.array([[0.2, 0.8, 0.2]], dtype=np.float64), (N, 1))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(rgb)
    return ls


def visualize_interactive(mesh: o3d.geometry.TriangleMesh,
                          query_points: np.ndarray,
                          penalty_lines: o3d.geometry.LineSet,
                          show_wireframe: bool):
    # Query points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(query_points)
    pcd.paint_uniform_color([1.0, 0.3, 0.3])

    # Wireframe overlay (optional)
    geoms = []

    # Semi-transparent mesh (classic visualizer does not truly support alpha reliably)
    # We'll still set a light color; you can toggle wireframe overlay for clarity.
    mesh_vis = copy.deepcopy(mesh)
    mesh_vis.paint_uniform_color([0.75, 0.75, 0.75])
    geoms.append(mesh_vis)

    if show_wireframe:
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_vis)
        wire.paint_uniform_color([0.05, 0.05, 0.05])
        geoms.append(wire)

    geoms.extend([penalty_lines, pcd])

    # Use classic draw_geometries for maximum compatibility
    o3d.visualization.draw_geometries(
        geoms,
        window_name="Open3D Contact Probe (Option B)",
        mesh_show_back_face=True,
    )


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Path to triangle mesh file (ply/obj/stl/glb/...).")
    ap.add_argument("--start", required=True, help='Start xyz, e.g. "0 0 0" or "0,0,0"')
    ap.add_argument("--end", required=True, help='End xyz, e.g. "0 0 1"')
    ap.add_argument("--num_query", type=int, default=150)
    ap.add_argument("--radius", type=float, default=0.01)

    ap.add_argument("--merge_tol", type=float, default=1e-6,
                    help="Merge-close-vertices tolerance (mesh units).")
    ap.add_argument("--remove_non_manifold", action="store_true",
                    help="Attempt to remove non-manifold edges (can change topology).")
    ap.add_argument("--no_clean", action="store_true", help="Disable mesh cleanup.")

    ap.add_argument("--eps_dist", type=float, default=1e-9,
                    help="If unsigned distance <= eps_dist, use triangle normal fallback.")
    ap.add_argument("--eps_surface", type=float, default=1e-6,
                    help="Deadband for occupancy sign near surface (prevents flicker).")
    ap.add_argument("--occupancy_nsamples", type=int, default=1,
                    help="Occupancy nsamples (2..4 may be steadier but slower).")

    ap.add_argument("--force_surface", action="store_true",
                    help="Force surface-only mode even if mesh is watertight.")
    ap.add_argument("--show_wireframe", action="store_true")
    ap.add_argument("--min_draw", type=float, default=0.0,
                    help="Skip drawing penalty vectors smaller than this magnitude.")
    args = ap.parse_args()

    start = parse_xyz(args.start)
    end = parse_xyz(args.end)

    if args.num_query < 2:
        raise ValueError("--num_query must be >= 2")

    # Sample points along the line
    ts = np.linspace(0.0, 1.0, args.num_query, dtype=np.float64)
    points = start[None, :] * (1.0 - ts[:, None]) + end[None, :] * ts[:, None]

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh.is_empty() or not mesh.has_triangles():
        raise RuntimeError(f"Failed to load a valid triangle mesh: {args.mesh}")

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    wt_before = bool(mesh.is_watertight())
    print(f"[mesh] watertight before clean: {wt_before}")
    print(f"[mesh] #verts={len(mesh.vertices)} #tris={len(mesh.triangles)}")

    # Clean mesh
    if not args.no_clean:
        mesh = clean_mesh(mesh, merge_tol=float(args.merge_tol), remove_non_manifold=bool(args.remove_non_manifold))
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

    wt_after = bool(mesh.is_watertight())
    print(f"[mesh] watertight after clean:  {wt_after}")
    print(f"[mesh] #verts={len(mesh.vertices)} #tris={len(mesh.triangles)}")

    # Build scene
    scene = make_scene(mesh)

    # Decide mode
    trust_occ_sign = wt_after and (not args.force_surface)
    print(f"[contact] mode = {'OptionB(watertight: occupancy sign)' if trust_occ_sign else 'SurfaceOnly(non-watertight)'}")

    # Probe
    res = contact_probe_option_b(
        scene=scene,
        mesh_legacy=mesh,
        points=points,
        radius=float(args.radius),
        trust_occupancy_sign=trust_occ_sign,
        eps_dist=float(args.eps_dist),
        eps_surface=float(args.eps_surface),
        occupancy_nsamples=int(args.occupancy_nsamples),
    )

    # Build colored penalty vectors
    penalty_lines = build_lineset_from_vectors(
        origins=res["points"],
        vecs=res["penalty_vec"],
        mags=res["penalty_mag"],
        min_draw=float(args.min_draw),
    )

    # Visualize interactive Open3D window
    visualize_interactive(
        mesh=mesh,
        query_points=res["points"],
        penalty_lines=penalty_lines,
        show_wireframe=bool(args.show_wireframe),
    )


if __name__ == "__main__":
    main()
