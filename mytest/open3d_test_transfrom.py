#!/usr/bin/env python3
import numpy as np
import open3d as o3d


def load_bunny_legacy():
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(f"Loaded bunny: {len(mesh.triangles)} triangles, {len(mesh.vertices)} vertices")
    return mesh


def make_example_rigid_transform():
    """
    Rigid SE(3) transform T_world_from_mesh:
      rotate around Z then translate.
    """
    angle = np.deg2rad(35.0)
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    t = np.array([0.02, -0.015, 0.01], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rz
    T[:3, 3] = t
    return T


def apply_transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    pts = np.asarray(pts)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert T.shape == (4, 4)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    out = (T @ pts_h.T).T
    return out[:, :3]


def build_scene_from_arrays(V_np: np.ndarray, F_np: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    """
    Build RaycastingScene directly from arrays so triangle indexing is exactly F_np.
    V_np: (N,3) float32/float64
    F_np: (M,3) uint32/int
    """
    scene = o3d.t.geometry.RaycastingScene()
    V = o3d.core.Tensor(V_np.astype(np.float32))
    F = o3d.core.Tensor(F_np.astype(np.uint32))
    scene.add_triangles(V, F)
    return scene


def query_scene(scene: o3d.t.geometry.RaycastingScene, query_pts: np.ndarray):
    """
    Returns:
      closest_pts: (N,3) float32
      tri_ids: (N,) int64
      uvs: (N,2) float32  (barycentric u,v; w = 1-u-v)
      sdf: (N,) float32  (signed distance; for non-watertight meshes sign may be unreliable)
    """
    q = o3d.core.Tensor(np.asarray(query_pts).astype(np.float32))
    ans = scene.compute_closest_points(q)
    closest = ans["points"].numpy()
    tri_ids = ans["primitive_ids"].numpy().astype(np.int64)
    uvs = ans["primitive_uvs"].numpy()
    sdf = scene.compute_signed_distance(q).numpy()
    return closest, tri_ids, uvs, sdf


def uvw_from_uv(uv: np.ndarray):
    u, v = float(uv[0]), float(uv[1])
    w = 1.0 - u - v
    return u, v, w


def near_boundary(uv: np.ndarray, eps=1e-6) -> bool:
    u, v, w = uvw_from_uv(uv)
    return (u <= eps) or (v <= eps) or (w <= eps)


def sample_world_queries_from_transformed_bbox(mesh_legacy, T_world_from_mesh, n=5, pad=0.01, seed=7):
    np.random.seed(seed)
    bbox = mesh_legacy.get_axis_aligned_bounding_box()
    minb = bbox.min_bound - pad
    maxb = bbox.max_bound + pad

    corners = np.array([
        [minb[0], minb[1], minb[2]],
        [minb[0], minb[1], maxb[2]],
        [minb[0], maxb[1], minb[2]],
        [minb[0], maxb[1], maxb[2]],
        [maxb[0], minb[1], minb[2]],
        [maxb[0], minb[1], maxb[2]],
        [maxb[0], maxb[1], minb[2]],
        [maxb[0], maxb[1], maxb[2]],
    ], dtype=np.float64)

    corners_w = apply_transform_points(corners, T_world_from_mesh)
    minw = corners_w.min(axis=0)
    maxw = corners_w.max(axis=0)

    return np.random.uniform(low=minw, high=maxw, size=(n, 3)).astype(np.float64)


def main():
    print("Open3D version:", o3d.__version__)
    mesh = load_bunny_legacy()

    # Extract arrays ONCE (canonical indexing)
    V0 = np.asarray(mesh.vertices).astype(np.float32)
    F0 = np.asarray(mesh.triangles).astype(np.uint32)

    # Rigid transform
    T_world_from_mesh = make_example_rigid_transform()
    T_mesh_from_world = np.linalg.inv(T_world_from_mesh)

    # Build scene in mesh frame ONCE (BVH built once)
    scene_mesh = build_scene_from_arrays(V0, F0)

    # Queries in world
    query_world = sample_world_queries_from_transformed_bbox(
        mesh, T_world_from_mesh, n=5, pad=0.01, seed=7
    )

    # ---------- A) FAST PATH (no BVH rebuild) ----------
    # world -> mesh
    query_mesh = apply_transform_points(query_world, T_mesh_from_world).astype(np.float32)

    closest_mesh_A, tri_A, uvs_A, sdf_A = query_scene(scene_mesh, query_mesh)
    # closest mesh -> world
    closest_world_A = apply_transform_points(closest_mesh_A.astype(np.float64), T_world_from_mesh).astype(np.float64)
    dist_A = np.linalg.norm(query_world - closest_world_A, axis=1)

    # ---------- B) SLOW PATH (transform mesh + rebuild) ----------
    Vw = apply_transform_points(V0.astype(np.float64), T_world_from_mesh).astype(np.float32)
    scene_world = build_scene_from_arrays(Vw, F0)

    closest_world_B, tri_B, uvs_B, sdf_B = query_scene(scene_world, query_world.astype(np.float32))
    closest_world_B = closest_world_B.astype(np.float64)
    dist_B = np.linalg.norm(query_world - closest_world_B, axis=1)

    # ---------- Print BOTH result sets ----------
    print("\n=== A) Fast path (inverse-transform queries; NO BVH rebuild) ===")
    for i in range(len(query_world)):
        u, v, w = uvw_from_uv(uvs_A[i])
        print(
            f"[{i}] q_w={query_world[i]} | "
            f"cp_w={closest_world_A[i]} | "
            f"dist={dist_A[i]:.6f} | sdf={float(sdf_A[i]):.6f} | "
            f"tri_id={int(tri_A[i])} | uvw=({u:.6f},{v:.6f},{w:.6f})"
        )

    print("\n=== B) Slow path (transform mesh + rebuild BVH) ===")
    for i in range(len(query_world)):
        u, v, w = uvw_from_uv(uvs_B[i])
        print(
            f"[{i}] q_w={query_world[i]} | "
            f"cp_w={closest_world_B[i]} | "
            f"dist={dist_B[i]:.6f} | sdf={float(sdf_B[i]):.6f} | "
            f"tri_id={int(tri_B[i])} | uvw=({u:.6f},{v:.6f},{w:.6f})"
        )

    # ---------- Verification / comparison ----------
    cp_err = np.linalg.norm(closest_world_A - closest_world_B, axis=1)
    sdf_err = np.abs(sdf_A.astype(np.float64) - sdf_B.astype(np.float64))
    tri_match = (tri_A == tri_B)

    print("\n=== Comparison (A vs B) ===")
    print("max closest-point error:", float(cp_err.max()))
    print("max signed-distance error:", float(sdf_err.max()))
    print("triangle id matches:", tri_match.tolist())

    print("\nPer-point detailed comparison:")
    for i in range(len(query_world)):
        nbA = near_boundary(uvs_A[i], eps=1e-6)
        nbB = near_boundary(uvs_B[i], eps=1e-6)
        print(
            f"[{i}] cp_err={cp_err[i]:.3e}, sdf_err={sdf_err[i]:.3e}, "
            f"tri_match={bool(tri_match[i])}, near_boundary(A)={nbA}, near_boundary(B)={nbB} | "
            f"triA={int(tri_A[i])}, triB={int(tri_B[i])}"
        )

    # If you want a strict pass criterion:
    tol_pos = 1e-6
    tol_sdf = 1e-6
    ok_geom = (cp_err.max() <= tol_pos) and (sdf_err.max() <= tol_sdf)

    # Triangle IDs can legitimately differ when closest point lies on/near an edge/vertex.
    # We'll only require ID match when BOTH queries are not near boundaries.
    strict_id_ok = True
    for i in range(len(query_world)):
        if (not near_boundary(uvs_A[i])) and (not near_boundary(uvs_B[i])):
            if not tri_match[i]:
                strict_id_ok = False

    print("\n=== Verdict ===")
    print("geometry (closest point + sdf) within tol:", ok_geom)
    print("triangle-id match away from boundaries:", strict_id_ok)
    print("OVERALL:", "PASS" if (ok_geom and strict_id_ok) else "FAIL")


if __name__ == "__main__":
    main()
