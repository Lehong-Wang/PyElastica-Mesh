import numpy as np
import open3d as o3d


def build_colored_cube_vertexcolors():
    """
    Build a unit cube centered at origin using 24 vertices (4 per face) so that
    vertex colors can represent per-face coloring without blending.
    Face-id convention:
      0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
    """
    # Cube corners (±0.5)
    p = 0.5
    # Face colors (6)
    face_colors = np.array([
        [1.0, 0.2, 0.2],  # +X
        [0.7, 0.0, 0.0],  # -X
        [0.2, 1.0, 0.2],  # +Y
        [0.0, 0.7, 0.0],  # -Y
        [0.2, 0.2, 1.0],  # +Z
        [0.0, 0.0, 0.7],  # -Z
    ], dtype=np.float64)

    face_id_to_name = {
        0: "+X",
        1: "-X",
        2: "+Y",
        3: "-Y",
        4: "+Z",
        5: "-Z",
    }

    # Define 6 faces, each with 4 vertices in CCW order as seen from outside
    # Each face contributes 2 triangles: (0,1,2) and (0,2,3) in its local quad indexing.
    faces = [
        # +X (x = +p)
        (0, np.array([[+p, -p, -p],
                      [+p, +p, -p],
                      [+p, +p, +p],
                      [+p, -p, +p]], dtype=np.float64)),
        # -X (x = -p)  outward normal -X
        (1, np.array([[-p, -p, +p],
                      [-p, +p, +p],
                      [-p, +p, -p],
                      [-p, -p, -p]], dtype=np.float64)),
        # +Y (y = +p)
        (2, np.array([[-p, +p, -p],
                      [-p, +p, +p],
                      [+p, +p, +p],
                      [+p, +p, -p]], dtype=np.float64)),
        # -Y (y = -p)
        (3, np.array([[-p, -p, +p],
                      [-p, -p, -p],
                      [+p, -p, -p],
                      [+p, -p, +p]], dtype=np.float64)),
        # +Z (z = +p)
        (4, np.array([[-p, -p, +p],
                      [+p, -p, +p],
                      [+p, +p, +p],
                      [-p, +p, +p]], dtype=np.float64)),
        # -Z (z = -p)
        (5, np.array([[-p, -p, -p],
                      [-p, +p, -p],
                      [+p, +p, -p],
                      [+p, -p, -p]], dtype=np.float64)),
    ]

    vertices = []
    triangles = []
    vcolors = []
    tri_to_face = []

    vid = 0
    for face_id, quad in faces:
        # add 4 vertices
        vertices.append(quad)
        vcolors.append(np.tile(face_colors[face_id], (4, 1)))

        # two triangles for this quad
        triangles.append(np.array([
            [vid + 0, vid + 1, vid + 2],
            [vid + 0, vid + 2, vid + 3],
        ], dtype=np.int32))

        tri_to_face.extend([face_id, face_id])
        vid += 4

    vertices = np.vstack(vertices)
    triangles = np.vstack(triangles)
    vcolors = np.vstack(vcolors)
    tri_to_face = np.array(tri_to_face, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcolors)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh, tri_to_face, face_id_to_name



def load_bunny():
    """
    Load the Stanford Bunny mesh from Open3D's example dataset.
    Returns the legacy mesh and triangle count.
    """
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    print(f"Loaded bunny: {len(mesh.triangles)} triangles, {len(mesh.vertices)} vertices")
    return mesh


def closest_point_query(scene: o3d.t.geometry.RaycastingScene, query_pts_np: np.ndarray):
    q = o3d.core.Tensor(query_pts_np.astype(np.float32))
    ans = scene.compute_closest_points(q)
    closest_pts = ans["points"].numpy()
    prim_ids = ans["primitive_ids"].numpy().astype(np.int64)  # triangle ids
    dists = scene.compute_signed_distance(q).numpy()
    return closest_pts, prim_ids, dists


def main():
    print("Open3D version:", o3d.__version__)
    np.random.seed(10)

    mesh, tri_to_face, face_id_to_name = build_colored_cube_vertexcolors()
    # mesh = load_bunny()
    mesh = o3d.io.read_triangle_mesh("mytest/bunny_low_10_center.stl")

    mesh.merge_close_vertices(1e-6)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    mesh.scale(2.0, center=mesh.get_center())
    print(f"watertight: {mesh.is_watertight()}")
    # RaycastingScene uses the tensor mesh
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _geom_id = scene.add_triangles(tmesh)

    # 5 random query points
    query_pts = np.random.uniform(low=-1.2, high=1.2, size=(5, 3))

    query_colors = np.array([
        [1.0, 0.6, 0.0],  # orange
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 1.0, 1.0],  # cyan
        [1.0, 1.0, 0.0],  # yellow
        [0.7, 0.7, 0.7],  # gray
    ], dtype=np.float64)

    closest_pts, prim_ids, dists = closest_point_query(scene, query_pts)

    # face_ids = tri_to_face[prim_ids]
    # dists = np.linalg.norm(query_pts - closest_pts, axis=1)

    # print("\nLegend (face_id -> face):")
    # for fid in range(6):
    #     print(f"  {fid}: {face_id_to_name[fid]}")

    print("\nQueries:")
    for i in range(len(query_pts)):
        print(
            f"[{i}] q={query_pts[i]}  "
            f"closest={closest_pts[i]}  "
            f"dist={dists[i]:.6f}  "
            f"tri_id={prim_ids[i]}  "
            # f"face_id={face_ids[i]}({face_id_to_name[int(face_ids[i])]})"
        )

    # ---- Visualization ----
    geoms = [mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]

    # query spheres, closest spheres
    for i, (p, cp) in enumerate(zip(query_pts, closest_pts)):
        c = query_colors[i]

        q_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        q_sphere.paint_uniform_color(c)
        q_sphere.translate(p)

        cp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.028)
        cp_sphere.paint_uniform_color(c)
        cp_sphere.translate(cp)

        geoms.extend([q_sphere, cp_sphere])

    # lines query -> closest
    n = len(query_pts)
    line_points = np.vstack([query_pts, closest_pts])
    lines = np.array([[i, n + i] for i in range(n)], dtype=np.int32)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(query_colors)
    geoms.append(line_set)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Open3D Closest Point Query (RaycastingScene) - face IDs",
        width=1100,
        height=750,
    )


if __name__ == "__main__":
    main()
