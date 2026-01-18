import open3d as o3d

m = o3d.io.read_triangle_mesh("mytest/bunny_low_10.stl")  # or .stl/.ply


# Clean + weld
m.remove_degenerate_triangles()
m.remove_duplicated_triangles()
m.remove_unreferenced_vertices()

# This is the important one for your case (24 -> 8 typically)
m.merge_close_vertices(1e-6)   # you may try 1e-5 if units are large

# After merging, run again
m.remove_duplicated_vertices()
m.remove_unreferenced_vertices()


print("verts, tris:", len(m.vertices), len(m.triangles))
print("edge_manifold:", m.is_edge_manifold())
print("vertex_manifold:", m.is_vertex_manifold())
print("self_intersect:", m.is_self_intersecting())
print("watertight:", m.is_watertight())
