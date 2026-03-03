import numpy as np
import open3d as o3d

# --- Load polyline ---
data = np.load("/Users/lehongwang/Desktop/PyElastica-Mesh/demo_2/traj.npz")
positions = data["positions"]
n = len(positions)

# Black dots at each node
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(positions)
pcd.colors = o3d.utility.Vector3dVector(np.zeros((n, 3)))  # black

# Red lines connecting consecutive nodes
lines = [[i, i + 1] for i in range(n - 1)]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(positions)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(lines), 1)))  # red

# --- Load STL mesh ---
stl_path = "/Users/lehongwang/Desktop/PyElastica-Mesh/demo_2/bunny_small.stl"  # <-- change this
mesh = o3d.io.read_triangle_mesh(stl_path)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.7, 0.7, 0.7])  # light gray

# --- Visualize ---
# Make point size larger via render option
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(line_set)
vis.add_geometry(mesh)

opt = vis.get_render_option()
opt.point_size = 8.0
opt.background_color = np.array([1, 1, 1])  # white background
opt.line_width = 2.0

vis.run()
vis.destroy_window()