import bpy
import numpy as np

obj = bpy.context.active_object

if obj is None or obj.type != 'CURVE':
    raise ValueError("Please select a curve object")

mat = obj.matrix_world

# Resolution per segment (uses the curve's own resolution setting)
resolution = obj.data.resolution_u

points = []
for spline in obj.data.splines:
    if spline.type != 'BEZIER':
        continue
    bezier_points = spline.bezier_points
    n = len(bezier_points)
    if n < 2:
        continue

    num_segments = n if spline.use_cyclic_u else n - 1

    for i in range(num_segments):
        p0 = bezier_points[i]
        p1 = bezier_points[(i + 1) % n]

        # Cubic bezier: start, handle_right, handle_left, end
        a = p0.co
        b = p0.handle_right
        c = p1.handle_left
        d = p1.co

        for j in range(resolution):
            t = j / resolution
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt

            co = mt3 * a + 3 * mt2 * t * b + 3 * mt * t2 * c + t3 * d
            co = mat @ co
            points.append([co.x, co.y, co.z])

    # Add last point (or close loop)
    if not spline.use_cyclic_u:
        co = mat @ bezier_points[-1].co
        points.append([co.x, co.y, co.z])

positions = np.array(points, dtype=np.float64)
print(f"Exported {len(positions)} points, shape: {positions.shape}")

np.savez("/Users/lehongwang/Desktop/PyElastica-Mesh/demo_2/traj.npz", positions=positions)
print("Saved to traj.npz")