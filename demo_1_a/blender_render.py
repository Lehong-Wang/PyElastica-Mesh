# Blender script: load PyElastica state .npz with data["position"] shaped (T, 3, n)
# and create a single Bezier curve from one time index t (using the 3 x n slice).
#
# File you gave:
# /Users/lehongwang/Desktop/PyElastica-Mesh/demo_1_a/static_knot_2_t50_dt2e-05_n50_r0.0175_y4e+06_state.npz
#
# Usage:
# 1) Blender -> Scripting tab -> Text Editor -> paste -> Run Script
# 2) Set NPZ_PATH and T_INDEX below.

import osa
import bpy
import numpy as np
from mathutils import Vector

# ------------------- Settings -------------------
NPZ_PATH = "/Users/lehongwang/Desktop/PyElastica-Mesh/demo_1_a/static_knot_2_t50_dt2e-05_n50_r0.0175_y4e+06_state.npz"
KEY = "position"       # expects shape (T, 3, n)
T_INDEX = 650           # pick which time step you want (0-based)
#T_INDEX = 1620
#T_INDEX = 3250
SCALE = 1.0            # multiply coordinates by this (e.g. 0.001 if units are mm)
DECIMATE_EVERY = 1     # keep every k-th node (>=1)
HANDLE_TYPE = "AUTO"   # "AUTO", "ALIGNED", "FREE" etc.

CURVE_DATA_NAME = "PyElasticaBezierCurve"
CURVE_OBJ_NAME = "PyElasticaBezierCurveObj"
DELETE_EXISTING = True

# Optional bevel (tube thickness)
BEVEL_DEPTH = 0.0      # e.g. 0.002 to see a tube
BEVEL_RESOLUTION = 6

# Axis fixes if needed
SWAP_YZ = False
NEGATE_X = False
NEGATE_Y = False
NEGATE_Z = False


# ------------------- Helpers -------------------
def delete_existing(obj_name: str):
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        return
    for col in list(obj.users_collection):
        col.objects.unlink(obj)
    if obj.data and obj.data.users <= 1:
        bpy.data.curves.remove(obj.data, do_unlink=True)
    bpy.data.objects.remove(obj, do_unlink=True)

def load_points(npz_path: str, key: str, t_index: int) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    if key not in data:
        raise KeyError(f'Key "{key}" not found. Available keys: {list(data.keys())}')

    arr = np.asarray(data[key])
    # Expect (T, 3, n)
    if arr.ndim != 3 or arr.shape[1] != 3:
        raise ValueError(f'Expected shape (T, 3, n). Got {arr.shape}')

    T = arr.shape[0]
    t = int(np.clip(t_index, 0, T - 1))

    # slice is (3, n) -> convert to (n, 3)
    pts = arr[t].T  # (n, 3)
    pts = pts.astype(np.float64, copy=False)

    # decimate
    k = max(1, int(DECIMATE_EVERY))
    if k > 1:
        pts = pts[::k].copy()

    # scale
    pts *= float(SCALE)

    # axis ops
    if SWAP_YZ:
        pts = pts[:, [0, 2, 1]]
    if NEGATE_X:
        pts[:, 0] *= -1
    if NEGATE_Y:
        pts[:, 1] *= -1
    if NEGATE_Z:
        pts[:, 2] *= -1

    return pts

def make_bezier_curve(points: np.ndarray, curve_data_name: str, obj_name: str):
    if len(points) < 2:
        raise ValueError("Need at least 2 points to make a curve.")

    curve_data = bpy.data.curves.new(name=curve_data_name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 12

    curve_data.bevel_depth = float(BEVEL_DEPTH)
    curve_data.bevel_resolution = int(BEVEL_RESOLUTION)

    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(len(points) - 1)  # one already exists

    for i, p in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = Vector((float(p[0]), float(p[1]), float(p[2])))
        bp.handle_left_type = HANDLE_TYPE
        bp.handle_right_type = HANDLE_TYPE

    spline.use_cyclic_u = False

    obj = bpy.data.objects.new(obj_name, curve_data)
    bpy.context.collection.objects.link(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


# ------------------- Main -------------------
pts = load_points(NPZ_PATH, KEY, T_INDEX)

if DELETE_EXISTING:
    delete_existing(CURVE_OBJ_NAME)

obj = make_bezier_curve(pts, CURVE_DATA_NAME, CURVE_OBJ_NAME)
print(f"Created '{obj.name}' from {KEY} at t={T_INDEX}, points={len(pts)}, file={NPZ_PATH}")                                                                                                                                                        