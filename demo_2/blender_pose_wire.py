import bpy
import numpy as np
from mathutils import Matrix, Vector

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

NPZ_PATH = "/Users/lehongwang/Desktop/PyElastica-Mesh/demo_2/bunny_fixed_waypoint_rod_4view_n100_E200000_d0p4_sd0p04_cf5_state_1.npz"
ARMATURE_NAME = "Armature.001_CHAIN.001"

START_FRAME = 1
FPS = 24
USE_TIME_ARRAY = True
USE_DIRECTOR = True
APPLY_BONE_STRETCH = True
CLEAR_EXISTING_ANIMATION = True
ENFORCE_INDEPENDENT_BONES = True

# If sim coordinates differ from Blender world, set a transform here.
SIM_TO_WORLD = Matrix.Identity(4)


# ---------------------------------------------------------------------
# Helpers (kept consistent with tests/npz_to_wire.py logic)
# ---------------------------------------------------------------------

def np_vec3_to_world(v3_np):
    v = Vector((float(v3_np[0]), float(v3_np[1]), float(v3_np[2])))
    v4 = SIM_TO_WORLD @ Vector((v.x, v.y, v.z, 1.0))
    return Vector((v4.x, v4.y, v4.z))


def ordered_pose_bones(arm_obj, m_expected):
    bones = list(arm_obj.pose.bones)
    bones.sort(key=lambda pb: arm_obj.data.bones.find(pb.name))
    if len(bones) != m_expected:
        raise RuntimeError(f"{arm_obj.name}: has {len(bones)} bones, expected {m_expected}")
    return bones


def enforce_independent_bones(arm_obj):
    """
    npz_to_wire uses independent bones for world-absolute directors.
    If this armature is connected/parented chain style, disconnect it.
    """
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    for eb in arm_obj.data.edit_bones:
        eb.use_connect = False
        eb.parent = None
    bpy.ops.object.mode_set(mode="OBJECT")


def make_R_world_from_dir(D):
    """
    STRICT mapping from npz_to_wire:
      b = D[0], n = D[1], t = D[2] (world vectors)
    bone local axes: +X=b, +Y=t, +Z=n
    """
    bvec = Vector((float(D[0, 0]), float(D[0, 1]), float(D[0, 2])))
    nvec = Vector((float(D[1, 0]), float(D[1, 1]), float(D[1, 2])))
    tvec = Vector((float(D[2, 0]), float(D[2, 1]), float(D[2, 2])))

    if tvec.length < 1e-12:
        tvec = Vector((0, 1, 0))
    tvec.normalize()

    nvec = nvec - nvec.dot(tvec) * tvec
    if nvec.length < 1e-12:
        nvec = tvec.orthogonal()
    nvec.normalize()

    bvec = tvec.cross(nvec)
    if bvec.length < 1e-12:
        bvec = Vector((1, 0, 0))
    bvec.normalize()

    return Matrix((bvec, tvec, nvec)).transposed()


def make_R_world_from_tangent(tvec, n_hint=None):
    """
    Build a world rotation from tangent only:
      local +Y -> tangent
      local +Z -> computed normal (from hint / fallback up)
      local +X -> binormal
    """
    tvec = Vector((float(tvec[0]), float(tvec[1]), float(tvec[2])))
    if tvec.length < 1e-12:
        tvec = Vector((0.0, 1.0, 0.0))
    tvec.normalize()

    if n_hint is None or n_hint.length < 1e-12:
        nvec = Vector((0.0, 0.0, 1.0))
        if abs(nvec.dot(tvec)) > 0.99:
            nvec = Vector((1.0, 0.0, 0.0))
    else:
        nvec = n_hint.copy()

    # Gram-Schmidt against tangent
    nvec = nvec - nvec.dot(tvec) * tvec
    if nvec.length < 1e-12:
        nvec = tvec.orthogonal()
    nvec.normalize()

    bvec = tvec.cross(nvec)
    if bvec.length < 1e-12:
        bvec = Vector((1.0, 0.0, 0.0))
    bvec.normalize()

    # Recompute n for strict orthonormal basis
    nvec = bvec.cross(tvec)
    if nvec.length < 1e-12:
        nvec = Vector((0.0, 0.0, 1.0))
    nvec.normalize()

    return Matrix((bvec, tvec, nvec)).transposed(), nvec


def apply_pose_world_absolute(arm_obj, pos_3_mp1, dir_3_3_m=None, use_director=True, apply_stretch=True):
    mp1 = pos_3_mp1.shape[-1]
    m_pos = mp1 - 1

    if use_director:
        if dir_3_3_m is None:
            raise RuntimeError("use_director=True but no director array provided.")
        m = dir_3_3_m.shape[-1]
        if mp1 != m + 1:
            raise RuntimeError(f"pos mp1={mp1} but dirs m={m}")
    else:
        m = m_pos

    pose_bones = ordered_pose_bones(arm_obj, m)
    world_to_arm = arm_obj.matrix_world.inverted()
    # Keep rotation orthonormal if armature object scale is not identity.
    world_to_arm_rot = arm_obj.matrix_world.to_3x3().normalized().inverted()
    n_hint = None

    for i, pb in enumerate(pose_bones):
        pb.rotation_mode = "QUATERNION"
        ei = min(i, m - 1)

        head_w = np_vec3_to_world(pos_3_mp1[:, i])
        tail_w = np_vec3_to_world(pos_3_mp1[:, i + 1])

        if use_director:
            Rw = make_R_world_from_dir(dir_3_3_m[:, :, ei])
        else:
            seg_vec = tail_w - head_w
            Rw, n_hint = make_R_world_from_tangent(seg_vec, n_hint=n_hint)

        head_a = world_to_arm @ head_w
        Ra = world_to_arm_rot @ Rw

        M = Ra.to_4x4()
        M.translation = head_a
        pb.matrix = M

        if apply_stretch:
            seg_len = max((tail_w - head_w).length, 1e-12)
            rest_len = max(pb.bone.length, 1e-12)
            pb.scale = Vector((1.0, seg_len / rest_len, 1.0))
        else:
            pb.scale = Vector((1.0, 1.0, 1.0))


def keyframe_pose_bones(pose_bones, frame):
    for pb in pose_bones:
        pb.keyframe_insert(data_path="location", frame=frame)
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        pb.keyframe_insert(data_path="scale", frame=frame)


def load_rod_series(npz_path, use_director=True):
    data = np.load(npz_path)
    required = ["rod_position"]
    if use_director:
        required.append("rod_director")
    for k in required:
        if k not in data:
            raise KeyError(f"NPZ missing '{k}'. Keys={list(data.keys())}")

    pos = data["rod_position"]   # expected (T,3,M+1)
    dirs = data["rod_director"] if "rod_director" in data else None

    if pos.ndim != 3 or pos.shape[1] != 3:
        raise ValueError(f"rod_position must be (T,3,M+1), got {pos.shape}")

    T = pos.shape[0]
    if use_director:
        if dirs is None:
            raise ValueError("Expected rod_director array but it is missing.")
        if dirs.ndim != 4 or dirs.shape[1:3] != (3, 3):
            raise ValueError(f"rod_director must be (T,3,3,M), got {dirs.shape}")
        Td = dirs.shape[0]
        if T != Td:
            raise ValueError(f"time dim mismatch: rod_position T={T}, rod_director T={Td}")
        if pos.shape[2] != dirs.shape[3] + 1:
            raise ValueError(
                "Need rod_position last dim == rod_director last dim + 1. "
                f"Got {pos.shape[2]} vs {dirs.shape[3]}"
            )

    time = data["time"] if ("time" in data and np.asarray(data["time"]).ndim == 1) else None
    return pos, dirs, time


def frame_for_index(i, time_arr, start_frame, fps, use_time_array):
    if use_time_array and time_arr is not None and len(time_arr) > 0:
        t0 = float(time_arr[0])
        ti = float(time_arr[i])
        return start_frame + int(round((ti - t0) * fps))
    return start_frame + i


def main():
    arm_obj = bpy.data.objects.get(ARMATURE_NAME)
    if arm_obj is None or arm_obj.type != "ARMATURE":
        raise RuntimeError(f"Armature '{ARMATURE_NAME}' not found or not an ARMATURE object.")

    pos_t_3_mp1, dir_t_3_3_m, time_arr = load_rod_series(NPZ_PATH, use_director=USE_DIRECTOR)
    T = pos_t_3_mp1.shape[0]
    m = (dir_t_3_3_m.shape[-1] if USE_DIRECTOR else (pos_t_3_mp1.shape[-1] - 1))

    pose_bones = ordered_pose_bones(arm_obj, m)

    if ENFORCE_INDEPENDENT_BONES:
        enforce_independent_bones(arm_obj)
        pose_bones = ordered_pose_bones(arm_obj, m)

    if CLEAR_EXISTING_ANIMATION:
        arm_obj.animation_data_clear()

    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()

    action_name = f"{arm_obj.name}_PoseFromNPZ"
    arm_obj.animation_data.action = bpy.data.actions.new(name=action_name)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    first_frame = frame_for_index(0, time_arr, START_FRAME, FPS, USE_TIME_ARRAY)
    last_frame = first_frame

    for ti in range(T):
        frame = frame_for_index(ti, time_arr, START_FRAME, FPS, USE_TIME_ARRAY)
        last_frame = frame
        bpy.context.scene.frame_set(frame)
        apply_pose_world_absolute(
            arm_obj,
            pos_t_3_mp1[ti],
            (dir_t_3_3_m[ti] if USE_DIRECTOR else None),
            use_director=USE_DIRECTOR,
            apply_stretch=APPLY_BONE_STRETCH,
        )
        bpy.context.view_layer.update()
        keyframe_pose_bones(pose_bones, frame)

    bpy.context.scene.frame_start = first_frame
    bpy.context.scene.frame_end = last_frame
    bpy.context.scene.frame_set(first_frame)
    bpy.ops.object.mode_set(mode="OBJECT")

    print(
        f"[pose_wire] Done. Armature='{arm_obj.name}', frames={T}, "
        f"timeline=[{first_frame}, {last_frame}], npz='{NPZ_PATH}', use_director={USE_DIRECTOR}"
    )


if __name__ == "__main__":
    main()
