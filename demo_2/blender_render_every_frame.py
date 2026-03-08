import os
import bpy

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

OUTPUT_DIR = "/Users/lehongwang/Desktop/Wire Paper/assets/render_bunny"
FILE_PREFIX = "bunny_"
FILE_FORMAT = "PNG"

# Use scene.frame_start/frame_end when None.
START_FRAME = None
END_FRAME = None


def resolve_frame_range(scene):
    start = scene.frame_start if START_FRAME is None else int(START_FRAME)
    end = scene.frame_end if END_FRAME is None else int(END_FRAME)
    if end < start:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")
    return start, end


def render_all_frames():
    scene = bpy.context.scene
    render_settings = scene.render
    frame_start, frame_end = resolve_frame_range(scene)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    old_filepath = render_settings.filepath
    old_format = render_settings.image_settings.file_format
    old_use_file_extension = render_settings.use_file_extension
    old_use_overwrite = render_settings.use_overwrite

    try:
        render_settings.image_settings.file_format = FILE_FORMAT
        render_settings.use_file_extension = True
        render_settings.use_overwrite = True

        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            render_settings.filepath = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}{frame:04d}")
            bpy.ops.render.render(write_still=True)
            print(f"[render_frames] Rendered frame {frame} -> {render_settings.filepath}")

    finally:
        render_settings.filepath = old_filepath
        render_settings.image_settings.file_format = old_format
        render_settings.use_file_extension = old_use_file_extension
        render_settings.use_overwrite = old_use_overwrite

    print(
        f"[render_frames] Done. frames=[{frame_start}, {frame_end}], "
        f"output_dir='{OUTPUT_DIR}', format={FILE_FORMAT}"
    )


if __name__ == "__main__":
    render_all_frames()
