"""
Keyframe the camera to glide along the custom arc path:
    2y = -41 - sqrt(r^2 - x^2) for -28 <= x <= 28 (r = 41)
Frames: 1 to 120. Press Spacebar (Play) to watch the sweep!
"""
import bpy
import os
import sys

START_FRAME = 1
END_FRAME = 120

# 1. Ensure the rig is built
cam = bpy.data.objects.get("Camera")
fp = cam.constraints.get("Arc: Follow Path") if cam else None
if not fp:
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else (
        os.path.dirname(os.path.abspath(bpy.data.filepath)) if bpy.data.filepath else os.getcwd()
    )
    camera_rig_dir = os.path.join(base_dir, "camera_rig")
    if camera_rig_dir not in sys.path:
        sys.path.append(camera_rig_dir)
    import camera_arc_rig
    camera_arc_rig.build(bpy.context.scene)
    cam = bpy.data.objects.get("Camera")
    fp = cam.constraints.get("Arc: Follow Path")

# 2. Configure timeline
scene = bpy.context.scene
scene.frame_start = START_FRAME
scene.frame_end = END_FRAME

# 3. Keyframe offset_factor on the Follow Path constraint
if cam.animation_data:
    cam.animation_data_clear()

# Frame 1: start at x = -28 (offset_factor = 0.0)
scene.frame_set(START_FRAME)
fp.offset_factor = 0.0
cam.keyframe_insert(data_path='constraints["Arc: Follow Path"].offset_factor', frame=START_FRAME)

# Frame END: end at x = +28 (offset_factor = 1.0)
scene.frame_set(END_FRAME)
fp.offset_factor = 1.0
cam.keyframe_insert(data_path='constraints["Arc: Follow Path"].offset_factor', frame=END_FRAME)

# Reset to frame 1
scene.frame_set(START_FRAME)
print(f"Custom arc animation created! Frames {START_FRAME} to {END_FRAME}. Press Spacebar to play.")
