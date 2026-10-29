"""Constrain the scene camera to follow the custom arc for the 82x82 forest patch.

Equation:
    2y = -41 - sqrt(r^2 - x^2)  for -28 <= x <= 28
    with r = 41 (patch center at 0,0, size 82x82).

Inside Blender:
    Run this script in the Scripting workspace.
    It builds the Camera Arc Path, sets up the Camera Look Target at (0, 0, 16),
    and attaches the Camera using Follow Path (offset_factor from 0.0 to 1.0)
    and Track To constraints.
"""
import math
import sys
import bpy
from mathutils import Matrix, Vector

R = 41.0
X_MIN = -28.0
X_MAX = 28.0
LOOK_HEIGHT = 16.0
DEFAULT_CAMERA_HEIGHT = 25.0

CAMERA = "Camera"
TARGET = "Camera Look Target"
PATH = "Camera Arc Path"
CONTROLLER = "Camera Arc Controller"


def compute_y(x, r=R):
    """2y = -41 - sqrt(r^2 - x^2) => y = (-41 - sqrt(r^2 - x^2)) / 2"""
    disc = max(0.0, r**2 - x**2)
    return (-41.0 - math.sqrt(disc)) / 2.0


def remove_previous_rig(cam):
    """Remove previous rig constraints, parents, and objects cleanly."""
    if cam.parent:
        cam.parent = None
        cam.matrix_parent_inverse = Matrix.Identity(4)
    if cam.animation_data:
        for d in list(cam.animation_data.drivers):
            cam.animation_data.drivers.remove(d)
    for c in list(cam.constraints):
        cam.constraints.remove(c)
    cam.lock_location = cam.lock_rotation = cam.lock_scale = (False, False, False)
    cam.location = (0, 0, 0)
    cam.rotation_euler = (0, 0, 0)

    for name in ("Camera Arc Pivot", TARGET, PATH, CONTROLLER, "Camera Orbit Point"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def build_arc_curve(scene, r=R, cam_height=DEFAULT_CAMERA_HEIGHT, num_points=113):
    """Build a 3D NURBS/Poly curve following 2y = -41 - sqrt(r^2 - x^2) for -28 <= x <= 28."""
    curve = bpy.data.curves.new(PATH, 'CURVE')
    curve.dimensions = '3D'
    spline = curve.splines.new('POLY')
    spline.points.add(num_points - 1)

    for i in range(num_points):
        t = i / (num_points - 1)
        x = X_MIN + (X_MAX - X_MIN) * t
        y = compute_y(x, r)
        spline.points[i].co = (x, y, cam_height, 1.0)

    path_obj = bpy.data.objects.new(PATH, curve)
    scene.collection.objects.link(path_obj)
    path_obj.hide_render = True
    path_obj.color = (0.8, 0.1, 0.1, 1.0)
    return path_obj


def build(scene):
    cam = bpy.data.objects.get(CAMERA)
    if cam is None:
        # Find active camera or create one
        cam = scene.camera
        if cam is None:
            cam_data = bpy.data.cameras.new(CAMERA)
            cam = bpy.data.objects.new(CAMERA, cam_data)
            scene.collection.objects.link(cam)
    scene.camera = cam

    # Determine camera height from existing camera or default
    cam_height = cam.location.z if abs(cam.location.z) > 1.0 else DEFAULT_CAMERA_HEIGHT
    remove_previous_rig(cam)

    # 1. Target at center of 82x82 patch (0, 0, LOOK_HEIGHT)
    target = bpy.data.objects.new(TARGET, None)
    target.empty_display_type = 'PLAIN_AXES'
    target.empty_display_size = 5
    target.location = (0.0, 0.0, LOOK_HEIGHT)
    scene.collection.objects.link(target)

    # 2. Mathematical Arc Path Curve
    path_obj = build_arc_curve(scene, r=R, cam_height=cam_height)

    # 3. Follow Path Constraint
    fp = cam.constraints.new('FOLLOW_PATH')
    fp.name = "Arc: Follow Path"
    fp.target = path_obj
    fp.use_curve_follow = False
    fp.use_fixed_location = True
    fp.offset_factor = 0.5  # start at center (x = 0, y = -41)

    # 4. Track To Constraint (aim at patch center)
    track = cam.constraints.new('TRACK_TO')
    track.name = "Arc: Face Center"
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    bpy.context.view_layer.update()
    print(f"Custom arc rig built successfully! R={R}, x in [{X_MIN}, {X_MAX}], height={cam_height:.1f}")


if __name__ == "__main__":
    build(bpy.context.scene)
