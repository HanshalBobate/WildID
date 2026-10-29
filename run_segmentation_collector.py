"""
Tiger Labelled Segmentation Dataset Collector
==============================================
Collects pixel-accurate image pairs for training semantic segmentation models:
1. Realistic Textured Render:
   - White tiger with authentic fur texturing and shading.
   - Sculpted forest terrain, geometry nodes instanced trees, fallen leaves, and bark.
   - Natural HDR sunrise world lighting and dynamic sun angles.

2. Solid Color Segmentation Mask:
   - Tiger = Pure Solid RED   (RGB: 255, 0, 0   / #FF0000)
   - Background = Pure Solid BLUE  (RGB: 0, 0, 255   / #0000FF)  [Terrain, Trees, Leaves, Bark]
   - Sky = Pure Solid YELLOW (RGB: 255, 255, 0 / #FFFF00)  [Atmosphere, Horizon, Voids]

Features:
- Dedicated structured dataset folder:
    <out_dir>/
    ├── images/       (Realistic renders: 000001.png, 000002.png, ...)
    ├── masks/        (Pure 3-class segmentation masks: 000001.png, ...)
    ├── previews/     (Side-by-side visual validation: 000001_preview.jpg, ...)
    └── dataset_metadata.csv (All seeds, poses, coordinates, camera transforms)
- Full scene randomization:
  1. Ground Seed: procedural noise displacement for terrain.
  2. Ground Height Amplitude: variable elevation slope.
  3. Tiger Pose: chosen from 19 authentic tiger poses with canonical scale lock.
  4. 4-Paw Ground Collision: BVH raycasting snaps paws and aligns slope pitch/roll.
  5. Tree Seed: Geometry Nodes randomizes trees, fallen leaves, and bark distribution.
  6. Camera Arc Rig (animate_camera_arc.py): follows mathematical curve
     2y = -41 - sqrt(41^2 - x^2) for -28 <= x <= 28 across frames 1 to 120.
     Captures variable-angled photos (Left flank, Center frontal, Right flank)
     tracking the tiger with eye-level to elevated vantage heights.
  7. Sun Lighting: randomized sun azimuth and elevation.
- Post-quantization snapping: guarantees 100% of mask pixels are strictly one of
  the 3 exact class colors with zero anti-aliasing transition bleed.
- Dual execution:
  - Run from Windows PowerShell / Terminal: `python run_segmentation_collector.py`
  - Run headless with args: `blender -b target.blend --python run_segmentation_collector.py -- --count 50`
  - Run inside Blender GUI Scripting workspace.
"""
import os
import sys
import math
import random
import time
import csv
import argparse
from datetime import datetime

# ==============================================================================
# WRAPPER: IF RUN OUTSIDE BLENDER, AUTO-LAUNCH BLENDER
# ==============================================================================
try:
    import bpy
    # In Blender (GUI or headless), bpy.app.binary_path points to blender.exe.
    # When installed as a standalone Python pip module, binary_path is empty and no project scene is loaded.
    is_blender_proc = bool(getattr(bpy.app, "binary_path", "") and "blender" in os.path.basename(bpy.app.binary_path).lower())
    has_target_scene = bpy.data.objects.get("Forest Generator") is not None
    IN_BLENDER = is_blender_proc or has_target_scene
except ImportError:
    IN_BLENDER = False

if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
elif IN_BLENDER and bpy.data.filepath:
    BASE_DIR = os.path.dirname(os.path.abspath(bpy.data.filepath))
else:
    BASE_DIR = os.getcwd()

TARGET_BLEND = os.path.join(BASE_DIR, "target.blend")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "dataset_segmentation_pairs")

BLENDER_CANDIDATE_PATHS = [
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
]


def find_blender_binary():
    """Finds Blender executable across PATH and standard install directories."""
    import shutil
    which_b = shutil.which("blender")
    if which_b:
        return which_b
    for p in BLENDER_CANDIDATE_PATHS:
        if os.path.isfile(p):
            return p
    unix_candidates = [
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    for p in unix_candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Could not find blender executable! Please install Blender or add it to PATH."
    )


if not IN_BLENDER:
    import subprocess

    def run_via_blender():
        blender_exe = find_blender_binary()
        this_script = os.path.abspath(__file__)
        args = sys.argv[1:]
        cmd = [blender_exe, "-b", TARGET_BLEND, "--python", this_script, "--"] + args
        print(f"\n[Launcher] Launching Blender headless:\n{' '.join(cmd)}\n")
        sys.stdout.flush()
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n[Launcher] Stopped by user.")
        except subprocess.CalledProcessError as e:
            print(f"\n[Launcher] Blender exited with error code {e.returncode}")
            sys.exit(e.returncode)

    if __name__ == "__main__":
        run_via_blender()
        sys.exit(0)

# ==============================================================================
# BLENDER RUNTIME LOGIC (Runs inside Blender bundled Python)
# ==============================================================================
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

CAMERA_RIG_DIR = os.path.join(BASE_DIR, "camera_rig")
POSES_DIR = os.path.join(BASE_DIR, "poses")
if CAMERA_RIG_DIR not in sys.path:
    sys.path.append(CAMERA_RIG_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
import camera_arc_rig
import basecolor_extractor

TIGER_CANONICAL_SCALE = Vector((0.04908, 0.04908, 0.04908))
PAW_PAD_OFFSET = 0.45


def remap_missing_assets():
    """Finds and remaps any broken image/HDR filepaths to local textures and assets directories."""
    search_dirs = [
        os.path.join(BASE_DIR, "assets"),
        os.path.join(BASE_DIR, "textures"),
    ]
    filename_map = {}
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for root, _, files in os.walk(s_dir):
                for f in files:
                    lower = f.lower()
                    if lower not in filename_map:
                        filename_map[lower] = os.path.join(root, f)

    for img in bpy.data.images:
        if img.name == "Render Result" or img.packed_file:
            continue
        cur_abs = bpy.path.abspath(img.filepath) if img.filepath else ""
        if not cur_abs or not os.path.exists(cur_abs):
            basename = os.path.basename(img.filepath or img.name).lower()
            if basename in filename_map:
                new_path = filename_map[basename]
                img.filepath = bpy.path.relpath(new_path)
                try:
                    img.reload()
                except Exception:
                    pass


def fix_ground_occlusion():
    """Hides the flat Ground plane at Z=0 so it never occludes the forest terrain."""
    ground = bpy.data.objects.get("Ground")
    if ground:
        ground.hide_render = True
        ground.hide_viewport = True
        ground.location.z = -20.0


def set_modifier_socket(mod, socket_name, value):
    """Sets modifier socket value robustly across Blender 5.x interface items."""
    if mod and mod.node_group and hasattr(mod.node_group, 'interface'):
        for item in mod.node_group.interface.items_tree:
            if getattr(item, 'name', '') == socket_name:
                item.default_value = value
                return True
    try:
        if hasattr(mod, '__getitem__'):
            mod[socket_name] = value
            return True
    except Exception:
        pass
    return False


def preload_all_poses():
    """
    Preloads all 19 pose actions into memory once from the poses/ directory.
    Returns list of tuples: (pose_file_name, pose_action_name)
    """
    if not os.path.exists(POSES_DIR):
        raise FileNotFoundError(f"Poses directory not found: {POSES_DIR}")

    pose_files = sorted([f for f in os.listdir(POSES_DIR) if f.endswith(".blend")])
    pose_map = []

    print(f"[Collector] Preloading {len(pose_files)} poses into memory...")
    for f in pose_files:
        fpath = os.path.join(POSES_DIR, f)
        with bpy.data.libraries.load(fpath, link=False) as (data_from, data_to):
            pose_acts = [a for a in data_from.actions if a.startswith("Pose_")]
            if pose_acts:
                data_to.actions = [pose_acts[0]]
                pose_map.append((f, pose_acts[0]))

    print(f"[Collector] Successfully cached {len(pose_map)} tiger poses.")
    return pose_map


def plant_tiger_4paws(tiger_arm, fg, depsgraph, center_x, center_y, yaw):
    """
    Positions tiger at (center_x, center_y), aligns to slope, and snaps all 4 paws to ground.
    """
    mod_trees = fg.modifiers.get("GeometryNodes")
    orig_vp = mod_trees.show_viewport if mod_trees else True
    if mod_trees:
        mod_trees.show_viewport = False
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

    fg_eval = fg.evaluated_get(depsgraph)
    bvh = BVHTree.FromObject(fg_eval, depsgraph)

    if mod_trees:
        mod_trees.show_viewport = orig_vp
        bpy.context.view_layer.update()

    paw_bone_names = {
        'FL': 'l_toesFront_00_uJnt',
        'FR': 'r_toesFront_00_uJnt',
        'BL': 'l_toes_00_uJnt',
        'BR': 'r_toes_00_uJnt',
    }

    # 1. Place tiger at target center_x, center_y with specified yaw
    tiger_arm.location.x = center_x
    tiger_arm.location.y = center_y
    tiger_arm.rotation_euler = (0.0, 0.0, yaw)
    bpy.context.view_layer.update()

    # 2. Raycast to find terrain height under each paw
    ray_dir = Vector((0.0, 0.0, -1.0))
    ray_height = 80.0

    paw_hits = {}
    for key, bname in paw_bone_names.items():
        pb = tiger_arm.pose.bones.get(bname)
        w_head = tiger_arm.matrix_world @ pb.head if pb else tiger_arm.location.copy()
        hit, norm, _, _ = bvh.ray_cast(Vector((w_head.x, w_head.y, ray_height)), ray_dir)
        if not hit:
            hit, norm, _, _ = bvh.ray_cast(Vector((center_x, center_y, ray_height)), ray_dir)
            if not hit:
                hit = Vector((w_head.x, w_head.y, -3.5))
                norm = Vector((0.0, 0.0, 1.0))
        paw_hits[key] = (hit, norm)

    # 3. Compute pitch & roll of terrain
    z_FL = paw_hits['FL'][0].z
    z_FR = paw_hits['FR'][0].z
    z_BL = paw_hits['BL'][0].z
    z_BR = paw_hits['BR'][0].z

    z_front = (z_FL + z_FR) / 2.0
    z_back = (z_BL + z_BR) / 2.0
    z_left = (z_FL + z_BL) / 2.0
    z_right = (z_FR + z_BR) / 2.0

    pb_FL = tiger_arm.matrix_world @ tiger_arm.pose.bones[paw_bone_names['FL']].head
    pb_BL = tiger_arm.matrix_world @ tiger_arm.pose.bones[paw_bone_names['BL']].head
    pb_FR = tiger_arm.matrix_world @ tiger_arm.pose.bones[paw_bone_names['FR']].head

    dy = max(0.5, abs(pb_FL.y - pb_BL.y))
    dx = max(0.3, abs(pb_FL.x - pb_FR.x))

    pitch = math.atan2((z_front - z_back), dy)
    roll = math.atan2((z_right - z_left), dx)

    tiger_arm.rotation_euler.x = pitch
    tiger_arm.rotation_euler.y = roll
    bpy.context.view_layer.update()

    # 4. Vertical alignment: Paw pads sit ~0.45m below toe bone head
    diffs = []
    for key, bname in paw_bone_names.items():
        pb = tiger_arm.pose.bones.get(bname)
        curr_paw_z = (tiger_arm.matrix_world @ pb.head).z - PAW_PAD_OFFSET
        diffs.append(paw_hits[key][0].z - curr_paw_z)

    # Embed 3cm into dirt/moss for realistic contact
    tiger_arm.location.z += (max(diffs) - 0.03)
    bpy.context.view_layer.update()


def setup_camera_arc_sweep(scene, cam, tiger_arm, cam_height=2.8):
    """
    Sets up the mathematical camera arc rig from animate_camera_arc.py:
        2y = -41 - sqrt(r^2 - x^2) for -28 <= x <= 28 (r = 41)
    Keyframes Follow Path constraint offset_factor:
        Frame 1:   offset_factor = 0.0 (x = -28.0)
        Frame 120: offset_factor = 1.0 (x = +28.0)
    Points camera dynamically at the tiger upper body with Track To constraint.
    """
    camera_arc_rig.remove_previous_rig(cam)
    camera_arc_rig.build_arc_curve(scene, r=camera_arc_rig.R, cam_height=cam_height)

    fp = cam.constraints.new('FOLLOW_PATH')
    fp.name = "Arc: Follow Path"
    fp.target = bpy.data.objects.get(camera_arc_rig.PATH)
    fp.use_curve_follow = False
    fp.use_fixed_location = True

    # Aim directly at tiger upper body/chest
    look_target = bpy.data.objects.get("Camera Look Target")
    if not look_target:
        look_target = bpy.data.objects.new("Camera Look Target", None)
        scene.collection.objects.link(look_target)

    jitter_x = random.uniform(-0.2, 0.2)
    jitter_z = random.uniform(-0.1, 0.2)
    look_target.location = (
        tiger_arm.location.x + jitter_x,
        tiger_arm.location.y,
        tiger_arm.location.z + 1.15 + jitter_z
    )

    track = cam.constraints.new('TRACK_TO')
    track.name = "Arc: Face Center"
    track.target = look_target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    # Clear prior animation data and keyframe across frames 1 to 120 as in animate_camera_arc.py
    if cam.animation_data:
        cam.animation_data_clear()

    START_FRAME = 1
    END_FRAME = 120
    scene.frame_start = START_FRAME
    scene.frame_end = END_FRAME

    scene.frame_set(START_FRAME)
    fp.offset_factor = 0.0
    cam.keyframe_insert(data_path='constraints["Arc: Follow Path"].offset_factor', frame=START_FRAME)

    scene.frame_set(END_FRAME)
    fp.offset_factor = 1.0
    cam.keyframe_insert(data_path='constraints["Arc: Follow Path"].offset_factor', frame=END_FRAME)

    scene.frame_set(START_FRAME)
    bpy.context.view_layer.update()
    return fp, look_target


def get_arc_angles_frames(angles_per_scene=3):
    """
    Selects variable angle frames (1 to 120) along the camera arc sweep.
    - If 1: 1 random angle across the sweep [15, 105].
    - If 3: Left flank (~20-28), Frontal center (~56-64), Right flank (~92-100).
    - If K: K evenly distributed points between 15 and 105 with small random jitter.
    """
    if angles_per_scene <= 1:
        return [random.randint(15, 105)]
    elif angles_per_scene == 3:
        return [
            random.randint(18, 28),   # Left flank
            random.randint(56, 64),   # Frontal center
            random.randint(92, 102),  # Right flank
        ]
    else:
        frames = []
        step = (105 - 15) / (angles_per_scene - 1)
        for i in range(angles_per_scene):
            base_f = 15 + i * step
            f = int(round(base_f + random.uniform(-2.0, 2.0)))
            frames.append(max(1, min(120, f)))
        return frames


def get_or_create_emission_mat(name, color):
    """Creates a flat unlit emission shader material (no shadows, pure constant color)."""
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (300, 0)
    emit_node = nodes.new(type="ShaderNodeEmission")
    emit_node.location = (0, 0)
    emit_node.inputs['Color'].default_value = color
    emit_node.inputs['Strength'].default_value = 1.0
    mat.node_tree.links.new(emit_node.outputs['Emission'], out_node.inputs['Surface'])
    return mat


def get_or_create_yellow_world():
    """Creates a pure unlit yellow sky background for the segmentation pass."""
    world = bpy.data.worlds.get("MASK_World_Yellow")
    if not world:
        world = bpy.data.worlds.new(name="MASK_World_Yellow")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    out_node = nodes.new(type="ShaderNodeOutputWorld")
    out_node.location = (300, 0)
    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.location = (0, 0)
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 0.0, 1.0)
    bg_node.inputs['Strength'].default_value = 1.0
    world.node_tree.links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])
    return world


def snap_mask_to_exact_classes(mask_filepath):
    """
    Quantizes segmentation mask using native Blender image buffers and NumPy:
    Guarantees 100% of pixels are strictly one of the 3 class RGBs:
    - Tiger: [1.0, 0.0, 0.0, 1.0] (Pure Red: 255, 0, 0)
    - Background: [0.0, 0.0, 1.0, 1.0] (Pure Blue: 0, 0, 255)
    - Sky: [1.0, 1.0, 0.0, 1.0] (Pure Yellow: 255, 255, 0)
    Eliminates all edge anti-aliasing interpolation artifacts for model training.
    """
    img = bpy.data.images.load(mask_filepath)
    w, h = img.size
    total_pixels = w * h

    px = np.empty(total_pixels * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape((total_pixels, 4))

    rgb = px[:, :3]
    target_red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    target_blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    target_yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    d_red = np.sum((rgb - target_red) ** 2, axis=-1)
    d_blue = np.sum((rgb - target_blue) ** 2, axis=-1)
    d_yellow = np.sum((rgb - target_yellow) ** 2, axis=-1)

    nearest = np.argmin(np.stack([d_red, d_blue, d_yellow], axis=-1), axis=-1)
    palette = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0]
    ], dtype=np.float32)

    clean_px = palette[nearest].ravel()
    img.pixels.foreach_set(clean_px)

    img.filepath_raw = mask_filepath
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def create_side_by_side_preview(render_filepath, mask_filepath, preview_filepath):
    """Creates a side-by-side preview image using native Blender image buffers."""
    r_img = bpy.data.images.load(render_filepath)
    m_img = bpy.data.images.load(mask_filepath)
    w, h = r_img.size

    r_px = np.empty(w * h * 4, dtype=np.float32)
    m_px = np.empty(w * h * 4, dtype=np.float32)
    r_img.pixels.foreach_get(r_px)
    m_img.pixels.foreach_get(m_px)

    r_px = r_px.reshape((h, w, 4))
    m_px = m_px.reshape((h, w, 4))

    # Combine horizontally: Left = Render, Right = Mask
    combined = np.concatenate([r_px, m_px], axis=1)

    prev_img = bpy.data.images.new("Preview_Temp", width=w * 2, height=h, alpha=False)
    prev_img.pixels.foreach_set(combined.ravel())
    prev_img.filepath_raw = preview_filepath
    prev_img.file_format = 'JPEG'
    prev_img.save()

    bpy.data.images.remove(r_img)
    bpy.data.images.remove(m_img)
    bpy.data.images.remove(prev_img)


def render_segmentation_pair(render_path, mask_path, preview_path=None):
    """
    Renders the paired images:
    Pass 1: Fully textured realistic render.
    Pass 2: Unlit solid color segmentation mask (Red Tiger, Blue Background, Yellow Sky).
    """
    scene = bpy.context.scene

    # ------------------------------------------------------------------
    # PASS 1: REALISTIC TEXTURED RENDER
    # ------------------------------------------------------------------
    scene.render.filepath = render_path
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)

    # ------------------------------------------------------------------
    # PASS 2: SOLID COLOR SEGMENTATION MASK
    # ------------------------------------------------------------------
    mat_red = get_or_create_emission_mat("MASK_Tiger_Red", (1.0, 0.0, 0.0, 1.0))
    mat_blue = get_or_create_emission_mat("MASK_BG_Blue", (0.0, 0.0, 1.0, 1.0))
    world_yellow = get_or_create_yellow_world()

    # Backup Scene state
    orig_world = scene.world
    orig_view_transform = scene.view_settings.view_transform
    orig_look = scene.view_settings.look
    orig_exposure = scene.view_settings.exposure

    # Store and replace materials
    orig_obj_mats = {}
    try:
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                orig_obj_mats[obj.name] = [slot.material for slot in obj.material_slots]
                # White Tiger Body -> Red; Everything else (Forest Generator, Trees, Leaves, Barks) -> Blue
                target_mat = mat_red if ("White Tiger" in obj.name or "Tiger" in obj.name) else mat_blue
                for slot in obj.material_slots:
                    slot.material = target_mat
                if len(obj.material_slots) == 0:
                    obj.data.materials.append(target_mat)

        # Set Raw view transform to preserve exact RGB color values without tone curve alteration
        scene.world = world_yellow
        scene.view_settings.view_transform = 'Raw'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0

        scene.render.filepath = mask_path
        bpy.ops.render.render(write_still=True)

    finally:
        # Always restore original scene materials and view settings even on interruption
        for obj_name, mats in orig_obj_mats.items():
            obj = bpy.data.objects.get(obj_name)
            if obj:
                for idx, mat in enumerate(mats):
                    if idx < len(obj.material_slots):
                        obj.material_slots[idx].material = mat

        scene.world = orig_world
        scene.view_settings.view_transform = orig_view_transform
        scene.view_settings.look = orig_look
        scene.view_settings.exposure = orig_exposure

    # Clean mask to strictly 3 exact colors
    snap_mask_to_exact_classes(mask_path)

    # Generate side-by-side preview
    if preview_path:
        create_side_by_side_preview(render_path, mask_path, preview_path)


def get_next_pair_id(images_dir):
    """Finds next sequential integer ID from existing dataset files."""
    os.makedirs(images_dir, exist_ok=True)
    existing_nums = []
    for fname in os.listdir(images_dir):
        if fname.endswith(".png"):
            base = os.path.splitext(fname)[0]
            if base.isdigit():
                existing_nums.append(int(base))
    return max(existing_nums, default=0) + 1


def run_pipeline(
    count=0,
    scenes_count=0,
    angles_per_scene=3,
    out_dir=DEFAULT_OUT_DIR,
    res_x=1920,
    res_y=1080,
    make_previews=True,
    make_basecolor=True,
    min_face_pixels=1,
    tiger_ids=4
):
    """
    Executes the automated segmentation data collection pipeline using the
    custom mathematical camera arc rig from animate_camera_arc.py.
    Captures variable-angled photos for each randomized scene.
    count = 0 runs indefinitely.
    """
    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    previews_dir = os.path.join(out_dir, "previews")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    if make_previews:
        os.makedirs(previews_dir, exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100

    fix_ground_occlusion()
    remap_missing_assets()

    fg = bpy.data.objects.get("Forest Generator")
    tiger_arm = bpy.data.objects.get("White Tiger Skeleton")
    tiger_mesh = bpy.data.objects.get("White Tiger Body")
    sun = bpy.data.objects.get("Sun")

    if (not fg or not tiger_arm) and os.path.isfile(TARGET_BLEND):
        print(f"[Collector] Loading {TARGET_BLEND}...")
        bpy.ops.wm.open_mainfile(filepath=TARGET_BLEND)
        fg = bpy.data.objects.get("Forest Generator")
        tiger_arm = bpy.data.objects.get("White Tiger Skeleton")
        tiger_mesh = bpy.data.objects.get("White Tiger Body")
        sun = bpy.data.objects.get("Sun")

    if not fg or not tiger_arm:
        raise RuntimeError("Missing 'Forest Generator' or 'White Tiger Skeleton' in scene!")

    # Ensure Terrain Seed Displace modifier
    mod_ground = fg.modifiers.get("Terrain Seed Displace")
    if not mod_ground:
        import add_ground_seed_controller
        add_ground_seed_controller.setup_terrain_seed()
        mod_ground = fg.modifiers.get("Terrain Seed Displace")

    mod_trees = fg.modifiers.get("GeometryNodes")
    pose_list = preload_all_poses()

    # Camera Arc Rig setup
    cam = bpy.data.objects.get("Camera")
    if not cam:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
    scene.camera = cam

    # CSV metadata log
    csv_path = os.path.join(out_dir, "dataset_metadata.csv")
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "image_id", "scene_id", "angle_index",
            "render_file", "mask_file", "preview_file",
            "ground_seed", "ground_height", "tree_seed",
            "tiger_pose", "pose_file",
            "tiger_x", "tiger_y", "tiger_z", "tiger_yaw",
            "cam_arc_frame", "cam_offset",
            "cam_x", "cam_y", "cam_z",
            "cam_rot_x_deg", "cam_rot_y_deg", "cam_rot_z_deg", "cam_height",
            "resolution", "timestamp"
        ])
        csv_file.flush()

    extractor = None
    if make_basecolor:
        if not tiger_mesh:
            raise RuntimeError("Missing 'White Tiger Body' in scene (needed for basecolor extraction)!")
        extractor = basecolor_extractor.BasecolorExtractor(
            scene, tiger_arm, tiger_mesh, out_dir, min_face_pixels=min_face_pixels,
            n_identities=tiger_ids
        )

    pair_id = get_next_pair_id(images_dir)
    scene_id = 1
    pairs_collected = 0
    is_indefinite = (count == 0 and scenes_count == 0)

    print("\n" + "=" * 75)
    print("🐅 TIGER SEGMENTATION DATASET COLLECTOR (CAMERA ARC SWEEP)")
    print("=" * 75)
    if is_indefinite:
        mode_desc = f"INDEFINITE ({angles_per_scene} angles/scene - Press Ctrl+C to stop)"
    elif scenes_count > 0:
        mode_desc = f"{scenes_count} SCENES x {angles_per_scene} ANGLES = {scenes_count * angles_per_scene} PAIRS"
    else:
        mode_desc = f"{count} PAIRS ({angles_per_scene} angles per scene)"
    print(f"Mode:             {mode_desc}")
    print(f"Camera Arc Path:  2y = -41 - sqrt(41^2 - x^2) [Frames 1-120 sweep]")
    print(f"Angles / Scene:   {angles_per_scene}")
    print(f"Output Directory: {out_dir}")
    print(f"Resolution:       {res_x} x {res_y}")
    print(f"Starting at ID:   {pair_id:06d}")
    print(f"Classes:          Tiger=RED (255,0,0) | BG=BLUE (0,0,255) | Sky=YELLOW (255,255,0)")
    print("=" * 75 + "\n")
    sys.stdout.flush()

    try:
        while is_indefinite or (count > 0 and pairs_collected < count) or (scenes_count > 0 and scene_id <= scenes_count):
            t_scene_start = time.time()
            print(f"\n>>> [Scene {scene_id}] Randomizing terrain, pose, placement & sun lighting...")

            # ------------------------------------------------------------------
            # 1. RANDOMIZE SCENE PARAMETERS
            # ------------------------------------------------------------------
            # Terrain Seeds
            s_ground = random.randint(1, 99999999)
            s_height = round(random.uniform(-0.6, 0.6), 3)
            set_modifier_socket(mod_ground, 'Ground Seed', s_ground)
            set_modifier_socket(mod_ground, 'Ground Height', s_height)

            # Forest / Trees Seed
            s_tree = random.randint(1, 99999999)
            set_modifier_socket(mod_trees, 'Random Seed', s_tree)

            # Tiger Pose & Canonical Scale Lock
            pose_file, pose_name = random.choice(pose_list)
            act = bpy.data.actions.get(pose_name)
            if not tiger_arm.animation_data:
                tiger_arm.animation_data_create()
            tiger_arm.animation_data.action = act

            # Tiger identity (distinct stripe pattern) for this scene
            tiger_id = random.randrange(tiger_ids) if extractor else 0
            if extractor:
                extractor.select_identity(tiger_id)

            tiger_arm.scale = TIGER_CANONICAL_SCALE
            if tiger_mesh:
                tiger_mesh.scale = Vector((1.0, 1.0, 1.0))
            for pb in tiger_arm.pose.bones:
                if pb.name != 'c_breath_00_uJnt':
                    pb.scale = Vector((1.0, 1.0, 1.0))

            # Tiger Position in clearing & Yaw
            pos_x = round(random.uniform(-3.5, 3.5), 2)
            pos_y = round(random.uniform(-20.0, -15.0), 2)
            yaw = round(random.uniform(-math.pi * 0.85, math.pi * 0.85), 3)

            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()

            # Plant tiger with 4-paw collision raycasting
            plant_tiger_4paws(tiger_arm, fg, depsgraph, center_x=pos_x, center_y=pos_y, yaw=yaw)

            # Sun lighting variation (azimuth and elevation)
            if sun:
                sun.rotation_euler.z = random.uniform(0.0, 2 * math.pi)
                sun.rotation_euler.x = math.radians(random.uniform(25.0, 60.0))

            # ------------------------------------------------------------------
            # 2. SETUP CAMERA ARC RIG (from animate_camera_arc.py)
            # ------------------------------------------------------------------
            cam_height = round(random.uniform(2.0, 4.8), 2)
            fp, look_target = setup_camera_arc_sweep(scene, cam, tiger_arm, cam_height=cam_height)

            # Determine variable angle frames along the arc
            arc_frames = get_arc_angles_frames(angles_per_scene)

            # ------------------------------------------------------------------
            # 3. CAPTURE VARIABLE ANGLED PHOTOS ALONG THE ARC
            # ------------------------------------------------------------------
            for angle_idx, frame_num in enumerate(arc_frames):
                if count > 0 and pairs_collected >= count:
                    break

                t_pair_start = time.time()
                id_str = f"{pair_id:06d}"

                # Set frame along the camera arc sweep
                scene.frame_set(frame_num)
                bpy.context.view_layer.update()
                dg = bpy.context.evaluated_depsgraph_get()

                cam_eval = cam.evaluated_get(dg)
                cam_loc = cam_eval.matrix_world.translation
                cam_rot = cam_eval.matrix_world.to_euler()
                cam_offset = round(fp.offset_factor, 4)

                # Filepaths
                render_filename = f"{id_str}.png"
                mask_filename = f"{id_str}.png"
                preview_filename = f"{id_str}_preview.jpg"

                render_filepath = os.path.join(images_dir, render_filename)
                mask_filepath = os.path.join(masks_dir, mask_filename)
                preview_filepath = os.path.join(previews_dir, preview_filename) if make_previews else None

                # Render realistic image + solid segmentation mask
                render_segmentation_pair(render_filepath, mask_filepath, preview_filepath)

                # Visible-only basecolor atlas + joint table (same frame/camera as the render)
                bc_info = extractor.process_pair(pair_id, id_str) if extractor else None

                # Log metadata
                now_iso = datetime.now().isoformat()
                csv_writer.writerow([
                    pair_id, scene_id, angle_idx + 1,
                    f"images/{render_filename}",
                    f"masks/{mask_filename}",
                    f"previews/{preview_filename}" if make_previews else "",
                    s_ground, s_height, s_tree,
                    pose_name, pose_file,
                    pos_x, pos_y, round(tiger_arm.location.z, 3), yaw,
                    frame_num, cam_offset,
                    round(cam_loc.x, 3), round(cam_loc.y, 3), round(cam_loc.z, 3),
                    round(math.degrees(cam_rot.x), 1), round(math.degrees(cam_rot.y), 1), round(math.degrees(cam_rot.z), 1),
                    cam_height, f"{res_x}x{res_y}", now_iso
                ])
                csv_file.flush()

                pair_elapsed = time.time() - t_pair_start
                print(f"  📷 [Pair {id_str}] Scene {scene_id} | Angle {angle_idx + 1}/{len(arc_frames)} (Frame {frame_num:3d}, t={cam_offset:.3f}) | "
                      f"Cam=({cam_loc.x:+.1f},{cam_loc.y:+.1f},{cam_loc.z:.1f}) | Pose={pose_name} | Time={pair_elapsed:.1f}s")
                if bc_info:
                    print(f"     🎨 Tiger #{bc_info['tiger_id']} | Basecolor: {bc_info['visible_faces']}/{bc_info['total_faces']} faces visible "
                          f"({bc_info['coverage'] * 100:.1f}% of skin atlas) | tiger px={bc_info['tiger_pixels']} | "
                          f"joints on tiger={bc_info['joints_on_tiger']}/{bc_info['joints_in_frame']} in frame | "
                          f"invalid-id px={bc_info['invalid_id_pixels']}")
                sys.stdout.flush()

                pair_id += 1
                pairs_collected += 1

            scene_elapsed = time.time() - t_scene_start
            print(f">>> [Scene {scene_id} Done] Captured {len(arc_frames)} angles in {scene_elapsed:.1f}s")
            scene_id += 1

    except KeyboardInterrupt:
        print("\n[Collector] Data collection paused by user (Ctrl+C).")
    finally:
        csv_file.close()
        if extractor:
            extractor.close()

    print(f"\n[Collector] Complete! Captured {pairs_collected} segmentation dataset pairs stored in:\n{out_dir}\n")


def parse_cli_args():
    """Parses command line arguments passed either directly or after Blender's '--'."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Tiger Segmentation Dataset Collector (Camera Arc Rig)")
    parser.add_argument("--count", type=int, default=None, help="Number of image pairs (0 = indefinite)")
    parser.add_argument("--scenes", type=int, default=0, help="Number of distinct scenes to capture")
    parser.add_argument("--angles_per_scene", type=int, default=3, help="Number of camera angles along the arc per scene (default: 3)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("--res_x", type=int, default=1920, help="Render width (default: 1920)")
    parser.add_argument("--res_y", type=int, default=1080, help="Render height (default: 1080)")
    parser.add_argument("--no_previews", action="store_true", help="Disable side-by-side preview images")
    parser.add_argument("--no_basecolor", action="store_true", help="Disable visible-only basecolor atlases and the joints database")
    parser.add_argument("--tiger_ids", type=int, default=4, help="Number of distinct tiger identities (stripe patterns); id 0 = original skin (default: 4, needs basecolor enabled)")
    parser.add_argument("--min_face_pixels", type=int, default=1, help="Min visible pixels for a face to count as visible in the basecolor atlas (default: 1)")
    return parser.parse_known_args(argv)[0]


def main():
    args = parse_cli_args()

    # Interactive prompt if count/out_dir not specified via CLI
    if args.count is None and args.scenes == 0:
        try:
            c_in = input("Enter number of pairs to capture (0 for indefinite, default 12): ").strip()
            count = int(c_in) if c_in else 12
            a_in = input(f"Camera angles per scene along the arc (default {args.angles_per_scene}): ").strip()
            angles_per_scene = int(a_in) if a_in else args.angles_per_scene
        except (EOFError, KeyboardInterrupt):
            count = 12
            angles_per_scene = args.angles_per_scene
    else:
        count = args.count if args.count is not None else 0
        angles_per_scene = args.angles_per_scene

    if args.out_dir is None:
        try:
            o_in = input(f"Output folder (press Enter for '{DEFAULT_OUT_DIR}'): ").strip()
            out_dir = o_in if o_in else DEFAULT_OUT_DIR
        except (EOFError, KeyboardInterrupt):
            out_dir = DEFAULT_OUT_DIR
    else:
        out_dir = args.out_dir

    run_pipeline(
        count=count,
        scenes_count=args.scenes,
        angles_per_scene=angles_per_scene,
        out_dir=os.path.abspath(out_dir),
        res_x=args.res_x,
        res_y=args.res_y,
        make_previews=not args.no_previews,
        make_basecolor=not args.no_basecolor,
        min_face_pixels=args.min_face_pixels,
        tiger_ids=max(1, args.tiger_ids)
    )


if __name__ == "__main__":
    main()

