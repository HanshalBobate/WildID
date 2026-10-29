"""
Tiger All Poses Diagnostic Renderer
====================================
Renders a fully textured image for all 19 tiger poses against a fixed, identical
terrain and lighting scene.

Purpose:
- Inspect every pose for misconfigurations, armature distortions, scale glitches,
  terrain clipping, or floating issues before running large dataset collection runs.

Output:
- Dedicated folder: test_all_poses_renders/
  - 01_Pose_Crouch_Ready_To_Pounce.png
  - 02_Pose_Drinking.png
  - ...
  - 19_Pose_Walking.png
  - poses_gallery.html (Interactive visual grid to inspect all 19 poses in browser)
  - poses_summary.csv

Usage:
  # Run from terminal (auto-launches Blender headless):
  python test_all_poses.py

  # Or with custom resolution/folder:
  python test_all_poses.py --out_dir ./all_poses_renders --res_x 1920 --res_y 1080
"""
import os
import sys
import math
import time
import csv
import argparse

# ==============================================================================
# WRAPPER: AUTO-LAUNCH BLENDER IF RUN OUTSIDE BLENDER
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
POSES_DIR = os.path.join(BASE_DIR, "poses")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "test_all_poses_renders")

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
        print(f"\n[Launcher] Launching Blender headless for Pose Diagnostics:\n{' '.join(cmd)}\n")
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
# BLENDER RUNTIME LOGIC
# ==============================================================================
from mathutils import Vector
from mathutils.bvhtree import BVHTree

CAMERA_RIG_DIR = os.path.join(BASE_DIR, "camera_rig")
if CAMERA_RIG_DIR not in sys.path:
    sys.path.append(CAMERA_RIG_DIR)
import camera_arc_rig

TIGER_CANONICAL_SCALE = Vector((0.04908, 0.04908, 0.04908))
PAW_PAD_OFFSET = 0.45

# Fixed baseline terrain seeds for controlled 1-to-1 comparison
FIXED_GROUND_SEED = 42891
FIXED_GROUND_HEIGHT = 0.15
FIXED_TREE_SEED = 77123
FIXED_TIGER_POS = (0.0, -18.0)
FIXED_TIGER_YAW = math.radians(25.0)  # Gentle 3/4 turn to camera


def fix_ground_occlusion():
    ground = bpy.data.objects.get("Ground")
    if ground:
        ground.hide_render = True
        ground.hide_viewport = True
        ground.location.z = -20.0


def set_modifier_socket(mod, socket_name, value):
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


def load_all_pose_actions():
    """
    Loads all 19 tiger pose actions from poses/ directory.
    Returns list of tuples: (index, filename, action_name, action_object)
    """
    if not os.path.exists(POSES_DIR):
        raise FileNotFoundError(f"Poses directory not found: {POSES_DIR}")

    pose_files = sorted([f for f in os.listdir(POSES_DIR) if f.endswith(".blend")])
    poses = []

    print(f"\n[Poses] Loading {len(pose_files)} pose actions from: {POSES_DIR}")
    for idx, f in enumerate(pose_files, start=1):
        fpath = os.path.join(POSES_DIR, f)
        with bpy.data.libraries.load(fpath, link=False) as (data_from, data_to):
            pose_acts = [a for a in data_from.actions if a.startswith("Pose_")]
            if pose_acts:
                data_to.actions = [pose_acts[0]]
                act_name = pose_acts[0]
            else:
                act_name = None

        if act_name:
            act_obj = bpy.data.actions.get(act_name)
            poses.append((idx, f, act_name, act_obj))
            print(f"  [{idx:02d}/19] {act_name} (from {f})")

    return poses


def plant_tiger_4paws(tiger_arm, fg, depsgraph, center_x=0.0, center_y=-18.0, yaw=0.0):
    """
    Aligns tiger to terrain collision.
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

    tiger_arm.location.x = center_x
    tiger_arm.location.y = center_y
    tiger_arm.rotation_euler = (0.0, 0.0, yaw)
    bpy.context.view_layer.update()

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

    diffs = []
    for key, bname in paw_bone_names.items():
        pb = tiger_arm.pose.bones.get(bname)
        curr_paw_z = (tiger_arm.matrix_world @ pb.head).z - PAW_PAD_OFFSET
        diffs.append(paw_hits[key][0].z - curr_paw_z)

    # Embed 3cm into dirt/moss for realistic contact
    tiger_arm.location.z += (max(diffs) - 0.03)
    bpy.context.view_layer.update()


def build_html_gallery(out_dir, pose_records):
    """Generates an HTML contact sheet to review all 19 renders at once."""
    html_path = os.path.join(out_dir, "poses_gallery.html")
    cards = []
    for r in pose_records:
        cards.append(f"""
        <div class="card">
            <div class="card-header">
                <span class="badge">#{r['index']:02d}</span>
                <strong>{r['action_name']}</strong>
            </div>
            <div class="img-wrap">
                <img src="{r['filename']}" alt="{r['action_name']}" loading="lazy"/>
            </div>
            <div class="card-footer">
                <small>{r['file']}</small><br/>
                <code>Scale: {r['scale']} | Z: {r['pos_z']}m</code>
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tiger All Poses Diagnostic Gallery (19 Poses)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f1117;
            color: #e2e8f0;
            margin: 0;
            padding: 24px;
        }}
        h1 {{
            margin-top: 0;
            color: #f8fafc;
            font-size: 24px;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 12px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .card {{
            background: #1e222d;
            border: 1px solid #334155;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            border-color: #38bdf8;
        }}
        .card-header {{
            padding: 10px 14px;
            background: #181b24;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid #282f3f;
        }}
        .badge {{
            background: #0284c7;
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 12px;
        }}
        .img-wrap {{
            width: 100%;
            background: #000;
            cursor: pointer;
        }}
        .img-wrap img {{
            width: 100%;
            display: block;
            height: auto;
        }}
        .card-footer {{
            padding: 8px 14px;
            background: #181b24;
            font-size: 11px;
            color: #94a3b8;
            border-top: 1px solid #282f3f;
        }}
        code {{
            color: #38bdf8;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <h1>🐅 Tiger Pose Diagnostics & Inspection Gallery (19 Poses)</h1>
    <p style="color: #94a3b8; margin: 0 0 16px 0;">
        Fixed scene parameters: Ground Seed={FIXED_GROUND_SEED}, Height={FIXED_GROUND_HEIGHT}, Tree Seed={FIXED_TREE_SEED}.
        Verify body proportions, scale stability, 4-paw terrain contact, and orientation.
    </p>
    <div class="grid">
        {''.join(cards)}
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


def run_poses_test(out_dir=DEFAULT_OUT_DIR, res_x=1280, res_y=720):
    """
    Renders all 19 tiger poses against identical fixed terrain seeds.
    """
    os.makedirs(out_dir, exist_ok=True)
    fix_ground_occlusion()

    scene = bpy.context.scene
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    fg = bpy.data.objects.get("Forest Generator")
    tiger_arm = bpy.data.objects.get("White Tiger Skeleton")
    tiger_mesh = bpy.data.objects.get("White Tiger Body")

    if (not fg or not tiger_arm) and os.path.isfile(TARGET_BLEND):
        print(f"[Diagnostics] Loading {TARGET_BLEND}...")
        bpy.ops.wm.open_mainfile(filepath=TARGET_BLEND)
        fg = bpy.data.objects.get("Forest Generator")
        tiger_arm = bpy.data.objects.get("White Tiger Skeleton")
        tiger_mesh = bpy.data.objects.get("White Tiger Body")

    if not fg or not tiger_arm:
        raise RuntimeError("Missing 'Forest Generator' or 'White Tiger Skeleton' in scene!")

    # Set fixed terrain seed
    mod_ground = fg.modifiers.get("Terrain Seed Displace")
    if not mod_ground:
        import add_ground_seed_controller
        add_ground_seed_controller.setup_terrain_seed()
        mod_ground = fg.modifiers.get("Terrain Seed Displace")

    set_modifier_socket(mod_ground, 'Ground Seed', FIXED_GROUND_SEED)
    set_modifier_socket(mod_ground, 'Ground Height', FIXED_GROUND_HEIGHT)

    # Set fixed tree seed
    mod_trees = fg.modifiers.get("GeometryNodes")
    set_modifier_socket(mod_trees, 'Random Seed', FIXED_TREE_SEED)

    # Setup camera framing
    cam = bpy.data.objects.get("Camera")
    if not cam:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
    scene.camera = cam

    # Build camera arc curve at 2.6m eye-level height
    camera_arc_rig.remove_previous_rig(cam)
    camera_arc_rig.build_arc_curve(scene, r=camera_arc_rig.R, cam_height=2.6)

    fp = cam.constraints.new('FOLLOW_PATH')
    fp.name = "Arc: Follow Path"
    fp.target = bpy.data.objects.get(camera_arc_rig.PATH)
    fp.use_curve_follow = False
    fp.use_fixed_location = True
    fp.offset_factor = 0.52  # Centered front-left view

    look_target = bpy.data.objects.get("Camera Look Target")
    if not look_target:
        look_target = bpy.data.objects.new("Camera Look Target", None)
        scene.collection.objects.link(look_target)
    look_target.location = (FIXED_TIGER_POS[0], FIXED_TIGER_POS[1], -2.0)

    track = cam.constraints.new('TRACK_TO')
    track.name = "Arc: Face Center"
    track.target = look_target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    poses = load_all_pose_actions()
    total_poses = len(poses)

    # CSV log
    csv_path = os.path.join(out_dir, "poses_summary.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "index", "action_name", "pose_file", "image_file",
        "tiger_scale_x", "tiger_scale_y", "tiger_scale_z",
        "tiger_pos_x", "tiger_pos_y", "tiger_pos_z", "render_time_s"
    ])

    print("\n" + "=" * 70)
    print("🐅 RENDERING ALL 19 TIGER POSES (DIAGNOSTIC TEST)")
    print("=" * 70)
    print(f"Output Directory: {out_dir}")
    print(f"Resolution:       {res_x} x {res_y}")
    print(f"Fixed Terrain:    Ground={FIXED_GROUND_SEED} (h={FIXED_GROUND_HEIGHT}) | Tree={FIXED_TREE_SEED}")
    print(f"Fixed Tiger Pos:  {FIXED_TIGER_POS} | Yaw={math.degrees(FIXED_TIGER_YAW):.1f}°")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    records = []
    try:
        for idx, pose_file, act_name, act_obj in poses:
            t0 = time.time()
            filename = f"{idx:02d}_{act_name}.png"
            filepath = os.path.join(out_dir, filename)

            # 1. Assign action
            if not tiger_arm.animation_data:
                tiger_arm.animation_data_create()
            tiger_arm.animation_data.action = act_obj

            # 2. Strict scale enforcement
            tiger_arm.scale = TIGER_CANONICAL_SCALE
            if tiger_mesh:
                tiger_mesh.scale = Vector((1.0, 1.0, 1.0))
            for pb in tiger_arm.pose.bones:
                if pb.name != 'c_breath_00_uJnt':
                    pb.scale = Vector((1.0, 1.0, 1.0))

            # 3. Ground collision alignment
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            plant_tiger_4paws(
                tiger_arm, fg, depsgraph,
                center_x=FIXED_TIGER_POS[0],
                center_y=FIXED_TIGER_POS[1],
                yaw=FIXED_TIGER_YAW
            )

            # Update look target to current tiger body height
            look_target.location = (
                tiger_arm.location.x,
                tiger_arm.location.y,
                tiger_arm.location.z + 1.15
            )
            bpy.context.view_layer.update()

            # 4. Render textured image
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)
            elapsed = time.time() - t0

            # Record
            sx, sy, sz = [round(v, 5) for v in tiger_arm.scale]
            tx, ty, tz = [round(v, 3) for v in tiger_arm.location]
            csv_writer.writerow([
                idx, act_name, pose_file, filename,
                sx, sy, sz, tx, ty, tz, round(elapsed, 2)
            ])
            csv_file.flush()

            records.append({
                "index": idx,
                "action_name": act_name,
                "file": pose_file,
                "filename": filename,
                "scale": f"({sx}, {sy}, {sz})",
                "pos_z": tz,
            })

            print(f"[{idx:02d}/{total_poses}] Saved: {filename} (Z={tz:.2f}m, time={elapsed:.1f}s)")
            sys.stdout.flush()

    finally:
        csv_file.close()

    # Build HTML Gallery
    html_gallery_path = build_html_gallery(out_dir, records)

    print("\n" + "=" * 70)
    print("ALL 19 POSES RENDERED SUCCESSFULLY!")
    print(f"HTML Gallery: {html_gallery_path}")
    print(f"CSV Summary:  {csv_path}")
    print("=" * 70 + "\n")


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Render all 19 tiger poses for visual inspection")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--res_x", type=int, default=1280, help="Render width (default: 1280)")
    parser.add_argument("--res_y", type=int, default=720, help="Render height (default: 720)")
    return parser.parse_known_args(argv)[0]


def main():
    args = parse_args()
    run_poses_test(
        out_dir=os.path.abspath(args.out_dir),
        res_x=args.res_x,
        res_y=args.res_y
    )


if __name__ == "__main__":
    main()
