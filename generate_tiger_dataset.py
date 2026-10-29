"""
Tiger Dataset Generator with Ground Randomizer and 4-Foot Collision Alignment.

Features:
1. Generates a random integer Ground Seed for 'Terrain Seed Displace'.
2. Eliminates 'Ground' plane occlusion so the sculpted forest terrain is fully visible.
3. Raycasts the evaluated terrain to plant the Tiger with all 4 feet firmly touching the ground.
4. Aligns tiger pitch and roll to match the terrain slope.
5. Loops across 10 random Tree Seeds.
6. For each tree seed, captures N images (variable, default 3) along the configured arc path.
7. Saves renders to the output folder.

Can be run via:
- The 3D Viewport N-panel ('Tiger Dataset' tab) using the 'Generate & Capture Dataset' button.
- Running this script directly in the Scripting workspace.
"""
import bpy
import math
import os
import random
import sys
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
elif bpy.data.filepath:
    BASE_DIR = os.path.dirname(os.path.abspath(bpy.data.filepath))
else:
    BASE_DIR = os.getcwd()

OUTPUT_DIR_DEFAULT = os.path.join(BASE_DIR, "dataset_renders")
PAW_PAD_OFFSET = 0.45  # Distance from toe bone head to the bottom of the paw pad on the mesh


def fix_ground_occlusion():
    """
    Ensures the flat 'Ground' object at Z=0 does not occlude the sculpted terrain
    or bury the tiger.
    """
    ground = bpy.data.objects.get("Ground")
    if ground:
        ground.hide_render = True
        ground.hide_viewport = True
        ground.location.z = -20.0


def get_pure_terrain_bvh(fg, depsgraph):
    """
    Build BVH tree of evaluated Forest Generator mesh, temporarily disabling tree
    geometry nodes so raycasts hit pure terrain and never get blocked by tree branches.
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

    return bvh


def plant_tiger_on_ground(tiger_arm, fg, depsgraph, center_x=0.0, center_y=-18.0):
    """
    Raycast downwards to align the tiger's 4 feet to the terrain surface.
    Adjusts armature location, pitch, and roll so all 4 paws make solid, natural contact.
    """
    fix_ground_occlusion()
    bvh = get_pure_terrain_bvh(fg, depsgraph)

    paw_bone_names = {
        'FL': 'l_toesFront_00_uJnt',
        'FR': 'r_toesFront_00_uJnt',
        'BL': 'l_toes_00_uJnt',
        'BR': 'r_toes_00_uJnt',
    }

    # 1. Place tiger at target center_x, center_y while preserving yaw
    tiger_arm.location.x = center_x
    tiger_arm.location.y = center_y
    tiger_arm.rotation_euler = (0.0, 0.0, tiger_arm.rotation_euler.z)
    bpy.context.view_layer.update()

    # 2. Raycast to find terrain height under each paw
    ray_dir = Vector((0.0, 0.0, -1.0))
    ray_height = 80.0

    paw_hits = {}
    for key, bname in paw_bone_names.items():
        pb = tiger_arm.pose.bones.get(bname)
        if pb:
            w_head = tiger_arm.matrix_world @ pb.head
        else:
            w_head = tiger_arm.location.copy()

        origin = Vector((w_head.x, w_head.y, ray_height))
        hit, norm, _, _ = bvh.ray_cast(origin, ray_dir)
        if not hit:
            # Fallback raycast at tiger center
            c_orig = Vector((center_x, center_y, ray_height))
            hit, norm, _, _ = bvh.ray_cast(c_orig, ray_dir)
            if not hit:
                hit = Vector((w_head.x, w_head.y, -3.5))
                norm = Vector((0.0, 0.0, 1.0))
        paw_hits[key] = (hit, norm)

    # 3. Compute pitch & roll of terrain under the paws
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

    # 4. Vertical alignment: Paw pads/claws are ~0.45m below bone head in world space
    diffs = []
    for key, bname in paw_bone_names.items():
        pb = tiger_arm.pose.bones.get(bname)
        curr_paw_z = (tiger_arm.matrix_world @ pb.head).z - PAW_PAD_OFFSET
        target_ground_z = paw_hits[key][0].z
        diffs.append(target_ground_z - curr_paw_z)

    # Set elevation so paws touch ground firmly (embed 3cm for realistic dirt/moss contact)
    target_dz = max(diffs) - 0.03
    tiger_arm.location.z += target_dz
    bpy.context.view_layer.update()

    print(f"Tiger firmly planted at ({tiger_arm.location.x:.2f}, {tiger_arm.location.y:.2f}, {tiger_arm.location.z:.2f})")
    print(f"Terrain slope alignment -> Pitch: {math.degrees(pitch):.1f} deg, Roll: {math.degrees(roll):.1f} deg")


def set_modifier_socket(mod, socket_name, value):
    """Robustly sets modifier input socket value in Blender 5.x."""
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


def run_generation_pipeline(ground_seed=None, num_tree_seeds=10, images_per_tree=3, output_dir=OUTPUT_DIR_DEFAULT):
    """
    Executes the full generation & capture process:
    1. Hides occluding Ground plane
    2. Randomizes Ground Seed & deforms terrain
    3. Plants Tiger with 4-feet collision and fixes in place
    4. Loops across tree seeds
    5. Captures N views along the configured camera arc path
    """
    os.makedirs(output_dir, exist_ok=True)
    scene = bpy.context.scene

    # Fix ground plane occlusion
    fix_ground_occlusion()

    fg = bpy.data.objects.get("Forest Generator")
    if not fg:
        raise RuntimeError("Object 'Forest Generator' not found in scene!")

    tiger_arm = bpy.data.objects.get("White Tiger Skeleton")
    if not tiger_arm:
        tiger_arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
        if not tiger_arm:
            raise RuntimeError("White Tiger Skeleton (Armature) not found in scene!")

    cam = bpy.data.objects.get("Camera")
    if not cam or not cam.constraints.get("Arc: Follow Path"):
        camera_rig_dir = os.path.join(BASE_DIR, "camera_rig")
        if camera_rig_dir not in sys.path:
            sys.path.append(camera_rig_dir)
        import camera_arc_rig
        camera_arc_rig.build(scene)
        cam = bpy.data.objects.get("Camera")

    fp = cam.constraints.get("Arc: Follow Path")
    scene.camera = cam

    # 1. Randomize Ground Seed
    if ground_seed is None or ground_seed <= 0:
        ground_seed = random.randint(1, 999999)

    # Ensure Terrain Seed Displace modifier exists
    mod_ground = fg.modifiers.get("Terrain Seed Displace")
    if not mod_ground:
        if BASE_DIR not in sys.path:
            sys.path.append(BASE_DIR)
        import add_ground_seed_controller
        add_ground_seed_controller.setup_terrain_seed()
        mod_ground = fg.modifiers.get("Terrain Seed Displace")

    # Set integer ground seed
    set_modifier_socket(mod_ground, 'Ground Seed', int(ground_seed))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    print(f"\n==================================================")
    print(f"STEP 1: Ground Seed set to {ground_seed}")

    # 2. Plant Tiger with 4-feet collision and fix in place
    # Place tiger nicely framed in clearing (X in [-3.0, 3.0], Y in [-20.0, -16.0])
    tiger_x = random.uniform(-2.5, 2.5)
    tiger_y = random.uniform(-19.0, -16.0)
    plant_tiger_on_ground(tiger_arm, fg, depsgraph, center_x=tiger_x, center_y=tiger_y)
    print("STEP 2: Tiger placed with 4-feet contact and fixed.")

    # 3. Setup Tree Seeds Loop
    mod_trees = fg.modifiers.get("GeometryNodes")
    tree_seeds = [random.randint(1, 999999) for _ in range(num_tree_seeds)]

    total_images = num_tree_seeds * images_per_tree
    current_count = 0
    print(f"STEP 3 & 4: Capturing {total_images} images across {num_tree_seeds} tree seeds ({images_per_tree} per seed)...")
    sys.stdout.flush()

    for t_idx, t_seed in enumerate(tree_seeds):
        # Set tree random seed
        set_modifier_socket(mod_trees, 'Random Seed', int(t_seed))
        bpy.context.view_layer.update()

        print(f"\n--- Tree Seed [{t_idx+1}/{num_tree_seeds}]: {t_seed} ---")

        # Capture N images along the camera arc path
        for img_idx in range(images_per_tree):
            t = img_idx / (images_per_tree - 1) if images_per_tree > 1 else 0.5
            fp.offset_factor = t
            bpy.context.view_layer.update()

            out_filename = f"tiger_g{ground_seed}_tree{t_seed:06d}_v{img_idx+1:02d}.jpg"
            out_filepath = os.path.join(output_dir, out_filename)

            scene.render.filepath = out_filepath
            bpy.ops.render.render(write_still=True)

            current_count += 1
            print(f"[{current_count}/{total_images}] Saved: {out_filename}")
            sys.stdout.flush()

    print(f"\n==================================================")
    print(f"DATASET GENERATION COMPLETE! {total_images} images saved to:")
    print(output_dir)
    return total_images


# ==============================================================================
# BLENDER UI PANEL & OPERATOR
# ==============================================================================

class TigerGeneratorProperties(bpy.types.PropertyGroup):
    ground_seed: bpy.props.IntProperty(
        name="Ground Seed",
        description="Ground terrain random seed (0 = auto-randomize)",
        default=0,
        min=0
    )
    num_tree_seeds: bpy.props.IntProperty(
        name="Tree Seeds Count",
        description="Number of different tree layout seeds to generate",
        default=10,
        min=1,
        max=100
    )
    images_per_tree_seed: bpy.props.IntProperty(
        name="Images per Tree Seed",
        description="Number of camera angles along the arc to capture per tree seed",
        default=3,
        min=1,
        max=50
    )
    output_dir: bpy.props.StringProperty(
        name="Output Folder",
        description="Directory to save rendered dataset images",
        default=OUTPUT_DIR_DEFAULT,
        subtype='DIR_PATH'
    )


    collector_count: bpy.props.IntProperty(
        name="Collector Count",
        description="Number of image pairs to capture (0 = indefinite loop)",
        default=0,
        min=0
    )
    angles_per_scene: bpy.props.IntProperty(
        name="Angles per Scene",
        description="Number of camera angles along the arc to capture per scene",
        default=3,
        min=1,
        max=30
    )


class OBJECT_OT_GenerateTigerDataset(bpy.types.Operator):
    bl_idname = "object.generate_tiger_dataset"
    bl_label = "Generate & Capture Dataset"
    bl_description = "Generate ground seed, snap tiger with 4-feet collision, and capture images across tree seeds"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.tiger_gen_props
        g_seed = props.ground_seed if props.ground_seed > 0 else None

        try:
            total = run_generation_pipeline(
                ground_seed=g_seed,
                num_tree_seeds=props.num_tree_seeds,
                images_per_tree=props.images_per_tree_seed,
                output_dir=bpy.path.abspath(props.output_dir)
            )
            self.report({'INFO'}, f"Successfully generated {total} images!")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Generation failed: {str(e)}")
            return {'CANCELLED'}


class OBJECT_OT_PlantTigerOnGround(bpy.types.Operator):
    bl_idname = "object.plant_tiger_on_ground"
    bl_label = "Plant Tiger (4 Paws)"
    bl_description = "Align tiger 4 paws to current ground surface"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        fg = bpy.data.objects.get("Forest Generator")
        tiger_arm = bpy.data.objects.get("White Tiger Skeleton")
        if not fg or not tiger_arm:
            self.report({'ERROR'}, "Forest Generator or White Tiger Skeleton not found!")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        plant_tiger_on_ground(tiger_arm, fg, depsgraph, center_x=tiger_arm.location.x, center_y=tiger_arm.location.y)
        self.report({'INFO'}, "Tiger planted firmly on all 4 paws!")
        return {'FINISHED'}


class OBJECT_OT_Run5SeedCollector(bpy.types.Operator):
    bl_idname = "object.run_5seed_collector"
    bl_label = "Start 5-Seed Collector"
    bl_description = "Runs 5-seed collector (ground seed, height, pose, tree seed, position) saving as number.png"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.tiger_gen_props
        import run_collector
        try:
            run_collector.run_collector(
                count=props.collector_count,
                out_dir=bpy.path.abspath(props.output_dir)
            )
            self.report({'INFO'}, "Collector finished successfully!")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Collector error: {str(e)}")
            return {'CANCELLED'}


class OBJECT_OT_RunSegmentationCollector(bpy.types.Operator):
    bl_idname = "object.run_segmentation_collector"
    bl_label = "Run Segmentation Collector (Image Pairs)"
    bl_description = "Collects paired realistic renders and solid color segmentation masks (Red=Tiger, Blue=BG, Yellow=Sky)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.tiger_gen_props
        if BASE_DIR not in sys.path:
            sys.path.append(BASE_DIR)
        import run_segmentation_collector
        try:
            run_segmentation_collector.run_pipeline(
                count=props.collector_count,
                angles_per_scene=props.angles_per_scene,
                out_dir=bpy.path.abspath(props.output_dir)
            )
            self.report({'INFO'}, "Segmentation collector finished successfully!")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Segmentation collector error: {str(e)}")
            return {'CANCELLED'}


class VIEW3D_PT_TigerDatasetGenerator(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tiger Dataset'
    bl_label = "Tiger & Forest Generator"

    def draw(self, context):
        layout = self.layout
        props = context.scene.tiger_gen_props

        # Labelled Segmentation Section (Image Pairs)
        box_seg = layout.box()
        box_seg.label(text="Labelled Segmentation Collector:", icon='IMAGE_RGB_ALPHA')
        box_seg.label(text="Pairs: Red=Tiger | Blue=BG | Yellow=Sky", icon='COLOR')
        box_seg.prop(props, "collector_count", text="Pair Count (0=Indefinite)")
        box_seg.prop(props, "angles_per_scene", text="Angles per Scene")
        box_seg.prop(props, "output_dir", text="Folder")
        box_seg.operator("object.run_segmentation_collector", text="🎯 Capture Segmentation Pairs", icon='RENDER_STILL')

        layout.separator()

        # 5-Seed Collector Section
        box = layout.box()
        box.label(text="5-Seed Automated Collector:", icon='FILE_REFRESH')
        box.prop(props, "collector_count", text="Count (0=Indefinite)")
        box.prop(props, "output_dir", text="Folder")
        box.operator("object.run_5seed_collector", text="▶ Run 5-Seed Collector", icon='PLAY')

        layout.separator()

        col = layout.column(align=True)
        col.label(text="Tree Sweep Dataset:", icon='CAMERA_DATA')
        col.prop(props, "ground_seed")
        col.prop(props, "num_tree_seeds")
        col.prop(props, "images_per_tree_seed")
        col.operator("object.generate_tiger_dataset", text="▶ Generate & Capture Sweep", icon='RENDER_STILL')

        layout.separator()
        layout.operator("object.plant_tiger_on_ground", text="🐾 Snap Tiger to Ground (4 Paws)", icon='SNAP_ON')


classes = (
    TigerGeneratorProperties,
    OBJECT_OT_GenerateTigerDataset,
    OBJECT_OT_PlantTigerOnGround,
    OBJECT_OT_Run5SeedCollector,
    OBJECT_OT_RunSegmentationCollector,
    VIEW3D_PT_TigerDatasetGenerator,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.tiger_gen_props = bpy.props.PointerProperty(type=TigerGeneratorProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.tiger_gen_props


if __name__ == "__main__":
    # If run in Blender, register the UI panel
    register()
    print("Tiger Dataset Generator UI Panel registered! Check the 3D Viewport N-panel under 'Tiger Dataset' tab.")
