"""
Per-image Visible Basecolor Extractor
=====================================
For every rendered pair (e.g. images/000001.png) this produces:

1. basecolors/000001.png
   The tiger's 2048x2048 basecolor (skin) atlas where ONLY the mesh faces that are
   visible in images/000001.png keep their skin texels. Everything else is solid white.
   (Whole visible faces are filled, so patches are dense and hole-free.)

2. basecolor_masks/000001.png
   Black/white map of exactly which atlas texels were filled. Needed because the skin is
   itself mostly white, so "white" alone cannot tell "empty" from "white fur".

3. dataset_joints.sqlite  (stdlib sqlite3, keyed by image_id = 1, 2, ...)
   - pairs         one row per image: visible/total faces, atlas coverage, tiger pixel bbox ...
   - bones         bone index -> name / parent (fixed order)
   - joints        one row per image x bone: armature-local, world and camera-space positions
                   of head/tail, projected pixel (px, py), depth, in_frame, on_tiger
   - pose_vectors  one flat float32 BLOB per image (local / camera / pixel) for similarity search

How visibility is found (joints alone cannot say which skin is visible):
- The tiger mesh gets a per-corner colour attribute that encodes its face index in RGB.
- A flat-shaded pass is rendered from the SAME frame/camera: tiger = face-index colour,
  everything else = black, with alpha-cutout leaves kept as cutouts so they occlude correctly.
- Every face index found in that image is "visible". Its UV triangle is looked up in a
  precomputed UV->face map and the skin texels are copied into the output atlas.
"""
import hashlib
import os
import sqlite3
import time

import numpy as np
import bpy
from bpy_extras.object_utils import world_to_camera_view

import tiger_identities

IDENTITY_DIRNAME = "identities"
# Vertex-group name fragments -> how much of the stripe change a face may receive (0 = protected).
PROTECTED_PARTS = ("eye", "nose", "lip", "tounge", "tongue", "jaw", "mouth")
PARTIAL_PARTS = ("head", "ear", "brow", "cheek", "neckinf")
PARTIAL_WEIGHT = 0.3

FACE_ATTR = "fid_col"
CACHE_DIRNAME = "cache"
DB_NAME = "dataset_joints.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs (
    image_id INTEGER PRIMARY KEY,
    basecolor_file TEXT, basecolor_mask_file TEXT,
    visible_faces INTEGER, total_faces INTEGER, atlas_coverage REAL,
    tiger_pixels INTEGER,
    bbox_x0 INTEGER, bbox_y0 INTEGER, bbox_x1 INTEGER, bbox_y1 INTEGER,
    joints_in_frame INTEGER, joints_on_tiger INTEGER,
    res_x INTEGER, res_y INTEGER, invalid_id_pixels INTEGER,
    tiger_id INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS identities (
    tiger_id INTEGER PRIMARY KEY, skin_file TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS bones (
    bone_index INTEGER PRIMARY KEY, name TEXT, parent TEXT
);
CREATE TABLE IF NOT EXISTS joints (
    image_id INTEGER, bone_index INTEGER, bone TEXT,
    head_lx REAL, head_ly REAL, head_lz REAL,
    head_wx REAL, head_wy REAL, head_wz REAL,
    tail_wx REAL, tail_wy REAL, tail_wz REAL,
    head_cx REAL, head_cy REAL, head_cz REAL,
    px REAL, py REAL, depth REAL,
    in_frame INTEGER, on_tiger INTEGER,
    PRIMARY KEY (image_id, bone_index)
);
CREATE TABLE IF NOT EXISTS pose_vectors (
    image_id INTEGER PRIMARY KEY, n_bones INTEGER,
    local_vec BLOB, cam_vec BLOB, px_vec BLOB
);
"""


def _upstream_image(socket, depth=0):
    """Walks upstream from a shader socket to the first Image Texture node. Returns (image, node) or (None, None)."""
    if depth > 8 or not socket.is_linked:
        return None, None
    node = socket.links[0].from_node
    if node.bl_idname == 'ShaderNodeTexImage':
        return node.image, node
    for inp in node.inputs:
        img, found = _upstream_image(inp, depth + 1)
        if img:
            return img, found
    return None, None


def _rasterize_uv_faces(uv_tris, width, height):
    """
    uv_tris: (F, 3, 2) UV triangles in 0..1. Returns int32 (H, W): face index per texel, -1 = none.
    Texel rows follow Blender's image order (row 0 = bottom, v = 0).
    """
    n_faces = len(uv_tris)
    out = np.full((height, width), -1, dtype=np.int32)
    p = uv_tris.astype(np.float64) * np.array([width, height], dtype=np.float64)
    x_lo = np.clip(np.floor(p[..., 0].min(axis=1)).astype(np.int64), 0, width - 1)
    x_hi = np.clip(np.floor(p[..., 0].max(axis=1)).astype(np.int64), 0, width - 1)
    y_lo = np.clip(np.floor(p[..., 1].min(axis=1)).astype(np.int64), 0, height - 1)
    y_hi = np.clip(np.floor(p[..., 1].max(axis=1)).astype(np.int64), 0, height - 1)
    eps = 1e-5

    for f in range(n_faces):
        a, b, c = p[f]
        x0, x1, y0, y1 = x_lo[f], x_hi[f], y_lo[f], y_hi[f]
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        covered = False
        if abs(d) > 1e-12:
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
            w2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
            w3 = 1.0 - w1 - w2
            m = (w1 >= -eps) & (w2 >= -eps) & (w3 >= -eps)
            if m.any():
                out[y0:y1 + 1, x0:x1 + 1][m] = f
                covered = True
        if not covered:
            # Sub-texel or degenerate triangle: stamp the texel under its centroid so it is not lost.
            cx = int(np.clip((a[0] + b[0] + c[0]) / 3.0, 0, width - 1))
            cy = int(np.clip((a[1] + b[1] + c[1]) / 3.0, 0, height - 1))
            out[cy, cx] = f
    return out


class BasecolorExtractor:
    def __init__(self, scene, tiger_arm, tiger_mesh, out_dir, min_face_pixels=1, keep_debug=False,
                 n_identities=1):
        self.scene = scene
        self.arm = tiger_arm
        self.obj = tiger_mesh
        self.me = tiger_mesh.data
        self.out_dir = out_dir
        self.min_face_pixels = max(1, int(min_face_pixels))
        self.keep_debug = keep_debug

        self.basecolor_dir = os.path.join(out_dir, "basecolors")
        self.mask_dir = os.path.join(out_dir, "basecolor_masks")
        self.cache_dir = os.path.join(out_dir, CACHE_DIRNAME)
        for d in (self.basecolor_dir, self.mask_dir, self.cache_dir):
            os.makedirs(d, exist_ok=True)

        self._check_triangles()
        self.n_faces = len(self.me.polygons)
        self.skin = self._load_skin_pixels()            # (H, W, 3) float32, Non-Color
        self.height, self.width = self.skin.shape[:2]
        self.uv_face_map = self._load_or_build_uv_face_map()
        self.atlas_tiger_texels = int((self.uv_face_map >= 0).sum())
        self.n_identities = max(1, int(n_identities))
        self.current_id = 0
        self.skins = {0: self.skin}
        self.identity_images = {0: self.skin_image}
        self.identity_dir = os.path.join(out_dir, IDENTITY_DIRNAME)
        if self.n_identities > 1:
            self._setup_identities()

        self._install_face_id_attribute()
        self._occluder_cache = {}
        self.tiger_mat = self._make_tiger_id_material()
        self.black_mat = self._make_black_material("FID_Black")
        self.black_world = self._make_black_world()

        self.db = sqlite3.connect(os.path.join(out_dir, DB_NAME))
        self.db.executescript(SCHEMA)
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(pairs)")]
        if "tiger_id" not in cols:                      # database written by an older version
            self.db.execute("ALTER TABLE pairs ADD COLUMN tiger_id INTEGER DEFAULT 0")
        self._write_bones_table()
        self._write_identities_table()
        self.db.commit()

        print(f"[Basecolor] Ready: {self.n_faces} faces, atlas {self.width}x{self.height}, "
              f"{self.atlas_tiger_texels / (self.width * self.height) * 100:.1f}% of atlas is skin.")

    # ------------------------------------------------------------------
    # One-time setup
    # ------------------------------------------------------------------
    def _check_triangles(self):
        totals = np.empty(len(self.me.polygons), dtype=np.int32)
        self.me.polygons.foreach_get("loop_total", totals)
        if not (totals == 3).all():
            raise RuntimeError("Tiger mesh must be triangulated for the face-id pass.")
        starts = np.empty(len(self.me.polygons), dtype=np.int32)
        self.me.polygons.foreach_get("loop_start", starts)
        self._loop_starts = starts

    def _load_skin_pixels(self):
        mat = self.obj.material_slots[0].material
        img = None
        self.skin_node = None
        for node in mat.node_tree.nodes:
            if node.bl_idname == 'ShaderNodeBsdfPrincipled':
                img, self.skin_node = _upstream_image(node.inputs['Base Color'])
                break
        if img is None:
            raise RuntimeError("Could not find the tiger Base Color image texture.")
        self.skin_image = img
        tmp = img.copy()
        try:
            tmp.colorspace_settings.name = 'Non-Color'
            w, h = tmp.size
            buf = np.empty(w * h * 4, dtype=np.float32)
            tmp.pixels.foreach_get(buf)
        finally:
            bpy.data.images.remove(tmp)
        print(f"[Basecolor] Skin image: '{img.name}' ({w}x{h})")
        return buf.reshape(h, w, 4)[..., :3].copy()

    def _uv_triangles(self):
        n = np.empty(len(self.me.loops) * 2, dtype=np.float32)
        self.me.uv_layers.active.uv.foreach_get("vector", n)
        uv = n.reshape(-1, 2)
        idx = self._loop_starts[:, None] + np.arange(3)[None, :]
        return uv[idx]                                   # (F, 3, 2)

    def _load_or_build_uv_face_map(self):
        tris = self._uv_triangles()
        key = hashlib.md5(tris.tobytes()).hexdigest()[:10]
        path = os.path.join(self.cache_dir, f"uv_face_map_{self.n_faces}_{self.width}x{self.height}_{key}.npy")
        if os.path.exists(path):
            print(f"[Basecolor] Loaded cached UV face map: {os.path.basename(path)}")
            return np.load(path)
        t0 = time.time()
        print("[Basecolor] Rasterizing UV face map (one-time, cached)...")
        uv_map = _rasterize_uv_faces(tris, self.width, self.height)
        np.save(path, uv_map)
        print(f"[Basecolor] UV face map built in {time.time() - t0:.1f}s")
        return uv_map

    def _install_face_id_attribute(self):
        """Per-corner colour attribute: RGB = bytes of (face_index + 1); 0 is reserved for 'no tiger'."""
        if FACE_ATTR in self.me.attributes:
            self.me.attributes.remove(self.me.attributes[FACE_ATTR])
        attr = self.me.attributes.new(FACE_ATTR, 'FLOAT_COLOR', 'CORNER')
        ids = np.repeat(np.arange(1, self.n_faces + 1, dtype=np.int64), 3)
        col = np.ones((len(ids), 4), dtype=np.float32)
        col[:, 0] = ((ids >> 16) & 255) / 255.0
        col[:, 1] = ((ids >> 8) & 255) / 255.0
        col[:, 2] = (ids & 255) / 255.0
        attr.data.foreach_set("color", col.ravel())
        self.me.update()

    def _make_tiger_id_material(self):
        mat = bpy.data.materials.get("FID_Tiger") or bpy.data.materials.new("FID_Tiger")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = 'GEOMETRY'
        attr.attribute_name = FACE_ATTR
        emit.inputs['Strength'].default_value = 1.0
        nt.links.new(attr.outputs['Color'], emit.inputs['Color'])
        nt.links.new(emit.outputs['Emission'], out.inputs['Surface'])
        mat.use_backface_culling = False
        return mat

    def _make_black_material(self, name):
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        nt.links.new(emit.outputs['Emission'], out.inputs['Surface'])
        return mat

    def _make_black_world(self):
        world = bpy.data.worlds.get("FID_World_Black") or bpy.data.worlds.new("FID_World_Black")
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld")
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
        return world

    def _occluder_material(self, mat):
        """
        Black stand-in for any non-tiger material. Alpha-cutout materials (leaves) keep their cutout
        (hard threshold at 0.5) so gaps between leaves do not hide the tiger.
        """
        if mat is None or not mat.use_nodes:
            return self.black_mat
        if mat.name in self._occluder_cache:
            return self._occluder_cache[mat.name]

        has_alpha = any(
            n.bl_idname == 'ShaderNodeBsdfPrincipled' and n.inputs['Alpha'].is_linked
            for n in mat.node_tree.nodes
        )
        if not has_alpha:
            self._occluder_cache[mat.name] = self.black_mat
            return self.black_mat

        new = mat.copy()
        new.name = f"FID_occ_{mat.name}"
        nt = new.node_tree
        alpha_src = next(
            n.inputs['Alpha'].links[0].from_socket
            for n in nt.nodes
            if n.bl_idname == 'ShaderNodeBsdfPrincipled' and n.inputs['Alpha'].is_linked
        )
        clip = nt.nodes.new("ShaderNodeMath")
        clip.operation = 'GREATER_THAN'
        clip.inputs[1].default_value = 0.5
        nt.links.new(alpha_src, clip.inputs[0])
        transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(clip.outputs['Value'], mix.inputs['Fac'])
        nt.links.new(transparent.outputs['BSDF'], mix.inputs[1])
        nt.links.new(emit.outputs['Emission'], mix.inputs[2])
        for out in [n for n in nt.nodes if n.bl_idname == 'ShaderNodeOutputMaterial']:
            for link in list(out.inputs['Surface'].links):
                nt.links.remove(link)
            nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
        self._occluder_cache[mat.name] = new
        return new

    def _write_bones_table(self):
        rows = [(i, pb.name, pb.parent.name if pb.parent else None)
                for i, pb in enumerate(self.arm.pose.bones)]
        self.db.executemany("INSERT OR REPLACE INTO bones VALUES (?,?,?)", rows)

    # ------------------------------------------------------------------
    # Tiger identities (stripe variants of the skin)
    # ------------------------------------------------------------------
    def _face_change_weights(self):
        """Per-face 0..1: how much a face may be altered. Eyes/nose/lips/tongue/jaw = 0, rest of head partial."""
        n_verts = len(self.me.vertices)
        group_names = [g.name.lower() for g in self.obj.vertex_groups]
        group_w = np.ones(len(group_names), dtype=np.float32)
        for i, name in enumerate(group_names):
            if any(p in name for p in PROTECTED_PARTS):
                group_w[i] = 0.0
            elif any(p in name for p in PARTIAL_PARTS):
                group_w[i] = PARTIAL_WEIGHT
        vert_w = np.ones(n_verts, dtype=np.float32)
        for v in self.me.vertices:
            if v.groups:
                best = max(v.groups, key=lambda g: g.weight)
                vert_w[v.index] = group_w[best.group]
        verts = np.empty(self.n_faces * 3, dtype=np.int32)
        self.me.polygons.foreach_get("vertices", verts)
        return vert_w[verts.reshape(-1, 3)].min(axis=1)

    def _read_rgb_png(self, path):
        img = bpy.data.images.load(path)
        try:
            img.colorspace_settings.name = 'Non-Color'
            w, h = img.size
            buf = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(buf)
        finally:
            bpy.data.images.remove(img)
        return buf.reshape(h, w, 4)[..., :3].copy()

    def _setup_identities(self):
        """tiger_id 0 = original skin; 1..N-1 = reproducible stripe variants, saved to <out_dir>/identities/."""
        os.makedirs(self.identity_dir, exist_ok=True)
        missing = [k for k in range(1, self.n_identities)
                   if not os.path.exists(os.path.join(self.identity_dir, f"tiger_{k:02d}.png"))]
        if missing:
            t0 = time.time()
            print(f"[Identities] Generating stripe variants for tiger ids {missing} (one-time per output folder)...")
            variants = tiger_identities.build_variants(
                self.skin, self.uv_face_map, self._face_change_weights(), missing)
            for k in missing:
                self._save_rgb_png(os.path.join(self.identity_dir, f"tiger_{k:02d}.png"), variants[k])
            print(f"[Identities] Done in {time.time() - t0:.1f}s")
        if not os.path.exists(os.path.join(self.identity_dir, "tiger_00.png")):
            self._save_rgb_png(os.path.join(self.identity_dir, "tiger_00.png"),
                               tiger_identities.quantize(self.skin))
        self.skins[0] = tiger_identities.quantize(self.skin)
        self._loaded_identity_images = []
        for k in range(1, self.n_identities):
            path = os.path.join(self.identity_dir, f"tiger_{k:02d}.png")
            self.skins[k] = self._read_rgb_png(path)            # exact texels, same as the saved file
            img = bpy.data.images.load(path)                    # sRGB, used by the render
            img.name = f"TigerID_{k:02d}"
            self.identity_images[k] = img
            self._loaded_identity_images.append(img)

    def _write_identities_table(self):
        rows = [(k, f"{IDENTITY_DIRNAME}/tiger_{k:02d}.png",
                 "original skin" if k == 0 else "stripe variant")
                for k in range(self.n_identities)]
        self.db.executemany("INSERT OR REPLACE INTO identities VALUES (?,?,?)", rows)

    def select_identity(self, tiger_id):
        """Swaps the tiger's Base Color texture. Call before rendering the scene's pairs."""
        if tiger_id not in self.identity_images:
            raise ValueError(f"tiger_id {tiger_id} out of range (0..{self.n_identities - 1})")
        self.skin_node.image = self.identity_images[tiger_id]
        self.current_id = tiger_id

    # ------------------------------------------------------------------
    # Per-pair work
    # ------------------------------------------------------------------
    def _render_face_ids(self):
        """Renders the flat face-id pass at the current frame/camera. Returns (H, W) int64, -1 = none (top-left origin)."""
        scene = self.scene
        r = scene.render
        vs = scene.view_settings
        ims = r.image_settings
        tmp_path = os.path.join(self.cache_dir, "_face_id_tmp.png")

        saved = dict(
            filepath=r.filepath, file_format=ims.file_format, color_mode=ims.color_mode,
            color_depth=ims.color_depth, dither=r.dither_intensity, filter_size=r.filter_size,
            taa=scene.eevee.taa_render_samples, world=scene.world,
            view=vs.view_transform, look=vs.look, exposure=vs.exposure, gamma=vs.gamma,
        )
        orig_mats = {}
        added_slots = []
        try:
            for obj in bpy.data.objects:
                # Include hide_render objects too: they can still be Geometry Nodes instancing sources
                # (e.g. 'Leaves'), and their original lit materials would leak into the pass.
                if obj.type != 'MESH':
                    continue
                is_tiger = obj == self.obj
                orig_mats[obj.name] = [s.material for s in obj.material_slots]
                if not obj.material_slots:
                    obj.data.materials.append(self.black_mat)
                    added_slots.append(obj)
                    continue
                for slot in obj.material_slots:
                    slot.material = self.tiger_mat if is_tiger else self._occluder_material(slot.material)

            scene.world = self.black_world
            vs.view_transform = 'Raw'
            vs.look = 'None'
            vs.exposure = 0.0
            vs.gamma = 1.0
            r.dither_intensity = 0.0
            r.filter_size = 0.0
            scene.eevee.taa_render_samples = 1
            ims.file_format = 'PNG'
            ims.color_mode = 'RGB'
            ims.color_depth = '8'
            r.filepath = tmp_path
            bpy.ops.render.render(write_still=True)
        finally:
            for obj in added_slots:
                obj.data.materials.pop()
            for name, mats in orig_mats.items():
                obj = bpy.data.objects.get(name)
                if obj:
                    for i, m in enumerate(mats):
                        if i < len(obj.material_slots):
                            obj.material_slots[i].material = m
            scene.world = saved["world"]
            vs.view_transform = saved["view"]
            vs.look = saved["look"]
            vs.exposure = saved["exposure"]
            vs.gamma = saved["gamma"]
            r.dither_intensity = saved["dither"]
            r.filter_size = saved["filter_size"]
            scene.eevee.taa_render_samples = saved["taa"]
            ims.file_format = saved["file_format"]
            ims.color_mode = saved["color_mode"]
            ims.color_depth = saved["color_depth"]
            r.filepath = saved["filepath"]

        img = bpy.data.images.load(tmp_path)
        try:
            img.colorspace_settings.name = 'Non-Color'
            w, h = img.size
            buf = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(buf)
        finally:
            bpy.data.images.remove(img)
        px = buf.reshape(h, w, 4)[::-1]                  # flip to top-left origin
        rgb = np.rint(px[..., :3] * 255.0).astype(np.int64)
        ids = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        if not self.keep_debug:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return ids - 1                                    # -1 = no tiger, else face index

    def _project_joints(self, cam_eval, face_ids):
        scene = self.scene
        res_x = scene.render.resolution_x * scene.render.resolution_percentage // 100
        res_y = scene.render.resolution_y * scene.render.resolution_percentage // 100
        arm_mw = self.arm.matrix_world
        cam_inv = cam_eval.matrix_world.inverted()
        h, w = face_ids.shape

        rows, local_vec, cam_vec, px_vec = [], [], [], []
        in_frame_n = on_tiger_n = 0
        for i, pb in enumerate(self.arm.pose.bones):
            head_l = pb.head.copy()
            head_w = arm_mw @ pb.head
            tail_w = arm_mw @ pb.tail
            head_c = cam_inv @ head_w
            nx, ny, depth = world_to_camera_view(scene, cam_eval, head_w)
            px, py = nx * res_x, (1.0 - ny) * res_y
            in_frame = int(depth > 0.0 and 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0)
            on_tiger = 0
            if in_frame:
                ix, iy = min(w - 1, int(px)), min(h - 1, int(py))
                on_tiger = int(face_ids[iy, ix] >= 0)
            in_frame_n += in_frame
            on_tiger_n += on_tiger
            rows.append((
                i, pb.name,
                head_l.x, head_l.y, head_l.z,
                head_w.x, head_w.y, head_w.z,
                tail_w.x, tail_w.y, tail_w.z,
                head_c.x, head_c.y, head_c.z,
                px, py, depth, in_frame, on_tiger,
            ))
            local_vec.append(tuple(head_l))
            cam_vec.append(tuple(head_c))
            px_vec.append((px, py))
        return rows, local_vec, cam_vec, px_vec, in_frame_n, on_tiger_n, res_x, res_y

    def _save_rgb_png(self, path, rgb):
        h, w = rgb.shape[:2]
        img = bpy.data.images.new("BC_tmp", w, h, alpha=False)
        try:
            img.colorspace_settings.name = 'Non-Color'
            rgba = np.ones((h, w, 4), dtype=np.float32)
            rgba[..., :3] = rgb
            img.pixels.foreach_set(rgba.ravel())
            img.filepath_raw = path
            img.file_format = 'PNG'
            img.save()
        finally:
            bpy.data.images.remove(img)

    def process_pair(self, image_id, id_str):
        """Call right after the pair was rendered, with the same frame and camera still active."""
        t0 = time.time()
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        cam_eval = self.scene.camera.evaluated_get(dg)

        face_ids = self._render_face_ids()
        invalid = (face_ids >= self.n_faces)
        invalid_n = int(invalid.sum())
        face_ids[invalid] = -1

        on_tiger = face_ids >= 0
        counts = np.bincount(face_ids[on_tiger], minlength=self.n_faces)
        visible = counts >= self.min_face_pixels
        lut = np.concatenate([[False], visible])          # index = face_map + 1 so "-1" -> False
        atlas_vis = lut[self.uv_face_map + 1]

        atlas = np.where(atlas_vis[..., None], self.skins[self.current_id], 1.0).astype(np.float32)
        bc_name, mask_name = f"{id_str}.png", f"{id_str}.png"
        self._save_rgb_png(os.path.join(self.basecolor_dir, bc_name), atlas)
        self._save_rgb_png(os.path.join(self.mask_dir, mask_name),
                           np.repeat(atlas_vis[..., None].astype(np.float32), 3, axis=2))

        (rows, local_vec, cam_vec, px_vec,
         in_frame_n, on_tiger_n, res_x, res_y) = self._project_joints(cam_eval, face_ids)

        ys, xs = np.nonzero(on_tiger)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else (-1, -1, -1, -1)
        coverage = float(atlas_vis.sum()) / max(1, self.atlas_tiger_texels)
        n_vis = int(visible.sum())

        self.db.execute(
            "INSERT OR REPLACE INTO pairs (image_id, basecolor_file, basecolor_mask_file, visible_faces,"
            " total_faces, atlas_coverage, tiger_pixels, bbox_x0, bbox_y0, bbox_x1, bbox_y1,"
            " joints_in_frame, joints_on_tiger, res_x, res_y, invalid_id_pixels, tiger_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (image_id, f"basecolors/{bc_name}", f"basecolor_masks/{mask_name}",
             n_vis, self.n_faces, coverage, int(on_tiger.sum()), *bbox,
             in_frame_n, on_tiger_n, res_x, res_y, invalid_n, self.current_id))
        self.db.execute("DELETE FROM joints WHERE image_id=?", (image_id,))
        self.db.executemany(
            "INSERT INTO joints VALUES (" + ",".join(["?"] * 20) + ")",
            [(image_id, *row) for row in rows])
        self.db.execute(
            "INSERT OR REPLACE INTO pose_vectors VALUES (?,?,?,?,?)",
            (image_id, len(rows),
             np.asarray(local_vec, dtype=np.float32).tobytes(),
             np.asarray(cam_vec, dtype=np.float32).tobytes(),
             np.asarray(px_vec, dtype=np.float32).tobytes()))
        self.db.commit()

        return dict(tiger_id=self.current_id,
                    visible_faces=n_vis, total_faces=self.n_faces, coverage=coverage,
                    tiger_pixels=int(on_tiger.sum()), invalid_id_pixels=invalid_n,
                    joints_in_frame=in_frame_n, joints_on_tiger=on_tiger_n,
                    elapsed=time.time() - t0)

    def close(self):
        """Restores the original skin, removes the temporary face-id attribute and closes the database."""
        try:
            self.skin_node.image = self.skin_image
            for img in getattr(self, "_loaded_identity_images", []):
                bpy.data.images.remove(img)
        except Exception:
            pass
        try:
            if FACE_ATTR in self.me.attributes:
                self.me.attributes.remove(self.me.attributes[FACE_ATTR])
        except Exception:
            pass
        try:
            self.db.commit()
            self.db.close()
        except Exception:
            pass
