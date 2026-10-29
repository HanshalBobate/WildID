# 🐅 Tiger Semantic Segmentation Dataset Generator

A self-contained, production-grade 3D procedural dataset generation pipeline for training semantic segmentation models on tigers in complex forest environments.

---

## 🌟 Highlights

- **Paired Data Generation**:
  1. **Photorealistic Renders**: White tiger with authentic fur texturing, 19 natural poses, 4-paw slope-aligned ground contact, procedurally displaced terrain, instanced trees/leaves/bark, and HDR sunrise lighting.
  2. **Solid Color 3-Class Segmentation Masks**:
     - 🔴 **Tiger**: Pure Solid Red `(255, 0, 0)` / `#FF0000`
     - 🔵 **Background (Terrain & Trees)**: Pure Solid Blue `(0, 0, 255)` / `#0000FF`
     - 🟡 **Sky & Horizon**: Pure Solid Yellow `(255, 255, 0)` / `#FFFF00`
  3. **Zero Anti-Aliasing Bleed**: Post-quantization snapping guarantees 100% of mask pixels are strictly mapped to one of the 3 exact RGB classes.
  4. **Side-by-Side Validation Previews**: Instant visual inspection saved as lightweight JPEG previews.
  5. **Comprehensive Metadata**: Every pair logged to `dataset_metadata.csv` with terrain seeds, tree seeds, tiger pose, position, rotation, camera arc frame, and 3D camera coordinates.

- **Variable-Angled Camera Arc Sweep**:
  Implements the mathematical arc trajectory from `animate_camera_arc.py`:
  $$2y = -41 - \sqrt{41^2 - x^2} \quad \text{for } -28 \le x \le 28 \quad (R = 41)$$
  Captures variable perspectives (Left flank, Center frontal, Right flank) along the 120-frame timeline with eye-level to elevated vantage heights ($2.0\text{m} - 4.8\text{m}$) tracking the tiger.

---

## 📁 Package Structure

```
Best_Work_Yet/
├── target.blend                # Master 3D scene (Forest, Tiger Armature, Geometry Nodes, Shaders)
├── run_segmentation_collector.py # Primary automated segmentation pair collector
├── basecolor_extractor.py      # Visible-only basecolor atlas + joints database (called by the collector)
├── tiger_identities.py         # Reproducible stripe-pattern variants of the skin (tiger_id 0..N-1)
├── merge_basecolors.py         # Merges visible-only atlases into one FULL basecolor per tiger (standalone)
├── generate_tiger_dataset.py   # Blender 3D Viewport N-panel interactive UI add-on
├── animate_camera_arc.py       # Camera arc keyframer & timeline animator (frames 1 to 120)
├── add_ground_seed_controller.py # Geometry Nodes terrain displacement setup script
├── test_all_poses.py           # Verification script rendering all 19 tiger poses
├── camera_rig/
│   └── camera_arc_rig.py       # Mathematical curve & camera rig generator
├── poses/                      # 19 authentic white tiger pose blend files
├── textures/                   # 101 PBR textures for bark, leaves, and forest ground
├── assets/                     # HDR environment skies and ground materials
└── README.md                   # This documentation file
```

---

## 🚀 Quick Start

### 1. Requirements
- **Blender**: 4.0 or newer (tested with Blender 5.2 LTS). Make sure `blender` is in PATH or installed in standard location (`C:\Program Files\Blender Foundation\Blender 5.x\`).
- **Python**: 3.10+ (Standard Python library + `numpy`).

### 2. Run from Terminal / PowerShell (Auto-Launcher)
Run the script directly with Python. It automatically locates Blender and launches headless rendering:

```powershell
# Interactive mode (prompts for pairs, angles, and output directory):
python run_segmentation_collector.py

# Collect 10 scenes with 3 camera arc angles each (30 image pairs total):
python run_segmentation_collector.py --scenes 10 --angles_per_scene 3

# Run continuously until stopped (Ctrl+C):
python run_segmentation_collector.py --count 0 --angles_per_scene 3

# Custom resolution:
python run_segmentation_collector.py --scenes 5 --angles_per_scene 3 --res_x 1920 --res_y 1080
```

### 3. Run Directly with Blender Headless
```powershell
blender -b target.blend --python run_segmentation_collector.py -- --scenes 10 --angles_per_scene 3
```

### 4. Interactive GUI inside Blender
1. Open `target.blend` in Blender.
2. In the **Scripting** workspace, open and run `generate_tiger_dataset.py`.
3. In the **3D Viewport**, press <kbd>N</kbd> to open the sidebar.
4. Go to the **Tiger Dataset** tab:
   - Set **Pair Count** (0 = Indefinite)
   - Set **Angles per Scene** (Default: 3)
   - Click **🎯 Capture Segmentation Pairs**

---

## 📊 Dataset Output Format

Dataset output folder structure:

```
dataset_segmentation_pairs/
├── images/
│   ├── 000001.png       # Realistic textured render
│   ├── 000002.png
│   └── ...
├── masks/
│   ├── 000001.png       # Solid 3-class segmentation mask
│   ├── 000002.png
│   └── ...
├── previews/
│   ├── 000001_preview.jpg # Side-by-side visual validation (Render + Mask)
│   ├── 000002_preview.jpg
│   └── ...
└── dataset_metadata.csv  # Complete ground-truth CSV log
```

### Metadata Fields (`dataset_metadata.csv`)

| Column | Description |
| :--- | :--- |
| `image_id` | Sequential dataset pair ID (`1`, `2`, ...) |
| `scene_id` | Scene identifier (`1`, `2`, ...) |
| `angle_index` | Angle index within the scene (`1`, `2`, `3`) |
| `render_file` | Relative path to realistic render PNG |
| `mask_file` | Relative path to solid color mask PNG |
| `preview_file` | Relative path to side-by-side preview JPG |
| `ground_seed` | Integer procedural noise seed for terrain displacement |
| `ground_height` | Terrain elevation amplitude scale |
| `tree_seed` | Integer seed for Geometry Nodes forest instancing |
| `tiger_pose` | Active action name (e.g. `Pose_Stalking`, `Pose_Sleeping_On_Side`) |
| `pose_file` | Source pose blend file in `poses/` |
| `tiger_x`, `tiger_y`, `tiger_z` | Armature world location coordinates |
| `tiger_yaw` | Armature yaw orientation (radians) |
| `cam_arc_frame` | Keyframe along the 1–120 arc animation |
| `cam_offset` | Normalized arc curve parameter $t \in [0.0, 1.0]$ |
| `cam_x`, `cam_y`, `cam_z` | Camera evaluated world coordinates |
| `cam_rot_x_deg`, `cam_rot_y_deg`, `cam_rot_z_deg` | Camera Euler rotation (degrees) |
| `cam_height` | Camera arc rig elevation height |
| `resolution` | Render dimensions (e.g. `1920x1080`) |
| `timestamp` | ISO-8601 generation timestamp |

---

## 🧬 Visible-Only Basecolor Atlas + Joints Database

For every pair (e.g. `images/000001.png`) the collector also writes the tiger's 2048x2048 skin atlas with **only the body parts visible in that image** filled in (everything else solid white). Disable with `--no_basecolor`.

```
basecolors/000001.png        # skin atlas, visible faces only, rest = white
basecolor_masks/000001.png   # white = filled texel (the skin is itself mostly white, so use this to tell "empty" from "white fur")
dataset_joints.sqlite        # joints / pose vectors, keyed by image_id (= 000001 -> 1)
cache/uv_face_map_*.npy      # one-time UV -> face lookup (safe to delete)
```

**How visibility is decided:** joints alone cannot say which skin is visible, so each pair gets an extra flat-shaded pass from the same frame and camera in which every tiger face is drawn in a colour that encodes its face index, and everything else is black (alpha-cutout leaves keep their cutouts, so gaps between leaves do not hide the tiger). Every face index found in that pass is visible, and its whole UV triangle is copied from the original skin (`--min_face_pixels N` sets how many pixels a face needs, default 1).

**`dataset_joints.sqlite` tables**

| Table | Contents |
| :--- | :--- |
| `pairs` | one row per image: visible/total faces, atlas coverage, tiger pixel count + bbox, joints in frame / on tiger, resolution |
| `bones` | bone index -> name / parent (fixed order, 120 bones) |
| `joints` | one row per image x bone: armature-local, world and camera-space head position, tail world position, projected pixel `px, py`, `depth`, `in_frame`, `on_tiger` |
| `pose_vectors` | one flat float32 BLOB per image: `local_vec` (pose only), `cam_vec` (pose + view), `px_vec` (120x3 / 120x3 / 120x2) |

`on_tiger` means the joint's projected pixel lies on a visible tiger pixel. The root/helper bones (`Tiger`, `joints_grp`, `Root`) sit at the armature origin and are normally `0`.

### Tiger identities (distinct stripe patterns)

Real tigers are told apart by their stripes, but the project has one skin. `--tiger_ids N` (default 4) gives each scene one of N tigers: **id 0 is the original skin**, ids 1..N-1 are reproducible variants of it whose stripes are warped (positions/shapes), thickened/thinned (stripes can break or merge), and differ left vs right. Warping fades to zero at every UV island border so seams stay clean; eyes, nose, lips, tongue and jaw are never touched and the rest of the head changes only slightly. The same `tiger_id` always yields the same tiger, and the skins are saved as `identities/tiger_00.png`, `tiger_01.png`, ... (generated once per output folder; delete the folder's `identities/` to regenerate). Each pair's tiger is stored in `pairs.tiger_id`. Use `--tiger_ids 1` to render only the original skin.

### Full basecolor per tiger (merge many views)

```powershell
pip install numpy pillow          # once; merge_basecolors.py does not need Blender
python merge_basecolors.py --dataset .\dataset_segmentation_pairs
```

All pairs with the same `tiger_id` are combined, so every extra angle adds more of that tiger's skin. Re-run it any time, it rebuilds from all pairs. Output in `full_basecolors/`:

```
tiger_00.png                 # merged full basecolor (never-seen texels stay white)
tiger_00_seen_mask.png       # white = seen in at least one image
tiger_00_observations.png    # how many images saw each texel
tiger_00_missing.png         # merged image with still-unseen skin painted red
merge_report.json            # per tiger: images, coverage %, conflicts, coverage-vs-images curve
```

The console table shows coverage per tiger. `conflicts` counts texels where two images disagree (should be 0) and `vs identities/` checks the merged texels against the tiger's real skin (should be 0 bad texels). Coverage is measured over texels that belong to the tiger mesh only. It reaches 100% only if every part of the skin has been seen, and areas such as the belly and inner legs may stay missing until more varied poses and angles are collected.

```python
import sqlite3, numpy as np
db = sqlite3.connect("dataset_segmentation_pairs/dataset_joints.sqlite")
vec = np.frombuffer(db.execute("select cam_vec from pose_vectors where image_id=1").fetchone()[0], np.float32).reshape(-1, 3)
```

---

## 🎨 Segmentation Classes Specification

| Class | RGB Color | Hex Code | Normalization `[0, 1]` | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Tiger** | `(255, 0, 0)` | `#FF0000` | `[1.0, 0.0, 0.0]` | Full tiger body, head, legs, paws, tail, whiskers |
| **Background** | `(0, 0, 255)` | `#0000FF` | `[0.0, 0.0, 1.0]` | Sculpted terrain, instanced trees, bark, branches, fallen leaves |
| **Sky** | `(255, 255, 0)` | `#FFFF00` | `[1.0, 1.0, 0.0]` | Atmosphere, horizon, sky gaps through the forest canopy |

---

## 🧪 Testing All Poses

To inspect all 19 authentic tiger poses against a fixed terrain background:

```powershell
python test_all_poses.py
```

This generates `test_all_poses_renders/` containing individual textured renders, a side-by-side HTML gallery (`poses_gallery.html`), and a summary CSV (`poses_summary.csv`).
