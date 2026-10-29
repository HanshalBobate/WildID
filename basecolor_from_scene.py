"""
Basecolor for an existing render (a frame of a .blend scene)
============================================================
Makes the visible-only basecolor atlas for an image that was rendered from a saved scene, e.g. a
frame of the camera-arc animation in target.blend. The scene is NOT randomised or modified: it is
opened as saved, the timeline is set to --frame, and the same extraction as the dataset collector runs.

    python basecolor_from_scene.py --image .\\1.png --frame 33
    python basecolor_from_scene.py --image .\\1.png --frame 33 --blend .\\target.blend --out_dir .\\from_image

Outputs in --out_dir (same layout as the collector):
    images/000001.png            copy of your image
    basecolors/000001.png        skin atlas, only faces visible in the image, rest white
    basecolor_masks/000001.png   white = filled texel
    dataset_joints.sqlite        joints + pose vectors for this image
    _check_render.png            our own render of the scene at that frame (for comparison)

The printed match score compares _check_render.png with your image. A near-zero difference means the
scene state + frame reproduce your image, so the basecolor is exact. A large difference means the saved
scene is not the one that produced your image (different pose / tiger position / camera): the basecolor
would then describe the wrong view, so do not trust it.
"""
import argparse
import os
import shutil
import sys

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

BLENDER_CANDIDATE_PATHS = [
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
]


def find_blender_binary():
    import shutil as _sh
    found = _sh.which("blender")
    if found:
        return found
    for p in BLENDER_CANDIDATE_PATHS + ["/usr/bin/blender", "/usr/local/bin/blender",
                                        "/Applications/Blender.app/Contents/MacOS/Blender"]:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Could not find blender executable! Install Blender or add it to PATH.")


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Visible-only basecolor for an existing render of a saved scene")
    ap.add_argument("--image", required=True, help="The rendered image (PNG) to make a basecolor for")
    ap.add_argument("--frame", type=int, default=None, help="Timeline frame the image was rendered at (see the PNG's 'Frame' metadata)")
    ap.add_argument("--blend", default=os.path.join(BASE_DIR, "target.blend"), help="Scene to open (default: target.blend)")
    ap.add_argument("--out_dir", default=os.path.join(BASE_DIR, "from_image"), help="Output folder")
    ap.add_argument("--image_id", type=int, default=1, help="Id used for file names (default 1 -> 000001.png)")
    ap.add_argument("--min_face_pixels", type=int, default=1)
    ap.add_argument("--no_check_render", action="store_true", help="Skip the comparison render")
    return ap.parse_args(argv)


if not IN_BLENDER:
    import subprocess

    def launch():
        args = parse_args(sys.argv[1:])
        forwarded = sys.argv[1:]
        # make --image/--blend/--out_dir absolute so Blender's cwd cannot change their meaning
        for flag in ("--image", "--blend", "--out_dir"):
            if flag in forwarded:
                i = forwarded.index(flag) + 1
                forwarded[i] = os.path.abspath(forwarded[i])
        cmd = [find_blender_binary(), "-b", os.path.abspath(args.blend), "--disable-autoexec",
               "--python", os.path.abspath(__file__), "--"] + forwarded
        print("[Launcher]", " ".join(cmd))
        sys.exit(subprocess.run(cmd).returncode)

    if __name__ == "__main__":
        launch()
    sys.exit(0)

# ==============================================================================
# Inside Blender
# ==============================================================================
import numpy as np

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
import basecolor_extractor


def read_rgb(path):
    img = bpy.data.images.load(path)
    try:
        img.colorspace_settings.name = 'Non-Color'
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
    finally:
        bpy.data.images.remove(img)
    return (buf.reshape(h, w, 4)[::-1, :, :3] * 255.0)


def main():
    # Under blender.exe the args follow "--"; under the pip 'bpy' module they are plain sys.argv.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)

    # With the pip 'bpy' module nothing is loaded yet, so open the scene ourselves.
    want = os.path.normcase(os.path.abspath(args.blend))
    have = os.path.normcase(os.path.abspath(bpy.data.filepath)) if bpy.data.filepath else ""
    if want != have:
        print(f"[Scene] Opening {args.blend}")
        bpy.ops.wm.open_mainfile(filepath=args.blend)
    scene = bpy.context.scene

    # same broken-path repair the collector uses (missing HDR / textures), so lighting matches
    import run_segmentation_collector as collector
    collector.remap_missing_assets()

    probe = bpy.data.images.load(args.image)
    res_x, res_y = probe.size
    bpy.data.images.remove(probe)
    scene.render.resolution_x, scene.render.resolution_y = res_x, res_y
    scene.render.resolution_percentage = 100

    if args.frame is not None:
        scene.frame_set(args.frame)
    bpy.context.view_layer.update()

    arm = bpy.data.objects.get("White Tiger Skeleton")
    mesh = bpy.data.objects.get("White Tiger Body")
    if not arm or not mesh:
        raise RuntimeError("Scene has no 'White Tiger Skeleton' / 'White Tiger Body'.")
    action = arm.animation_data.action.name if arm.animation_data and arm.animation_data.action else None
    print(f"[Scene] frame={scene.frame_current} camera={scene.camera.name if scene.camera else None} "
          f"tiger action={action} loc={tuple(round(v, 2) for v in arm.location)} "
          f"rot={tuple(round(v, 3) for v in arm.rotation_euler)} res={res_x}x{res_y}")

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    id_str = f"{args.image_id:06d}"
    shutil.copy(args.image, os.path.join(args.out_dir, "images", f"{id_str}.png"))

    extractor = basecolor_extractor.BasecolorExtractor(
        scene, arm, mesh, args.out_dir, min_face_pixels=args.min_face_pixels, n_identities=1)
    try:
        info = extractor.process_pair(args.image_id, id_str)
    finally:
        extractor.close()
    print(f"[Basecolor] {info['visible_faces']}/{info['total_faces']} faces visible "
          f"({info['coverage'] * 100:.1f}% of skin atlas) | tiger px={info['tiger_pixels']} | "
          f"joints on tiger={info['joints_on_tiger']}/{info['joints_in_frame']} | invalid-id px={info['invalid_id_pixels']}")

    if not args.no_check_render:
        check_path = os.path.join(args.out_dir, "_check_render.png")
        scene.render.filepath = check_path
        scene.render.image_settings.file_format = 'PNG'
        scene.eevee.taa_render_samples = min(scene.eevee.taa_render_samples, 32)
        bpy.ops.render.render(write_still=True)
        ours, theirs = read_rgb(check_path), read_rgb(args.image)
        if ours.shape != theirs.shape:
            print("[Match] size mismatch, cannot compare")
        else:
            diff = np.abs(ours - theirs).mean(axis=2)
            print(f"[Match] mean abs pixel difference vs your image (0-255): {diff.mean():.2f}  "
                  f"| pixels differing by >40: {(diff > 40).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
