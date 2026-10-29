"""
Merge visible-only basecolor atlases into one FULL basecolor image per tiger identity.
=====================================================================================
Every pair in a dataset has basecolors/000001.png (only the skin visible in that image is filled,
the rest is white) and basecolor_masks/000001.png (which texels are filled). Pairs that show the same
tiger (same tiger_id in dataset_joints.sqlite) are combined here, so the more angles are collected the
more of the tiger's skin / stripe pattern the full image contains.

Run any time, as often as you like (it always rebuilds from all pairs):

    python merge_basecolors.py --dataset .\\dataset_segmentation_pairs

Needs: Python 3.9+, numpy, pillow  (pip install numpy pillow). No Blender required.

Writes <dataset>/full_basecolors/:
    tiger_00.png                  merged full basecolor (never-seen texels stay white)
    tiger_00_seen_mask.png        white = seen in at least one image
    tiger_00_observations.png     how many images saw each texel (brighter = more)
    tiger_00_missing.png          merged image with still-unseen SKIN texels painted red
    merge_report.json             per tiger: images, coverage, conflicts, coverage-vs-images curve
and the table full_basecolors in dataset_joints.sqlite.

Coverage is measured over texels that belong to the tiger mesh (the white padding between UV islands
is not counted), using the UV face map the collector caches in <dataset>/cache/.
"""
import argparse
import glob
import json
import os
import sqlite3
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(HERE, "dataset_segmentation_pairs")


def load_skin_texels(dataset):
    """Bool (H, W) of texels covered by the mesh, from the collector's UV face map. Row 0 = image top."""
    files = sorted(glob.glob(os.path.join(dataset, "cache", "uv_face_map_*.npy")), key=os.path.getmtime)
    if not files:
        return None
    uv_map = np.load(files[-1])                      # Blender order: row 0 = bottom
    return (uv_map >= 0)[::-1]


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def save_png(path, arr):
    Image.fromarray(arr).save(path)


def merge_identity(dataset, tiger_id, rows, skin_texels, tol):
    first = read_rgb(os.path.join(dataset, rows[0][1]))
    h, w = first.shape[:2]
    merged = np.full((h, w, 3), 255, dtype=np.uint8)
    seen = np.zeros((h, w), dtype=bool)
    obs = np.zeros((h, w), dtype=np.int32)
    conflicts = 0
    curve = []
    skin_total = int(skin_texels.sum()) if skin_texels is not None else None

    for image_id, bc_file, mask_file in rows:
        bc = read_rgb(os.path.join(dataset, bc_file))
        vm = read_mask(os.path.join(dataset, mask_file))
        overlap = vm & seen
        if overlap.any():
            diff = np.abs(bc[overlap].astype(np.int16) - merged[overlap].astype(np.int16)).max(axis=1)
            conflicts += int((diff > tol).sum())
        new = vm & ~seen
        merged[new] = bc[new]
        seen |= vm
        obs += vm
        covered = int((seen & skin_texels).sum()) if skin_texels is not None else int(seen.sum())
        curve.append((image_id, covered / skin_total if skin_total else covered))
    return merged, seen, obs, conflicts, curve


def main():
    ap = argparse.ArgumentParser(description="Merge visible-only basecolor atlases into a full basecolor per tiger.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset folder written by run_segmentation_collector.py")
    ap.add_argument("--out", default=None, help="Output folder (default: <dataset>/full_basecolors)")
    ap.add_argument("--conflict_tol", type=int, default=2, help="Max per-channel difference (0-255) before two views count as conflicting")
    args = ap.parse_args()

    dataset = os.path.abspath(args.dataset)
    db_path = os.path.join(dataset, "dataset_joints.sqlite")
    if not os.path.exists(db_path):
        sys.exit(f"No dataset_joints.sqlite in {dataset} - run the collector first (basecolor extraction must be on).")
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(dataset, "full_basecolors")
    os.makedirs(out_dir, exist_ok=True)

    db = sqlite3.connect(db_path)
    cols = [r[1] for r in db.execute("PRAGMA table_info(pairs)")]
    id_col = "tiger_id" if "tiger_id" in cols else "0"
    rows = db.execute(
        f"SELECT image_id, basecolor_file, basecolor_mask_file, COALESCE({id_col}, 0) FROM pairs ORDER BY image_id"
    ).fetchall()
    if not rows:
        sys.exit("The database has no pairs yet.")

    skin_texels = load_skin_texels(dataset)
    if skin_texels is None:
        print("Warning: no cache/uv_face_map_*.npy found; coverage will be relative to the whole atlas.")

    by_tiger = {}
    for image_id, bc, mk, tid in rows:
        by_tiger.setdefault(int(tid), []).append((image_id, bc, mk))

    db.execute("""CREATE TABLE IF NOT EXISTS full_basecolors (
        tiger_id INTEGER PRIMARY KEY, n_images INTEGER, seen_texels INTEGER, skin_texels INTEGER,
        coverage REAL, conflicts INTEGER, gt_mismatch_texels INTEGER, file TEXT)""")

    report = {}
    print(f"\n{'tiger':>5} {'images':>7} {'coverage':>9} {'conflicts':>10} {'vs identities/':>16}")
    for tid in sorted(by_tiger):
        merged, seen, obs, conflicts, curve = merge_identity(dataset, tid, by_tiger[tid], skin_texels, args.conflict_tol)

        name = f"tiger_{tid:02d}"
        save_png(os.path.join(out_dir, f"{name}.png"), merged)
        save_png(os.path.join(out_dir, f"{name}_seen_mask.png"), (seen * 255).astype(np.uint8))
        scale = 255.0 / max(1, int(obs.max()))
        save_png(os.path.join(out_dir, f"{name}_observations.png"), np.clip(obs * scale, 0, 255).astype(np.uint8))
        missing = merged.copy()
        if skin_texels is not None:
            missing[skin_texels & ~seen] = (255, 0, 0)
        save_png(os.path.join(out_dir, f"{name}_missing.png"), missing)

        skin_total = int(skin_texels.sum()) if skin_texels is not None else int(seen.size)
        seen_in_skin = int((seen & skin_texels).sum()) if skin_texels is not None else int(seen.sum())
        coverage = seen_in_skin / skin_total

        # Ground-truth check: merged texels must equal the identity's real skin
        gt_mismatch = None
        gt_path = os.path.join(dataset, "identities", f"{name}.png")
        if os.path.exists(gt_path):
            gt = read_rgb(gt_path)
            diff = np.abs(merged[seen].astype(np.int16) - gt[seen].astype(np.int16)).max(axis=1)
            gt_mismatch = int((diff > args.conflict_tol).sum())

        report[name] = dict(
            tiger_id=tid, images=len(by_tiger[tid]), coverage=round(coverage, 4),
            seen_texels=seen_in_skin, skin_texels=skin_total, conflicts=conflicts,
            gt_mismatch_texels=gt_mismatch,
            coverage_by_image=[[int(i), round(float(c), 4)] for i, c in curve],
        )
        db.execute("INSERT OR REPLACE INTO full_basecolors VALUES (?,?,?,?,?,?,?,?)",
                   (tid, len(by_tiger[tid]), seen_in_skin, skin_total, coverage, conflicts, gt_mismatch,
                    f"full_basecolors/{name}.png"))
        gt_txt = "n/a" if gt_mismatch is None else f"{gt_mismatch} bad texels"
        print(f"{tid:>5} {len(by_tiger[tid]):>7} {coverage * 100:>8.1f}% {conflicts:>10} {gt_txt:>16}")

    db.commit()
    db.close()
    with open(os.path.join(out_dir, "merge_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full basecolors to: {out_dir}")


if __name__ == "__main__":
    main()
