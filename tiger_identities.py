"""
Tiger identities: distinct stripe-pattern variants of the one white-tiger skin atlas.
=====================================================================================
Real individual tigers are told apart by their stripes. The project ships a single skin, so
this module derives extra, reproducible "individuals" from it (pure numpy, no scipy/cv2, so it
runs inside Blender's bundled Python):

  tiger_id 0      the original skin, untouched
  tiger_id k >= 1 original skin with its markings changed by
      1. smooth warp   - stripes are displaced by a low + mid frequency noise field, so their
                         positions and shapes change. The displacement fades to zero towards
                         every UV island border, so seams between islands stay consistent.
      2. width change  - stripes are locally thickened / thinned (and can break or merge) by
                         shifting the stripe threshold with fine noise.
  Eyes, nose, lips, tongue and jaw are never touched; the rest of the head only gets a fraction
  of the change so faces do not distort.

Left and right flanks are warped independently, so each identity is also left/right asymmetric.
Variants are fully determined by (tiger_id, BASE_SEED): the same id always gives the same tiger.
All outputs are quantised to 8-bit so the in-memory arrays equal the saved PNG exactly.
"""
import os

import numpy as np

BASE_SEED = 20240611
BORDER_FADE_TEXELS = 24      # displacement ramps from 0 at the island border to full this far inside
BORDER_ZERO_TEXELS = 3       # nothing moves within this many texels of a border
WARP_LOW = (300, 26.0)       # (wavelength texels, amplitude texels)
WARP_MID = (110, 9.0)
WIDTH_AMP = 0.22             # luminance shift used to thicken / thin stripes
STRIPE_E0, STRIPE_E1 = 0.55, 0.28   # luminance 0.55 -> not stripe, 0.28 -> full stripe


def quantize(rgb):
    return (np.rint(np.clip(rgb, 0.0, 1.0) * 255.0) / 255.0).astype(np.float32)


def _box_blur_axis(a, r, axis):
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode='edge'), axis=axis, dtype=np.float64)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def box_blur(a, r):
    return _box_blur_axis(_box_blur_axis(a.astype(np.float32), r, 0), r, 1)


def distance_to_outside(covered):
    """City-block distance (texels) from each covered texel to the nearest uncovered texel."""
    h, w = covered.shape
    big = np.float32(h + w)
    d = np.where(covered, big, np.float32(0.0)).astype(np.float32)
    for y in range(1, h):                                   # vertical, downwards
        np.minimum(d[y], d[y - 1] + 1.0, out=d[y])
    for y in range(h - 2, -1, -1):                          # vertical, upwards
        np.minimum(d[y], d[y + 1] + 1.0, out=d[y])
    x = np.arange(w, dtype=np.float32)[None, :]
    d = np.minimum(np.minimum.accumulate(d - x, axis=1) + x,                       # left -> right
                   (np.minimum.accumulate((d + x)[:, ::-1], axis=1)[:, ::-1]) - x)  # right -> left
    return d


def _smooth_noise(h, w, wavelength, rng):
    """Smooth value noise in [-1, 1] with the given feature size (texels)."""
    gh, gw = h // wavelength + 3, w // wavelength + 3
    g = rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
    ys = np.arange(h, dtype=np.float32) / wavelength
    xs = np.arange(w, dtype=np.float32) / wavelength
    y0, x0 = ys.astype(np.int64), xs.astype(np.int64)
    fy, fx = ys - y0, xs - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    fy, fx = fy[:, None], fx[None, :]
    y0, x0 = y0[:, None], x0[None, :]
    top = g[y0, x0] * (1 - fx) + g[y0, x0 + 1] * fx
    bot = g[y0 + 1, x0] * (1 - fx) + g[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def _remap_bilinear(img, map_x, map_y):
    h, w = img.shape[:2]
    x = np.clip(map_x, 0, w - 1.001)
    y = np.clip(map_y, 0, h - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def make_variant(skin, inner, weight, seed):
    """
    skin:   (H, W, 3) float32 in 0..1 (raw texel values)
    inner:  (H, W) 0..1, 0 at island borders and outside, 1 deep inside an island
    weight: (H, W) 0..1 how much this texel may change (0 = protected, 1 = free)
    """
    h, w = skin.shape[:2]
    rng = np.random.default_rng(seed)
    strength = rng.uniform(0.85, 1.15)
    field = inner * weight

    # 1. Smooth warp (backward mapping, so no holes)
    dx = WARP_LOW[1] * _smooth_noise(h, w, WARP_LOW[0], rng) + WARP_MID[1] * _smooth_noise(h, w, WARP_MID[0], rng)
    dy = WARP_LOW[1] * _smooth_noise(h, w, WARP_LOW[0], rng) + WARP_MID[1] * _smooth_noise(h, w, WARP_MID[0], rng)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = _remap_bilinear(skin, gx + dx * field * strength, gy + dy * field * strength)

    # 2. Stripe width / topology change on the stripe layer
    lum = warped @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    s = _smoothstep(STRIPE_E0, STRIPE_E1, lum)
    noise = 0.55 * _smooth_noise(h, w, 60, rng) + 0.45 * _smooth_noise(h, w, 24, rng)
    s2 = _smoothstep(STRIPE_E0, STRIPE_E1, lum + WIDTH_AMP * strength * noise * field)

    dark_sel = s > 0.9
    fur_sel = s < 0.1
    dark_col = warped[dark_sel].mean(axis=0) if dark_sel.any() else np.array([0.05, 0.05, 0.05], np.float32)
    fur_col = np.median(warped[fur_sel], axis=0) if fur_sel.any() else np.array([0.85, 0.85, 0.85], np.float32)
    num = box_blur(warped * (1.0 - s)[..., None], 8)
    den = box_blur(1.0 - s, 8)[..., None]
    fur_est = np.where(den > 0.08, num / np.maximum(den, 1e-6), fur_col)
    out = warped + (s2 - s)[..., None] * (dark_col - fur_est)
    return quantize(out)


def prepare_masks(uv_face_map, face_weight):
    """inner (border falloff) and weight (protection) maps from the UV face map and per-face weights."""
    covered = uv_face_map >= 0
    dist = distance_to_outside(covered)
    inner = np.clip((dist - BORDER_ZERO_TEXELS) / BORDER_FADE_TEXELS, 0.0, 1.0)
    inner[~covered] = 0.0
    lut = np.concatenate([[0.0], np.asarray(face_weight, dtype=np.float32)])
    weight = box_blur(lut[uv_face_map + 1], 6)             # soften protection edges
    return inner, weight


def build_variants(skin, uv_face_map, face_weight, tiger_ids, out_dir=None, save_png=None):
    """
    Returns {tiger_id: (H, W, 3) float32 array}. tiger_id 0 is the original skin.
    save_png(path, rgb) is called for every identity when out_dir is given.
    """
    skin = quantize(skin)
    result = {0: skin}
    inner = weight = None
    for k in tiger_ids:
        if k == 0:
            continue
        if inner is None:
            inner, weight = prepare_masks(uv_face_map, face_weight)
        result[k] = make_variant(skin, inner, weight, BASE_SEED + 7919 * k)
    if out_dir and save_png:
        os.makedirs(out_dir, exist_ok=True)
        for k, arr in result.items():
            save_png(os.path.join(out_dir, f"tiger_{k:02d}.png"), arr)
    return result
