# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""Offscreen render self-test for the M1 path tracer.

Renders a curved triangle sheet (a stand-in for the cloth) over the analytic
sphere and ground with :class:`shaderbang.pathtracer.renderer.PathTracer`,
accumulates a few samples, tone-maps, and writes ``pathtracer_offscreen.png``
(or ``.ppm`` if Pillow is absent). It exercises the whole render path -- indexed
GAS build *and* in-place refit, primary-ray trace, analytic sphere/ground,
accumulation, HDR denoise, ACES tone-map, device->host readback -- with **no GL
context**, so it is the on-target validation step for the renderer before it is
wired into cloth.py's live loop.

Run:
    python -m shaderbang.pathtracer.offscreen

Exit status is 0 iff the render completes and looks sane (finite, and the scene
actually covers the frame rather than rendering pure background).
"""

import os
import sys
import traceback


def _make_sheet(n, extent, height, sag):
    """An (n+1)x(n+1) vertex grid in XZ at ``height``, dished down by ``sag``
    toward the middle so it catches directional light. Returns (verts, indices).
    """
    import numpy as np
    xs = np.linspace(-extent, extent, n + 1, dtype=np.float32)
    zs = np.linspace(-extent, extent, n + 1, dtype=np.float32)
    gx, gz = np.meshgrid(xs, zs, indexing="xy")
    r = np.sqrt(gx * gx + gz * gz)
    gy = height - sag * np.cos(np.clip(r / extent, 0.0, 1.0) * (np.pi * 0.5))
    verts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

    idx = []
    stride = n + 1
    for j in range(n):
        for i in range(n):
            a = j * stride + i
            b = a + 1
            c = a + stride
            d = c + 1
            idx.append((a, c, b))
            idx.append((b, c, d))
    indices = np.array(idx, dtype=np.int32)
    return verts, indices


def _save_image(img, path_base):
    """img: (H, W, 4) uint8, row 0 == bottom. Save flipped so it looks upright.
    Returns the written path."""
    import numpy as np
    upright = np.flipud(img)
    try:
        from PIL import Image
        p = path_base + ".png"
        Image.fromarray(upright, "RGBA").save(p)
        return p
    except Exception:  # noqa: BLE001 -- no Pillow: fall back to a PPM (P6).
        h, w = upright.shape[:2]
        p = path_base + ".ppm"
        rgb = np.ascontiguousarray(upright[..., :3])
        with open(p, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(rgb.tobytes())
        return p


def run():
    print("=" * 70)
    print("shaderbang path-tracer M1 offscreen render self-test")
    print("=" * 70)
    width, height = 512, 512
    samples = 32

    try:
        import numpy as np
        import warp as wp
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] import numpy/warp -- {e}")
        return 1

    try:
        import cupy as cp
        wp.init()
        wp.set_device("cuda:0")
        cp.cuda.Device(0).use()
        cp.cuda.runtime.free(0)  # ensure the primary context exists
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] GPU init (warp/cupy) -- {e}")
        traceback.print_exc()
        return 1

    try:
        from shaderbang.pathtracer.renderer import PathTracer
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] import PathTracer -- {e}")
        traceback.print_exc()
        return 1

    try:
        verts_np, idx_np = _make_sheet(n=64, extent=1.4, height=2.2, sag=0.6)
        verts = wp.array(verts_np, dtype=wp.vec3, device="cuda:0")
        indices = wp.array(idx_np, dtype=wp.int32, device="cuda:0")
        print(f"  [ OK ] sheet: {len(verts)} verts, {len(indices)} tris")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] build geometry -- {e}")
        traceback.print_exc()
        return 1

    try:
        pt = PathTracer(width, height, device="cuda:0", exposure=1.2)
        print("  [ OK ] PathTracer constructed (pipeline compiled)")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] PathTracer construction -- {e}")
        traceback.print_exc()
        return 1

    try:
        pt.set_geometry(verts, indices)
        pt.set_camera_lookat(eye=(0.0, 1.6, 5.0), target=(0.0, 1.3, 0.0),
                             up=(0.0, 1.0, 0.0), fov_y_deg=40.0)
        pt.set_sphere(center=(0.0, 1.5, 0.0), radius=0.5, albedo=(0.8, 0.25, 0.2))
        pt.set_ground(y=0.0, albedo=(0.55, 0.55, 0.6))
        pt.set_light(direction=(0.5, 1.0, 0.4), color=(1.1, 1.05, 0.95))
        pt.set_cloth_albedo(front=(0.2, 0.45, 0.85), back=(0.85, 0.6, 0.2))
        print("  [ OK ] scene configured")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] scene setup -- {e}")
        traceback.print_exc()
        return 1

    try:
        # First sample resets the accumulator; the rest refine it (AA).
        pt.render(reset=True)
        # Exercise the in-place GAS refit path (topology unchanged).
        pt.refit()
        for _ in range(samples - 1):
            pt.render(reset=False)
        img = pt.download_ldr()
        print(f"  [ OK ] rendered {samples} samples + refit -> {img.shape}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] render -- {e}")
        traceback.print_exc()
        return 1

    # Sanity: finite, and the scene covers most of the frame (not all sky).
    finite = bool(np.all(np.isfinite(img)))
    # Background sky is bright blue-ish; count clearly non-sky (darker or warm)
    # pixels as "scene". A correct render fills the lower/central frame with the
    # sheet, sphere and ground, so a large fraction is non-sky.
    rgb = img[..., :3].astype(np.float32) / 255.0
    is_scene = ~((rgb[..., 2] > 0.55) & (rgb[..., 2] >= rgb[..., 0]) &
                 (rgb[..., 1] > 0.45))
    scene_frac = float(is_scene.mean())

    ok = finite and scene_frac > 0.25
    try:
        out = _save_image(img, os.path.abspath("pathtracer_offscreen"))
        print(f"  [ OK ] wrote {out}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not save image -- {e}")

    print()
    print(f"  finite={finite}  scene_fraction={scene_frac:.3f}")
    if ok:
        print("  RESULT: PASS")
        return 0
    print("  RESULT: FAIL (see image; check camera/geometry/shading)")
    return 1


def main():
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
