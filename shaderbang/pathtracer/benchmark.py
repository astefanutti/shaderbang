# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""On-target render benchmark for the path tracer (milestone M2).

Measures the per-frame cost of the render pipeline -- GAS refit, primary-ray
trace, HDR denoise, ACES tone-map -- at a set of resolutions on the target GPU,
and reports frames/s and primary rays/s. No GL context is required, so it runs
headless over SSH on the RTX 5090; it is the tool that turns the design-time
spp/bounce *hypotheses* into measured numbers.

The scene is a deforming triangle sheet sized to match ``cloth.py``'s default
(``num_x = num_y = 400`` -> 320k triangles), traced against the analytic sphere
and ground, so the GAS build/refit and trace costs are representative of the live
example rather than a toy.

Timing uses wall-clock around an explicit stream sync. Each phase is measured in
its own loop with a *single* sync after ``iters`` launches, so the number is the
pipelined (steady-state) per-op cost -- i.e. the throughput the loop actually
sustains, not a sync-serialized worst case. ``trace`` is derived as
``render - denoise`` (``render()`` fuses the OptiX launch and the denoiser).

Run:
    python -m shaderbang.pathtracer.benchmark
    python -m shaderbang.pathtracer.benchmark --tris 320000 --iters 200
    python -m shaderbang.pathtracer.benchmark --res 1920x1080 --res 3840x2160
"""

import argparse
import math
import sys
import traceback
from time import perf_counter


def _make_sheet(n, extent, height, sag):
    """An (n+1)x(n+1) vertex grid in XZ at ``height``, dished by ``sag`` toward
    the middle. Returns (verts (N,3) float32, indices (2*n*n,3) int32). Same
    generator as the offscreen self-test so the two stay comparable."""
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


def _sheet_subdivisions(tris):
    """Grid subdivisions ``n`` giving ~``tris`` triangles (2*n*n)."""
    n = int(round(math.sqrt(max(tris, 2) / 2.0)))
    return max(n, 2)


def _time(fn, sync, iters, warmup):
    """Mean wall-clock seconds per call of ``fn`` over ``iters`` iterations,
    after ``warmup`` warmup calls, with a single ``sync()`` at the end."""
    for _ in range(warmup):
        fn()
    sync()
    t0 = perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (perf_counter() - t0) / iters


def _parse_res(s):
    w, _, h = s.lower().partition("x")
    return (int(w), int(h))


def _fmt_hz(v):
    if v >= 1e9:
        return f"{v / 1e9:.2f} G"
    if v >= 1e6:
        return f"{v / 1e6:.2f} M"
    if v >= 1e3:
        return f"{v / 1e3:.2f} k"
    return f"{v:.2f} "


def run(argv=None):
    parser = argparse.ArgumentParser(
        description="Path-tracer render benchmark (M2)")
    parser.add_argument("--tris", type=int, default=320_000,
                        help="approximate cloth triangle count (default: 320000, "
                             "== cloth.py's 400x400 sheet)")
    parser.add_argument("--res", action="append", default=None,
                        help="WxH resolution to benchmark (repeatable); "
                             "default: 1920x1080 and 3840x2160")
    parser.add_argument("--iters", type=int, default=200,
                        help="timed iterations per phase (default: 200)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="warmup iterations per phase (default: 20)")
    parser.add_argument("--exposure", type=float, default=1.2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    resolutions = ([_parse_res(r) for r in args.res] if args.res
                   else [(1920, 1080), (3840, 2160)])

    print("=" * 78)
    print("shaderbang path-tracer M2 render benchmark")
    print("=" * 78)

    try:
        import numpy as np
        import warp as wp
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] import numpy/warp -- {e}")
        return 1

    try:
        import cupy as cp
        wp.init()
        wp.set_device(args.device)
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

    n = _sheet_subdivisions(args.tris)
    verts_np, idx_np = _make_sheet(n=n, extent=3.0, height=2.2, sag=0.6)
    num_tris = len(idx_np)
    num_verts = len(verts_np)
    verts = wp.array(verts_np, dtype=wp.vec3, device=args.device)
    indices = wp.array(idx_np, dtype=wp.int32, device=args.device)
    print(f"  scene: {num_verts:,} verts  {num_tris:,} triangles "
          f"(sheet n={n})  + analytic sphere & ground")
    print(f"  iters={args.iters}  warmup={args.warmup}  device={args.device}")
    print()

    header = (f"  {'resolution':>11} | {'refit':>8} {'trace':>8} "
              f"{'denoise':>8} {'tonemap':>8} {'frame':>8} | "
              f"{'fps':>7} {'rays/s':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    ok = True
    for (w, h) in resolutions:
        try:
            pt = PathTracer(w, h, device=args.device, exposure=args.exposure)
            cloth_mat = pt.add_material(base_color=(0.2, 0.45, 0.85),
                                        base_color_back=(0.85, 0.6, 0.2),
                                        roughness=0.6)
            sphere_mat = pt.add_material(base_color=(0.8, 0.25, 0.2),
                                         roughness=0.3)
            ground_mat = pt.add_material(base_color=(0.55, 0.55, 0.6),
                                         roughness=0.9)
            pt.add_mesh(verts, indices, material_id=cloth_mat, deformable=True)
            pt.set_camera_lookat(eye=(0.0, 1.8, 6.0), target=(0.0, 1.3, 0.0),
                                 up=(0.0, 1.0, 0.0), fov_y_deg=40.0,
                                 aspect=w / float(h))
            pt.add_sphere(center=(0.0, 1.5, 0.0), radius=0.5,
                          material_id=sphere_mat)
            pt.add_plane(normal=(0.0, 1.0, 0.0), offset=0.0,
                         material_id=ground_mat)
            pt.add_light("directional", direction=(0.5, 1.0, 0.4),
                         color=(1.1, 1.05, 0.95))

            stream = pt._wp_stream
            sync = lambda: wp.synchronize_stream(stream)

            # Prime accumulation/denoise buffers before isolating phases.
            pt.render(reset=True)
            sync()

            t_refit = _time(lambda: pt.refit(), sync, args.iters, args.warmup)
            t_render = _time(lambda: pt.render(reset=True), sync,
                             args.iters, args.warmup)
            t_denoise = _time(lambda: pt._denoise(), sync,
                              args.iters, args.warmup)
            t_tonemap = _time(lambda: pt._tonemap_into(pt.d_ldr), sync,
                              args.iters, args.warmup)

            t_trace = max(t_render - t_denoise, 0.0)
            t_frame = t_refit + t_render + t_tonemap
            fps = 1.0 / t_frame if t_frame > 0 else float("inf")
            rays = (w * h) / t_trace if t_trace > 0 else float("inf")

            def ms(t):
                return f"{t * 1e3:7.3f}m"

            print(f"  {w:5d}x{h:<5d} | {ms(t_refit)} {ms(t_trace)} "
                  f"{ms(t_denoise)} {ms(t_tonemap)} {ms(t_frame)} | "
                  f"{fps:6.1f}  {_fmt_hz(rays):>9}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {w}x{h}: [FAIL] {e}")
            traceback.print_exc()

    print()
    print("  columns: per-frame GPU time (ms). trace = render - denoise "
          "(render fuses launch+denoise).")
    print("  frame = refit + render + tonemap (= trace + denoise + refit + "
          "tonemap); present's PBO upload/quad is a GL cost, not measured here.")
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
