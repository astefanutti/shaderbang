#!/usr/bin/env python
"""Headless test/benchmark harness for the cloth simulation (examples/cloth.py).

Runs the Warp cloth solve with NO DRM/KMS and NO OpenGL, so the physics
(XPBD + PDT self-collision + swept-CCD sphere/ground) can be exercised,
profiled and regression-checked on any Linux box with an NVIDIA GPU -- without
taking over the display.

It imports examples/cloth.py (whose main/DRM block is guarded by __main__),
builds a Cloth of a configurable size, installs a static module-level `sphere`
(step() reads it), runs the GL-free `init_headless()`, and drives frames.

Modes
  (default) bench    real per-frame path: simulate() re-captures the CUDA graph
                     every frame (exactly what pre_render() does in the app).
  --reuse-graph      capture the substep graph ONCE and replay it across frames
                     (valid while the sphere is static) -- isolates recapture cost.
  --profile          per-kernel GPU-time breakdown of a single uncaptured substep,
                     taken after --warmup frames so the cloth is in contact/folding.

Each frame prints: index, wall ms, max|pos|, and a NaN flag; stdout is flushed
per line so running under `timeout N` cleanly localizes a hang ("stuck").

Examples
  # fast iteration (small grid)
  .venv/bin/python examples/cloth_headless.py --nx 64 --ny 64 --frames 120
  # reproduce the production config
  timeout 300 .venv/bin/python examples/cloth_headless.py --nx 400 --ny 400 --y-offset 2.2 --frames 60
  # find the slow kernel
  .venv/bin/python examples/cloth_headless.py --nx 200 --ny 200 --profile --warmup 15
  # A/B the self-collision cost
  .venv/bin/python examples/cloth_headless.py --nx 200 --ny 200 --frames 60 --no-self-collision
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                      # so `import cloth` resolves
sys.path.insert(0, os.path.dirname(_HERE))     # repo root -> so `import shaderbang` resolves

import numpy as np
import warp as wp

import cloth as C  # examples/cloth.py -- imports cleanly headless (main is __main__-guarded)


def build(nx, ny, spacing, y_offset, sphere_y, sphere_r):
    """Construct the cloth + sphere and do GL-free init. Returns the Cloth."""
    sphere = C.Sphere(center=wp.vec3(0.0, sphere_y, 0.0), radius=sphere_r)
    C.sphere = sphere  # step()/collider_project read the module global

    t0 = time.perf_counter()
    cloth = C.Cloth(y_offset=y_offset, num_x=nx, num_y=ny, spacing=spacing)
    cloth.init_headless()
    wp.synchronize()
    print(f"[build] {time.perf_counter() - t0:.2f}s  "
          f"particles={cloth.numParticles} tris={cloth.numTris} edges={cloth.numEdges}",
          flush=True)
    return cloth


def stats(cloth):
    """(max_abs_coord, has_nan) from the host position buffer (updated by simulate)."""
    hp = cloth.hostPos.numpy()
    return float(np.nanmax(np.abs(hp))), bool(np.isnan(hp).any())


def penetration(cloth):
    """Objective collider-correctness metrics against the END-of-frame sphere pose.

    Returns (sphere_pen, n_in_sphere, edge_pen, ground_pen). sphere_pen is the deepest
    a PARTICLE sits below the (offset) sphere surface; edge_pen the deepest any cloth
    EDGE segment cuts into it -- the fabric-surface metric. A vertex-only collider can
    show sphere_pen=0 while the sphere passes between particles through stretched
    edges (edge_pen large), so both must be ~0 for a correct collider.
    """
    hp = cloth.hostPos.numpy()
    s = C.sphere
    c = np.array([s.center[0] + s.dc[0], s.center[1] + s.dc[1], s.center[2] + s.dc[2]])
    r = s.radius + s.dr + C.thickness
    d = np.linalg.norm(hp - c, axis=1)
    inside = d < r
    sphere_pen = float((r - d[inside]).max()) if inside.any() else 0.0
    E = cloth.edgeIds.numpy()
    pa, pb = hp[E[:, 0]], hp[E[:, 1]]
    ab = pb - pa
    t = np.clip(np.einsum('ij,ij->i', c - pa, ab)
                / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12), 0.0, 1.0)
    de = np.linalg.norm(pa + t[:, None] * ab - c, axis=1)
    ein = de < r
    edge_pen = float((r - de[ein]).max()) if ein.any() else 0.0
    ground_pen = float(np.maximum(C.thickness - hp[:, 1], 0.0).max())
    return sphere_pen, int(inside.sum()), edge_pen, ground_pen


_min_dist = None
_n_viol = None


def self_contact_stats(cloth, thresh_frac=0.5):
    """(min_gap, n_violations): smallest non-neighbor vertex-triangle gap and count
    below thresh_frac*d_offset. Correct self-collision -> min_gap ~ d_offset, n ~ 0."""
    global _min_dist, _n_viol
    if _min_dist is None:
        _min_dist = wp.zeros(1, dtype=float)
        _n_viol = wp.zeros(1, dtype=wp.int32)
    _min_dist.fill_(1.0e30)
    _n_viol.zero_()
    # The sim no longer keeps a wp.Mesh (the hash grid does collision, picking is
    # brute-force); build a throwaway BVH here for the diagnostic query only.
    mesh = wp.Mesh(cloth.pos, cloth.triIds.flatten(), bvh_constructor="lbvh")
    wp.launch(C.Cloth.count_self_contacts, dim=cloth.numParticles,
              inputs=[mesh.id, cloth.pos, cloth.gridRC,
                      thresh_frac * C.d_offset, _min_dist, _n_viol])
    wp.synchronize()
    return float(_min_dist.numpy()[0]), int(_n_viol.numpy()[0])


def run_bench(cloth, frames, self_collision, solve, abort_on_nan, slow_ms, sphere_vel=None,
              undulation_start=None, iterations=None):
    """Real path: simulate() per frame (re-captures the graph each frame).

    sphere_vel: optional (dx, dy, dz) per-frame sphere translation, applied via
    the same dc/swept-CCD path the interactive app uses (drag the sphere through
    the cloth) -- the scenario most likely to trigger the interactive "stuck".

    undulation_start: when set, from that frame on, compute per-frame settling
    motion stats |pos_t - pos_(t-1)| (the objective "undulation" metric: a
    settled drape should have ~0 frame-to-frame motion; residual elastic waves
    show up as a persistent step size). Printed per frame and summarized.
    """
    times = []
    prev_hp = None
    und_mean, und_max, und_maxpos = [], [], []
    for f in range(frames):
        if sphere_vel is not None:
            C.sphere.dc = wp.vec3(*sphere_vel)  # collider_project swept CCD reads this
        wp.synchronize()
        t0 = time.perf_counter()
        cloth.simulate(steps=C.numSubsteps, self_collision=self_collision,
                       solve_constraints=solve,
                       iterations=iterations if iterations is not None else C.numIterations)
        wp.synchronize()
        if sphere_vel is not None:  # commit the frame delta, mirroring Sphere.post_render
            C.sphere.center += C.sphere.dc
            C.sphere.dc = wp.vec3()
        dt_ms = (time.perf_counter() - t0) * 1e3
        times.append(dt_ms)
        mx, nan = stats(cloth)
        spen, nin, epen, gpen = penetration(cloth)
        und = ""
        if undulation_start is not None and f >= undulation_start:
            hp = cloth.hostPos.numpy().copy()
            if prev_hp is not None:
                d = np.linalg.norm(hp - prev_hp, axis=1)
                und_mean.append(float(d.mean()))
                und_max.append(float(d.max()))
                und_maxpos.append(mx)
                und = f"  step_mean={d.mean():.3e} step_max={d.max():.3e}"
            prev_hp = hp
        flag = "  <<< NaN" if nan else ("  <<< SLOW" if dt_ms > slow_ms else "")
        print(f"frame {f:4d}  {dt_ms:8.1f} ms  max|pos|={mx:8.3f}  "
              f"sphere_pen={spen:.4f}(n={nin})  edge_pen={epen:.4f}  "
              f"ground_pen={gpen:.4f}{und}{flag}", flush=True)
        if nan and abort_on_nan:
            print("[abort] NaN detected", flush=True)
            break
    summarize(times)
    if und_mean:
        a_mean = np.array(und_mean)
        a_max = np.array(und_max)
        a_mp = np.array(und_maxpos)
        print(f"[undulation] window={undulation_start + 1}..{frames - 1} n={len(a_mean)}  "
              f"step_mean(avg)={a_mean.mean():.3e}  step_mean(last)={a_mean[-1]:.3e}  "
              f"step_max(avg)={a_max.mean():.3e}  step_max(last)={a_max[-1]:.3e}  "
              f"std_max|pos|={a_mp.std():.3e}", flush=True)
    if self_collision:
        mg, nv = self_contact_stats(cloth)
        gap = "none-in-range" if mg > 1.0 else f"{mg:.5f}"
        ovf = int(cloth.selfCollisionOverflow.numpy()[0])  # candidate-buffer drops last frame
        print(f"[self-contact] min_gap={gap} (d_offset={C.d_offset:.5f})  "
              f"violations(<0.5*d_offset)={nv}  buffer_overflow={ovf}", flush=True)


def run_reuse_graph(cloth, frames, self_collision, solve, abort_on_nan, slow_ms):
    """Capture the substep graph once, replay it numSubsteps x per frame."""
    dt = C.timeStep / C.numSubsteps
    wp.copy(cloth.pos, cloth.hostPos)
    wp.copy(cloth.invMass, cloth.hostInvMass)
    with wp.ScopedCapture() as cap:
        cloth.step(dt, self_collision=self_collision, solve_constraints=solve)
    graph = cap.graph
    wp.synchronize()
    print("[reuse-graph] captured one substep; replaying across frames", flush=True)

    times = []
    for f in range(frames):
        wp.synchronize()
        t0 = time.perf_counter()
        wp.copy(cloth.pos, cloth.hostPos)
        wp.copy(cloth.invMass, cloth.hostInvMass)
        for _ in range(C.numSubsteps):
            wp.capture_launch(graph)
        wp.copy(cloth.hostPos, cloth.pos)
        wp.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3
        times.append(dt_ms)
        mx, nan = stats(cloth)
        flag = "  <<< NaN" if nan else ("  <<< SLOW" if dt_ms > slow_ms else "")
        print(f"frame {f:4d}  {dt_ms:8.1f} ms  max|pos|={mx:10.3f}{flag}", flush=True)
        if nan and abort_on_nan:
            print("[abort] NaN detected", flush=True)
            break
    summarize(times)


def run_profile(cloth, warmup, self_collision, solve):
    """Warm up to a contact state, then per-kernel GPU-time breakdown of 1 substep."""
    dt = C.timeStep / C.numSubsteps
    print(f"[profile] warming up {warmup} frames to reach contact...", flush=True)
    for f in range(warmup):
        cloth.simulate(steps=C.numSubsteps, self_collision=self_collision,
                       solve_constraints=solve)
    wp.synchronize()
    mx, nan = stats(cloth)
    print(f"[profile] warm state: max|pos|={mx:.3f} nan={nan}", flush=True)

    # Load the current host state onto the device (mirrors simulate() setup) so the
    # profiled substep runs on the warmed geometry rather than the flat rest state.
    wp.copy(cloth.pos, cloth.hostPos)
    wp.copy(cloth.invMass, cloth.hostInvMass)
    wp.synchronize()

    print("[profile] per-kernel GPU time for ONE uncaptured substep:", flush=True)
    with wp.ScopedTimer("substep", cuda_filter=wp.TIMING_KERNEL, synchronize=True):
        cloth.step(dt, self_collision=self_collision, solve_constraints=solve)
    wp.synchronize()


def summarize(times):
    if not times:
        return
    a = np.array(times)
    print(f"[summary] frames={len(a)}  mean={a.mean():.1f}ms  "
          f"median={np.median(a):.1f}ms  min={a.min():.1f}ms  max={a.max():.1f}ms  "
          f"~fps={1000.0 / a.mean():.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--spacing", type=float, default=0.015)
    ap.add_argument("--y-offset", type=float, default=2.2, help="cloth start height")
    ap.add_argument("--sphere-y", type=float, default=1.5)
    ap.add_argument("--sphere-r", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--no-self-collision", dest="self_collision", action="store_false")
    ap.add_argument("--no-solve", dest="solve", action="store_false",
                    help="disable XPBD constraint solve (isolates collision cost)")
    ap.add_argument("--reuse-graph", action="store_true",
                    help="capture the substep graph once and replay (static sphere)")
    ap.add_argument("--profile", action="store_true",
                    help="per-kernel GPU breakdown of one substep after --warmup frames")
    ap.add_argument("--warmup", type=int, default=15, help="profile warmup frames")
    ap.add_argument("--abort-on-nan", action="store_true")
    ap.add_argument("--slow-ms", type=float, default=100.0, help="flag frames slower than this")
    ap.add_argument("--sphere-dx", type=float, default=0.0, help="per-frame sphere translation x")
    ap.add_argument("--sphere-dy", type=float, default=0.0, help="per-frame sphere translation y (e.g. drag up through cloth)")
    ap.add_argument("--sphere-dz", type=float, default=0.0, help="per-frame sphere translation z")
    ap.add_argument("--substeps", type=int, default=None, help="override numSubsteps")
    ap.add_argument("--undulation-start", type=int, default=None,
                    help="from this frame on, report frame-to-frame settling motion "
                         "(the undulation metric) and a [undulation] summary")
    ap.add_argument("--dist-kd", type=float, default=None,
                    help="override kd on ALL distance constraints (root-cause toggle)")
    ap.add_argument("--bend-kd", type=float, default=None,
                    help="override kd on ALL bending constraints (root-cause toggle)")
    ap.add_argument("--bend-ke", type=float, default=None,
                    help="override ke on ALL bending constraints (1e-6 ~ disabled)")
    ap.add_argument("--dist-ke", type=float, default=None, nargs=3,
                    help="override ke per distance group: one_ cross two_ (1e-6 ~ disabled)")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override numIterations for the XPBD solve")
    ap.add_argument("--bend-relax", type=float, default=None,
                    help="override relaxation on bending constraints")
    ap.add_argument("--dump-pos", type=str, default=None,
                    help="save final host positions to this .npy path")
    args = ap.parse_args()

    if args.substeps is not None:
        C.numSubsteps = args.substeps

    sphere_vel = None
    if args.sphere_dx or args.sphere_dy or args.sphere_dz:
        sphere_vel = (args.sphere_dx, args.sphere_dy, args.sphere_dz)

    print(f"[config] nx={args.nx} ny={args.ny} spacing={args.spacing} "
          f"y_offset={args.y_offset} sphere=(y={args.sphere_y},r={args.sphere_r}) "
          f"self_collision={args.self_collision} solve={args.solve} "
          f"substeps={C.numSubsteps} iters={C.numIterations}", flush=True)

    cloth = build(args.nx, args.ny, args.spacing, args.y_offset, args.sphere_y, args.sphere_r)

    # kd is a runtime kernel argument (not a baked constant), so overriding the
    # Constraint objects here takes effect on the next graph capture -- no recompile.
    if args.dist_kd is not None:
        for c in cloth.distConstraints.constraints:
            c.kd = args.dist_kd
        print(f"[override] distance kd={args.dist_kd}", flush=True)
    if args.bend_kd is not None:
        for c in cloth.bendConstraints.constraints:
            c.kd = args.bend_kd
        print(f"[override] bending kd={args.bend_kd}", flush=True)
    if args.bend_ke is not None:
        for c in cloth.bendConstraints.constraints:
            c.ke = args.bend_ke
        print(f"[override] bending ke={args.bend_ke}", flush=True)
    if args.dist_ke is not None:
        for c, ke in zip(cloth.distConstraints.constraints, args.dist_ke):
            c.ke = ke
        print(f"[override] distance ke={args.dist_ke}", flush=True)
    if args.iterations is not None:
        C.numIterations = args.iterations
        print(f"[override] iterations={args.iterations}", flush=True)
    if args.bend_relax is not None:
        for c in cloth.bendConstraints.constraints:
            c.relaxation = args.bend_relax
        print(f"[override] bending relaxation={args.bend_relax}", flush=True)

    if args.profile:
        run_profile(cloth, args.warmup, args.self_collision, args.solve)
    elif args.reuse_graph:
        run_reuse_graph(cloth, args.frames, args.self_collision, args.solve,
                        args.abort_on_nan, args.slow_ms)
    else:
        run_bench(cloth, args.frames, args.self_collision, args.solve,
                  args.abort_on_nan, args.slow_ms, sphere_vel,
                  undulation_start=args.undulation_start, iterations=args.iterations)
        if args.dump_pos:
            wp.synchronize()
            np.save(args.dump_pos, cloth.hostPos.numpy())
            print(f"[dump] positions -> {args.dump_pos}", flush=True)


if __name__ == "__main__":
    main()
