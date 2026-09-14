#!/usr/bin/env python3

# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Plasma globe M1 micro-benchmarks
================================

Headless GL 4.6 + Warp measurements behind the render budget of ``docs/plasma_globe.md`` (plan section 9, M1).
Runs without DRM master through the EGL device platform (``plasma.glctx``); every number is a GL
``GL_TIME_ELAPSED`` (GPU) or CUDA-event / wall-clock median as stated, medians of ``--iters`` dispatches after
a warm-up, repeated ``--repeats`` times (median of medians) because other work shares the GPU.

    python examples/plasma_globe_bench.py --bench all --res 1920x1080 --out /tmp/plasma_bench

Benchmarks and thresholds
-------------------------
capsule   closed-form capsule emission integral, N segments per ray at the internal resolution; slope of
          time(N) = a + b N fitted ``--slope-runs`` times (median, min, max reported; the median is the
          verdict): b <= 6 us per candidate per ray per frame for the 1/(d^2+eps^2)^(3/2) kernel with every
          pixel walking the segments in the same order (the plan's protocol); also the local slope over
          N in [32, 512] (the app's operating point), a divergent variant (each pixel starts at a hashed offset,
          the DDA regime) and the atan (1/(d^2+eps^2)) kernel for comparison; GLSL result checked against the
          numpy closed form, the closed form against numeric integration, and a zero-length segment must
          produce no NaN
dda       Amanatides-Woo over a 96^3 CSR of a synthetic 13 x 150-segment electrode star, mailbox 8, hard cap
          270; dilation r in {0, 1, 2}: mean <= 40-60 and p99 <= 270 candidates per globe ray at r = 1 for the
          ``--star`` preset (default ``tortuous``, the plan's walk), plus r = 1 for the other preset as a bracket
glass     deterministic 3-branch glass split, no filaments: <= 1.5 ms
taau      stand-in temporal upscale to mode resolution: <= 1.2 ms
present   fullscreen AgX + dither present to RGBA8 at mode resolution: <= 0.3 ms (+ 1080p -> 4K blit)
interop   batched 2-resource map + unmap including the GL <-> CUDA handoff (wp.synchronize) <= 0.2 ms, the
          CPU submission time as a secondary column, Warp's nested / sequential per-resource paths and the
          forced fallback interleaved in the same loop; wp.copy 5.5 MB <= 0.02 ms; cuMemcpy3DAsync and
          Texture3D.copy_from 96^3 RGBA16F <= 0.1 ms; mapped-pointer stability over 1000 map/unmap cycles
calib     effective FP32 rate inferred from the capsule slope (the display calibration of plan M1 needs DRM)

Outputs: PNGs and ``results.json`` under ``--out``.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plasma.glctx import gl_limits, make_headless_context  # noqa: E402
from plasma import glutil  # noqa: E402
from plasma import interop  # noqa: E402
from plasma.glutil import (  # noqa: E402
    Buffer, Framebuffer, FullscreenTriangle, Image2D, Image3D, TimerQuery, camera_rays, compile_compute,
    compile_program, perspective_camera, save_heatmap_png, save_png,
)
from OpenGL.GL import GL_LINEAR, GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY, glFinish  # noqa: E402

import warp as wp  # noqa: E402

# Globe geometry (metres), plan 4.1
R1 = 0.015
R2I = 0.075
R2O = 0.0775

# Synthetic star and render grid
CHAINS = 13
SEGMENTS_PER_CHAIN = 150
NODE_SPACING = 1.5e-3
CORE_EPS = 0.5e-3
GRID_N = 96
CELL_SIZE = 2.0 * R2I / GRID_N
GRID_MIN = -R2I
ITEM_CAPACITY = 1 << 22
CAND_CAP = 270
HIST_BINS = 64
HIST_BIN_WIDTH = 5

# Synthetic star presets: (persistence, sigma) of the tangential random walk.  "tortuous" is the plan's star
# (turning angle ~24 deg per step median), "straight" brackets a smoother tree (<= 10 deg per step).  Both keep
# the plan's 13 x 150 segments of 1.5 mm between R1 and R2i, so the path/chord tortuosity (~3.7) is fixed by
# construction; only the local curvature differs.
STAR_PRESETS = {"tortuous": (0.85, 0.55), "straight": (0.98, 0.08)}

# Camera: 36 deg vertical FOV at 0.30 m -> the globe fills ~83 % of the frame height
CAMERA_EYE = (0.0, 0.02, 0.30)
CAMERA_TARGET = (0.0, 0.0, 0.0)
CAMERA_FOV = 36.0

FRAME_SSBO_BYTES = 5_500_000
PEAK_FP32_INSTR_PER_S = 52.44e12

THRESHOLDS = {
    "capsule.slope_us_k3": ("<=", 6.0),
    "dda.r1.mean_evals": ("<=", 60.0),
    "dda.r1.p99_evals": ("<=", 270.0),
    "glass.ms": ("<=", 1.5),
    "taau.ms": ("<=", 1.2),
    "present.ms": ("<=", 0.3),
    "interop.batched_map_unmap_sync_ms": ("<=", 0.2),
    "interop.wp_copy_5p5MB_ms": ("<=", 0.02),
    "interop.memcpy3d_96_ms": ("<=", 0.1),
    "interop.copy_from_96_ms": ("<=", 0.1),
}


# ----------------------------------------------------------------------------------------------------
# Timing helpers
# ----------------------------------------------------------------------------------------------------

def timed(fn, iters: int, warmup: int, repeats: int, between=glutil.barrier_image) -> dict:
    """GPU time of ``fn`` per call: median of ``iters`` GL timer samples, median over ``repeats`` runs."""
    timer = TimerQuery(depth=iters)
    for _ in range(warmup):
        fn()
        between()
    medians = []
    for _ in range(repeats):
        for _ in range(iters):
            between()
            timer.begin()
            fn()
            timer.end()
        glFinish()
        samples = timer.collect(wait=True)
        medians.append(statistics.median(samples))
    timer.delete()
    return {"ms": statistics.median(medians), "runs_ms": medians}


def wall_timed(fn, iters: int, warmup: int, repeats: int, before=None) -> dict:
    """Wall time per call of ``fn``; ``before`` (untimed) runs ahead of every call, e.g. ``wp.synchronize`` so
    that a CPU-only measurement does not absorb the previous call's pending GPU work."""
    for _ in range(warmup):
        fn()
    medians = []
    for _ in range(repeats):
        samples = []
        for _ in range(iters):
            if before is not None:
                before()
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
        medians.append(statistics.median(samples))
    return {"ms": statistics.median(medians), "runs_ms": medians}


def wall_timed_interleaved(fns: dict, iters: int, warmup: int, repeats: int) -> dict:
    """Wall time per call of several functions measured in the same loop with a rotating order, so that no
    variant systematically inherits the state left behind by another (the GL <-> CUDA map/unmap handoff, for
    one, shows up in whichever call next waits on the stream)."""
    names = list(fns)
    for _ in range(warmup):
        for n in names:
            fns[n]()
    medians = {n: [] for n in names}
    for _ in range(repeats):
        samples = {n: [] for n in names}
        for i in range(iters):
            k = i % len(names)
            for n in names[k:] + names[:k]:
                t0 = time.perf_counter()
                fns[n]()
                samples[n].append((time.perf_counter() - t0) * 1e3)
        for n in names:
            medians[n].append(statistics.median(samples[n]))
    return {n: {"ms": statistics.median(m), "runs_ms": m} for n, m in medians.items()}


def event_timed(fn, iters: int, warmup: int, repeats: int) -> dict:
    """GPU time between two CUDA events around ``fn`` on Warp's current stream."""
    for _ in range(warmup):
        fn()
    wp.synchronize()
    e0 = wp.Event(enable_timing=True)
    e1 = wp.Event(enable_timing=True)
    medians = []
    for _ in range(repeats):
        samples = []
        for _ in range(iters):
            wp.record_event(e0)
            fn()
            wp.record_event(e1)
            wp.synchronize()
            samples.append(wp.get_event_elapsed_time(e0, e1))
        medians.append(statistics.median(samples))
    return {"ms": statistics.median(medians), "runs_ms": medians}


def set_camera(program, cam: dict):
    program.set("uRes", (cam["width"], cam["height"]))
    program.set("uEye", cam["eye"])
    program.set("uFwd", cam["fwd"])
    program.set("uRight", cam["right"])
    program.set("uUp", cam["up"])
    program.set("uTanAspect", (cam["tan_half"], cam["aspect"]))


def verdict(key: str, value: float) -> str:
    if key not in THRESHOLDS:
        return "N/A"
    op, limit = THRESHOLDS[key]
    ok = value <= limit if op == "<=" else value >= limit
    return "PASS" if ok else "FAIL"


# ----------------------------------------------------------------------------------------------------
# Synthetic electrode-centred star (13 chains x 150 segments of 1.5 mm from R1 to R2i)
# ----------------------------------------------------------------------------------------------------

def make_star(chains: int = CHAINS, segments: int = SEGMENTS_PER_CHAIN, h: float = NODE_SPACING, seed: int = 7,
              persistence: float = 0.85, sigma: float = 0.55) -> np.ndarray:
    """Persistent random walk per chain; the radial component is prescribed so node k sits at
    r = R1 + (R2i - R1) k / segments exactly, the tangential remainder of each 1.5 mm step wanders with a
    persistent direction (tortuosity ~ 3.7 as in the plan's synthetic star; ``persistence`` / ``sigma`` set
    the turning angle per step, see ``STAR_PRESETS``).  Returns ``(N, 8)`` float32 rows
    ``p0.xyz, eps, p1.xyz, power`` matching ``Segment`` in common.glsl."""
    rng = np.random.default_rng(seed)
    nodes = segments + 1
    rows = []
    for c in range(chains):
        d0 = rng.normal(size=3)
        d0 /= np.linalg.norm(d0)
        tang = rng.normal(size=3)
        tang -= d0 * np.dot(tang, d0)
        tang /= np.linalg.norm(tang)
        p = R1 * d0
        pts = [p.copy()]
        for k in range(1, nodes):
            r_now = np.linalg.norm(p)
            radial = p / r_now
            r_target = R1 + (R2I - R1) * k / segments
            dr = min(r_target - r_now, h)
            tang_mag = math.sqrt(max(h * h - dr * dr, 0.0))
            tang -= radial * np.dot(tang, radial)
            noise = rng.normal(size=3)
            noise -= radial * np.dot(noise, radial)
            tang = persistence * tang + sigma * noise
            tang -= radial * np.dot(tang, radial)
            tang /= np.linalg.norm(tang)
            p = p + dr * radial + tang_mag * tang
            pts.append(p.copy())
        pts = np.array(pts)
        power = 0.6 + 0.8 * rng.random()
        for k in range(segments):
            rows.append([*pts[k], CORE_EPS, *pts[k + 1], power])
    return np.asarray(rows, dtype=np.float32)


def star_stats(segments: np.ndarray, chains: int = CHAINS, per_chain: int = SEGMENTS_PER_CHAIN) -> dict:
    """Geometry of a star: segment length, path/chord tortuosity per chain, turning angle between consecutive
    segments (median / p90 in degrees) -- so a DDA figure is always reported with the shape it was measured on."""
    p0 = segments[:, 0:3].astype(np.float64)
    p1 = segments[:, 4:7].astype(np.float64)
    d = p1 - p0
    lengths = np.linalg.norm(d, axis=1)
    tort, turns = [], []
    for c in range(chains):
        sl = slice(c * per_chain, (c + 1) * per_chain)
        tort.append(lengths[sl].sum() / np.linalg.norm(p1[sl][-1] - p0[sl][0]))
        u = d[sl] / lengths[sl][:, None]
        turns.append(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u[:-1], u[1:]), -1.0, 1.0))))
    turns = np.concatenate(turns)
    return {"segment_mm": float(lengths.mean() * 1e3), "tortuosity_min": float(min(tort)),
            "tortuosity_max": float(max(tort)), "turn_deg_median": float(np.median(turns)),
            "turn_deg_p90": float(np.percentile(turns, 90))}


# ----------------------------------------------------------------------------------------------------
# Uniform-grid CSR in Warp (count -> exclusive scan -> scatter), dilation r cells
# ----------------------------------------------------------------------------------------------------

@wp.func
def segment_box_overlap(a: wp.vec3, b: wp.vec3, bmin: wp.vec3, bmax: wp.vec3) -> bool:
    d = b - a
    tmin = float(0.0)
    tmax = float(1.0)
    for axis in range(3):
        if wp.abs(d[axis]) < 1e-12:
            if a[axis] < bmin[axis] or a[axis] > bmax[axis]:
                return False
        else:
            inv = 1.0 / d[axis]
            t1 = (bmin[axis] - a[axis]) * inv
            t2 = (bmax[axis] - a[axis]) * inv
            lo = wp.min(t1, t2)
            hi = wp.max(t1, t2)
            tmin = wp.max(tmin, lo)
            tmax = wp.min(tmax, hi)
            if tmin > tmax:
                return False
    return True


@wp.func
def cell_range(a: wp.vec3, b: wp.vec3, grid_min: float, cell: float, n: int, r: int, axis: int):
    lo = wp.min(a[axis], b[axis])
    hi = wp.max(a[axis], b[axis])
    ilo = wp.clamp(int(wp.floor((lo - grid_min) / cell)) - r, 0, n - 1)
    ihi = wp.clamp(int(wp.floor((hi - grid_min) / cell)) + r, 0, n - 1)
    return ilo, ihi


@wp.kernel
def k_cell_count(p0: wp.array(dtype=wp.vec3), p1: wp.array(dtype=wp.vec3), count: wp.array(dtype=int),
                 grid_min: float, cell: float, n: int, r: int, counts: wp.array(dtype=int)):
    i = wp.tid()
    if i >= count[0]:
        return
    a = p0[i]
    b = p1[i]
    xlo, xhi = cell_range(a, b, grid_min, cell, n, r, 0)
    ylo, yhi = cell_range(a, b, grid_min, cell, n, r, 1)
    zlo, zhi = cell_range(a, b, grid_min, cell, n, r, 2)
    grow = float(r) * cell
    for z in range(zlo, zhi + 1):
        for y in range(ylo, yhi + 1):
            for x in range(xlo, xhi + 1):
                bmin = wp.vec3(grid_min + float(x) * cell - grow, grid_min + float(y) * cell - grow,
                               grid_min + float(z) * cell - grow)
                bmax = bmin + wp.vec3(cell + 2.0 * grow)
                if segment_box_overlap(a, b, bmin, bmax):
                    wp.atomic_add(counts, (z * n + y) * n + x, 1)


@wp.kernel
def k_cell_scatter(p0: wp.array(dtype=wp.vec3), p1: wp.array(dtype=wp.vec3), count: wp.array(dtype=int),
                   grid_min: float, cell: float, n: int, r: int, cell_start: wp.array(dtype=int),
                   cursor: wp.array(dtype=int), items: wp.array(dtype=int), capacity: int,
                   overflow: wp.array(dtype=int)):
    i = wp.tid()
    if i >= count[0]:
        return
    a = p0[i]
    b = p1[i]
    xlo, xhi = cell_range(a, b, grid_min, cell, n, r, 0)
    ylo, yhi = cell_range(a, b, grid_min, cell, n, r, 1)
    zlo, zhi = cell_range(a, b, grid_min, cell, n, r, 2)
    grow = float(r) * cell
    for z in range(zlo, zhi + 1):
        for y in range(ylo, yhi + 1):
            for x in range(xlo, xhi + 1):
                bmin = wp.vec3(grid_min + float(x) * cell - grow, grid_min + float(y) * cell - grow,
                               grid_min + float(z) * cell - grow)
                bmax = bmin + wp.vec3(cell + 2.0 * grow)
                if segment_box_overlap(a, b, bmin, bmax):
                    ci = (z * n + y) * n + x
                    slot = cell_start[ci] + wp.atomic_add(cursor, ci, 1)
                    if slot < capacity:
                        items[slot] = i
                    else:
                        wp.atomic_add(overflow, 0, 1)


def build_csr(segments: np.ndarray, dilation: int, device="cuda:0") -> dict:
    """Bin ``(N, 8)`` segments into the 96^3 grid: returns ``cell_start`` (uint32, N^3 + 1) and ``items``."""
    n_seg = segments.shape[0]
    p0 = wp.array(np.ascontiguousarray(segments[:, 0:3]), dtype=wp.vec3, device=device)
    p1 = wp.array(np.ascontiguousarray(segments[:, 4:7]), dtype=wp.vec3, device=device)
    count = wp.array([n_seg], dtype=int, device=device)
    ncells = GRID_N ** 3
    counts = wp.zeros(ncells + 1, dtype=int, device=device)
    cell_start = wp.zeros(ncells + 1, dtype=int, device=device)
    cursor = wp.zeros(ncells + 1, dtype=int, device=device)
    items = wp.zeros(ITEM_CAPACITY, dtype=int, device=device)
    overflow = wp.zeros(1, dtype=int, device=device)
    wp.launch(k_cell_count, dim=n_seg, inputs=[p0, p1, count, GRID_MIN, CELL_SIZE, GRID_N, dilation, counts],
              device=device)
    wp.utils.array_scan(counts, cell_start, inclusive=False)
    wp.launch(k_cell_scatter, dim=n_seg,
              inputs=[p0, p1, count, GRID_MIN, CELL_SIZE, GRID_N, dilation, cell_start, cursor, items,
                      ITEM_CAPACITY, overflow], device=device)
    wp.synchronize_device(device)
    start_np = cell_start.numpy()
    total = int(start_np[-1])
    if int(overflow.numpy()[0]) > 0:
        raise RuntimeError(f"CSR item capacity {ITEM_CAPACITY} exceeded (dilation {dilation}, {total} items)")
    counts_np = counts.numpy()[:ncells]
    occupied = int(np.count_nonzero(counts_np))
    return {
        "cell_start": start_np.astype(np.uint32),
        "items": items.numpy()[:max(total, 1)].astype(np.uint32),
        "total_items": total,
        "occupied_cells": occupied,
        "occupancy": occupied / ncells,
        "max_items_per_cell": int(counts_np.max()),
        "mean_items_per_occupied_cell": float(total / max(occupied, 1)),
    }


# ----------------------------------------------------------------------------------------------------
# numpy references for the capsule integral
# ----------------------------------------------------------------------------------------------------

def _quad3(k, b, c, ta, tb):
    q = lambda t: (k * t + 2.0 * b) * t + c  # noqa: E731
    det = k * c - b * b
    parallel = k < 1e-6
    safe_det = np.where(parallel, 1.0, det)
    closed = ((k * tb + b) / np.sqrt(q(tb)) - (k * ta + b) / np.sqrt(q(ta))) / safe_det
    return np.where(parallel, (tb - ta) * c ** -1.5, closed)


def _quad2(k, b, c, ta, tb):
    det = k * c - b * b
    parallel = k < 1e-6
    s = 1.0 / np.sqrt(np.where(parallel, 1.0, det))
    closed = (np.arctan((k * tb + b) * s) - np.arctan((k * ta + b) * s)) * s
    return np.where(parallel, (tb - ta) / c, closed)


def capsule_closed_np(ro, rd, t0, t1, a, b, eps, kernel=3):
    """Port of common.glsl capsuleEmission3/2 for ``(M, 3)`` rays; returns ``(M,)``."""
    ab = b - a
    L = math.sqrt(max(float(ab @ ab), 1e-14))   # CAPSULE_L2_MIN: a zero-length segment is a point source
    u = ab / L
    w = ro - a
    s0 = w @ u
    m = rd @ u
    wperp = w - s0[:, None] * u
    rperp = rd - m[:, None] * u
    kb = np.einsum("ij,ij->i", rperp, rperp)
    bb = np.einsum("ij,ij->i", wperp, rperp)
    cb = np.einsum("ij,ij->i", wperp, wperp) + eps * eps
    wb = ro - b
    bA, cA = np.einsum("ij,ij->i", w, rd), np.einsum("ij,ij->i", w, w) + eps * eps
    bB, cB = np.einsum("ij,ij->i", wb, rd), np.einsum("ij,ij->i", wb, wb) + eps * eps
    ms = np.where(m >= 0.0, np.maximum(m, 1e-7), np.minimum(m, -1e-7))
    tA = -s0 / ms
    tB = (L - s0) / ms
    lo = np.minimum(tA, tB)
    hi = np.maximum(tA, tB)
    quad = _quad3 if kernel == 3 else _quad2
    out = np.zeros(ro.shape[0])
    b0, b1 = np.maximum(t0, lo), np.minimum(t1, hi)
    mask = b1 > b0
    out[mask] += quad(kb[mask], bb[mask], cb[mask], b0[mask], b1[mask])
    l0, l1 = np.full_like(lo, t0), np.minimum(t1, lo)
    bl = np.where(ms > 0, bA, bB)
    cl = np.where(ms > 0, cA, cB)
    mask = l1 > l0
    out[mask] += quad(np.ones(mask.sum()), bl[mask], cl[mask], l0[mask], l1[mask])
    h0, h1 = np.maximum(t0, hi), np.full_like(hi, t1)
    bh = np.where(ms > 0, bB, bA)
    ch = np.where(ms > 0, cB, cA)
    mask = h1 > h0
    out[mask] += quad(np.ones(mask.sum()), bh[mask], ch[mask], h0[mask], h1[mask])
    return out


def capsule_numeric_np(ro, rd, t0, t1, a, b, eps, kernel=3, samples=200_001):
    """Simpson integration of the profile of the true point-to-segment distance along each ray."""
    t = np.linspace(t0, t1, samples)
    ab = b - a
    L2 = max(float(ab @ ab), 1e-14)   # a == b: the projection is 0, the distance is to the point a
    out = np.empty(ro.shape[0])
    wgt = np.ones(samples)
    wgt[1:-1:2] = 4.0
    wgt[2:-1:2] = 2.0
    wgt *= (t1 - t0) / (samples - 1) / 3.0
    for i in range(ro.shape[0]):
        x = ro[i][None, :] + t[:, None] * rd[i][None, :]
        s = np.clip(((x - a) @ ab) / L2, 0.0, 1.0)
        d2 = np.sum((x - (a + s[:, None] * ab)) ** 2, axis=1)
        f = (d2 + eps * eps) ** (-1.5 if kernel == 3 else -1.0)
        out[i] = np.sum(f * wgt)
    return out


def check_capsule_formula(segments: np.ndarray, rng, kernel: int, rays: int = 48, degenerate: int = 8) -> float:
    """Max relative error of the closed form vs numeric integration over random rays near random segments; the
    last ``degenerate`` rays use a zero-length segment (b = a), which must give the finite point-source integral."""
    worst = 0.0
    for i in range(rays):
        seg = segments[rng.integers(segments.shape[0])]
        a, b = seg[0:3].astype(np.float64), seg[4:7].astype(np.float64)
        if i >= rays - degenerate:
            b = a.copy()
        eps = float(seg[3])
        centre = 0.5 * (a + b)
        offset = rng.normal(size=3) * 4.0 * eps
        rd = rng.normal(size=3)
        if rng.random() < 0.25 and i < rays - degenerate:  # some nearly parallel rays
            rd = (b - a) + rng.normal(size=3) * 1e-4
        rd /= np.linalg.norm(rd)
        ro = centre + offset - 0.05 * rd
        ro, rd = ro[None, :], rd[None, :]
        t0, t1 = 0.0, 0.1
        closed = capsule_closed_np(ro, rd, t0, t1, a, b, eps, kernel)[0]
        numeric = capsule_numeric_np(ro, rd, t0, t1, a, b, eps, kernel)[0]
        if not (math.isfinite(closed) and math.isfinite(numeric)):
            raise AssertionError(f"non-finite capsule integral (kernel {kernel}, degenerate {i >= rays - degenerate})")
        worst = max(worst, abs(closed - numeric) / max(abs(numeric), 1e-12))
    return worst


def globe_pixel_rays(cam: dict, rng, n: int = 400):
    """``n`` random pixels whose ray crosses the inner sphere: ``(ys, xs, ro, rd, tlen)`` with the origin re-based
    at the globe entry point exactly like the shaders do."""
    o, d = camera_rays(cam)
    oc = -o
    bq = np.einsum("ijk,ijk->ij", oc, d)
    cq = np.einsum("ijk,ijk->ij", oc, oc) - R2I * R2I
    disc = bq * bq - cq
    ys, xs = np.nonzero(disc > 0)
    pick = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    ys, xs = ys[pick], xs[pick]
    rd = d[ys, xs]
    sq = np.sqrt(disc[ys, xs])
    t0 = np.maximum(bq[ys, xs] - sq, 0.0)
    t1 = bq[ys, xs] + sq
    ro = o[ys, xs] + t0[:, None] * rd
    return ys, xs, ro, rd, t1 - t0


# ----------------------------------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------------------------------

def fit_slope(counts, times) -> tuple[float, float]:
    """Least squares time = a + b N: returns ``(a [ms], b [us])``."""
    A = np.vstack([np.ones(len(counts)), counts]).T
    (a, b), *_ = np.linalg.lstsq(A, np.array(times), rcond=None)
    return float(a), float(b * 1e3)


def bench_capsule(args, cam, out_dir: Path, results: dict, segments: np.ndarray):
    print("\n== capsule: closed-form emission integral, time(N) = a + b N ==")
    rng = np.random.default_rng(3)
    for kernel in (3, 2):
        err = check_capsule_formula(segments, rng, kernel)
        print(f"  numpy closed form vs Simpson, kernel {kernel} (48 rays incl. near-parallel and 8 zero-length "
              f"segments): max rel err {err:.2e}")
        results[f"capsule.formula_check_k{kernel}_max_rel_err"] = err

    w, h = cam["width"], cam["height"]
    seg_buffer = Buffer(segments)
    out = Image2D(w, h, "RGBA16F")
    counts = [1, 8, 32, 128, 512, 1950]
    counts = [min(n, segments.shape[0]) for n in counts]
    local = [n for n in counts if 32 <= n <= 512]
    limit = THRESHOLDS["capsule.slope_us_k3"][1]

    def sweep(prog):
        times = []
        for n in counts:
            prog.set("uCount", n)
            iters = args.iters if n <= 512 else max(16, args.iters // 4)
            times.append(timed(lambda: prog.dispatch(w, h), iters, args.warmup, args.repeats)["ms"])
        return times

    # (kernel, divergent): the plan's broadcast protocol for both kernels, the divergent regime for the chosen one
    for kernel, divergent in ((3, 0), (3, 1), (2, 0)):
        tag = f"k{kernel}" + ("_divergent" if divergent else "")
        prog = compile_compute("bench_capsule.comp", {"KERNEL": kernel, "DIVERGENT": divergent})
        set_camera(prog, cam)
        prog.set("uTLen", 2.0 * R2I)
        prog.set("uScale", 1.0)
        prog.set("uClip", 0)
        seg_buffer.bind_base(0)
        out.bind_image(0, GL_WRITE_ONLY)
        slopes, local_slopes, intercepts = [], [], []
        for _ in range(args.slope_runs):
            times = sweep(prog)
            a, b = fit_slope(counts, times)
            _, bl = fit_slope(local, [t for n, t in zip(counts, times) if n in local])
            slopes.append(b)
            local_slopes.append(bl)
            intercepts.append(a)
        for n, t in zip(counts, times):
            print(f"  {tag:13s} N={n:5d}  {t:8.3f} ms  ({t * 1e3 / n:8.3f} us/segment)   [last run]")
        slope = statistics.median(slopes)
        note = ""
        if kernel == 3 and not divergent:
            note = "  <- AT THRESHOLD (within 10 %)" if 0.9 * limit <= slope <= limit else ("  <- FAIL" if slope > limit else "")
        print(f"  {tag}: slope over {args.slope_runs} runs median {slope:.3f} us (min {min(slopes):.3f}, "
              f"max {max(slopes):.3f}); local slope N=32..512 median {statistics.median(local_slopes):.3f} us; "
              f"intercept {statistics.median(intercepts):.3f} ms{note}")
        results[f"capsule.times_ms_{tag}"] = dict(zip(map(str, counts), times))
        results[f"capsule.slope_us_{tag}"] = slope
        results[f"capsule.slope_us_{tag}_runs"] = slopes
        results[f"capsule.slope_local_us_{tag}"] = statistics.median(local_slopes)
        results[f"capsule.intercept_ms_{tag}"] = statistics.median(intercepts)
        if divergent:
            prog.delete()
            continue

        # globe-clipped variant at the largest N: rays missing the globe do nothing
        prog.set("uClip", 1)
        prog.set("uCount", counts[-1])
        t = timed(lambda: prog.dispatch(w, h), max(16, args.iters // 4), args.warmup, args.repeats)["ms"]
        results[f"capsule.clipped_ms_k{kernel}_N{counts[-1]}"] = t
        print(f"  kernel {kernel}: N={counts[-1]} clipped to the globe: {t:.3f} ms")
        save_png(out, out_dir / f"capsule_k{kernel}.png", exposure=0.25 / max(counts[-1] / 64.0, 1.0))

        # GLSL vs numpy closed form on the clipped image with a handful of segments
        n_check = 16
        prog.set("uCount", n_check)
        prog.dispatch(w, h)
        glutil.barrier_image()
        glFinish()
        img = out.read()[..., 0].astype(np.float64)
        ys, xs, ro, rd, tlen = globe_pixel_rays(cam, rng)
        ref = np.zeros(len(ys))
        for s in segments[:n_check]:
            a3, b3, eps, power = s[0:3].astype(np.float64), s[4:7].astype(np.float64), float(s[3]), float(s[7])
            val = capsule_closed_np(ro, rd, np.zeros(len(ys)), tlen, a3, b3, eps, kernel)
            ref += power * (eps * eps if kernel == 3 else eps) * val
        got = img[ys, xs]
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-2)
        print(f"  kernel {kernel}: GLSL vs numpy on {len(ys)} globe pixels, {n_check} segments: "
              f"max rel err {rel.max():.2e} (rgba16f output), mean {rel.mean():.2e}")
        results[f"capsule.glsl_vs_numpy_k{kernel}_max_rel_err"] = float(rel.max())

        # a zero-length segment through the shader: no NaN anywhere, equal to the point-source integral
        point = np.array([[0.01, 0.0, 0.0, CORE_EPS, 0.01, 0.0, 0.0, 1.0]], dtype=np.float32)
        point_buffer = Buffer(point)
        point_buffer.bind_base(0)
        prog.set("uCount", 1)
        prog.dispatch(w, h)
        glutil.barrier_image()
        glFinish()
        img = out.read()[..., 0].astype(np.float64)
        nan_pixels = int(np.isnan(img).sum())
        a3 = point[0, 0:3].astype(np.float64)
        ref = (CORE_EPS * CORE_EPS if kernel == 3 else CORE_EPS) * capsule_closed_np(
            ro, rd, np.zeros(len(ys)), tlen, a3, a3, CORE_EPS, kernel)
        rel = np.abs(img[ys, xs] - ref) / np.maximum(np.abs(ref), 1e-2)
        print(f"  kernel {kernel}: zero-length segment: NaN pixels {nan_pixels} of {w * h}, GLSL vs numpy point "
              f"source max rel err {rel.max():.2e}")
        results[f"capsule.zero_length_nan_pixels_k{kernel}"] = nan_pixels
        results[f"capsule.zero_length_max_rel_err_k{kernel}"] = float(rel.max())
        if nan_pixels:
            raise AssertionError(f"zero-length segment produced {nan_pixels} NaN pixels (kernel {kernel})")
        point_buffer.delete()
        seg_buffer.bind_base(0)
        prog.delete()
    seg_buffer.delete()
    out.delete()


def bench_dda(args, cam, out_dir: Path, results: dict, stars: dict):
    print("\n== dda: 96^3 CSR, Amanatides-Woo, mailbox 8, cap 270 ==")
    w, h = cam["width"], cam["height"]
    out = Image2D(w, h, "RGBA16F")
    stats = Buffer((4 + HIST_BINS) * 4)
    progs = {
        (1, 1): compile_compute("bench_dda.comp", {"GRID_N": GRID_N, "CAND_CAP": CAND_CAP, "HISTOGRAM": 1, "EVAL": 1}),
        (0, 1): compile_compute("bench_dda.comp", {"GRID_N": GRID_N, "CAND_CAP": CAND_CAP, "HISTOGRAM": 0, "EVAL": 1}),
        (0, 0): compile_compute("bench_dda.comp", {"GRID_N": GRID_N, "CAND_CAP": CAND_CAP, "HISTOGRAM": 0, "EVAL": 0}),
    }
    for prog in progs.values():
        set_camera(prog, cam)
        prog.set("uGridMin", (GRID_MIN, GRID_MIN, GRID_MIN))
        prog.set("uCellSize", CELL_SIZE)
        prog.set("uScale", 1.0)

    for preset, segments in stars.items():
        main_preset = preset == args.star
        base = "dda" if main_preset else f"dda.{preset}"
        geo = star_stats(segments)
        print(f"  star '{preset}': segment {geo['segment_mm']:.2f} mm, tortuosity {geo['tortuosity_min']:.2f}-"
              f"{geo['tortuosity_max']:.2f}, turning angle median {geo['turn_deg_median']:.1f} deg, "
              f"p90 {geo['turn_deg_p90']:.1f} deg")
        results[f"{base}.star"] = {"preset": preset, **geo}
        seg_buffer = Buffer(segments)
        for dilation in ((0, 1, 2) if main_preset else (1,)):
            csr = build_csr(segments, dilation)
            print(f"  {preset} r={dilation}: {csr['total_items']} items, {csr['occupied_cells']} occupied cells "
                  f"({100 * csr['occupancy']:.2f} %), max {csr['max_items_per_cell']} / cell, "
                  f"mean {csr['mean_items_per_occupied_cell']:.1f} / occupied cell")
            start_buffer = Buffer(csr["cell_start"])
            items_buffer = Buffer(csr["items"])
            seg_buffer.bind_base(0)
            start_buffer.bind_base(1)
            items_buffer.bind_base(2)
            stats.bind_base(3)
            out.bind_image(0, GL_WRITE_ONLY)

            def run(prog=progs[(1, 1)]):
                prog.dispatch(w, h)

            def between():
                glutil.barrier_all()
                stats.clear()

            t_full = timed(lambda: run(progs[(1, 1)]), args.iters, args.warmup, args.repeats, between)["ms"]
            t_nohist = timed(lambda: run(progs[(0, 1)]), args.iters, args.warmup, args.repeats)["ms"]
            t_trav = timed(lambda: run(progs[(0, 0)]), args.iters, args.warmup, args.repeats)["ms"]

            # statistics from the last full dispatch: per-ray counters from the image, totals from the SSBO
            stats.clear()
            run(progs[(1, 1)])
            glutil.barrier_all()
            glFinish()
            st = stats.read(np.uint32)
            img = out.read()
            evals = img[..., 0].astype(np.float64)
            steps = img[..., 1].astype(np.float64)
            tests = img[..., 2].astype(np.float64)
            inside = (steps > 0) | (evals > 0) | (tests > 0)
            flags = evals >= CAND_CAP
            e_in, s_in, t_in = evals[inside], steps[inside], tests[inside]
            hist = st[4:4 + HIST_BINS]
            rays_inside = int(st[0])
            prefix = f"{base}.r{dilation}"
            res = {
                "ms": t_full, "ms_no_histogram": t_nohist, "ms_traversal_only": t_trav,
                "rays_inside": rays_inside, "inside_fraction": float(inside.mean()),
                "mean_evals": float(e_in.mean()), "p99_evals": float(np.percentile(e_in, 99)),
                "max_evals": float(e_in.max()),
                "mean_tests": float(t_in.mean()), "p99_tests": float(np.percentile(t_in, 99)),
                "max_tests": float(t_in.max()), "tests_per_eval": float(t_in.sum() / max(e_in.sum(), 1.0)),
                "flagged": int(flags[inside].sum()), "flagged_fraction": float(flags[inside].mean()),
                "mean_steps": float(s_in.mean()), "p99_steps": float(np.percentile(s_in, 99)),
                "mean_evals_all_pixels": float(evals.mean()),
                "histogram": hist.tolist(), "items": csr["total_items"], "occupied_cells": csr["occupied_cells"],
            }
            for k, v in res.items():
                results[f"{prefix}.{k}"] = v
            print(f"  {preset} r={dilation}: frame {t_full:.3f} ms (no histogram {t_nohist:.3f}, traversal only "
                  f"{t_trav:.3f}); globe rays {rays_inside} ({100 * inside.mean():.1f} % of pixels); "
                  f"evaluations/ray mean {res['mean_evals']:.1f} p99 {res['p99_evals']:.0f} max "
                  f"{res['max_evals']:.0f}; tests/ray mean {res['mean_tests']:.1f} p99 {res['p99_tests']:.0f} max "
                  f"{res['max_tests']:.0f} ({res['tests_per_eval']:.2f} tests per evaluation); steps/ray mean "
                  f"{res['mean_steps']:.1f} p99 {res['p99_steps']:.0f}; flagged {res['flagged']} "
                  f"({100 * res['flagged_fraction']:.3f} %)")
            assert int(hist.sum()) == rays_inside, "histogram does not sum to the ray count"
            assert int(inside.sum()) == rays_inside, "image / SSBO ray counts differ"
            assert res["flagged"] == int(st[3]), "image-derived cap flags differ from the SSBO count"
            assert int(round(e_in.sum())) == int(st[2]), "image evaluations differ from the SSBO total"
            # colour scale: 1.5x the p99 so the structure is readable; the cap (270) is far off-scale at r <= 1
            tag = f"{dilation}" if main_preset else f"{dilation}_{preset}"
            save_heatmap_png(np.where(inside, evals, -1.0), out_dir / f"dda_heat_r{tag}.png",
                             vmax=max(1.5 * res["p99_evals"], 20.0))
            if dilation == 1:
                save_heatmap_png(np.where(inside, tests, -1.0), out_dir / f"dda_tests_r{tag}.png",
                                 vmax=max(1.5 * res["p99_tests"], 20.0))
            if dilation == 1 and main_preset:
                radiance = np.repeat(img[..., 3:4], 3, axis=2) * np.array([1.0, 0.55, 0.85], dtype=np.float32)
                tmp = Image2D(w, h, "RGBA16F")
                tmp.upload(np.concatenate([radiance, np.ones_like(radiance[..., :1])], axis=2))
                save_png(tmp, out_dir / "dda_radiance_r1.png", exposure=0.5)
                tmp.delete()
            start_buffer.delete()
            items_buffer.delete()
        seg_buffer.delete()
    for prog in progs.values():
        prog.delete()
    out.delete()
    stats.delete()


def bench_glass(args, cam, out_dir: Path, results: dict) -> Image2D:
    print("\n== glass: deterministic 3-branch split, no filaments ==")
    w, h = cam["width"], cam["height"]
    out = Image2D(w, h, "RGBA16F")
    for far_wall in (1, 0):
        prog = compile_compute("bench_glass.comp", {"FAR_WALL_REFLECTION": far_wall})
        set_camera(prog, cam)
        out.bind_image(0, GL_WRITE_ONLY)
        t = timed(lambda: prog.dispatch(w, h), args.iters, args.warmup, args.repeats)["ms"]
        key = "glass.ms" if far_wall else "glass.ms_without_far_wall_reflection"
        results[key] = t
        print(f"  far-wall reflection {'on ' if far_wall else 'off'}: {t:.3f} ms")
        if far_wall:
            save_png(out, out_dir / "glass.png", exposure=1.0)
        prog.delete()
    return out


def synthetic_hdr(w: int, h: int, seed: int = 11) -> np.ndarray:
    """HDR test content: soft gradient, a few bright emissive streaks, noise (rgba float32, (h, w, 4))."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / w, y / h
    img = np.zeros((h, w, 4), dtype=np.float32)
    img[..., 0] = 0.05 + 0.1 * u
    img[..., 1] = 0.05 + 0.1 * v
    img[..., 2] = 0.08
    for _ in range(12):
        cx, cy = rng.random() * w, rng.random() * h
        ang = rng.random() * math.pi
        dx, dy = math.cos(ang), math.sin(ang)
        d = np.abs(-(x - cx) * dy + (y - cy) * dx)
        along = (x - cx) * dx + (y - cy) * dy
        streak = np.exp(-d * d / (2.0 * 2.5 ** 2)) * (np.abs(along) < 0.25 * h)
        img[..., 0] += 6.0 * streak
        img[..., 1] += 3.0 * streak
        img[..., 2] += 5.0 * streak
    img[..., :3] += rng.random((h, w, 1)).astype(np.float32) * 0.02
    img[..., 3] = 1.0
    return img


def bench_taau(args, cam, out_dir: Path, results: dict) -> Image2D:
    print("\n== taau: stand-in 2x temporal upscale ==")
    w, h = cam["width"], cam["height"]
    W, H = 2 * w, 2 * h
    color = Image2D(w, h, "RGBA16F")
    color.upload(synthetic_hdr(w, h))
    motion = Image2D(w, h, "RG16F")
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    mv = np.zeros((h, w, 2), dtype=np.float32)
    mv[..., 0] = 1.5 / w * np.sin(y / h * 6.0)
    mv[..., 1] = 1.5 / h * np.cos(x / w * 6.0)
    motion.upload(mv)
    history = [Image2D(W, H, "RGBA16F"), Image2D(W, H, "RGBA16F")]
    for img in history:
        img.clear()
    prog = compile_compute("bench_taau.comp")
    prog.set("uOutRes", (W, H))
    prog.set("uInRes", (w, h))
    prog.set("uBlend", 0.1)
    prog.set("uGamma", 1.0)
    color.bind_sampler(0)
    motion.bind_sampler(1)
    frame = [0]

    def run():
        src = history[frame[0] & 1]
        dst = history[(frame[0] + 1) & 1]
        src.bind_sampler(2)
        dst.bind_image(0, GL_WRITE_ONLY)
        prog.dispatch(W, H)
        frame[0] += 1

    t = timed(run, args.iters, args.warmup, args.repeats)["ms"]
    results["taau.ms"] = t
    print(f"  {W}x{H} from {w}x{h}: {t:.3f} ms")
    final = history[frame[0] & 1]
    save_png(final, out_dir / "taau.png", exposure=0.3)
    prog.delete()
    color.delete()
    motion.delete()
    history[(frame[0] + 1) & 1].delete()
    return final


def blue_noise_64(seed: int = 5, iterations: int = 12) -> np.ndarray:
    """Quick blue noise: white noise re-ranked after repeated high-pass filtering (toroidal Gaussian)."""
    rng = np.random.default_rng(seed)
    n = 64
    v = rng.random((n, n))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    sigma = 1.6
    gauss = np.exp(-2.0 * (math.pi * sigma) ** 2 * (fx * fx + fy * fy))
    for _ in range(iterations):
        low = np.real(np.fft.ifft2(np.fft.fft2(v) * gauss))
        hp = v - low
        ranks = np.argsort(np.argsort(hp.ravel())).reshape(n, n)
        v = (ranks + 0.5) / (n * n)
    return v.astype(np.float32)


def bench_present(args, cam, out_dir: Path, results: dict, history: Image2D | None, glow_src: Image2D | None):
    print("\n== present: AgX + blue-noise dither to RGBA8 at mode resolution ==")
    w, h = cam["width"], cam["height"]
    W, H = 2 * w, 2 * h
    own_history = history is None
    if own_history:
        history = Image2D(W, H, "RGBA16F")
        history.upload(synthetic_hdr(W, H, seed=21))
    own_glow = glow_src is None
    if own_glow:
        glow_src = Image2D(w, h, "RGBA16F")
        glow_src.upload(synthetic_hdr(w, h, seed=31) * 0.2)
    noise = Image2D(64, 64, "R8", linear=False)
    noise.upload((blue_noise_64() * 255.0 + 0.5).astype(np.uint8)[..., None])
    target = Image2D(W, H, "RGBA8")
    fbo = Framebuffer([target])
    prog = compile_program("fullscreen.vert", "bench_present.frag")
    prog.set("uOutRes", (W, H))
    prog.set("uExposure", 1.0)
    prog.set("uGlowWeight", 0.5)
    tri = FullscreenTriangle()

    def run():
        fbo.bind()
        prog.use()
        history.bind_sampler(0)
        glow_src.bind_sampler(1)
        noise.bind_sampler(2)
        tri.draw()
        Framebuffer.unbind()

    t = timed(run, args.iters, args.warmup, args.repeats)["ms"]
    results["present.ms"] = t
    print(f"  {W}x{H} present: {t:.3f} ms")
    save_png(target, out_dir / "present.png")

    # 1080p -> mode-resolution blit of an RGBA16F FBO into the RGBA8 target
    src_fbo = Framebuffer([glow_src])
    t_blit = timed(lambda: src_fbo.blit_to(fbo, W, H, GL_LINEAR), args.iters, args.warmup, args.repeats)["ms"]
    results["present.blit_1080p_to_4k_ms"] = t_blit
    print(f"  glBlitFramebuffer {w}x{h} RGBA16F -> {W}x{H} RGBA8, linear: {t_blit:.3f} ms")
    save_png(target, out_dir / "blit.png")

    src_fbo.delete()
    tri.delete()
    prog.delete()
    fbo.delete()
    target.delete()
    noise.delete()
    if own_history:
        history.delete()
    if own_glow:
        glow_src.delete()


def bench_interop(args, out_dir: Path, results: dict):
    print("\n== interop: batched map/unmap, wp.copy into an SSBO, 3D texture upload ==")
    device = wp.get_device("cuda:0")
    n_floats = FRAME_SSBO_BYTES // 4
    ssbo = Buffer(FRAME_SSBO_BYTES, target=GL_SHADER_STORAGE_BUFFER)
    volumes = {96: Image3D(96, 96, 96, "RGBA16F"), 128: Image3D(128, 128, 128, "RGBA16F")}
    buf_res = interop.register_buffer(ssbo.id, flags=wp.RegisteredGLBuffer.WRITE_DISCARD, device=device)
    tex_res = {n: interop.register_image(v.id, interop.GL_TEXTURE_3D, device=device) for n, v in volumes.items()}
    print(f"  slots: RegisteredGLBuffer.resource=0x{buf_res.resource:x} -> CUgraphicsResource 0x{interop.resource_handle(buf_res):x}, "
          f"GLTextureResource._resource=0x{tex_res[96]._resource:x} -> 0x{interop.resource_handle(tex_res[96]):x}; "
          f"libcuda via ctypes: {interop.driver_available()}")

    staging = wp.array(np.arange(n_floats, dtype=np.float32), dtype=wp.float32, device=device)
    vol_np = {}
    vol_staging = {}
    for n in volumes:
        rng = np.random.default_rng(n)
        data = rng.random((n, n, n, 4)).astype(np.float16)
        vol_np[n] = data
        vol_staging[n] = wp.array(data, dtype=wp.vec4h, device=device)

    # Map/unmap cost.  The CPU submission of the two driver calls is ~0.013 ms, but the GL <-> CUDA handoff adds
    # ~0.13 ms of latency on the stream that only shows once something waits on it (the next kernel, the frame's
    # glFinish) -- so the frame-relevant figure, and the threshold figure, is map + unmap + wp.synchronize().
    # The variants run interleaved in one loop (rotating order) so that none inherits another's pending handoff.
    batch = interop.BatchedMap([buf_res, tex_res[96]], device=device)
    fallback = interop.BatchedMap([buf_res, tex_res[96]], device=device)
    fallback.batched = False   # Warp's per-resource wp_cuda_graphics_map/unmap in nested order

    def batched_sync():
        batch.map()
        batch.unmap()
        wp.synchronize()

    def fallback_sync():
        fallback.map()
        fallback.unmap()
        wp.synchronize()

    def warp_nested_sync():
        buf_res.map(dtype=wp.float32, shape=(n_floats,))
        tex_res[96].map()
        tex_res[96].unmap()
        buf_res.unmap()
        wp.synchronize()

    def warp_sequential_sync():
        buf_res.map(dtype=wp.float32, shape=(n_floats,))
        buf_res.unmap()
        tex_res[96].map()
        tex_res[96].unmap()
        wp.synchronize()

    t_sync = wall_timed_interleaved({"batched": batched_sync, "fallback": fallback_sync, "nested": warp_nested_sync,
                                     "sequential": warp_sequential_sync}, args.frames, 20, args.repeats)
    t_cpu = wall_timed(lambda: (batch.map(), batch.unmap()), args.frames, 20, args.repeats, before=wp.synchronize)["ms"]
    t_cpu_fb = wall_timed(lambda: (fallback.map(), fallback.unmap()), args.frames, 20, args.repeats,
                          before=wp.synchronize)["ms"]
    t_sync_only = wall_timed(wp.synchronize, args.frames, 20, args.repeats)["ms"]
    results["interop.batched"] = batch.batched
    results["interop.batched_map_unmap_sync_ms"] = t_sync["batched"]["ms"]
    results["interop.batched_map_unmap_cpu_ms"] = t_cpu
    results["interop.fallback_nested_map_unmap_sync_ms"] = t_sync["fallback"]["ms"]
    results["interop.fallback_nested_map_unmap_cpu_ms"] = t_cpu_fb
    results["interop.warp_nested_map_unmap_sync_ms"] = t_sync["nested"]["ms"]
    results["interop.warp_sequential_map_unmap_sync_ms"] = t_sync["sequential"]["ms"]
    results["interop.synchronize_alone_ms"] = t_sync_only
    print(f"  map + unmap of 2 resources (SSBO 5.5 MB + 96^3 RGBA16F) + wp.synchronize, interleaved, wall:")
    print(f"    batched ctypes call pair        {t_sync['batched']['ms']:.4f} ms  (CPU submission alone "
          f"{t_cpu:.4f} ms; batched={batch.batched})")
    print(f"    BatchedMap fallback (nested)    {t_sync['fallback']['ms']:.4f} ms  (CPU submission alone "
          f"{t_cpu_fb:.4f} ms)")
    print(f"    Warp nested map/map/unmap/unmap {t_sync['nested']['ms']:.4f} ms")
    print(f"    Warp sequential pairs           {t_sync['sequential']['ms']:.4f} ms  (two handoffs)")
    print(f"    wp.synchronize alone            {t_sync_only:.4f} ms")

    batch.map()
    dst = batch.array(0, wp.float32, (n_floats,))
    t_copy = event_timed(lambda: wp.copy(dst, staging), args.frames, 10, args.repeats)["ms"]
    results["interop.wp_copy_5p5MB_ms"] = t_copy
    print(f"  wp.copy 5.5 MB staging -> mapped SSBO: {t_copy:.4f} ms (CUDA events)")
    arr96 = batch.cuarray(1)
    t_m3d = event_timed(lambda: interop.upload_volume(arr96, vol_staging[96], 96, 96, 96, 8, device=device),
                        args.frames, 10, args.repeats)["ms"]
    results["interop.memcpy3d_96_ms"] = t_m3d
    print(f"  cuMemcpy3DAsync 96^3 RGBA16F (7.1 MB): {t_m3d:.4f} ms")
    tex96 = batch.texture(1)
    t_cf = event_timed(lambda: tex96.copy_from(vol_staging[96]), args.frames, 10, args.repeats)["ms"]
    results["interop.copy_from_96_ms"] = t_cf
    print(f"  wp.Texture3D.copy_from 96^3 RGBA16F: {t_cf:.4f} ms (dtype {tex96.dtype}, channels {tex96.num_channels})")
    wp.synchronize()
    del tex96   # per-map wrapper: never keep it across unmap
    batch.unmap()
    t_texobj = wall_timed(lambda: (batch.map(), batch.texture(1), batch.unmap(), wp.synchronize()), args.frames, 10,
                          args.repeats)["ms"]
    results["interop.texture_wrapper_overhead_ms"] = t_texobj - t_sync["batched"]["ms"]
    print(f"  BatchedMap.texture(i) wrapper create + drop per map: {t_texobj - t_sync['batched']['ms']:.4f} ms extra")

    batch128 = interop.BatchedMap([tex_res[128]], device=device)
    batch128.map()
    arr128 = batch128.cuarray(0)
    t_m3d128 = event_timed(lambda: interop.upload_volume(arr128, vol_staging[128], 128, 128, 128, 8, device=device),
                           args.frames, 10, args.repeats)["ms"]
    t_cf128 = event_timed(lambda: batch128.texture(0).copy_from(vol_staging[128]), args.frames, 10, args.repeats)["ms"]
    results["interop.memcpy3d_128_ms"] = t_m3d128
    results["interop.copy_from_128_ms"] = t_cf128
    print(f"  128^3 RGBA16F (16.8 MB): cuMemcpy3DAsync {t_m3d128:.4f} ms, copy_from {t_cf128:.4f} ms")
    wp.synchronize()
    batch128.unmap()

    # full P0b sequence: map -> copy -> volume upload -> unmap, wall time including the GPU work
    def frame():
        batch.map()
        wp.copy(batch.array(0, wp.float32, (n_floats,)), staging)
        interop.upload_volume(batch.cuarray(1), vol_staging[96], 96, 96, 96, 8, device=device)
        batch.unmap()
        wp.synchronize()

    t_frame = wall_timed(frame, args.frames, 10, args.repeats)["ms"]
    results["interop.p0b_frame_wall_ms"] = t_frame
    print(f"  P0b frame (map, wp.copy, memcpy3d, unmap, synchronize): {t_frame:.4f} ms wall")

    # correctness: read both resources back through GL
    got = ssbo.read(np.float32)
    ok_buf = np.array_equal(got, staging.numpy())
    got_vol = volumes[96].read()
    ref_vol = vol_np[96].astype(np.float32)
    ok_vol = np.allclose(got_vol, ref_vol, atol=1e-3)
    got_vol128 = volumes[128].read()
    ok_vol128 = np.allclose(got_vol128, vol_np[128].astype(np.float32), atol=1e-3)
    results["interop.ssbo_roundtrip_ok"] = bool(ok_buf)
    results["interop.volume96_roundtrip_ok"] = bool(ok_vol)
    results["interop.volume128_roundtrip_ok"] = bool(ok_vol128)
    print(f"  GL read-back matches staging: SSBO {ok_buf}, 96^3 RGBA16F {ok_vol}, 128^3 RGBA16F {ok_vol128}")

    stab = interop.pointer_stability(batch, cycles=1000)
    results["interop.pointer_stability"] = stab
    print(f"  pointer stability over {stab['cycles']} map/unmap cycles: changes per resource {stab['changes']} "
          f"-> {'stable' if stab['stable'] else 'CHANGED'}")

    # teardown order the app must follow: close the maps (unregisters the resources) before the GL objects go
    fallback.close(unregister_resources=False)
    batch.close()
    batch128.close()
    assert buf_res.resource is None and all(t._resource is None for t in tex_res.values()), "unregister failed"
    ssbo.delete()
    for v in volumes.values():
        v.delete()


def bench_calib(results: dict):
    print("\n== calib: effective FP32 rate inferred from the capsule slope ==")
    slope = results.get("capsule.slope_us_k3")
    if slope is None:
        print("  run the capsule benchmark first (--bench all or --bench capsule calib)")
        return
    pixels = results["resolution"][0] * results["resolution"][1]
    print(f"  the display calibration of plan M1 (plasma_globe.glsl through shaderbang) needs DRM master and is "
          f"not run here")
    for instr in (50, 75, 100):
        rate = instr * pixels / (slope * 1e-6)
        results[f"calib.fp32_rate_at_{instr}_instr"] = rate
        print(f"  {instr:3d} instr/candidate: {rate / 1e12:6.2f} T instr/s = {100 * rate / PEAK_FP32_INSTR_PER_S:5.1f} % "
              f"of the 52.44 T FP32 peak")
    slope2 = results.get("capsule.slope_us_k2")
    if slope2:
        print(f"  atan kernel penalty: {slope2 / slope:.2f}x the algebraic kernel")
        results["calib.atan_penalty"] = slope2 / slope


# ----------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plasma globe M1 micro-benchmarks (headless GL 4.6 + Warp)")
    parser.add_argument("--bench", nargs="+", default=["all"],
                        choices=["all", "capsule", "dda", "glass", "taau", "present", "interop", "calib"])
    parser.add_argument("--res", default="1920x1080", help="internal resolution; the mode resolution is 2x")
    parser.add_argument("--out", type=Path, default=Path("/tmp/plasma_bench"))
    parser.add_argument("--iters", type=int, default=64, help="timed dispatches per run")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3, help="runs per measurement (median of medians)")
    parser.add_argument("--frames", type=int, default=200, help="frames per interop measurement")
    parser.add_argument("--slope-runs", type=int, default=5, help="full N sweeps per capsule slope (median reported)")
    parser.add_argument("--star", choices=sorted(STAR_PRESETS), default="tortuous",
                        help="synthetic star preset for the capsule bench and the full DDA sweep (the other preset "
                             "is run at r = 1 as a bracket)")
    parser.add_argument("--persistence", type=float, default=None, help="override the --star preset's persistence")
    parser.add_argument("--sigma", type=float, default=None, help="override the --star preset's sigma")
    args = parser.parse_args()
    benches = set(args.bench)
    if "all" in benches:
        benches = {"capsule", "dda", "glass", "taau", "present", "interop", "calib"}
    w, h = (int(v) for v in args.res.lower().split("x"))
    args.out.mkdir(parents=True, exist_ok=True)

    wp.init()
    wp.set_device("cuda:0")
    results_path = args.out / "results.json"
    # partial runs (--bench dda ...) update the sections of an existing results.json instead of replacing it
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update({"resolution": [w, h], "mode_resolution": [2 * w, 2 * h]})
    with make_headless_context() as ctx:
        results["gl"] = {"version": ctx.version, "glsl": ctx.glsl, "renderer": ctx.renderer}
        results["gl_limits"] = gl_limits()
        print("GL limits:", json.dumps(results["gl_limits"]))
        cam = perspective_camera(CAMERA_EYE, CAMERA_TARGET, (0.0, 1.0, 0.0), CAMERA_FOV, w, h)
        stars = {}
        for preset in [args.star] + [p for p in STAR_PRESETS if p != args.star]:
            persistence, sigma = STAR_PRESETS[preset]
            if preset == args.star:
                persistence = args.persistence if args.persistence is not None else persistence
                sigma = args.sigma if args.sigma is not None else sigma
            stars[preset] = make_star(persistence=persistence, sigma=sigma)
            geo = star_stats(stars[preset])
            print(f"star '{preset}' (persistence {persistence}, sigma {sigma}): {stars[preset].shape[0]} segments of "
                  f"{geo['segment_mm']:.2f} mm, tortuosity {geo['tortuosity_min']:.2f}-{geo['tortuosity_max']:.2f}, "
                  f"turning angle median {geo['turn_deg_median']:.1f} deg / p90 {geo['turn_deg_p90']:.1f} deg")
        segments = stars[args.star]
        results["segments"] = int(segments.shape[0])
        results["star"] = {"preset": args.star, **star_stats(segments)}
        glass_img = None
        history = None
        if "capsule" in benches:
            bench_capsule(args, cam, args.out, results, segments)
        if "dda" in benches:
            bench_dda(args, cam, args.out, results, stars)
        if "glass" in benches:
            glass_img = bench_glass(args, cam, args.out, results)
        if "taau" in benches:
            history = bench_taau(args, cam, args.out, results)
        if "present" in benches:
            bench_present(args, cam, args.out, results, history, glass_img)
        if "interop" in benches:
            bench_interop(args, args.out, results)
        if "calib" in benches:
            bench_calib(results)
        if glass_img is not None:
            glass_img.delete()
        if history is not None:
            history.delete()

    print("\n== summary ==")
    rows = []
    for key, (op, limit) in THRESHOLDS.items():
        if key in results:
            v = results[key]
            rows.append((key, v, f"{op} {limit}", verdict(key, v)))
    for key, v, thr, verd in rows:
        print(f"  {key:36s} {v:10.4f}   {thr:10s} {verd}")
    results["verdicts"] = {k: verd for k, _, _, verd in rows}
    results_path.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nresults: {results_path}; PNGs in {args.out}")


if __name__ == "__main__":
    main()
