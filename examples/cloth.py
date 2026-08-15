#!/usr/bin/env -S uv run --script

# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "shaderbang",
#     "pyopengl",
#     "warp-lang",
# ]
#
# [tool.uv.sources]
# shaderbang = { git = "https://github.com/astefanutti/shaderbang.git", branch = "main" }
# ///

"""
Cloth Simulation
================

GPU-accelerated cloth simulation using NVIDIA Warp with XPBD physics,
adapted to run with Shaderbang.

Inspired by Matthias Mueller's Ten Minute Physics:
https://matthias-research.github.io/pages/tenMinutePhysics/
https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/16-GPUCloth.py

Keyboard Controls
-----------------
    P               Pause / resume the simulation
    Space / Right   Advance one simulation step (works while paused)
    S               Cycle step granularity:
                      Frame step -> Sub-step -> Contact step -> Frame step
    R               Reset the cloth to its initial state
    C               Toggle particle self-collision
    W               Toggle wireframe rendering
    F               Toggle back-face culling (shows front/back in different colors)
    Ctrl+A          Select all anchors
    Delete / Bksp   Remove selected anchors

Mouse Controls
--------------
    Left drag on cloth      Grab and drag a cloth particle
    Left drag elsewhere     Orbit the camera
    Right drag              Track (pan) the camera
    Scroll wheel            Dolly (zoom) the camera
    Ctrl + release          Lock the dragged particle as a persistent anchor
    Click (no drag)         Toggle anchor selection

Touchscreen Controls
--------------------
    1 finger on cloth       Grab and drag a cloth particle
    1 finger elsewhere      Orbit the camera (trackball)
    2-3 fingers             Track, dolly, and rotate the camera
    4 fingers               Translate the sphere
    5+ fingers              Rotate and resize the sphere

Trackpad Controls
-----------------
    2 fingers               Orbit, dolly, and rotate the camera
    3 fingers               Track, dolly, and rotate the camera
    4 fingers               Translate the sphere
    5+ fingers              Rotate and resize the sphere
"""


import argparse
import ctypes
import glob
import os
import math
import stat
import sys
import signal
import threading
import time

import numpy as np
import warp as wp

from dataclasses import dataclass, field
from enum import auto, Flag
from typing import Callable, Generic, Optional, Self, TypeVar

from contextlib import ExitStack
from pathlib import Path
from signal import pthread_sigmask, pthread_kill, sigwait
from threading import main_thread, Thread

from libevdev import Device, EV_ABS, EV_KEY, EV_REL, INPUT_PROP_DIRECT, INPUT_PROP_POINTER

import shaderbang.input
from shaderbang.inotify import INotify, IN_CREATE, IN_ATTRIB
from shaderbang.input import Input, TouchSlot
from shaderbang.gesture import homothety_and_rotation
from shaderbang import lib as sb, options

from OpenGL import setPlatform
setPlatform("egl")

from OpenGL.GL import *
from OpenGL.GLU import *


parser = argparse.ArgumentParser(description="Run cloth simulation")
parser.add_argument("-D", "--device", metavar="DEVICE", type=Path,
                    help="the DRM device")
parser.add_argument("-C", "--connector", metavar="CONNECTOR", type=int,
                    help="the DRM connector")
parser.add_argument("--mode", metavar="MODE", type=str,
                    help="the name of the video mode, e.g., 1920x1080")
parser.add_argument("--refresh", metavar="FREQ", type=int,
                    help="the vertical refresh rate in Hz")
parser.add_argument("--async-page-flip", action=argparse.BooleanOptionalAction,
                    help="use async page flipping")
parser.add_argument("--atomic-drm-mode", action=argparse.BooleanOptionalAction,
                    help="use atomic mode setting")
parser.add_argument("-n", "--frames", metavar="N", type=int,
                    help="run for N frames and exit")

gravity = wp.vec3(0.0, -9.80665, 0.0)

thickness = 0.001
particleRadius = 0.0045
fingerRadius = 0.06     # world radius of a touch/click grab: all cloth particles
                        # within this distance of the picked point are pinned and
                        # dragged together (see Particle.group / drag_anchor)
maxVelocity = 1e2   # m/s cap on cloth particle velocity (spike guard)

# Stability / broadphase-safety bounds (see the collision audit).
# maxDisplacement: hard cap on a free particle's per-substep trial displacement
#   |pos - prev_pos|, applied AFTER the XPBD solve and BEFORE self-collision. A
#   legitimate substep travels <= maxVelocity*dt (~0.056) plus small constraint
#   corrections, so 0.2 never fires on a contract-valid substep but bounds the
#   swept query AABB when an instability spike scatters positions.
# maxQueryExtent: if a self-collision query AABB edge exceeds this, skip the query.
#   A governed box tops out around edge_len + 2*maxDisplacement + 2*d_offset ~= 0.44,
#   so 1.0 is a pure no-op in normal operation and only fires on NaN/blown-up boxes,
#   turning a would-be multi-second (O(edges x tris)) frame into a bounded one.
maxDisplacement = 0.2
maxQueryExtent = 1.0

# Planar Divide-and-Truncate (PDT) collision parameters
d_offset = 2.0 * particleRadius  # cloth self-collision separation (matches old 2*particleRadius)
gamma_r = 0.9                    # conservative truncation safety ratio (Newton uses 0.85-0.95)

# Per-substep cap on the accumulated C<0 feasibility-recovery push magnitude. The
# push is an atomic_add over every overlapping pair (order-non-deterministic float
# add) applied AFTER the displacement governor, and it feeds straight back into
# velocity (push/dt). Uncapped, a dense-overlap substep can jump a particle
# arbitrarily far and pump energy, occasionally diverging the sim ("sometimes gets
# stuck"). Capping to a fraction of d_offset bounds the per-substep separation
# (overlaps still relax over several of the 60 substeps) and the injected velocity.
pushClamp = 0.5 * d_offset

# Self-collision is split into a hash-grid "detect" pass that caches candidate
# primitives and a query-free "narrowphase" pass that does the segment-segment /
# point-triangle truncation. The detect pass used to query warp's LBVH
# (mesh_query_aabb), whose fixed 32 KiB per-block shared-memory traversal stack
# capped occupancy at ~50% and pointer-chased the tree -- 63% of the substep. It
# now queries a wp.HashGrid over the frozen start-of-substep POINTS (built
# in-graph each substep; a build is ~0.07 ms vs ~1.2 ms of BVH queries) and
# expands each neighbor point into its precomputed incident faces (vertex-
# triangle) or incident edges (edge-edge). Query radii are derived per primitive
# from the exact narrowphase no-op criterion (gap <= d_offset + own sweep + max
# sweep), plus a Jung-type bound (longest-edge/sqrt(3), resp. /2) that converts
# "triangle/segment within range" into "some VERTEX within range"; the global
# longest-edge and max-sweep bounds are recomputed on the GPU every substep, so
# stretched cloth or fast motion enlarge the radii instead of missing candidates.
# vtBuf caches candidate FACE ids (dedup'd, 2-ring culled -- identical semantics
# to the old BVH cull); eeBuf now caches candidate EDGE ids directly (shared-
# vertex + ring culls applied at detect time), which removes the face->3-edge
# expansion and canonicalization from the hot edge-edge narrowphase. The
# `selfCollisionOverflow` counter still flags any capacity drop (which would
# allow penetration).
# The grid detect's acceptance reach (~d_offset + longest_edge slack, see
# detect_expand) is ~2x the old BVH boxes' +/-d_offset in the contact-normal
# direction, so compressed stacks (a sphere squeezing the draped cloth) yield
# 2-3x the old candidate counts. Peaks are hit MID-frame (worst substep), well
# above the end-of-frame counts test_counts.py reports.
maxVT = 256   # cached candidate faces per vertex (non-ring, dedup'd)
maxEE = 768   # cached candidate edges per edge (culled, dedup'd)

# Frozen-edge length cap for the vertex-centered broadphase walk radii (see
# detect_walk_radius). The radii used to add the GLOBAL longest frozen edge
# (bounds[0]) so stretched cloth widened the search -- but that lets ONE
# locally stretched edge (a fast sphere drag reaches 0.3-2.0 m) inflate EVERY
# vertex's walk radius; combined with multi-layer squeeze states this
# saturated the fixed candidate buffers (silent drops = potential tunneling).
# The radii now use L = min(bounds[0], edgeLenCap), which bounds every walk:
# primitives whose frozen edges all fit under the cap are covered by the
# capped walk exactly as before (the at-rest longest edge is the 0.0212 m quad
# diagonal, far below the cap, so rest behavior is unchanged), while OVERSIZED
# primitives (any frozen edge > edgeLenCap) are partitioned EXCLUSIVELY to a
# fallback: detect_walk_radius/detect_expand skip them (same `> edgeLenCap`
# comparison everywhere, so no pair is ever processed twice -- the C<0 push is
# not idempotent) and scatter_oversized / oversized_pairs append them into
# the same candidate buffers (see those kernels). 4x the 0.015 rest spacing.
edgeLenCap = 4.0 * 0.015

# Strain limiting: hard cap on edge elongation, enforced by a post-solve Jacobi
# projection (strain_limit kernel, strainLimitIters passes/substep). Real cloth
# stretches < ~10%; without a hard cap, dragging the sphere through a wrapped
# drape overwhelms the 2-iteration XPBD solve and edges stretch 25-100x rest --
# which (a) looks like the cloth "diverging", (b) explodes the self-collision
# contact density (the perf cliff: >1s frames + millions of dropped candidates
# in stretch wads), and (c) is what created oversized primitives in the first
# place. With strain capped at 1.2x, the longest possible edge is
# 1.2 * 0.0212 (rest diagonal) = 0.0255 << edgeLenCap, so the oversized
# fallback becomes a never-firing safety net and detect density stays bounded
# under any drag speed. The limiter engages ONLY beyond maxStrain (the XPBD
# solve keeps normal strain ~1-2%), so settled/draping behavior is unchanged.
maxStrain = 1.2          # max edge length as a multiple of rest length
minStrain = 0.5          # compression floor (settled drapes stay >= ~0.92x rest, so
                         # this is inert normally; crushing collapsed edges to ~0.1x
                         # rest = degenerate "inverted" triangles + contact-density
                         # blowup until this floor forces buckling instead)
strainLimitIters = 8     # Jacobi passes after the solve (pre-collision)
strainLimitPostIters = 4 # Jacobi passes after the collider, before update_velocity:
                         # the collider's nearest-surface ejection can split an edge
                         # across the sphere within one substep; limiting again before
                         # velocities are derived keeps that stretch out of vel (else
                         # (pos-prev)/dt bakes it in and integrate re-creates it).
strainRelax = 0.6        # under-relaxation (shared vertices, up to 8 edges each)

# Hash-grid cell width for the self-collision broadphase. Queries may use any
# radius (the grid walks more cells); the cell size just tunes performance. The
# quiet-state walk radius is ~0.022 (max of the vertex-triangle reach
# d_offset + rest_diag/sqrt(3) and the edge-edge reach
# sqrt(halflen^2 + (d_offset + rest_diag/2)^2)), so 0.024 keeps the walk at
# 3^3 cells with the tightest cell-granular over-return (validated by the
# broadphase microbench on the real crumpled state).
gridCellSize = 0.024

# Thread count for the single-slot detection-bounds reductions (grid-stride loops).
boundsReduceThreads = 16384

# Cached non-ring neighbor vertices per vertex (detect_gather -> detect_expand).
# Crumpled-plateau counts are ~20-30; sized for a compressed squeeze; the
# selfCollisionOverflow counter flags drops. Capacities are sized ~1.5-2x the
# worst MID-substep peaks measured on the 200x200 rising-sphere squeeze with
# the capped radii (nbr ~210, vt ~52, ee ~490): the squeeze density is real
# (multi-layer stack x fast ejection sweeps), the caps just need to clear it.
# Memory at 400x400: nbr 247 MB + vt 165 MB + ee 1.48 GB ~= 1.9 GB.
maxNbr = 384

# Fixed-size per-thread caches for detect_expand: the per-vertex incident-edge
# data is loop-invariant across the cached neighbors (high-water ~15 per vertex
# at a deep-settled 400x400 pile), and a neighbor's incident-edge data is
# invariant across the (up to 6) own edges it is tested against. Hoisting both into registers/local arrays removes the
# redundant global gathers that dominated the kernel. Grid-mesh vertex degree is
# <= 6 (faces and edges), asserted at build time.
expandDeg = 6
vec6i = wp.types.vector(length=expandDeg, dtype=wp.int32)
vec6f = wp.types.vector(length=expandDeg, dtype=wp.float32)
mat6x3f = wp.types.matrix(shape=(expandDeg, 3), dtype=wp.float32)
# detect_expand parallelizes over (vertex, neighbor slot): one thread per
# vertex left the whole ~(neighbors x incident) expansion as ONE serial
# dependent-load chain per thread, and 160K threads is barely a single wave on
# a large GPU -- no latency hiding and the pile's slowest vertices bound the
# kernel. expandK threads per vertex stride the cached neighbor list instead
# (the gather high-water at a deep-settled 400x400 pile is ~15, so each thread
# usually owns at most one neighbor).
expandK = 16

numIterations = 2
numSubsteps = 30  # ~2x frame rate vs 60. The "visible undulation" 30 used to cause was
                  # root-caused to (a) the bending constraint's per-substep impulse
                  # crossing its explicit-regime stability boundary at dt > timeStep/60
                  # (see the bending stability guard in step()) and (b) the XPBD damping
                  # term dotting gradients with absolute previous POSITIONS (a spurious
                  # dt-dependent bias, fixed in distance/bending_constraints). With both
                  # fixes a settled drape's frame-to-frame motion at 30 substeps matches
                  # 60 (~2e-4 m mean vs ~2e-2 unfixed); 60-substep behavior is unchanged
                  # (the guard is exactly 1.0 there).
timeStep = 1.0 / 30.0
epsilon = sys.float_info.epsilon

# Reference substep size the bending stability guard is calibrated to (the substep
# count the simulation was tuned and visually validated at). See step(): for
# dt <= bendStabilityDt the guard is exactly 1.0 (a no-op); only LARGER substeps
# (fewer than 60 substeps/frame) scale the bending relaxation down to keep the
# per-substep bending impulse inside the overshoot boundary observed at 60.
bendStabilityDt = timeStep / 60.0

wp.init()
wp.set_device("cuda")


class State(Flag):
    RUN = auto()
    STEP = auto()
    FRAME_STEP = auto()
    SMALL_STEP = auto()
    CONTACT_STEP = auto()
    SOLVER_STEP = auto()
    SELF_COLLISION = auto()
    CULL_FACE = auto()
    WIREFRAME = auto()


STEPS = State.FRAME_STEP | State.SMALL_STEP | State.CONTACT_STEP | State.SOLVER_STEP

state = State.RUN | State.FRAME_STEP | State.SELF_COLLISION | State.WIREFRAME


class _Perf:
    # Rolling frame-time buckets for the [perf] line (see Cloth.render).
    PERIOD = 120
    frames = 0
    t_last = None
    total = 0.0
    sim = 0.0
    draw = 0.0


_perf = _Perf()


class AnchorFlag(Flag):
    ACTIVE = auto()
    SELECTED = auto()
    LOCKED = auto()

@dataclass
class Particle:
    id: int
    screen: wp.vec2
    mass: float
    depth: float
    flags = AnchorFlag.ACTIVE
    origin: wp.vec2 = field(init=False)
    time: float = field(init=False)
    # Fingertip grab group: (particle id, original inv-mass, world offset from the
    # primary particle at grab time). All members are pinned and move rigidly with
    # the primary -- a single-vertex pin transmits absurd force through one thread
    # of fabric (and, being inv_mass==0, ignores collision), so grabbing a
    # fingertip-sized patch both feels natural and distributes the pull.
    group: list = field(default_factory=list)

    def drag(self) -> Self:
        self.origin = wp.vec2(self.screen)
        self.time = time.time()
        return self

    def click(self) -> bool:
        return wp.length(self.screen - self.origin) < 1.0 and time.time() - self.time < 0.5

    def drop(self):
        self.flags &= ~(AnchorFlag.ACTIVE | AnchorFlag.LOCKED)

class Cloth(Input):

    def __init__(self, y_offset, num_x, num_y, spacing):
        super().__init__("cloth")

        # TODO: change for size
        self.spacing = spacing
        self.anchors: list[Particle] = []
        self._quad = None

        if num_x % 2 == 1:
            num_x = num_x + 1
        if num_y % 2 == 1:
            num_y = num_y + 1

        self.numParticles = (num_x + 1) * (num_y + 1)
        pos = np.zeros((self.numParticles, 3))
        inv_mass = np.zeros(self.numParticles)

        for xi in range(num_x + 1):
            for yi in range(num_y + 1):
                i = xi * (num_y + 1) + yi
                pos[i, 0] = (-num_x * 0.5 + xi) * spacing
                pos[i, 1] = y_offset
                pos[i, 2] = (-num_y * 0.5 + yi) * spacing
                inv_mass[i] = 1.0

        # distance constraints
        one_x = lambda xi, yi: (xi * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi)
        one_y = lambda xi, yi: (xi * (num_y + 1) + yi, xi * (num_y + 1) + yi + 1)
        two_x = lambda xi, yi: (xi * (num_y + 1) + yi, (xi + 2) * (num_y + 1) + yi)
        two_y = lambda xi, yi: (xi * (num_y + 1) + yi, xi * (num_y + 1) + yi + 2)
        cross = lambda xi, yi: [(xi * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi + 1),
                                ((xi + 1) * (num_y + 1) + yi, xi * (num_y + 1) + yi + 1)]

        self.distConstraints = DistConstraints(
            Constraint(
                (range(num_x + 1), range(0, num_y, 2), one_y),
                (range(num_x + 1), range(1, num_y, 2), one_y),
                (range(0, num_x, 2), range(num_y + 1), one_x),
                (range(1, num_x, 2), range(num_y + 1), one_x),
                parallel=False,
                ke=1.0e9,
                kd=10.0,
            ),
            Constraint(
                (range(0, num_x, 2), range(0, num_y, 2), cross),
                (range(0, num_x, 2), range(1, num_y, 2), cross),
                (range(1, num_x, 2), range(0, num_y, 2), cross),
                (range(1, num_x, 2), range(1, num_y, 2), cross),
                parallel=False,
                ke=1.0e7,
                kd=10.0,
            ),
            Constraint(
                (range(num_x + 1), range(0, num_y - 1, 3), two_y),
                (range(num_x + 1), range(1, num_y - 1, 3), two_y),
                (range(num_x + 1), range(2, num_y - 1, 3), two_y),
                (range(0, num_x - 1, 3), range(num_y + 1), two_x),
                (range(1, num_x - 1, 3), range(num_y + 1), two_x),
                (range(2, num_x - 1, 3), range(num_y + 1), two_x),
                parallel=True,
                relaxation=0.6,
                ke=1.0e7,
                kd=10.0,
            ),
        )

        # bending constraints
        square = lambda xi, yi: (xi * (num_y + 1) + yi + 1, (xi + 1) * (num_y + 1) + yi,
                                 xi * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi + 1)
        diam_x = lambda xi, yi: (xi * (num_y + 1) + yi, (xi + 2) * (num_y + 1) + yi + 1,
                                 (xi + 1) * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi + 1)
        diam_y = lambda xi, yi: (xi * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi + 2,
                                 (xi + 1) * (num_y + 1) + yi + 1, xi * (num_y + 1) + yi + 1)

        self.bendConstraints = BendConstraints(
            Constraint(
                (range(num_x), range(num_y), square),
                (range(num_x), range(num_y - 1), diam_y),
                (range(num_x - 1), range(num_y), diam_x),
                parallel=True,
                relaxation=0.6,
                ke=1.0e5,
                kd=10.0,
            ),
        )

        self.constraints = Constraints.Chain(self.distConstraints, self.bendConstraints)

        # triangles
        self.numTris = 2 * num_x * num_y
        self.triDist = wp.zeros(self.numTris, dtype=float)
        self.hostTriIds = np.zeros((self.numTris, 3), dtype=np.int32)

        i = 0
        for xi in range(num_x):
            for yi in range(num_y):
                id0 = xi * (num_y + 1) + yi
                id1 = (xi + 1) * (num_y + 1) + yi
                id2 = (xi + 1) * (num_y + 1) + yi + 1
                id3 = xi * (num_y + 1) + yi + 1
                self.hostTriIds[i, 0] = id0
                self.hostTriIds[i, 1] = id1
                self.hostTriIds[i, 2] = id2
                i += 1
                self.hostTriIds[i, 0] = id0
                self.hostTriIds[i, 1] = id2
                self.hostTriIds[i, 2] = id3
                i += 1

        self.prevPos = wp.array(pos, dtype=wp.vec3)
        self.restPos = wp.clone(self.prevPos)
        self.invMass = wp.array(inv_mass, dtype=float)
        self.vel = wp.zeros_like(self.restPos)
        self.deltas = wp.zeros_like(self.restPos)

        self.hostInvMass = wp.array(inv_mass, dtype=float, device="cpu", copy=False, pinned=True)
        self.hostPos = wp.array(pos, dtype=wp.vec3, device="cpu", copy=False, pinned=True)
        self.hostTriDist = wp.zeros(self.numTris, dtype=float, device="cpu", pinned=True)

        self.pos = None
        self.pos_gl_buffer = GLuint()
        self.normals = None
        self.normals_gl_buffer = GLuint()
        self.triIds = None
        self.triIds_gl_buffer = GLuint()

        self.numCols = num_y + 1  # grid stride, for topological-neighbor exclusion
        self.truncation_ts = wp.zeros(self.numParticles, dtype=float)  # per-vertex PDT scale (atomic_min)
        self.push = wp.zeros_like(self.restPos)  # C<0 feasibility-recovery separation (atomic_add)
        # Click-time picking scratch (brute-force ray cast; no BVH in the sim).
        self._pickDist = wp.zeros(1, dtype=float)
        self._pickFace = wp.zeros(1, dtype=wp.int32)

        # Per-vertex grid (row, col) precomputed once so the self-collision ring cull
        # is a couple of int loads instead of 4 integer divisions per call. within_ring
        # is invoked up to 4x per candidate in the hottest (edge-edge) kernel, so on the
        # ring-cull-bound crumpled plateau this removes the dominant integer-div traffic.
        idx = np.arange(self.numParticles, dtype=np.int32)
        grid_rc = np.stack((idx // self.numCols, idx % self.numCols), axis=1)
        self.gridRC = wp.array(grid_rc, dtype=wp.int32)  # [numParticles, 2] = (row, col)


        # Sphere collider pose as single-element device arrays so the captured graph
        # can ADVANCE the sphere per substep (advance_sphere increments them in-graph):
        # each substep sweeps center[0]..center[0]+dc/numSubsteps, so the swept test
        # sees the true per-frame sphere velocity AND the last substep lands on the
        # rendered end pose (center+dc). Set from the module `sphere` each frame in
        # simulate(); the graph records the pointers, not the values.
        self.colliderCenter = wp.zeros(1, dtype=wp.vec3)
        self.colliderRadius = wp.zeros(1, dtype=float)

        # Per-SUBSTEP collider delta slices (frame delta / numSubsteps), also as
        # device arrays. They used to be baked into the captured graph as kernel
        # constants, forcing a RECAPTURE every frame; as device memory the graph
        # is frame-invariant, so simulate() captures once per flag combination and
        # replays it across frames (the in-graph hash-grid build makes per-frame
        # re-instantiation prohibitively slow: its mempool alloc nodes cost ~0.5 s
        # per instantiation).
        self.colliderDeltaC = wp.zeros(1, dtype=wp.vec3)
        self.colliderDeltaR = wp.zeros(1, dtype=float)
        self.colliderDeltaQ = wp.zeros(1, dtype=wp.quat)
        self._graphs = {}  # (iterations, integrate, self_collision, solve) -> captured graph

        # Self-collision candidate caches (detect writes, narrowphase reads) + a global
        # overflow counter (>0 means a per-primitive buffer filled and dropped a
        # candidate -> possible penetration; grow maxVT/maxEE).
        self.vtCount = wp.zeros(self.numParticles, dtype=wp.int32)
        self.vtBuf = wp.zeros(self.numParticles * maxVT, dtype=wp.int32)
        self.selfCollisionOverflow = wp.zeros(1, dtype=wp.int32)

        # Unique mesh edges (sorted vertex pairs) for edge-edge self-collision.
        edge_set = set()
        for tri in self.hostTriIds:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, w in ((a, b), (b, c), (c, a)):
                edge_set.add((u, w) if u < w else (w, u))
        edge_arr = np.array(sorted(edge_set), dtype=np.int32)
        self.numEdges = len(edge_arr)
        self.edgeIds = wp.array(edge_arr, dtype=wp.int32)  # [numEdges, 2], va < vb
        self.eeCount = wp.zeros(self.numEdges, dtype=wp.int32)
        self.eeBuf = wp.zeros(self.numEdges * maxEE, dtype=wp.int32)
        # Rest length per unique mesh edge, for the strain limiter.
        rest_len = np.linalg.norm(pos[edge_arr[:, 1]] - pos[edge_arr[:, 0]], axis=1)
        self.edgeRestLen = wp.array(rest_len.astype(np.float32), dtype=float)

        # Static pair list for the ring_floor kernel: every vertex pair within
        # Chebyshev ring distance <= 2 on the grid (the self-collision blind zone),
        # EXCLUDING actual mesh edges (governed by the strain limiter's compression
        # floor instead). Built vectorized: for each of the 12 canonical (dr, dc)
        # offsets covering the 5x5 neighborhood half-plane, pair every vertex with
        # its offset neighbor.
        rows = idx // self.numCols
        cols = idx % self.numCols
        nrows = num_x + 1
        pair_chunks = []
        for dr in range(0, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc <= 0:
                    continue  # half-plane: each unordered pair once
                ok = (rows + dr < nrows) & (cols + dc >= 0) & (cols + dc < self.numCols)
                a = idx[ok]
                b = a + dr * self.numCols + dc
                pair_chunks.append(np.stack((a, b), axis=1))
        ring_pairs = np.concatenate(pair_chunks, axis=0)
        # exclude mesh edges via a sorted-key set difference (vectorized hashing)
        key = ring_pairs.min(axis=1).astype(np.int64) * self.numParticles + ring_pairs.max(axis=1)
        ekey = edge_arr.min(axis=1).astype(np.int64) * self.numParticles + edge_arr.max(axis=1)
        keep = ~np.isin(key, ekey)
        ring_pairs = ring_pairs[keep]
        self.numRingPairs = len(ring_pairs)
        self.ringPairs = wp.array(ring_pairs.astype(np.int32), dtype=wp.int32)
        print(str(self.numRingPairs) + " ring-floor pairs created")
        print(str(self.numEdges) + " edges created")

        # Per-vertex incident-primitive tables (CSR) for the hash-grid broadphase:
        # a neighbor POINT found by the grid expands into its incident FACES
        # (vertex-triangle candidates) or incident EDGES (edge-edge candidates).
        # Grid-mesh degrees are tiny (<= 6 faces / 6 edges per vertex).
        corner_v = self.hostTriIds.reshape(-1)
        corner_f = np.repeat(np.arange(self.numTris, dtype=np.int32), 3)
        order = np.argsort(corner_v, kind="stable")
        vf_off = np.zeros(self.numParticles + 1, dtype=np.int32)
        vf_off[1:] = np.cumsum(np.bincount(corner_v, minlength=self.numParticles))
        self.vertFaceOff = wp.array(vf_off, dtype=wp.int32)
        self.vertFaceIds = wp.array(corner_f[order], dtype=wp.int32)

        end_v = edge_arr.reshape(-1)
        end_e = np.repeat(np.arange(self.numEdges, dtype=np.int32), 2)
        order = np.argsort(end_v, kind="stable")
        ve_off = np.zeros(self.numParticles + 1, dtype=np.int32)
        ve_off[1:] = np.cumsum(np.bincount(end_v, minlength=self.numParticles))
        self.vertEdgeOff = wp.array(ve_off, dtype=wp.int32)
        self.vertEdgeIds = wp.array(end_e[order], dtype=wp.int32)

        # detect_expand caches a vertex's incident-edge data in fixed-size
        # per-thread arrays (expandDeg entries); a larger degree would silently
        # index out of bounds, so fail loudly if the topology ever changes.
        max_vf = int(np.diff(vf_off).max())
        max_ve = int(np.diff(ve_off).max())
        assert max_vf <= expandDeg and max_ve <= expandDeg, \
            f"vertex degree {max_vf} faces / {max_ve} edges exceeds expandDeg={expandDeg}"

        # Edge -> (up to 2) adjacent faces, for collect_oversized's announced-
        # face table ([numEdges, 2], -1 padded). Vectorized: the Python loop over
        # 320k triangles took ~3 s of startup at 400x400. edge_arr rows are sorted
        # pairs in lexicographic order, so the a*N+b keys are ascending and
        # searchsorted maps each triangle edge to its edge id; a lexsort by
        # (edge, face) groups each edge's faces ascending, matching the loop's
        # face-major fill order.
        tri = self.hostTriIds.astype(np.int64)
        te = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=0)
        tkey = te.min(axis=1) * self.numParticles + te.max(axis=1)
        ekeys = edge_arr[:, 0].astype(np.int64) * self.numParticles + edge_arr[:, 1]
        flat_e = np.searchsorted(ekeys, tkey)
        flat_f = np.tile(np.arange(self.numTris, dtype=np.int32), 3)
        order = np.lexsort((flat_f, flat_e))  # by edge, then ascending face id
        se, sf = flat_e[order], flat_f[order]
        counts = np.bincount(se, minlength=self.numEdges)
        starts = np.searchsorted(se, np.arange(self.numEdges))
        ef = np.full((self.numEdges, 2), -1, dtype=np.int32)
        ef[counts >= 1, 0] = sf[starts[counts >= 1]]
        ef[counts >= 2, 1] = sf[starts[counts >= 2] + 1]
        self.edgeFaceIds = wp.array(ef, dtype=wp.int32)

        # Oversized-primitive partition state, rebuilt per substep on the frozen
        # reference state (see step()): per-primitive frozen edge lengths gate
        # the expand-side skips (`> edgeLenCap`, one float load in the hot
        # loops) AND provide the per-candidate Jung slack that keeps squeeze
        # states from saturating the candidate buffers; the compact id list
        # drives the oversized fallback (scatter_oversized / oversized_pairs,
        # O(k) per thread over the compact list).
        self.edgeLen = wp.zeros(self.numEdges, dtype=float)      # frozen |b - a| per edge
        self.faceLongest = wp.zeros(self.numTris, dtype=float)   # longest frozen edge per face
        self.oversizedIds = wp.zeros(self.numEdges, dtype=wp.int32)
        self.oversizedCount = wp.zeros(1, dtype=wp.int32)
        # Per-SLOT announce data for the oversized list, precomputed once per
        # substep by collect_oversized so the vertex-parallel scatter reads a
        # compact L2-resident table instead of re-deriving per (vertex, edge):
        # the faces the edge announces (it is their LONGEST edge; -1 = none),
        # the max apex-to-segment distance of those faces, the edge's own max
        # endpoint sweep, and whether both endpoints are pinned.
        self.ovF0 = wp.zeros(self.numEdges, dtype=wp.int32)
        self.ovF1 = wp.zeros(self.numEdges, dtype=wp.int32)
        self.ovApex = wp.zeros(self.numEdges, dtype=float)
        self.ovSweep = wp.zeros(self.numEdges, dtype=float)
        self.ovPinned = wp.zeros(self.numEdges, dtype=wp.int32)

        # Global per-substep detection bound, recomputed on the GPU inside the
        # captured graph: [0] = longest edge length (frozen reference state),
        # capped by edgeLenCap wherever the walk radii use it (see
        # detect_walk_radius / detect_expand).
        self.detectBounds = wp.zeros(2, dtype=float)

        # Neighbor cache between the two broadphase stages (gather -> expand).
        self.nbrCount = wp.zeros(self.numParticles, dtype=wp.int32)
        self.nbrBuf = wp.zeros(self.numParticles * maxNbr, dtype=wp.int32)

        # Hash grid replacing the LBVH for the self-collision broadphase. Created
        # here (device-side table allocation happens at first build, warmed up in
        # init()/init_headless() outside graph capture).
        self.grid = wp.HashGrid(128, 128, 128)

        print(str(self.numParticles) + " particles created")
        print(str(self.numTris) + " triangles created")
        print(str(self.distConstraints.count) + " distance constraints created")
        print(str(self.bendConstraints.count) + " bending constraints created")

    @staticmethod
    @wp.kernel
    def rest_distances(
            pos: wp.array(dtype=wp.vec3),
            const_ids: wp.array2d(dtype=wp.int32),
            rest_lengths: wp.array(dtype=float)):
        tid = wp.tid()
        p0 = pos[const_ids[tid, 0]]
        p1 = pos[const_ids[tid, 1]]
        rest_lengths[tid] = wp.length(p1 - p0)

    @staticmethod
    @wp.kernel
    def add_normals(
            pos: wp.array(dtype=wp.vec3),
            tri_ids: wp.array2d(dtype=wp.int32),
            normals: wp.array(dtype=wp.vec3)):
        tid = wp.tid()
        id0 = tri_ids[tid, 0]
        id1 = tri_ids[tid, 1]
        id2 = tri_ids[tid, 2]
        normal = wp.cross(pos[id1] - pos[id0], pos[id2] - pos[id0])
        wp.atomic_add(normals, id0, normal)
        wp.atomic_add(normals, id1, normal)
        wp.atomic_add(normals, id2, normal)

    @staticmethod
    @wp.kernel
    def normalize_normals(normals: wp.array(dtype=wp.vec3)):
        tid = wp.tid()
        normals[tid] = wp.normalize(normals[tid])

    @staticmethod
    @wp.func
    def closest_point_on_triangle(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.vec3:
        # Ericson, Real-Time Collision Detection: closest point on triangle (a,b,c) to p.
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = wp.dot(ab, ap)
        d2 = wp.dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return a
        bp = p - b
        d3 = wp.dot(ab, bp)
        d4 = wp.dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return b
        cp = p - c
        d5 = wp.dot(ab, cp)
        d6 = wp.dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return c
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b)
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return a + ab * v + ac * w

    @staticmethod
    @wp.func
    def planar_truncation_t(x: wp.vec3, dx: wp.vec3, n: wp.vec3, p: wp.vec3) -> float:
        # Fraction of displacement dx that reaches the plane (n, through p) from x.
        # Returns 1.0 when moving parallel to or away from the plane (no limit).
        denom = wp.dot(n, dx)
        if wp.abs(denom) < epsilon:
            return 1.0
        t = wp.dot(n, p - x) / denom
        if t < 0.0:
            return 1.0
        gamma_min = 1.0e-3
        return wp.clamp(wp.min(t * gamma_r, t - gamma_min), 0.0, 1.0)

    @staticmethod
    @wp.func
    def within_ring(a: wp.int32, b: wp.int32, num_cols: wp.int32) -> bool:
        # True when grid vertices a and b are within the 2-ring of each other.
        # The mesh is a regular grid: index = row * num_cols + col.
        dr = a // num_cols - b // num_cols
        dc = a % num_cols - b % num_cols
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return wp.max(dr, dc) <= 2

    @staticmethod
    @wp.func
    def within_ring_rc(grid_rc: wp.array2d(dtype=wp.int32), a: wp.int32, b: wp.int32) -> bool:
        # Division-free within_ring: (row, col) are precomputed per vertex, so the
        # 2-ring test is two int loads + subtracts instead of 4 integer divisions.
        dr = grid_rc[a, 0] - grid_rc[b, 0]
        dc = grid_rc[a, 1] - grid_rc[b, 1]
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return wp.max(dr, dc) <= 2

    @staticmethod
    @wp.func
    def ring2_rc(ar: wp.int32, ac: wp.int32, br: wp.int32, bc: wp.int32) -> bool:
        # within_ring_rc on PRE-LOADED (row, col) pairs: identical predicate, no
        # grid_rc loads (detect_expand hoists the rows it reuses).
        dr = ar - br
        dc = ac - bc
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return wp.max(dr, dc) <= 2

    @staticmethod
    @wp.func
    def point_segment_distance(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
        # Distance from point p to segment [a, b].
        ab = b - a
        ab2 = wp.dot(ab, ab)
        t = 0.0
        if ab2 > epsilon:
            t = wp.clamp(wp.dot(p - a, ab) / ab2, 0.0, 1.0)
        return wp.length(a + t * ab - p)

    @staticmethod
    @wp.func
    def closest_point_segment_segment(p1: wp.vec3, q1: wp.vec3,
                                      p2: wp.vec3, q2: wp.vec3):
        # Ericson, Real-Time Collision Detection: closest points between segments
        # [p1,q1] and [p2,q2]. Returns (c1, c2, s, t) with c1 = p1 + s*(q1-p1) and
        # c2 = p2 + t*(q2-p2); s, t are the barycentric parameters along each edge.
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        a = wp.dot(d1, d1)
        e = wp.dot(d2, d2)
        f = wp.dot(d2, r)
        s = float(0.0)
        t = float(0.0)
        if a <= epsilon and e <= epsilon:
            s = 0.0
            t = 0.0
        elif a <= epsilon:
            s = 0.0
            t = wp.clamp(f / e, 0.0, 1.0)
        else:
            cc = wp.dot(d1, r)
            if e <= epsilon:
                t = 0.0
                s = wp.clamp(-cc / a, 0.0, 1.0)
            else:
                b = wp.dot(d1, d2)
                denom = a * e - b * b
                if wp.abs(denom) > epsilon:
                    s = wp.clamp((b * f - cc * e) / denom, 0.0, 1.0)
                else:
                    s = 0.0
                t = (b * s + f) / e
                if t < 0.0:
                    t = 0.0
                    s = wp.clamp(-cc / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = wp.clamp((b - cc) / a, 0.0, 1.0)
        c1 = p1 + d1 * s
        c2 = p2 + d2 * t
        return c1, c2, s, t

    @staticmethod
    @wp.func
    def swept_sphere_ccd(pos: wp.vec3,
                         vel: wp.vec3,
                         center: wp.vec3,
                         radius: float,
                         dc: wp.vec3,
                         dr: float,
                         dt: float):
        # Analytic continuous collision of a particle segment against a moving /
        # growing sphere: solve for the time-of-impact and return the contact point
        # on the swept surface. Exact for a convex analytic collider, so a fast
        # sphere drag cannot tunnel through the cloth regardless of speed.
        s = center - pos
        vc = dc / dt
        vr = dr / dt
        v = vc - vel

        c = wp.dot(s, s) - radius * radius
        if c < 0.0:
            # The particle is inside the sphere
            return False, wp.vec3()
        a = wp.dot(v, v) - vr * vr
        if wp.abs(a) < epsilon:
            # The particle is not moving relative to the sphere
            return False, wp.vec3()
        b = wp.dot(v, s) - radius * vr
        if b > 0.0:
            # The particle is not moving towards the sphere
            return False, wp.vec3()
        d = b * b - a * c
        if d < 0.0:
            # The particle segment does not intersect the sphere
            return False, wp.vec3()
        t = (-b - wp.sqrt(d)) / a
        if t >= dt:
            # The particle segment does not intersect the sphere
            return False, wp.vec3()

        p = pos + t * vel
        o = center + t * vc
        r = p - o

        # CARRY-ALONG reconstruction: keep the sphere-relative contact direction the
        # particle had at the time of impact and evaluate it at the END pose -- the
        # particle rides the sphere. The previous reconstruction placed hits on the
        # flank of the swept tube (dead-ahead hits were pushed to an ARBITRARY
        # perpendicular), so a fast sphere actively parted the cloth, bored a hole
        # through the drape and let it close behind -- zero measured penetration,
        # but visually the sphere "passed through" the fabric. Carrying hit
        # particles forward is a plow: physical for fast motion, and safe now that
        # the strain limiter bounds the stretch it creates (carrying was the
        # over-fling hazard before strain limiting existed).
        d_rel = wp.length(r)
        if d_rel < epsilon:
            # Degenerate (particle at the TOI center): let Pass 2 resolve it.
            return False, wp.vec3()
        n = r / d_rel
        return True, center + dc + (radius + dr) * n

    @staticmethod
    @wp.kernel
    def integrate(
            dt: float,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3)):
        tid = wp.tid()

        # Freeze the penetration-free reference state X = prev_pos for this substep.
        # Collisions (sphere, ground, self) are resolved afterwards by PDT truncation
        # and analytic collider projection, not here.
        if inv_mass[tid] == 0.0:
            prev_pos[tid] = pos[tid]
            return

        prev_pos[tid] = pos[tid]
        pos[tid] += (vel[tid] + gravity * dt) * dt

    @staticmethod
    @wp.kernel
    def max_edge_length(
            prev_pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),
            bounds: wp.array(dtype=float)):   # out: bounds[0] = longest edge (atomic_max)
        # Longest edge in the frozen reference state. Grid-stride loop: coalesced
        # loads and only `boundsReduceThreads` conflicting atomics.
        t = wp.tid()
        num = edge_ids.shape[0]
        m = float(0.0)
        for e in range(t, num, boundsReduceThreads):
            m = wp.max(m, wp.length(prev_pos[edge_ids[e, 1]] - prev_pos[edge_ids[e, 0]]))
        wp.atomic_max(bounds, 0, m)

    @staticmethod
    @wp.func
    def detect_walk_radius(
            v: wp.int32,
            xv: wp.vec3,
            sweep_v: float,
            free_v: bool,
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),
            vert_edge_off: wp.array(dtype=wp.int32),
            vert_edge_ids: wp.array(dtype=wp.int32),
            edge_len: wp.array(dtype=float),
            bounds: wp.array(dtype=float)):
        # Walk radius shared by gather and expand (must be the identical value):
        # r_vt reaches every face relevant to v (see detect_expand), and the
        # edge-edge reach covers e's rc-CAPSULE from its nearer endpoint -- any
        # capsule point is within sqrt(halflen^2 + rc^2) of the nearer endpoint
        # (the caps are within rc of an endpoint), NOT halflen + rc. Radii use
        # only the primitive's OWN sweep (baseline swept-box semantics: own swept
        # box vs frozen other side) plus the frozen longest-edge bound; a global
        # displacement term would let one fast particle inflate every query.
        # (Pinned vertices keep the edge reach: an edge with one free endpoint
        # still collides.)
        # The longest-edge bound is CAPPED at edgeLenCap: one locally stretched
        # edge must not inflate every vertex's walk radius (buffer saturation =
        # silent candidate drops). The capped walk therefore only guarantees
        # coverage of primitives whose frozen edges are all <= edgeLenCap;
        # oversized primitives are excluded here AND in detect_expand (the
        # incident-edge skip below keeps the exact halflen term of a stretched
        # edge from unbounding the radius too) and handled exclusively by
        # the oversized fallback (scatter_oversized / oversized_pairs).
        L = wp.min(bounds[0], edgeLenCap)
        r_vt = float(0.0)
        if free_v:
            r_vt = sweep_v + d_offset + 0.578 * L + 1.0e-4
        r_ee = float(0.0)
        for k in range(vert_edge_off[v], vert_edge_off[v + 1]):
            e = vert_edge_ids[k]
            if edge_len[e] > edgeLenCap:
                continue  # handled by the oversized fallback instead
            other = edge_ids[e, 0] + edge_ids[e, 1] - v  # the endpoint that is not v
            halflen = 0.5 * wp.length(prev_pos[other] - xv)
            sw_e = wp.max(sweep_v, wp.length(pos[other] - prev_pos[other]))
            rc = sw_e + d_offset + 0.5 * L + 1.0e-4
            r_ee = wp.max(r_ee, wp.sqrt(halflen * halflen + rc * rc))
        return wp.max(r_vt, r_ee)

    @staticmethod
    @wp.kernel
    def detect_gather(
            grid: wp.uint64,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),   # frozen reference X (grid was built on it)
            pos: wp.array(dtype=wp.vec3),        # X + accumulated displacement
            grid_rc: wp.array2d(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),
            vert_edge_off: wp.array(dtype=wp.int32),
            vert_edge_ids: wp.array(dtype=wp.int32),
            edge_len: wp.array(dtype=float),
            bounds: wp.array(dtype=float),       # [0] = longest frozen edge (capped use)
            nbr_count: wp.array(dtype=wp.int32),  # out: accepted neighbors per vertex
            nbr_buf: wp.array(dtype=wp.int32),    # out: neighbor vertex ids
            overflow: wp.array(dtype=wp.int32)):
        # Self-collision BROADPHASE stage 1: ONE hash-grid walk per VERTEX caches
        # the nearby non-ring vertices. The walk is the latency-critical part
        # (grid traversal + scattered point loads), so it lives in this LEAN
        # kernel that runs at high occupancy; the register-heavy candidate
        # expansion reads the cache in a separate query-free kernel
        # (detect_expand). Replaces the old per-vertex AND per-edge LBVH queries
        # (32 KiB traversal stacks, ~50% occupancy cap, 63% of the substep).
        v = wp.tid()
        xv = prev_pos[v]
        sweep_v = wp.length(pos[v] - xv)
        r = Cloth.detect_walk_radius(v, xv, sweep_v, inv_mass[v] != 0.0,
                                     prev_pos, pos, edge_ids,
                                     vert_edge_off, vert_edge_ids,
                                     edge_len, bounds)
        if not (r < 0.5 * maxQueryExtent):  # NaN/blown-up guard (was maxQueryExtent box)
            nbr_count[v] = 0
            return
        base = v * maxNbr
        n = wp.int32(0)
        query = wp.hash_grid_query(grid, xv, r)
        u = wp.int32(0)
        while wp.hash_grid_query_next(query, u):
            if wp.length(prev_pos[u] - xv) > r:
                continue  # the grid over-returns to cell granularity
            if Cloth.within_ring_rc(grid_rc, v, u):
                continue  # every face/edge candidate via u would fail the ring cull
            if n < maxNbr:
                nbr_buf[base + n] = u
                n += 1
            else:
                wp.atomic_add(overflow, 0, 1)
        nbr_count[v] = n

    @staticmethod
    @wp.kernel
    def detect_expand(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),   # frozen reference X
            pos: wp.array(dtype=wp.vec3),        # X + accumulated displacement
            grid_rc: wp.array2d(dtype=wp.int32),
            tri_ids: wp.array2d(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),     # [numEdges, 2], va < vb
            vert_face_off: wp.array(dtype=wp.int32),  # CSR: vertex -> incident faces
            vert_face_ids: wp.array(dtype=wp.int32),
            vert_edge_off: wp.array(dtype=wp.int32),  # CSR: vertex -> incident edges
            vert_edge_ids: wp.array(dtype=wp.int32),
            face_longest: wp.array(dtype=float),  # longest frozen edge per face
            edge_len: wp.array(dtype=float),      # frozen length per edge
            bounds: wp.array(dtype=float),       # [0] = longest frozen edge (capped use)
            nbr_count: wp.array(dtype=wp.int32),  # cached neighbors (detect_gather)
            nbr_buf: wp.array(dtype=wp.int32),
            vt_count: wp.array(dtype=wp.int32),  # out: per-vertex candidate count (PRE-ZEROED, atomic)
            vt_buf: wp.array(dtype=wp.int32),    # out: candidate face ids
            ee_count: wp.array(dtype=wp.int32),  # out: per-edge candidate count (PRE-ZEROED, atomic)
            ee_buf: wp.array(dtype=wp.int32),    # out: candidate EDGE ids per edge
            overflow: wp.array(dtype=wp.int32)):
        # Self-collision BROADPHASE stage 2 (query-free): expand each cached
        # neighbor u into both candidate sets.
        #
        # Vertex-triangle: the old detect paired v with every frozen face whose
        # AABB touched v's swept box (own sweep + d_offset). Any triangle point
        # is within longest_edge/sqrt(3) of one of the triangle's VERTICES
        # (circumradius bound), so accepting neighbors within
        # r_vt = sweep(v) + d_offset + 0.578 * longest_edge and expanding them
        # into their incident faces covers every face that comes within
        # sweep(v) + d_offset of the vertex -- the pairs the truncation can act
        # on this substep (a face APPROACHING a slow vertex is protected, as
        # before, by the face's own vertices' queries and the edge-edge pass).
        # Faces are dedup'd (accepted only from their smallest in-range vertex)
        # so the narrowphase C<0 push (atomic_add, NOT idempotent) counts each
        # pair exactly once; the 2-ring cull is identical to the old BVH detect.
        #
        # Edge-edge: the old detect paired e one-sidedly with the edges of every
        # frozen face whose AABB touched e's swept box (own endpoint sweeps +
        # d_offset); one-sided discovery is safe because the narrowphase
        # truncates all 4 endpoints of a discovered pair. An edge f within
        # d_offset + sweep(e) of segment e has an ENDPOINT u inside e's capsule
        # of radius rc(e) = sweep(e) + d_offset + longest_edge/2 (endpoint gap
        # <= segment gap + halflen(f)), and that u is guaranteed to be in the
        # nearer e-endpoint's neighbor cache (see detect_walk_radius): each
        # vertex v expands every cached u into candidates (edges f incident to
        # u) for each of v's OWN incident edges e, appending to ee_buf[e]
        # through an atomic counter. Dedup: only the e-endpoint nearer to u
        # appends (both endpoints compute identical distances), and only from
        # f's smallest in-capsule endpoint -- each pair lands in ee_buf[e]
        # exactly once. The shared-vertex and pairwise 2-ring culls (the exact
        # predicates the old detect applied) run here, so the narrowphase
        # iterates clean EDGE ids.
        #
        # OVERSIZED partition: the longest-edge slack in both radii is capped at
        # edgeLenCap (see detect_walk_radius), so the coverage arguments above
        # only hold for candidates whose frozen edges fit under the cap. Any
        # oversized candidate face/edge -- and any OWN edge e that is oversized
        # (the gather cache no longer spans its capsule) -- is skipped here with
        # the same `> edgeLenCap` predicate the oversized fallback uses, so the
        # partition is exclusive: no pair is counted by both kernels (the C<0
        # push is not idempotent).
        #
        # PER-CANDIDATE slack: acceptance uses each candidate's OWN frozen
        # longest edge (face_longest / edge_len), not the capped global bound --
        # the Jung / halflen arguments are per-candidate, so this is the exact
        # original criterion, and it keeps a multi-layer squeeze (where the
        # global bound sits at the cap but local edges are at rest length ~1/3
        # of it) from tripling every acceptance radius and saturating the
        # buffers. The capped r_vt / rc below remain the coarse per-neighbor
        # pretests (and the gather radius, which must dominate them).
        v, j = wp.tid()  # (vertex, neighbor slot): slot j strides the neighbor cache
        n_nbr = wp.min(nbr_count[v], maxNbr)  # count may exceed capacity on overflow
        if j >= n_nbr:
            return  # vt_count/ee_count are pre-zeroed; appends are atomic

        xv = prev_pos[v]
        xpv = pos[v]
        sweep_v = wp.length(xpv - xv)

        L = wp.min(bounds[0], edgeLenCap)
        r_vt = float(0.0)
        if inv_mass[v] != 0.0:
            r_vt = sweep_v + d_offset + 0.578 * L + 1.0e-4

        v_row = grid_rc[v, 0]
        v_col = grid_rc[v, 1]

        # HOIST v's incident-edge data (loop-invariant across the cached
        # neighbors; reloading it per (neighbor x edge) was the kernel's main
        # memory traffic). Oversized own edges are excluded here once (the
        # oversized fallback owns their pairs), so the neighbor loop iterates a
        # compact list. All cached values are bitwise-identical to the loads
        # they replace, so the responsibility / dedup tie-breaks still agree
        # across endpoint threads.
        e_id = vec6i()
        e_other = vec6i()
        e_xo = mat6x3f()   # prev_pos[other]
        e_rc = vec6f()     # coarse capsule radius (sw_e + d_offset + 0.5*L + eps)
        e_swe = vec6f()    # own-sweep term sw_e (per-candidate rcf below)
        e_or = vec6i()     # grid_rc[other] (pairwise ring culls)
        e_oc = vec6i()
        n_e = wp.int32(0)
        ee_skip_r = float(0.0)
        for k in range(vert_edge_off[v], vert_edge_off[v + 1]):
            e = vert_edge_ids[k]
            elen = edge_len[e]
            if elen > edgeLenCap:
                continue  # capped cache no longer spans e's capsule; the oversized fallback owns e's pairs
            other = edge_ids[e, 0] + edge_ids[e, 1] - v
            xo = prev_pos[other]
            sw_e = wp.max(sweep_v, wp.length(pos[other] - xo))
            rc = sw_e + d_offset + 0.5 * L + 1.0e-4
            e_id[n_e] = e
            e_other[n_e] = other
            e_xo[n_e] = xo
            e_rc[n_e] = rc
            e_swe[n_e] = sw_e
            e_or[n_e] = grid_rc[other, 0]
            e_oc[n_e] = grid_rc[other, 1]
            n_e += 1
            # d(u, seg e) >= d(u, v) - len(e), so a neighbor beyond rc + len(e)
            # cannot pass e's du_seg <= rc pretest: max over the own edges gives
            # a whole-loop skip radius for far neighbors (pure pruning of cases
            # the original rejected at du_seg > rc; no acceptance change).
            ee_skip_r = wp.max(ee_skip_r, rc + elen)

        base = v * maxVT
        nbase = v * maxNbr

        # Per-NEIGHBOR incident-edge cache (lazy: filled on u's first own edge
        # that passes the capsule pretest, then reused across the remaining own
        # edges -- the same candidate f was previously re-fetched per own edge).
        f_id = vec6i()
        f_w = vec6i()      # f's other endpoint (f = (u, w))
        f_lf = vec6f()     # frozen length of f
        f_wr = vec6i()     # grid_rc[w]
        f_wc = vec6i()

        for ni in range(j, n_nbr, expandK):
            u = nbr_buf[nbase + ni]
            xu = prev_pos[u]
            duv = wp.length(xu - xv)

            # -- vertex-triangle expansion --
            if duv <= r_vt:
                # v's relevance region is the CAPSULE around its own displacement
                # segment [xv, pos(v)] (the old swept-box semantics; truncation
                # keeps the committed point on that segment), NOT the ball of
                # radius sweep+slack around xv: for a fast vertex the ball
                # over-accepts by ~(sweep/slack)^2 and saturates vtBuf exactly in
                # the ejection substeps of a multi-layer squeeze. duv <= r_vt
                # (the ball) stays as the coarse per-neighbor pretest.
                du_path = Cloth.point_segment_distance(xu, xv, xpv)
                for k in range(vert_face_off[u], vert_face_off[u + 1]):
                    face = vert_face_ids[k]
                    fl = face_longest[face]
                    if fl > edgeLenCap:
                        continue  # announced by its longest edge (scatter_oversized)
                    # Per-face acceptance radius (exact Jung slack for THIS face).
                    rf = d_offset + 0.578 * fl + 1.0e-4
                    if du_path > rf:
                        continue
                    i0 = tri_ids[face, 0]
                    i1 = tri_ids[face, 1]
                    i2 = tri_ids[face, 2]
                    if (Cloth.ring2_rc(v_row, v_col, grid_rc[i0, 0], grid_rc[i0, 1])
                            or Cloth.ring2_rc(v_row, v_col, grid_rc[i1, 0], grid_rc[i1, 1])
                            or Cloth.ring2_rc(v_row, v_col, grid_rc[i2, 0], grid_rc[i2, 1])):
                        continue
                    # Dedup: accept the face only from its smallest in-range vertex
                    # (same segment and per-face radius for all of the face's
                    # vertices, so the smallest in-range one is well defined
                    # within this thread).
                    if i0 < u and Cloth.point_segment_distance(prev_pos[i0], xv, xpv) <= rf:
                        continue
                    if i1 < u and Cloth.point_segment_distance(prev_pos[i1], xv, xpv) <= rf:
                        continue
                    if i2 < u and Cloth.point_segment_distance(prev_pos[i2], xv, xpv) <= rf:
                        continue
                    slot_vt = wp.atomic_add(vt_count, v, 1)
                    if slot_vt < maxVT:
                        vt_buf[base + slot_vt] = face
                    else:
                        wp.atomic_add(overflow, 0, 1)

            # -- edge-edge expansion: u gates its incident edges f, tested
            #    against each of v's (cached) incident edges e --
            if duv > ee_skip_r:
                continue  # u beyond every own edge's capsule pretest
            u_row = grid_rc[u, 0]
            u_col = grid_rc[u, 1]
            n_f = wp.int32(-1)  # u's incident-edge cache not filled yet
            for ke in range(n_e):
                other = e_other[ke]
                # (u == other is impossible: `other` is a mesh neighbor of v,
                # within grid ring 1, and the gather ring cull kept only
                # neighbors beyond ring 2.)
                xo = e_xo[ke]
                # Responsibility: only e's endpoint NEARER to u appends (ties -> va).
                # Both endpoint threads compute bitwise-identical distances here
                # (same subtractions), so exactly one appends.
                dou = wp.length(xu - xo)
                if duv > dou or (duv == dou and v > other):
                    continue  # (v != va <=> v > other under canonical va < vb)
                # Canonical (va, vb) argument order: the capsule and dedup
                # distances must be BITWISE identical no matter which endpoint's
                # thread evaluates them, or a pair could be double-counted or
                # dropped by both.
                pa = xv
                pb = xo
                if v > other:
                    pa = xo
                    pb = xv
                du_seg = Cloth.point_segment_distance(xu, pa, pb)
                if du_seg > e_rc[ke]:
                    continue  # u outside e's widest capsule (coarse pretest)
                if Cloth.ring2_rc(e_or[ke], e_oc[ke], u_row, u_col):
                    continue  # pairwise ring cull ((v,u) already culled by gather)
                if n_f < 0:
                    # First passing own edge: cache u's incident edges once
                    # (oversized candidates excluded here, same predicate as
                    # before: scatter_oversized appends them instead).
                    n_f = wp.int32(0)
                    for kk in range(vert_edge_off[u], vert_edge_off[u + 1]):
                        f = vert_edge_ids[kk]
                        lf = edge_len[f]
                        if lf > edgeLenCap:
                            continue
                        w = edge_ids[f, 0] + edge_ids[f, 1] - u
                        f_id[n_f] = f
                        f_w[n_f] = w
                        f_lf[n_f] = lf
                        f_wr[n_f] = grid_rc[w, 0]
                        f_wc[n_f] = grid_rc[w, 1]
                        n_f += 1
                sw_e = e_swe[ke]
                for kf in range(n_f):
                    w = f_w[kf]
                    # Skip pairs that share a vertex (adjacent edges never
                    # separate). u is not an endpoint of e (see above), so
                    # sharing a vertex means w is one of e's endpoints.
                    if w == v or w == other:
                        continue
                    # Per-candidate capsule radius (exact halflen slack for f).
                    rcf = sw_e + d_offset + 0.5 * f_lf[kf] + 1.0e-4
                    if du_seg > rcf:
                        continue
                    # Pairwise 2-ring cull: (va,u)/(vb,u) are done above; check
                    # both e endpoints against f's OTHER endpoint.
                    if (Cloth.ring2_rc(v_row, v_col, f_wr[kf], f_wc[kf])
                            or Cloth.ring2_rc(e_or[ke], e_oc[ke], f_wr[kf], f_wc[kf])):
                        continue
                    # Dedup: accept f only from its smallest in-capsule endpoint
                    # (same per-candidate radius from both endpoint threads).
                    # u == vd (f's larger endpoint) <=> u > w, and then vc == w.
                    if u > w and Cloth.point_segment_distance(prev_pos[w], pa, pb) <= rcf:
                        continue
                    # EXACT acceptance: keep the pair only if the FROZEN
                    # segment-segment gap is closable by the pair's OWN sweeps
                    # (d_offset + sweep(e) + sweep(f), the same swept-magnitude
                    # criterion scatter_oversized applies; rcf stays as the
                    # coarse pretest and the dedup radius above). BOTH sweeps
                    # matter: with sweep(e) alone, two edges approaching each
                    # other could each drop the pair from their own list while
                    # their combined motion closes the gap this substep. A
                    # dropped pair satisfies gap - sw_e - sw_f > d_offset, and
                    # interior displacement is a convex combination of endpoint
                    # displacements, so gap(t) > d_offset for the whole
                    # substep: no truncation can bind and no C<0 push is
                    # possible. The filter is symmetric in the pair and
                    # runs only in the thread the dedup selected, so the
                    # exactly-once accept becomes once-or-zero -- no pair can
                    # be double-counted. Canonical (vc, vd) = (min, max)
                    # argument order keeps the value bitwise-identical no
                    # matter which endpoint thread evaluates it.
                    xw = prev_pos[w]
                    qc = xu
                    qd = xw
                    if u > w:
                        qc = xw
                        qd = xu
                    sw_f = wp.max(wp.length(pos[u] - xu), wp.length(pos[w] - xw))
                    cse, csf, s_e, s_f = Cloth.closest_point_segment_segment(pa, pb, qc, qd)
                    if wp.length(cse - csf) > sw_e + sw_f + d_offset + 1.0e-4:
                        continue
                    e = e_id[ke]
                    slot = wp.atomic_add(ee_count, e, 1)
                    if slot < maxEE:
                        ee_buf[e * maxEE + slot] = f_id[kf]
                    else:
                        wp.atomic_add(overflow, 0, 1)

    @staticmethod
    @wp.kernel
    def collect_oversized(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),
            tri_ids: wp.array2d(dtype=wp.int32),
            edge_face_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2], -1 pad
            edge_len: wp.array(dtype=float),            # out: frozen length per edge
            oversized_ids: wp.array(dtype=wp.int32),    # out: compact oversized id list
            oversized_count: wp.array(dtype=wp.int32),  # out: list length (PRE-ZEROED)
            ov_f0: wp.array(dtype=wp.int32),            # out, per slot: announced faces (-1 = none)
            ov_f1: wp.array(dtype=wp.int32),
            ov_apex: wp.array(dtype=float),             # out, per slot: max announced apex-to-segment dist
            ov_sweep: wp.array(dtype=float),            # out, per slot: max endpoint sweep
            ov_pinned: wp.array(dtype=wp.int32)):       # out, per slot: both endpoints pinned
        # Cache every edge's FROZEN length (detect's per-candidate slack reads
        # it) and build the compact list of OVERSIZED edges (> edgeLenCap) the
        # fallback kernels iterate. `edge_len[..] > edgeLenCap` is THE partition
        # predicate: detect_walk_radius / detect_expand skip such primitives,
        # scatter_oversized / oversized_pairs handle exactly those (same
        # comparison on the same frozen lengths -> exclusive, no pair is ever
        # double-counted).
        #
        # Faces an oversized edge ANNOUNCES: E must be the face's LONGEST edge
        # (length ties -> smallest sorted vertex pair), so a face with several
        # oversized edges is announced by exactly one slot and each (vertex,
        # face) pair is appended once by the scatter (the C<0 push is not
        # idempotent). The <= 3 threads that evaluate a face compare
        # bitwise-identical lengths (same subtractions up to sign), so the
        # winner is unique -- and announcing from the LONGEST edge minimizes
        # the apex distance (height <= the face's shortest side).
        e = wp.tid()
        a = edge_ids[e, 0]
        b = edge_ids[e, 1]
        xa = prev_pos[a]
        xb = prev_pos[b]
        l = wp.length(xb - xa)
        edge_len[e] = l
        if not (l > edgeLenCap):
            return
        slot = wp.atomic_add(oversized_count, 0, 1)
        oversized_ids[slot] = e
        ov_sweep[slot] = wp.max(wp.length(pos[a] - xa), wp.length(pos[b] - xb))
        pinned = wp.int32(0)
        if inv_mass[a] == 0.0 and inv_mass[b] == 0.0:
            pinned = 1
        ov_pinned[slot] = pinned
        f0 = wp.int32(-1)
        f1 = wp.int32(-1)
        apex = float(0.0)
        for j in range(2):
            f = edge_face_ids[e, j]
            if f < 0:
                continue
            c3 = tri_ids[f, 0] + tri_ids[f, 1] + tri_ids[f, 2] - a - b
            xc = prev_pos[c3]
            la = wp.length(xc - xa)  # edge (a, c3)
            lb = wp.length(xc - xb)  # edge (b, c3)
            wins = wp.int32(0)
            if l > la or (l == la and Cloth.pair_less(a, b, wp.min(a, c3), wp.max(a, c3))):
                if l > lb or (l == lb and Cloth.pair_less(a, b, wp.min(b, c3), wp.max(b, c3))):
                    wins = 1
            if wins == 1:
                apex = wp.max(apex, Cloth.point_segment_distance(xc, xa, xb))
                if f0 < 0:
                    f0 = f
                else:
                    f1 = f
        ov_f0[slot] = f0
        ov_f1[slot] = f1
        ov_apex[slot] = apex

    @staticmethod
    @wp.kernel
    def face_longest_edges(
            prev_pos: wp.array(dtype=wp.vec3),
            tri_ids: wp.array2d(dtype=wp.int32),
            face_longest: wp.array(dtype=float)):  # out: longest frozen edge per face
        # Longest frozen edge per face: detect_expand's per-face Jung slack
        # reads it, and `> edgeLenCap` marks the faces the capped walk no longer
        # covers. Same lengths / same comparison as collect_oversized (|x-y| and
        # |y-x| are bitwise-equal), so the face partition is exactly "some edge
        # of the face is oversized", and collect_oversized's longest-edge
        # announcer is always an oversized edge.
        f = wp.tid()
        p0 = prev_pos[tri_ids[f, 0]]
        p1 = prev_pos[tri_ids[f, 1]]
        p2 = prev_pos[tri_ids[f, 2]]
        face_longest[f] = wp.max(wp.length(p1 - p0),
                                 wp.max(wp.length(p2 - p1), wp.length(p0 - p2)))

    @staticmethod
    @wp.func
    def pair_less(a0: wp.int32, a1: wp.int32, b0: wp.int32, b1: wp.int32) -> bool:
        # Lexicographic order on sorted vertex pairs == edge id order (edgeIds is
        # built sorted), used for deterministic tie-breaks on equal edge lengths.
        return a0 < b0 or (a0 == b0 and a1 < b1)

    @staticmethod
    @wp.kernel
    def scatter_oversized(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),   # frozen reference X
            pos: wp.array(dtype=wp.vec3),        # X + accumulated displacement
            grid_rc: wp.array2d(dtype=wp.int32),
            tri_ids: wp.array2d(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2], va < vb
            vert_edge_off: wp.array(dtype=wp.int32),
            vert_edge_ids: wp.array(dtype=wp.int32),
            edge_len: wp.array(dtype=float),
            oversized_ids: wp.array(dtype=wp.int32),
            oversized_count: wp.array(dtype=wp.int32),
            ov_f0: wp.array(dtype=wp.int32),
            ov_f1: wp.array(dtype=wp.int32),
            ov_apex: wp.array(dtype=float),
            ov_sweep: wp.array(dtype=float),
            ov_pinned: wp.array(dtype=wp.int32),
            vt_count: wp.array(dtype=wp.int32),   # in/out: appended ATOMICALLY (runs after expand)
            vt_buf: wp.array(dtype=wp.int32),
            ee_count: wp.array(dtype=wp.int32),
            ee_buf: wp.array(dtype=wp.int32),
            overflow: wp.array(dtype=wp.int32)):
        # FALLBACK broadphase, vertex side: OVERSIZED primitives (frozen edge >
        # edgeLenCap) are excluded from the capped vertex walk, so every VERTEX
        # scans the compact oversized list directly -- O(numParticles * k) with
        # a tiny L2-resident table, gated by an early-out when k == 0 (the
        # normal state). An earlier design walked the hash grid along each
        # oversized edge instead (one thread per edge, sampled sub-queries);
        # with a drag-front wad of thousands of stretched edges every such
        # thread iterated most of the wad through fat global-sweep radii, and
        # ONE substep cost ~450 ms. The scan needs no query geometry at all:
        # every accept below uses the vertex's OWN sweep plus the pair's own
        # primitives, so nothing global inflates anything.
        # MUST run AFTER detect_expand: both append atomically to the same
        # pre-zeroed counters, and stream order keeps the fallback's candidates
        # after the expand ones within each per-primitive list.
        v = wp.tid()
        n_over = oversized_count[0]
        if n_over == 0:
            return
        xv = prev_pos[v]
        sweep_v = wp.length(pos[v] - xv)
        free_v = inv_mass[v] != 0.0
        for j in range(n_over):
            E = oversized_ids[j]
            a = edge_ids[E, 0]
            b = edge_ids[E, 1]
            if v == a or v == b:
                continue
            xa = prev_pos[a]
            xb = prev_pos[b]
            du = Cloth.point_segment_distance(xv, xa, xb)
            # Per-pair reach: d_offset + E's own sweep + v's own sweep (+ the
            # slack converting segment distance into primitive distance).
            t0 = d_offset + ov_sweep[j] + sweep_v + 1.0e-4
            apex = ov_apex[j]
            if du > t0 + wp.max(edgeLenCap, apex):
                continue

            # -- vertex-triangle: E's announced faces vs v --
            # (pinned v gets no VT candidates, matching expand's r_vt = 0; the
            # apex slack converts distance-to-face into distance-to-segment,
            # the exact triangle test below is the accept.)
            if free_v and du <= t0 + apex:
                for jj in range(2):
                    f = ov_f0[j]
                    if jj == 1:
                        f = ov_f1[j]
                    if f < 0:
                        continue
                    i0 = tri_ids[f, 0]
                    i1 = tri_ids[f, 1]
                    i2 = tri_ids[f, 2]
                    if v == i0 or v == i1 or v == i2:
                        continue
                    if (Cloth.within_ring_rc(grid_rc, v, i0)
                            or Cloth.within_ring_rc(grid_rc, v, i1)
                            or Cloth.within_ring_rc(grid_rc, v, i2)):
                        continue
                    # Exact relevance (swept-magnitude criterion, >= expand's
                    # one-sided parity): v's own motion plus E's own sweep must
                    # be able to close the frozen gap to THIS face.
                    cp = Cloth.closest_point_on_triangle(
                        prev_pos[i0], prev_pos[i1], prev_pos[i2], xv)
                    if wp.length(xv - cp) > t0:
                        continue
                    slot = wp.atomic_add(vt_count, v, 1)
                    if slot < maxVT:
                        vt_buf[v * maxVT + slot] = f
                    else:
                        wp.atomic_add(overflow, 0, 1)

            # -- edge-edge: E vs v's SHORT incident edges --
            # (edgeLenCap slack: an edge within reach of E has an endpoint
            # within segment-gap + its own length of E's segment, so the
            # faster endpoint always accepts; the exact gap test is the
            # accept.)
            if du > t0 + edgeLenCap:
                continue
            for k in range(vert_edge_off[v], vert_edge_off[v + 1]):
                eu = vert_edge_ids[k]
                vc = edge_ids[eu, 0]
                vd = edge_ids[eu, 1]
                if vc == a or vc == b or vd == a or vd == b:
                    continue  # shares a vertex with E
                if edge_len[eu] > edgeLenCap:
                    continue  # oversized-vs-oversized: oversized_pairs
                w = vc + vd - v
                # Pairwise 2-ring culls, identical predicates to expand.
                if (Cloth.within_ring_rc(grid_rc, a, v)
                        or Cloth.within_ring_rc(grid_rc, b, v)
                        or Cloth.within_ring_rc(grid_rc, a, w)
                        or Cloth.within_ring_rc(grid_rc, b, w)):
                    continue
                # Exact relevance (swept-magnitude criterion, matching expand):
                # the frozen SEGMENT gap must be closable by the pair's own
                # sweeps. Canonical (vc, vd) / (xa, xb) order keeps the value
                # bitwise identical from either endpoint's thread.
                yc = prev_pos[vc]
                yd = prev_pos[vd]
                sw_eu = wp.max(wp.length(pos[vc] - yc), wp.length(pos[vd] - yd))
                cc1, cc2, s_e, t_e = Cloth.closest_point_segment_segment(xa, xb, yc, yd)
                if wp.length(cc1 - cc2) > d_offset + ov_sweep[j] + sw_eu + 1.0e-4:
                    continue
                if v > w:
                    # Dedup: w's thread appends iff w passes ITS accept (the
                    # mirrored expression on the same values -> bitwise equal
                    # across threads), so append from v only when w is out of
                    # range.
                    xw = prev_pos[w]
                    t0_w = d_offset + ov_sweep[j] + wp.length(pos[w] - xw) + 1.0e-4
                    if Cloth.point_segment_distance(xw, xa, xb) <= t0_w + edgeLenCap:
                        continue
                if inv_mass[vc] == 0.0 and inv_mass[vd] == 0.0:
                    # eu's narrowphase thread early-outs (both endpoints
                    # host-driven): hold the pair in E's OWN buffer instead so
                    # free E is still truncated against the pinned edge.
                    # (expand never writes ee_buf[E] for oversized E, so no
                    # double-count.)
                    if ov_pinned[j] == 0:
                        slot = wp.atomic_add(ee_count, E, 1)
                        if slot < maxEE:
                            ee_buf[E * maxEE + slot] = eu
                        else:
                            wp.atomic_add(overflow, 0, 1)
                    continue
                slot = wp.atomic_add(ee_count, eu, 1)
                if slot < maxEE:
                    ee_buf[eu * maxEE + slot] = E
                else:
                    wp.atomic_add(overflow, 0, 1)

    @staticmethod
    @wp.kernel
    def oversized_pairs(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            grid_rc: wp.array2d(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),
            edge_len: wp.array(dtype=float),
            oversized_ids: wp.array(dtype=wp.int32),
            oversized_count: wp.array(dtype=wp.int32),
            ov_sweep: wp.array(dtype=float),
            ee_count: wp.array(dtype=wp.int32),
            ee_buf: wp.array(dtype=wp.int32),
            overflow: wp.array(dtype=wp.int32)):
        # FALLBACK broadphase, oversized-vs-oversized side: two locally
        # stretched edges can cross mid-span with all four endpoints far from
        # the other segment (the classic "X" between long edges) -- neither the
        # capped vertex walk nor the vertex scan (which anchors on ENDPOINTS)
        # reaches those, so pair the oversized set directly: O(k) per oversized
        # thread over the compact list, ~free outside pathological stretch.
        # Each thread appends into its OWN buffer and its counterpart does the
        # converse, so both sides hold the pair once (the both-sided
        # narrowphase norm).
        E = wp.tid()
        if not (edge_len[E] > edgeLenCap):
            return
        a = edge_ids[E, 0]
        b = edge_ids[E, 1]
        if inv_mass[a] == 0.0 and inv_mass[b] == 0.0:
            return  # own narrowphase thread would early-out; the free side appends
        xa = prev_pos[a]
        xb = prev_pos[b]
        own_sweep = wp.max(wp.length(pos[a] - xa), wp.length(pos[b] - xb))
        ebase = E * maxEE
        n_over = oversized_count[0]
        for j in range(n_over):
            F = oversized_ids[j]
            if F == E:
                continue
            vc = edge_ids[F, 0]
            vd = edge_ids[F, 1]
            if vc == a or vc == b or vd == a or vd == b:
                continue
            if (Cloth.within_ring_rc(grid_rc, a, vc)
                    or Cloth.within_ring_rc(grid_rc, a, vd)
                    or Cloth.within_ring_rc(grid_rc, b, vc)
                    or Cloth.within_ring_rc(grid_rc, b, vd)):
                continue
            yc = prev_pos[vc]
            yd = prev_pos[vd]
            c1, c2, s_ab, t_cd = Cloth.closest_point_segment_segment(xa, xb, yc, yd)
            if wp.length(c1 - c2) > d_offset + own_sweep + ov_sweep[j] + 1.0e-4:
                continue  # frozen gap cannot close this substep
            slot = wp.atomic_add(ee_count, E, 1)
            if slot < maxEE:
                ee_buf[ebase + slot] = F
            else:
                wp.atomic_add(overflow, 0, 1)

    @staticmethod
    @wp.kernel
    def self_collision_truncate(
            tri_ids: wp.array2d(dtype=wp.int32),   # (numTris, 3)
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),  # frozen reference state X
            pos: wp.array(dtype=wp.vec3),       # X + accumulated displacement
            vt_count: wp.array(dtype=wp.int32),    # cached candidate counts (detect_expand)
            vt_buf: wp.array(dtype=wp.int32),      # cached candidate tri ids
            truncation_ts: wp.array(dtype=float),  # pre-filled with 1.0 (atomic_min)
            push: wp.array(dtype=wp.vec3)):        # pre-zeroed C<0 recovery (atomic_add)
        # Vertex-triangle NARROWPHASE (query-free): iterate the cached candidate faces
        # (detect_gather+detect_expand did the broadphase + 2-ring cull) and do the DIVIDE/TRUNCATE.
        # No BVH query here -> no 32 KiB traversal stack -> high occupancy.
        v = wp.tid()
        if inv_mass[v] == 0.0:
            return

        xv = prev_pos[v]
        dxv = pos[v] - xv

        n_cand = wp.min(vt_count[v], maxVT)  # scatter_oversized appends atomically; count may exceed capacity on overflow
        base = v * maxVT
        for kc in range(n_cand):
            face = vt_buf[base + kc]
            i0 = tri_ids[face, 0]
            i1 = tri_ids[face, 1]
            i2 = tri_ids[face, 2]

            # Narrowphase + DIVIDE: build one separating plane from the frozen state.
            p0 = prev_pos[i0]
            p1 = prev_pos[i1]
            p2 = prev_pos[i2]
            cp = Cloth.closest_point_on_triangle(p0, p1, p2, xv)
            n_hat = xv - cp
            d = wp.length(n_hat)
            if d < epsilon:
                continue
            n = n_hat / d
            c = d - d_offset  # signed gap (>= 0 at a feasible start)

            dt0 = pos[i0] - p0
            dt1 = pos[i1] - p1
            dt2 = pos[i2] - p2

            if c < 0.0:
                # Feasibility recovery: the pair starts already inside the offset,
                # so truncation cannot guarantee separation. Apply a one-sided push
                # to restore the offset (weighted by each side's approach), and skip
                # the truncation plane this substep.
                delta_v_n = wp.max(-wp.dot(n, dxv), 0.0)
                delta_t_n = wp.max(wp.max(wp.dot(n, dt0), wp.dot(n, dt1)),
                                   wp.max(wp.dot(n, dt2), 0.0))
                s = delta_v_n + delta_t_n
                if s == 0.0:
                    lmbd = 0.5
                else:
                    lmbd = wp.clamp(delta_t_n / s, 0.05, 0.95)
                depth = -c  # positive penetration
                wp.atomic_add(push, v, (1.0 - lmbd) * depth * n)
                tp = lmbd * depth / 3.0
                if inv_mass[i0] != 0.0:
                    wp.atomic_add(push, i0, -tp * n)
                if inv_mass[i1] != 0.0:
                    wp.atomic_add(push, i1, -tp * n)
                if inv_mass[i2] != 0.0:
                    wp.atomic_add(push, i2, -tp * n)
                continue

            # TRUNCATE: split the gap so the harder-approaching side gives more.
            delta_v_n = wp.max(-wp.dot(n, dxv), 0.0)
            delta_t_n = wp.max(wp.max(wp.dot(n, dt0), wp.dot(n, dt1)),
                               wp.max(wp.dot(n, dt2), 0.0))
            s = delta_v_n + delta_t_n
            if s == 0.0:
                lmbd = 0.5
            else:
                lmbd = wp.clamp(delta_t_n / s, 0.05, 0.95)
            p_plane = cp + (d_offset + lmbd * c) * n

            wp.atomic_min(truncation_ts, v, Cloth.planar_truncation_t(xv, dxv, n, p_plane))
            if inv_mass[i0] != 0.0:
                wp.atomic_min(truncation_ts, i0, Cloth.planar_truncation_t(p0, dt0, -n, p_plane))
            if inv_mass[i1] != 0.0:
                wp.atomic_min(truncation_ts, i1, Cloth.planar_truncation_t(p1, dt1, -n, p_plane))
            if inv_mass[i2] != 0.0:
                wp.atomic_min(truncation_ts, i2, Cloth.planar_truncation_t(p2, dt2, -n, p_plane))

    @staticmethod
    @wp.kernel(launch_bounds=(256, 4))  # cap regs (~80->64) -> 4 blocks/SM (67% occ) for
                                        # this hot query-free narrowphase; ptxas verified
                                        # to fit without local-memory spills.
    def self_collision_truncate_edges(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),  # frozen reference state X
            pos: wp.array(dtype=wp.vec3),       # X + accumulated displacement
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2], va < vb
            ee_count: wp.array(dtype=wp.int32),    # cached candidate counts (detect_expand)
            ee_buf: wp.array(dtype=wp.int32),      # cached candidate EDGE ids
            truncation_ts: wp.array(dtype=float),  # pre-filled with 1.0 (atomic_min)
            push: wp.array(dtype=wp.vec3)):        # pre-zeroed C<0 recovery (atomic_add)
        # Edge-edge NARROWPHASE (query-free): iterate the cached candidate EDGES
        # (detect_expand did the broadphase, the shared-vertex/2-ring culls and the dedup)
        # and do the DIVIDE/TRUNCATE. Catches folds where two edges cross with no vertex
        # near either face (the classic "X" configuration). No query -> high occupancy.
        # No pair-key dedup across THREADS: each edge processes every pair its own query
        # discovers (both sides usually find it); atomic_min makes the redundant
        # constraint order-independent, and the C<0 push touches only the thread's OWN
        # endpoints with complementary (1 - lmbd) weights, so there is no double-count.
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        if wa == 0.0 and wb == 0.0:
            return  # both endpoints host-driven

        xa = prev_pos[va]
        xb = prev_pos[vb]

        n_cand = wp.min(ee_count[e], maxEE)  # count may exceed capacity on overflow
        ebase = e * maxEE
        for ci in range(n_cand):
            f = ee_buf[ebase + ci]
            vc = edge_ids[f, 0]
            vd = edge_ids[f, 1]

            yc = prev_pos[vc]
            yd = prev_pos[vd]
            # DIVIDE: separating plane through the frozen-state closest points.
            ca, cb, se, sf = Cloth.closest_point_segment_segment(xa, xb, yc, yd)
            n_hat = ca - cb
            d = wp.length(n_hat)
            if d < epsilon:
                continue
            n = n_hat / d
            c = d - d_offset  # signed gap (>= 0 at a feasible start)

            # Per-ENDPOINT displacements: truncate each of the four endpoints by its
            # OWN displacement against the shared plane (newton-style per-vertex
            # atomic_min), not one blended-contact factor applied to both endpoints.
            d_va = pos[va] - xa
            d_vb = pos[vb] - xb
            d_vc = pos[vc] - yc
            d_vd = pos[vd] - yd

            # Normal approach of each side (toward the other), for the lambda split.
            delta_e_n = wp.max(wp.max(-wp.dot(n, d_va), -wp.dot(n, d_vb)), 0.0)  # ab side
            delta_f_n = wp.max(wp.max(wp.dot(n, d_vc), wp.dot(n, d_vd)), 0.0)    # cd side
            ssum = delta_e_n + delta_f_n
            if ssum == 0.0:
                lmbd = 0.5
            else:
                lmbd = wp.clamp(delta_f_n / ssum, 0.05, 0.95)

            if c < 0.0:
                # Feasibility recovery for an already-overlapping pair: push each edge's
                # OWN endpoints apart (weighted by the reference barycentric coord). The
                # two threads' (1 - lmbd) weights are complementary, so the pair
                # separates by ~depth without an atomic_add cross-push double-count.
                depth = -c
                if wa != 0.0:
                    wp.atomic_add(push, va, (1.0 - lmbd) * depth * (1.0 - se) * n)
                if wb != 0.0:
                    wp.atomic_add(push, vb, (1.0 - lmbd) * depth * se * n)
                continue

            # TRUNCATE: each endpoint against the shared plane by its own displacement
            # (planar_truncation_t is sign-symmetric in n, so both sides use +n).
            p_plane = cb + (d_offset + lmbd * c) * n
            if wa != 0.0:
                wp.atomic_min(truncation_ts, va, Cloth.planar_truncation_t(xa, d_va, n, p_plane))
            if wb != 0.0:
                wp.atomic_min(truncation_ts, vb, Cloth.planar_truncation_t(xb, d_vb, n, p_plane))
            if inv_mass[vc] != 0.0:
                wp.atomic_min(truncation_ts, vc, Cloth.planar_truncation_t(yc, d_vc, n, p_plane))
            if inv_mass[vd] != 0.0:
                wp.atomic_min(truncation_ts, vd, Cloth.planar_truncation_t(yd, d_vd, n, p_plane))

    @staticmethod
    @wp.kernel
    def count_self_contacts(
            mesh: wp.uint64,
            pos: wp.array(dtype=wp.vec3),
            grid_rc: wp.array2d(dtype=wp.int32),
            thresh: float,
            min_dist: wp.array(dtype=float),   # size 1, atomic_min (pre-filled large)
            n_violations: wp.array(dtype=wp.int32)):  # size 1, atomic_add (pre-zeroed)
        # DIAGNOSTIC ONLY (not part of the solve): vertex-triangle self-proximity on
        # the CURRENT geometry. Reports the smallest non-neighbor vertex-triangle gap
        # and how many are closer than `thresh` (separation violations). A correct
        # self-collision keeps this gap ~d_offset and n_violations ~0.
        v = wp.tid()
        xv = pos[v]
        r = thresh + d_offset
        lo = xv - wp.vec3(r, r, r)
        hi = xv + wp.vec3(r, r, r)
        query = wp.mesh_query_aabb(mesh, lo, hi)
        face = wp.int32(0)
        while wp.mesh_query_aabb_next(query, face):
            i0 = wp.mesh_get_index(mesh, 3 * face + 0)
            i1 = wp.mesh_get_index(mesh, 3 * face + 1)
            i2 = wp.mesh_get_index(mesh, 3 * face + 2)
            if (Cloth.within_ring_rc(grid_rc, v, i0)
                    or Cloth.within_ring_rc(grid_rc, v, i1)
                    or Cloth.within_ring_rc(grid_rc, v, i2)):
                continue
            cp = Cloth.closest_point_on_triangle(pos[i0], pos[i1], pos[i2], xv)
            dist = wp.length(xv - cp)
            wp.atomic_min(min_dist, 0, dist)
            if dist < thresh:
                wp.atomic_add(n_violations, 0, 1)

    @staticmethod
    @wp.kernel
    def strain_limit(
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),   # [numEdges, 2]
            rest_len: wp.array(dtype=float),
            deltas: wp.array(dtype=wp.vec3)):       # pre-zeroed, applied by add_deltas
        # Hard strain cap (Jacobi): pull the endpoints of any edge longer than
        # maxStrain * rest back toward the cap, mass-weighted, under-relaxed for
        # the shared-vertex Jacobi sum. Engages only past maxStrain (normal solve
        # strain is ~1-2%), so it is inert while draping/resting; under a violent
        # sphere drag it stops the fabric from stretching into the nonphysical
        # 25-100x "wads" that exploded self-collision density and frame time.
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        wsum = wa + wb
        if wsum == 0.0:
            return
        d = pos[vb] - pos[va]
        L = wp.length(d)
        rl = rest_len[e]
        if L < epsilon:
            return  # fully collapsed: no direction; neighbors resolve it
        Lmax = maxStrain * rl
        Lmin = minStrain * rl
        if L <= Lmax and L >= Lmin:
            return
        n = d / L
        # Step cap: with huge violations (a collider ejection can split an edge
        # 0.5 m across the sphere in one substep), an uncapped Jacobi pull on a
        # hub vertex (up to 8 edges) overshoots and diverges. Capping each edge's
        # per-iteration correction to a fraction of its rest length keeps the sum
        # bounded (~8 * 0.5 * rest * relax per iteration); convergence then comes
        # from iterations x substeps instead of step size.
        if L > Lmax:
            corr = wp.min((L - Lmax) * strainRelax, 0.5 * rl) / wsum
        else:
            # Compression floor: in-plane fabric buckles rather than compresses.
            # Under crushing (dragged grab plowing, sphere folding cloth against
            # the ground) edges collapsed to ~0.1x rest and triangles degenerated
            # -- the "inverted vertices" look -- and a collapsed region has
            # everything within contact range (a self-collision density bomb).
            # Pushing back to Lmin forces a buckle instead. Negative corr flips
            # the displacement direction below.
            corr = -wp.min((Lmin - L) * strainRelax, 0.5 * rl) / wsum
        if wa != 0.0:
            wp.atomic_add(deltas, va, n * (corr * wa))
        if wb != 0.0:
            wp.atomic_add(deltas, vb, -n * (corr * wb))

    @staticmethod
    @wp.kernel
    def ring_floor(
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),
            pair_ids: wp.array2d(dtype=wp.int32),  # [numRingPairs, 2] static list
            deltas: wp.array(dtype=wp.vec3)):      # shared with strain_limit's pass
        # Anti-fold-through: self-collision deliberately excludes the 2-ring
        # topological neighborhood (it would freeze bending), leaving a blind zone
        # where fabric can fold THROUGH itself at 1-2 cell scale -- flipped
        # triangles ("yellow face over red") and interpenetrated micro-regions
        # whose contact density hammers the broadphase. Enforce the SAME d_offset
        # separation the PDT applies everywhere else, on every non-edge vertex
        # pair within Chebyshev ring distance <= 2 (static list built in __init__;
        # rest distances are 0.015-0.042, all above d_offset, so this only fires
        # at fold-through; mesh edges are excluded -- the strain limiter's
        # compression floor governs those).
        t = wp.tid()
        i = pair_ids[t, 0]
        j = pair_ids[t, 1]
        wa = inv_mass[i]
        wb = inv_mass[j]
        wsum = wa + wb
        if wsum == 0.0:
            return
        d = pos[j] - pos[i]
        L = wp.length(d)
        if L >= d_offset or L < epsilon:
            return
        n = d / L
        corr = -wp.min((d_offset - L) * strainRelax, 0.5 * d_offset) / wsum
        if wa != 0.0:
            wp.atomic_add(deltas, i, n * (corr * wa))
        if wb != 0.0:
            wp.atomic_add(deltas, j, -n * (corr * wb))

    @staticmethod
    @wp.kernel
    def clamp_displacement(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),  # frozen penetration-free reference X
            pos: wp.array(dtype=wp.vec3),
            truncation_ts: wp.array(dtype=float),
            push: wp.array(dtype=wp.vec3)):
        # Displacement governor: bound each free particle's trial displacement
        # |pos - prev_pos| to maxDisplacement, after the XPBD solve and before
        # self-collision. A contract-valid substep travels far below it, so it is a
        # no-op in normal operation; it only sanitizes NaN/Inf and tames an instability
        # spike (XPBD overshoot at ke=1e9, or an accumulated push) so the swept
        # broadphase AABB cannot balloon and the query cannot degenerate to O(tris).
        # prev_pos (the frozen reference) is never modified, so the PDT contract holds.
        i = wp.tid()
        # Prologue: reset the self-collision accumulators for this substep here
        # (unconditional, before the anchor early-out) instead of paying two more
        # fill/zero launches -- clamp_displacement runs before any writer of
        # truncation_ts/push.
        truncation_ts[i] = 1.0
        push[i] = wp.vec3()
        if inv_mass[i] == 0.0:
            return  # anchors are host-driven
        dx = pos[i] - prev_pos[i]
        L = wp.length(dx)
        if not (L < 1.0e6):  # NaN/Inf -> snap back to the finite reference state
            pos[i] = prev_pos[i]
        elif L > maxDisplacement:
            pos[i] = prev_pos[i] + dx * (maxDisplacement / L)

    @staticmethod
    @wp.kernel
    def apply_truncation(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            truncation_ts: wp.array(dtype=float),
            push: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        if inv_mass[i] == 0.0:
            return  # leave host-driven anchors untouched
        # Bound the accumulated recovery push (see pushClamp): keeps a dense-overlap
        # substep from flinging a particle and pumping unbounded energy.
        p = push[i]
        lp = wp.length(p)
        if lp > pushClamp:
            p = p * (pushClamp / lp)
        pos[i] = prev_pos[i] + (pos[i] - prev_pos[i]) * truncation_ts[i] + p

    @staticmethod
    @wp.kernel
    def collider_project(
            dt: float,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            center: wp.array(dtype=wp.vec3),  # CURRENT substep sphere center (advanced in-graph)
            radius: wp.array(dtype=float),    # CURRENT substep sphere radius
            dc_arr: wp.array(dtype=wp.vec3),  # PER-SUBSTEP sphere translation (frame dc / numSubsteps)
            dr_arr: wp.array(dtype=float),    # PER-SUBSTEP radius change (frame dr / numSubsteps)
            dq_arr: wp.array(dtype=wp.quat)):  # PER-SUBSTEP rotation (frame dq ^ (1/numSubsteps))
        # Swept CCD against the moving / growing sphere: the analytic time-of-impact
        # catches approaching particles on the swept surface (no tunneling at any
        # drag speed); a static end-pose contact then resolves already-inside /
        # resting particles and applies depth-based Coulomb friction. Runs after the
        # constraint solve so its penetration-free result has the final say.
        # dc/dr/dq are per-substep slices: the sphere moves the frame delta over the
        # whole frame, so the swept velocity vc = dc/dt is the TRUE sphere speed
        # rather than numSubsteps x too fast (which flung hit particles and blew up).
        # They live in device arrays (set per frame by simulate) so the captured
        # graph stays frame-invariant.
        i = wp.tid()
        if inv_mass[i] == 0.0:
            return

        dc = dc_arr[0]
        dr = dr_arr[0]
        dq = dq_arr[0]

        sc = center[0]   # sphere center/radius at the START of this substep
        sr = radius[0]

        x = pos[i]
        vel_eff = (x - prev_pos[i]) / dt  # cloth substep velocity, before collision

        # Pass 1: swept sphere -> snap approaching particles onto the swept surface.
        hit, c = Cloth.swept_sphere_ccd(prev_pos[i], vel_eff, sc, sr, dc, dr, dt)
        if hit:
            n = wp.normalize(c - sc - dc)
            x = c + thickness * n

        # Pass 2: contact at this substep's end pose. The solved position's penetration
        # depth is the normal force signal for friction (rest contacts, where the
        # swept test does not fire because the per-substep travel is sub-margin).
        c_end = sc + dc
        r_end = sr + dr + thickness
        dv = x - c_end
        d = wp.length(dv)
        if d > epsilon and d < r_end:
            n = dv / d
            lambda_n = d - r_end  # < 0 when penetrating
            arm = r_end * n
            # dc/dq are per-substep, so the surface sweeps them over one substep dt.
            v_surf = (dc + wp.quat_rotate(dq, arm) - arm) / dt
            vrel = vel_eff - v_surf
            vt = vrel - wp.dot(n, vrel) * n
            lvt = wp.length(vt)
            x = c_end + r_end * n  # non-penetration: project onto the offset surface
            if lvt > epsilon:
                lambda_f = wp.max(0.4 * lambda_n, -lvt * dt)
                x += (vt / lvt) * lambda_f

        # Ground plane at y = thickness, with friction.
        if x[1] < thickness:
            n = Ground.NORMAL
            lambda_n = x[1] - thickness  # < 0 below ground
            vt = vel_eff - wp.dot(n, vel_eff) * n
            lvt = wp.length(vt)
            x = wp.vec3(x[0], thickness, x[2])
            if lvt > epsilon:
                lambda_f = wp.max(0.65 * lambda_n, -lvt * dt)
                x += (vt / lvt) * lambda_f

        pos[i] = x

    @staticmethod
    @wp.kernel
    def collider_project_edges(
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2]
            center: wp.array(dtype=wp.vec3),       # CURRENT substep sphere center
            radius: wp.array(dtype=float),
            dc_arr: wp.array(dtype=wp.vec3),  # PER-SUBSTEP sphere translation
            dr_arr: wp.array(dtype=float),    # PER-SUBSTEP radius change
            deltas: wp.array(dtype=wp.vec3)):      # pre-zeroed, applied by add_deltas
        # Edge-vs-sphere non-penetration: the vertex pass (collider_project) keeps every
        # PARTICLE out of the sphere, but under load the cloth can stretch until edges
        # span many times the particle spacing -- then the sphere passes BETWEEN
        # particles, through a stretched edge, with every vertex clear (fast lateral
        # drag through a large draped cloth: edges reached 100x rest length and the
        # sphere tunneled through the fabric). Project the closest point of each EDGE
        # segment out of the sphere, pushing the endpoints by their barycentric weights.
        # At rest this only trims the ~1e-4 chord sag between projected vertices.
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        if wa == 0.0 and wb == 0.0:
            return

        dc = dc_arr[0]
        dr = dr_arr[0]
        c_end = center[0] + dc            # same end pose as collider_project's Pass 2
        r_end = radius[0] + dr + thickness

        pa = pos[va]
        pb = pos[vb]
        ab = pb - pa
        ab2 = wp.dot(ab, ab)
        t = 0.0
        if ab2 > epsilon:
            t = wp.clamp(wp.dot(c_end - pa, ab) / ab2, 0.0, 1.0)
        cp = pa + t * ab
        dv = cp - c_end
        d = wp.length(dv)
        if d <= epsilon or d >= r_end:
            return
        n = dv / d
        depth = r_end - d

        # Move the contact point out by `depth` along n with the minimum-norm endpoint
        # displacements (weights 1-t and t), skipping pinned endpoints. Accumulated via
        # deltas + add_deltas so concurrent edges sharing a vertex read consistent pos;
        # any over-push from summing neighbors points AWAY from the sphere (safe) and
        # only occurs in the already-pathological stretched state.
        w = 0.0
        if wa != 0.0:
            w += (1.0 - t) * (1.0 - t)
        if wb != 0.0:
            w += t * t
        if w < epsilon:
            return
        # w >= 0.5 with both endpoints free; it only approaches 0 when the contact sits
        # next to a pinned endpoint -- clamp so the free endpoint's push stays bounded
        # (the remaining gap resolves over subsequent substeps).
        s = depth / wp.max(w, 0.1)
        if wa != 0.0:
            wp.atomic_add(deltas, va, (1.0 - t) * s * n)
        if wb != 0.0:
            wp.atomic_add(deltas, vb, t * s * n)

    @staticmethod
    @wp.kernel
    def advance_sphere(center: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
                       dc_arr: wp.array(dtype=wp.vec3), dr_arr: wp.array(dtype=float)):
        # Advance the collider pose by one substep slice. Runs (in the captured graph)
        # AFTER collider_project each substep, so replay k sees center_0 + k*dc and the
        # last substep lands exactly on the rendered end pose center_0 + numSubsteps*dc.
        center[0] = center[0] + dc_arr[0]
        radius[0] = radius[0] + dr_arr[0]

    @staticmethod
    @wp.kernel
    # copied from https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/xpbd/kernels.py
    def distance_constraints(
            dt: float,
            ke: float,
            kd: float,
            relaxation: float,
            offset: wp.int32,
            pos: wp.array(dtype=wp.vec3),
            prev_pos: wp.array(dtype=wp.vec3),
            inv_mass: wp.array(dtype=float),
            indices: wp.array2d(dtype=int),
            rest_lengths: wp.array(dtype=float),
            lambdas: wp.array(dtype=float),
            deltas: wp.array(dtype=wp.vec3),
    ):
        tid = offset + wp.tid()

        i = indices[tid, 0]
        j = indices[tid, 1]

        rest = rest_lengths[tid]

        xi = pos[i]
        xj = pos[j]

        pi = prev_pos[i]
        pj = prev_pos[j]

        xij = xi - xj
        # Substep displacement difference: the XPBD damping term is
        # gamma * grad_c . (x - x^n) (Macklin et al., eq. 26; newton uses
        # dt * grad_c . (vi - vj), identical since v = (x - x^n)/dt). Using
        # previous POSITIONS here (the old `vij = pi - pj`) is not a damping
        # term at all -- it is a constant ~rest-length bias scaled by
        # gamma = kd/(ke*dt), i.e. a spurious dt-DEPENDENT compression offset
        # with zero dissipation.
        vij = (xi - pi) - (xj - pj)

        l = wp.length(xij)
        if l < epsilon:
            return

        n = xij / l

        c = l - rest
        grad_c_xi = n
        grad_c_xj = -1.0 * n

        wi = inv_mass[i]
        wj = inv_mass[j]

        denom = wi + wj
        if denom == 0.0:
            return

        alpha = 1.0 / (ke * dt * dt)
        gamma = kd / (ke * dt)

        grad_c_dot_v = wp.dot(grad_c_xi, vij)
        dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha)

        dxi = wi * dlambda * grad_c_xi
        dxj = wj * dlambda * grad_c_xj

        lambdas[tid] = lambdas[tid] + dlambda

        wp.atomic_add(deltas, i, dxi * relaxation)
        wp.atomic_add(deltas, j, dxj * relaxation)

    @staticmethod
    @wp.kernel
    # copied from https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/xpbd/kernels.py
    def bending_constraints(
            dt: float,
            ke: float,
            kd: float,
            relaxation: float,
            offset: wp.int32,
            pos: wp.array(dtype=wp.vec3),
            prev_pos: wp.array(dtype=wp.vec3),
            inv_mass: wp.array(dtype=float),
            indices: wp.array2d(dtype=int),
            rest_angles: wp.array(dtype=float),
            lambdas: wp.array(dtype=float),
            deltas: wp.array(dtype=wp.vec3),
    ):
        tid = offset + wp.tid()

        # The edge lies between the particles indexed by 'k' and 'l',
        # and the two connected triangles with counter-clockwise winding: (i, k, l), (j, l, k).
        i = indices[tid, 0]
        j = indices[tid, 1]
        k = indices[tid, 2]
        l = indices[tid, 3]

        rest_angle = rest_angles[tid]

        x1 = pos[i]
        x2 = pos[j]
        x3 = pos[k]
        x4 = pos[l]

        p1 = prev_pos[i]
        p2 = prev_pos[j]
        p3 = prev_pos[k]
        p4 = prev_pos[l]

        w1 = inv_mass[i]
        w2 = inv_mass[j]
        w3 = inv_mass[k]
        w4 = inv_mass[l]

        n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1
        n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2

        n1_length = wp.length(n1)
        n2_length = wp.length(n2)

        if n1_length < epsilon or n2_length < epsilon:
            return

        n1 /= n1_length
        n2 /= n2_length

        # Clamp to [-1, 1]: dot of two unit normals can round just past 1.0 for
        # near-coplanar faces (creases during crumpling), and CUDA acosf(|x|>1)
        # returns NaN, which then spreads through deltas -> pos -> everything.
        cos_theta = wp.clamp(wp.dot(n1, n2), -1.0, 1.0)

        e = x4 - x3
        e_hat = wp.normalize(e)
        e_length = wp.length(e)

        derivative_flip = wp.sign(wp.dot(wp.cross(n1, n2), e))
        derivative_flip *= -1.0
        angle = wp.acos(cos_theta)

        grad_x1 = n1 * e_length * derivative_flip
        grad_x2 = n2 * e_length * derivative_flip
        grad_x3 = (n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)) * derivative_flip
        grad_x4 = (n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)) * derivative_flip
        c = angle - rest_angle
        denominator = (
                w1 * wp.length_sq(grad_x1)
                + w2 * wp.length_sq(grad_x2)
                + w3 * wp.length_sq(grad_x3)
                + w4 * wp.length_sq(grad_x4)
        )

        if denominator <= epsilon:
            return

        alpha = 1.0 / (ke * dt * dt)
        gamma = kd / (ke * dt)

        # XPBD damping: gamma * grad . (x - x^n), the substep DISPLACEMENT
        # (newton: dt * grad . v). The old form dotted the gradients with the
        # absolute previous POSITIONS -- translation-variant, zero actual
        # dissipation, and (scaled by gamma = kd/(ke*dt)) a bias that grows as
        # substeps drop. With the real term, kd damps the dihedral-angle rate:
        # this is what suppresses the residual bending oscillation ("undulation")
        # at lower substep counts, with the physically correct dt scaling.
        grad_dot_v = (wp.dot(grad_x1, x1 - p1) + wp.dot(grad_x2, x2 - p2)
                      + wp.dot(grad_x3, x3 - p3) + wp.dot(grad_x4, x4 - p4))
        dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha)

        delta0 = w1 * dlambda * grad_x1
        delta1 = w2 * dlambda * grad_x2
        delta2 = w3 * dlambda * grad_x3
        delta3 = w4 * dlambda * grad_x4

        lambdas[tid] = lambdas[tid] + dlambda

        wp.atomic_add(deltas, i, delta0 * relaxation)
        wp.atomic_add(deltas, j, delta1 * relaxation)
        wp.atomic_add(deltas, k, delta2 * relaxation)
        wp.atomic_add(deltas, l, delta3 * relaxation)

    @staticmethod
    @wp.kernel
    def add_deltas(
            pos: wp.array(dtype=wp.vec3),
            deltas: wp.array(dtype=wp.vec3)):
        # Applies AND re-zeroes: deltas is allocated zero and every producer only
        # atomic_adds, with add_deltas the sole consumer after each producer -- so
        # re-zeroing here keeps the all-zero-between-uses invariant and removes a
        # separate deltas.zero_() launch per iteration (19 launches/substep saved
        # across the solve, strain-limit and collider-edge blocks).
        tid = wp.tid()
        pos[tid] += deltas[tid]
        deltas[tid] = wp.vec3()

    @staticmethod
    @wp.kernel
    def update_velocity(
            dt: float,
            pos: wp.array(dtype=wp.vec3),
            prev_pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3)):
        tid = wp.tid()

        # pos is already collision-resolved (truncation + collider projection),
        # so velocity simply reflects the committed displacement.
        v = (pos[tid] - prev_pos[tid]) / dt
        mag = wp.length(v)
        # Sanitize first: the old one-sided `mag > maxVelocity` let NaN/Inf through
        # (IEEE comparisons with NaN are false), latching a dead particle forever.
        # Zeroing the velocity lets the sim self-heal (pos itself is repaired by the
        # displacement governor's isfinite reset before the next substep's solve).
        if not (mag < 1.0e6):
            v = wp.vec3(0.0, 0.0, 0.0)
        elif mag > maxVelocity:
            v *= maxVelocity / mag
        vel[tid] = v

    @staticmethod
    @wp.kernel
    def cast_ray(origin: wp.vec3,
                 direction: wp.vec3,
                 pos: wp.array(dtype=wp.vec3),
                 tri_ids: wp.array2d(dtype=wp.int32),
                 min_dist: wp.array(dtype=float)):
        # Picking, pass 1: brute-force Moller-Trumbore over all triangles,
        # atomic_min of the nearest hit distance. Runs once per CLICK (not per
        # frame), so a spatial structure is unnecessary -- this replaced the
        # wp.Mesh ray query and with it the last reason to keep (and refit) an
        # LBVH in the simulation loop.
        f = wp.tid()
        a = pos[tri_ids[f, 0]]
        e1 = pos[tri_ids[f, 1]] - a
        e2 = pos[tri_ids[f, 2]] - a
        h = wp.cross(direction, e2)
        det = wp.dot(e1, h)
        if wp.abs(det) < 1.0e-12:
            return
        inv = 1.0 / det
        s = origin - a
        u = wp.dot(s, h) * inv
        if u < 0.0 or u > 1.0:
            return
        q = wp.cross(s, e1)
        v = wp.dot(direction, q) * inv
        if v < 0.0 or u + v > 1.0:
            return
        t = wp.dot(e2, q) * inv
        if t > 1.0e-6:
            wp.atomic_min(min_dist, 0, t)

    @staticmethod
    @wp.kernel
    def cast_ray_face(origin: wp.vec3,
                      direction: wp.vec3,
                      pos: wp.array(dtype=wp.vec3),
                      tri_ids: wp.array2d(dtype=wp.int32),
                      min_dist: wp.array(dtype=float),
                      face_out: wp.array(dtype=wp.int32)):
        # Picking, pass 2: re-test and claim the face matching the winning
        # distance (ties race benignly -- any coincident face is a valid pick).
        f = wp.tid()
        d = min_dist[0]
        if d >= 1.0e6:
            return
        a = pos[tri_ids[f, 0]]
        e1 = pos[tri_ids[f, 1]] - a
        e2 = pos[tri_ids[f, 2]] - a
        h = wp.cross(direction, e2)
        det = wp.dot(e1, h)
        if wp.abs(det) < 1.0e-12:
            return
        inv = 1.0 / det
        s = origin - a
        u = wp.dot(s, h) * inv
        if u < 0.0 or u > 1.0:
            return
        q = wp.cross(s, e1)
        v = wp.dot(direction, q) * inv
        if v < 0.0 or u + v > 1.0:
            return
        t = wp.dot(e2, q) * inv
        if wp.abs(t - d) < 1.0e-6:
            face_out[0] = f

    def drag_anchor(self, screen_x, screen_y) -> Optional[Particle]:
        origin, direction = ray_from_screen(screen_x, screen_y)

        # Find the closest locked anchor hit by the ray
        near_anchor: Optional[Particle] = None
        min_anchor_dist = None
        host_pos = self.hostPos.numpy()
        for anchor in filter(lambda a: a.flags & AnchorFlag.LOCKED, self.anchors):
            dist = ray_to_sphere(origin, direction, wp.vec3f(host_pos[anchor.id]), 0.08)
            if dist and (not min_anchor_dist or dist[0] < min_anchor_dist):
                min_anchor_dist = dist[0]
                near_anchor = anchor

        # Find the closest triangle hit by the ray (brute force, click-time only)
        self._pickDist.fill_(1.0e6)
        self._pickFace.fill_(-1)
        wp.launch(kernel=Cloth.cast_ray,
                  dim=self.numTris,
                  inputs=[origin, direction, self.pos, self.triIds],
                  outputs=[self._pickDist])
        wp.launch(kernel=Cloth.cast_ray_face,
                  dim=self.numTris,
                  inputs=[origin, direction, self.pos, self.triIds, self._pickDist],
                  outputs=[self._pickFace])
        tri_id = int(self._pickFace.numpy()[0])
        min_tri_dist = float(self._pickDist.numpy()[0])
        hit = tri_id >= 0
        if not hit and not min_anchor_dist:
            return None

        # Check if with the sphere is hit by the ray before any hit anchor or triangle
        dist = ray_to_sphere(origin, direction, sphere.center, sphere.radius)
        if dist and (not near_anchor or dist[0] < min_anchor_dist) and (not hit or dist[0] < min_tri_dist):
            return None

        # Check if the hit locked anchor is closer than the hit triangle
        if near_anchor and (not hit or min_anchor_dist <= min_tri_dist):
            near_anchor.flags |= AnchorFlag.ACTIVE
            near_anchor.depth = min_anchor_dist
            near_anchor.screen = wp.vec2(screen_x, screen_y)
            return near_anchor.drag()

        # Otherwise return a new anchor for the triangle. Ids already claimed by
        # another anchor (primary or patch member) are off limits: their
        # hostInvMass currently reads 0.0, so grabbing one would record mass=0.0
        # and release would restore it as a permanent invisible pin.
        claimed = {a.id for a in self.anchors}
        for a in self.anchors:
            claimed.update(mid for mid, _, _ in a.group)
        particle_id = self.hostTriIds[tri_id, 0].item()
        if particle_id in claimed:
            return None
        inv_mass = self.hostInvMass.numpy()
        anchor = Particle(id=particle_id,
                          screen=wp.vec2(screen_x, screen_y),
                          mass=inv_mass[particle_id].item(),
                          depth=min_tri_dist)

        # Fingertip grab: pin every particle within fingerRadius of the picked one
        # and drag them as a conforming patch (world offsets from the primary at
        # grab time; see update_anchors), skipping claimed ids.
        center = host_pos[particle_id]
        d2 = np.einsum('ij,ij->i', host_pos - center, host_pos - center)
        for mid in np.nonzero(d2 < fingerRadius * fingerRadius)[0]:
            mid = int(mid)
            if mid == particle_id or mid in claimed:
                continue
            off = host_pos[mid] - center
            anchor.group.append((mid, inv_mass[mid].item(),
                                 wp.vec3f(off[0], off[1], off[2])))

        self.anchors.append(anchor)
        return anchor.drag()

    def update_anchors(self):
        inv_mass = self.hostInvMass.numpy()
        pos = self.hostPos.numpy()

        for particle in self.anchors:
            if not particle.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED):
                inv_mass[particle.id] = particle.mass
                for mid, mmass, _ in particle.group:
                    inv_mass[mid] = mmass
                continue

            if not particle.flags & AnchorFlag.ACTIVE:
                # host-side sphere push-out for released anchors (whole grab patch);
                # push depth |d| along the UNIT normal (r is not unit length), against
                # the sphere's committed-this-frame pose (center+dc, radius+dr) like
                # the ACTIVE branch below
                for mid in [particle.id] + [m for m, _, _ in particle.group]:
                    r = wp.vec3f(pos[mid]) - (sphere.center + sphere.dc)
                    d = wp.length(r) - (sphere.radius + sphere.dr) - thickness - particleRadius
                    if d < 0:
                        pos[mid] -= d * wp.normalize(r)
                continue

            inv_mass[particle.id] = 0.0
            screen_x, screen_y = particle.screen
            origin, direction = ray_from_screen(screen_x, screen_y)

            # Check intersection with the sphere
            dist = ray_to_sphere(origin, direction, sphere.center, sphere.radius + thickness)
            if dist:
                mid = (dist[0] + dist[1]) / 2.0
                if mid < particle.depth < dist[1]:
                    particle.depth = dist[1]
                elif dist[0] < particle.depth < mid:
                    particle.depth = dist[0]

            # Check intersection with the ground
            d = wp.dot(Ground.NORMAL, direction)
            if wp.abs(d) > 1e-3:
                depth = -wp.dot(Ground.NORMAL, origin) / d
                if camera.pos[1] >= 0.0 and 0.5 < depth < particle.depth:
                    particle.depth = depth
                elif (camera.pos[1] < 0.0
                      and wp.abs(pos[particle.id][1]) <= 2.0 * thickness
                      and depth > particle.depth):
                    particle.depth = depth

            pos[particle.id] = origin + direction * particle.depth
            # Move the grabbed patch with the primary. Pinned particles bypass the
            # collider, so members are pushed out of the sphere here (the primary
            # already avoids it via the ray-depth adjustment above). The grip is
            # CONFORMING, not rigid: each frame the stored offset relaxes toward
            # where the member actually ended up (i.e. the fabric may slip within
            # the grip). A rigid grab-time patch dragged across the sphere scrubs
            # a fixed plate over a curved surface with fabric sandwiched between
            # two hard constraints -- measured: it buckles the pinched fabric at
            # cell scale (flipped triangles, the "inverted" look) and the churn is
            # slow. Relaxation rate 0.15/frame keeps the grip firm at drag
            # timescales while letting the patch conform within ~half a second.
            new_group = []
            for mid, mmass, off in particle.group:
                inv_mass[mid] = 0.0
                p0 = pos[particle.id] + off
                r = wp.vec3f(p0) - (sphere.center + sphere.dc)
                d = wp.length(r) - (sphere.radius + sphere.dr + thickness + particleRadius)
                if d < 0.0:
                    p = p0 - d * wp.normalize(r)
                    # conform: fold half of the push-out into the stored offset, so
                    # the grip lets the fabric slip AROUND the sphere instead of
                    # scrubbing the grab-time shape against it (only the push-out
                    # conforms -- anchor motion never leaks in, so the grip stays
                    # firm during sustained drags away from the sphere)
                    off = off + 0.5 * (wp.vec3f(p) - wp.vec3f(p0))
                else:
                    p = p0
                pos[mid] = p
                new_group.append((mid, mmass, off))
            particle.group = new_group

        self.anchors[:] = [anchor for anchor in self.anchors if anchor.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED)]

    def simulate(self, steps=numSubsteps, iterations=numIterations, integrate=True, self_collision=True, solve_constraints=True):
        dt = timeStep / numSubsteps

        wp.copy(self.pos, self.hostPos)
        wp.copy(self.invMass, self.hostInvMass)

        # Capture one substep ONCE per flag combination and replay it across frames.
        # Everything per-frame (collider pose and delta slices) lives in device
        # arrays, so the graph is frame-invariant. This matters doubly with the
        # hash-grid build captured in-graph: the build's mempool alloc nodes make
        # graph INSTANTIATION cost ~0.5 s (a one-time hitch per flag combination),
        # while replaying it is ~0.06 ms -- and issuing the build on the stream
        # between replays instead serializes against the graph launches (~+2 ms per
        # substep), so in-graph + capture-once is the only fast arrangement.
        key = (iterations, integrate, self_collision, solve_constraints)
        graph = self._graphs.get(key)
        if graph is None:
            with wp.ScopedCapture() as capture:
                self.step(dt, iterations, integrate, self_collision, solve_constraints)
            graph = self._graphs[key] = capture.graph
        # Seed the collider pose at the frame START (outside the graph); advance_sphere
        # then walks it forward one substep slice per replay, landing on sphere.center+dc.
        # The delta slices feed the collider kernels through device memory.
        # Slice by the ACTUAL replay count: debug step modes call simulate(steps=1)
        # while Sphere.post_render commits the full frame delta -- slicing by
        # numSubsteps there would sweep only 1/numSubsteps of the sphere motion.
        inv_sub = 1.0 / float(steps)
        self.colliderDeltaC.fill_(sphere.dc * inv_sub)
        self.colliderDeltaR.fill_(sphere.dr * inv_sub)
        self.colliderDeltaQ.fill_(quat_fraction(sphere.dq, inv_sub))
        self.colliderCenter.fill_(sphere.center)
        self.colliderRadius.fill_(sphere.radius)
        self.selfCollisionOverflow.zero_()  # per-frame; accumulates over the replayed substeps
        for _ in range(steps):
            wp.capture_launch(graph)

        wp.copy(self.hostPos, self.pos)
        # hostPos is a PINNED cpu array, so the D2H copy above is issued as an
        # async cudaMemcpyAsync on the stream; .numpy() on a cpu array does NOT
        # sync the stream, so host readers (harnesses, update_anchors/drag_anchor)
        # would otherwise read the pinned buffer while the copy is still in flight,
        # yielding stale positions (looks like deep collider penetration during a
        # fast sphere drag). Sync so the buffer is complete before any host read.
        wp.synchronize_stream()

    def step(self, dt: float, iterations=numIterations, integrate=True, self_collision=True, solve_constraints=True,
             build_grid=True):
        # Phase 0: rebuild the hash grid over the start-of-substep points. Before
        # integrate runs, self.pos still holds the previous substep's end position,
        # which integrate is about to freeze into self.prevPos -> the grid bins the
        # EXACT reference positions the detect kernels distance-check against. A
        # build is ~0.07 ms of GPU work vs the ~1.2 ms the LBVH broadphase queries
        # used to take. HashGrid.build is graph-capturable on this stack (mempool
        # allocator; verified to rebuild correctly on every replay); the capture-
        # once structure in simulate() amortizes the expensive instantiation of
        # its alloc nodes. build_grid=False lets a caller that already built the
        # grid for this state skip it. The LBVH is no longer refit here -- it only
        # serves rendering/raycast/diagnostics (update_mesh / count_self_contacts
        # refit it on demand).
        if self_collision and build_grid:
            self.grid.build(self.pos, gridCellSize)

        if integrate:
            wp.launch(kernel=Cloth.integrate,
                      dim=self.numParticles,
                      inputs=[
                          dt,
                          self.invMass,
                          self.prevPos,
                          self.pos,
                          self.vel,
                      ])

        if solve_constraints:
            self.distConstraints.lambdas.zero_()
            self.bendConstraints.lambdas.zero_()
            # Bending stability guard. The bending constraint runs in XPBD's
            # compliance-dominated regime (alpha = 1/(ke*dt^2) ~ 8-32 >> the gradient
            # denominator ~ 1e-3), so its per-substep correction behaves like an
            # EXPLICIT force impulse whose stability product scales as
            # relaxation * ke * dt^2. The frame-level bending impulse is
            # dt-invariant (correction/substep ~ dt^2, substeps/frame ~ 1/dt), but at
            # dt > bendStabilityDt the per-substep overshoot crosses the Jacobi
            # stability boundary: a crumpled cloth then never settles -- it "undulates"
            # indefinitely (measured: settled drape frame-to-frame motion 2-5e-2 m at
            # 30 substeps vs 3e-4 at 60; with this guard 30 settles to ~1e-4). Scaling
            # the bending relaxation by (bendStabilityDt/dt)^2 keeps that stability
            # product exactly at its tuned-and-validated 60-substep value: a strict
            # no-op (factor 1.0) at >= 60 substeps, and the distance groups --
            # projection-dominated (denominator >> alpha), unconditionally stable --
            # are never touched.
            bend_relax_scale = min(1.0, (bendStabilityDt / dt) ** 2)
            for iteration in range(iterations):
                for offset, count, kernel, indices, rests, lambdas, ke, kd, parallel, relaxation in self.constraints:
                    if kernel is Cloth.bending_constraints:
                        relaxation = relaxation * bend_relax_scale
                    if parallel:
                        wp.launch(kernel=kernel,
                                  dim=count,
                                  inputs=[
                                      dt,
                                      ke,
                                      kd,
                                      relaxation,
                                      offset,
                                      self.pos,
                                      self.prevPos,
                                      self.invMass,
                                      indices,
                                      rests,
                                      lambdas,
                                      self.deltas,
                                  ])
                        wp.launch(kernel=Cloth.add_deltas,
                                  dim=self.numParticles,
                                  inputs=[self.pos, self.deltas])
                    else:
                        wp.launch(kernel=kernel,
                                  dim=count,
                                  inputs=[
                                      dt,
                                      ke,
                                      kd,
                                      relaxation,
                                      offset,
                                      self.pos,
                                      self.prevPos,
                                      self.invMass,
                                      indices,
                                      rests,
                                      lambdas,
                                      self.pos,
                                  ])

        # Strain limiter: hard-cap edge elongation at maxStrain before anything
        # downstream reads pos. Inert in normal states (solve strain ~1-2%); under
        # violent drags it prevents the 25-100x stretch "wads" that explode
        # self-collision candidate density (the >1s frame cliff) and read as the
        # cloth diverging. deltas is free here (solve batches re-zero before use).
        for _sl_it in range(strainLimitIters):
            wp.launch(kernel=Cloth.strain_limit,
                      dim=self.numEdges,
                      inputs=[self.invMass, self.pos, self.edgeIds, self.edgeRestLen],
                      outputs=[self.deltas])
            if _sl_it == 0:
                # ring_floor is a rare-fire guard (fold-through only): once per
                # block converges across substeps; 12x/substep cost ~3 ms/frame.
                wp.launch(kernel=Cloth.ring_floor,
                          dim=self.numRingPairs,
                          inputs=[self.invMass, self.pos, self.ringPairs],
                          outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])

        # Displacement governor: bound the post-solve trial displacement (and
        # sanitize NaN/Inf) before the self-collision broadphase reads pos, so an
        # instability spike cannot balloon the swept query AABB into an O(tris)
        # (multi-second) frame. No-op on contract-valid substeps.
        wp.launch(kernel=Cloth.clamp_displacement,
                  dim=self.numParticles,
                  inputs=[self.invMass, self.prevPos, self.pos,
                          self.truncation_ts, self.push])

        # Planar Divide-and-Truncate: clamp the net per-vertex displacement so no
        # vertex crosses a plane that separated it from a nearby triangle (vertex-
        # triangle) or a nearby edge (edge-edge) at the frozen reference state,
        # recovering feasibility where a pair already overlaps. Both passes write
        # the same truncation_ts (atomic_min) and push (atomic_add) buffers.
        if self_collision:
            # Global frozen-geometry bound for this substep (longest edge): the
            # grid query radii add min(bounds[0], edgeLenCap) -- moderate
            # stretch widens the search, while primitives stretched past the
            # cap go through the oversized fallback instead of inflating every
            # radius.
            self.detectBounds.zero_()
            wp.launch(kernel=Cloth.max_edge_length,
                      dim=boundsReduceThreads,
                      inputs=[self.prevPos, self.edgeIds],
                      outputs=[self.detectBounds])
            # Per-primitive frozen edge lengths (detect's per-candidate slack)
            # and the oversized partition for this substep's frozen state:
            # edges/faces with a frozen edge length > edgeLenCap are EXCLUDED
            # from the capped-radius vertex walk (gather/expand skip them) and
            # handled by the oversized fallback below instead.
            self.oversizedCount.zero_()
            wp.launch(kernel=Cloth.collect_oversized,
                      dim=self.numEdges,
                      inputs=[self.invMass, self.prevPos, self.pos,
                              self.edgeIds, self.triIds, self.edgeFaceIds],
                      outputs=[self.edgeLen, self.oversizedIds,
                               self.oversizedCount,
                               self.ovF0, self.ovF1, self.ovApex,
                               self.ovSweep, self.ovPinned])
            wp.launch(kernel=Cloth.face_longest_edges,
                      dim=self.numTris,
                      inputs=[self.prevPos, self.triIds],
                      outputs=[self.faceLongest])
            # DETECT (two stages; see the kernels for the radius derivations):
            # gather = one LEAN hash-grid walk per vertex -> neighbor cache;
            # expand = query-free expansion of the cache into both candidate
            # sets. vtCount/nbrCount are written for every vertex; eeCount is
            # filled by scattered atomic appends so it needs the pre-zero.
            self.eeCount.zero_()
            self.vtCount.zero_()
            wp.launch(kernel=Cloth.detect_gather,
                      dim=self.numParticles,
                      inputs=[self.grid.id, self.invMass, self.prevPos, self.pos,
                              self.gridRC, self.edgeIds,
                              self.vertEdgeOff, self.vertEdgeIds,
                              self.edgeLen, self.detectBounds],
                      outputs=[self.nbrCount, self.nbrBuf, self.selfCollisionOverflow])
            wp.launch(kernel=Cloth.detect_expand,
                      dim=(self.numParticles, expandK),
                      inputs=[self.invMass, self.prevPos, self.pos,
                              self.gridRC, self.triIds, self.edgeIds,
                              self.vertFaceOff, self.vertFaceIds,
                              self.vertEdgeOff, self.vertEdgeIds,
                              self.faceLongest, self.edgeLen,
                              self.detectBounds, self.nbrCount, self.nbrBuf],
                      outputs=[self.vtCount, self.vtBuf, self.eeCount, self.eeBuf,
                               self.selfCollisionOverflow])
            # FALLBACK: every vertex scans the compact oversized list (the
            # primitives the capped walk excluded) into the same candidate
            # buffers; a second edge-parallel pass pairs the oversized set
            # against itself (mid-span crossings). Both kernels append
            # atomically to the pre-zeroed counters (expand runs first in
            # stream order). Both early-out to ~one load when no edge is
            # stretched past edgeLenCap.
            wp.launch(kernel=Cloth.scatter_oversized,
                      dim=self.numParticles,
                      inputs=[self.invMass, self.prevPos, self.pos,
                              self.gridRC, self.triIds, self.edgeIds,
                              self.vertEdgeOff, self.vertEdgeIds,
                              self.edgeLen, self.oversizedIds,
                              self.oversizedCount,
                              self.ovF0, self.ovF1, self.ovApex,
                              self.ovSweep, self.ovPinned],
                      outputs=[self.vtCount, self.vtBuf, self.eeCount, self.eeBuf,
                               self.selfCollisionOverflow])
            wp.launch(kernel=Cloth.oversized_pairs,
                      dim=self.numEdges,
                      inputs=[self.invMass, self.prevPos, self.pos,
                              self.gridRC, self.edgeIds, self.edgeLen,
                              self.oversizedIds, self.oversizedCount,
                              self.ovSweep],
                      outputs=[self.eeCount, self.eeBuf,
                               self.selfCollisionOverflow])
            # NARROWPHASE (query-free -> high occupancy). Reads the caches, does the
            # DIVIDE/TRUNCATE; both passes write the same truncation_ts (atomic_min) and
            # push (atomic_add) buffers as before.
            wp.launch(kernel=Cloth.self_collision_truncate,
                      dim=self.numParticles,
                      inputs=[
                          self.triIds,
                          self.invMass,
                          self.prevPos,
                          self.pos,
                          self.vtCount,
                          self.vtBuf,
                      ],
                      outputs=[
                          self.truncation_ts,
                          self.push,
                      ])
            wp.launch(kernel=Cloth.self_collision_truncate_edges,
                      dim=self.numEdges,
                      inputs=[
                          self.invMass,
                          self.prevPos,
                          self.pos,
                          self.edgeIds,
                          self.eeCount,
                          self.eeBuf,
                      ],
                      outputs=[
                          self.truncation_ts,
                          self.push,
                      ])
            wp.launch(kernel=Cloth.apply_truncation,
                      dim=self.numParticles,
                      inputs=[self.invMass, self.prevPos, self.pos, self.truncation_ts, self.push])

        # Swept-CCD sphere projection + ground contact run last so their
        # penetration-free result has final say over the substep. The sphere's
        # per-frame delta (dc/dr/dq) is sliced to PER-SUBSTEP so the swept test sees
        # the true sphere velocity (dc/frame_dt) instead of numSubsteps x too fast
        # (which flung hit particles). collider_project reads the CURRENT substep
        # center/radius from self.collider* device arrays, which advance_sphere then
        # increments so each substep's end pose walks from the frame-start pose to the
        # rendered end pose (center+dc) -- no tunneling, no lag/penetration. The
        # per-substep delta slices are read from self.colliderDelta* device arrays
        # (set per frame by simulate) so the captured graph is frame-invariant.
        wp.launch(kernel=Cloth.collider_project,
                  dim=self.numParticles,
                  inputs=[
                      dt,
                      self.invMass,
                      self.prevPos,
                      self.pos,
                      self.colliderCenter,
                      self.colliderRadius,
                      self.colliderDeltaC,
                      self.colliderDeltaR,
                      self.colliderDeltaQ,
                  ])
        # Edge-vs-sphere pass: keeps the FABRIC (not just its vertices) out of the
        # sphere -- under load the cloth stretches until the sphere fits between
        # particles, and the vertex pass alone lets it tunnel through stretched edges.
        # Jacobi iterations: applying one edge's push can rotate a neighboring edge
        # into the sphere, and later passes catch it. The pass count is dt-scaled:
        # at the 60-substep reference dt, 2 passes left a 0.014 residual on one
        # extreme-stretch fast-drag trajectory and 3 measured 0 -- but a MORE
        # extreme chaotic rerun of the same gate (30 m/s, max edge 2.36 m) later
        # left 0.0003 with 3 passes, so the 60-substep floor is 5 (measured
        # 0.0000 there); at 30 substeps the per-substep sphere/cloth motion
        # doubles and convergence needs more passes (measured on the 9 m/s
        # 400x400 drag gate: 4 passes left 0.0001-0.0010, 10 measured 0.0000
        # across reruns). Each pass only pushes edges AWAY from the sphere, so
        # extra passes are safe; one pass is ~0.005 ms. deltas is free at this
        # point in the substep (only the constraint solve uses it).
        dt_ratio = max(1, int(math.ceil(dt / bendStabilityDt - 1.0e-6)))
        edge_passes = max(5, 8 * dt_ratio - 6)
        for _ in range(edge_passes):
            wp.launch(kernel=Cloth.collider_project_edges,
                      dim=self.numEdges,
                      inputs=[
                          self.invMass,
                          self.pos,
                          self.edgeIds,
                          self.colliderCenter,
                          self.colliderRadius,
                          self.colliderDeltaC,
                          self.colliderDeltaR,
                      ],
                      outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])
        # Post-collider strain limiting: the collider's nearest-surface ejection can
        # split an edge across the sphere within this substep; cap that stretch
        # BEFORE update_velocity derives velocities from it, or (pos-prev)/dt bakes
        # the stretch into vel and integrate re-creates it next substep (the runaway
        # that made drag frames explode).
        for _sl_it in range(strainLimitPostIters):
            wp.launch(kernel=Cloth.strain_limit,
                      dim=self.numEdges,
                      inputs=[self.invMass, self.pos, self.edgeIds, self.edgeRestLen],
                      outputs=[self.deltas])
            if _sl_it == 0:
                wp.launch(kernel=Cloth.ring_floor,
                          dim=self.numRingPairs,
                          inputs=[self.invMass, self.pos, self.ringPairs],
                          outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])
        # ...then let the edge-vs-sphere collider have FINAL say: the limiter's
        # step-capped pulls (<= ~0.5*rest per edge per iteration) can drag an edge
        # a few mm back into the sphere; without this closing block the frame ends
        # with fabric penetration (measured 0.003-0.027 at 9-30 m/s drags). The
        # re-ejection re-adds only that mm-scale stretch, which the next substep's
        # limiter absorbs -- alternating projections onto two compatible sets.
        for _ in range(3):
            wp.launch(kernel=Cloth.collider_project_edges,
                      dim=self.numEdges,
                      inputs=[
                          self.invMass,
                          self.pos,
                          self.edgeIds,
                          self.colliderCenter,
                          self.colliderRadius,
                          self.colliderDeltaC,
                          self.colliderDeltaR,
                      ],
                      outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])

        # Advance the collider pose only after every consumer of this substep's
        # pose (vertex pass, edge passes, post-limit closing block) has run.
        wp.launch(kernel=Cloth.advance_sphere,
                  dim=1,
                  inputs=[self.colliderCenter, self.colliderRadius,
                          self.colliderDeltaC, self.colliderDeltaR])

        wp.launch(kernel=Cloth.update_velocity,
                  dim=self.numParticles,
                  inputs=[
                      dt,
                      self.pos,
                      self.prevPos,
                  ],
                  outputs=[self.vel])

    def update_mesh(self):
        self.normals.zero_()
        wp.launch(kernel=Cloth.add_normals,
                  dim=self.numTris,
                  inputs=[self.pos, self.triIds, self.normals])
        wp.launch(kernel=Cloth.normalize_normals,
                  dim=self.numParticles,
                  inputs=[self.normals])

    def init(self, **kwargs):
        self._quad = gluNewQuadric()

        # Simulation state lives in PLAIN device arrays (identical to
        # init_headless): the captured graphs keep stable pointers, and CUDA
        # never holds a GL buffer mapped across the frame. Previously pos /
        # normals / triIds were permanently-mapped RegisteredGLBuffers that
        # glDrawElements sourced while still CUDA-mapped -- a pattern that can
        # force driver-level serialization between compute and raster. The GL
        # buffers are now refreshed in render() via a brief map -> copy -> unmap
        # (two ~2 MB device-device copies, ~0.1 ms).
        self.pos = wp.clone(self.restPos)
        self.normals = wp.zeros(self.numParticles, dtype=wp.vec3)
        self.triIds = wp.array(self.hostTriIds, dtype=wp.int32)

        host_pos = self.hostPos.numpy()
        glGenBuffers(1, ctypes.pointer(self.pos_gl_buffer))
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_gl_buffer)
        glBufferData(GL_ARRAY_BUFFER, host_pos.nbytes, host_pos, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._pos_gl = wp.RegisteredGLBuffer(int(self.pos_gl_buffer.value),
                                             flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                             fallback_to_copy=False)

        normals = np.zeros((self.numParticles, 3), dtype=np.float32)
        glGenBuffers(1, ctypes.pointer(self.normals_gl_buffer))
        glBindBuffer(GL_ARRAY_BUFFER, self.normals_gl_buffer)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._normals_gl = wp.RegisteredGLBuffer(int(self.normals_gl_buffer.value),
                                                 flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                                 fallback_to_copy=False)

        # The index buffer never changes: plain static GL upload, no interop.
        tri_ids = self.hostTriIds
        glGenBuffers(1, ctypes.pointer(self.triIds_gl_buffer))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.triIds_gl_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, tri_ids.nbytes, tri_ids, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        # Warm up the hash grid: the first build allocates the cell tables and
        # sort scratch (must happen outside CUDA-graph capture).
        self.grid.build(self.pos, gridCellSize)

        self.colliderCenter.fill_(sphere.center)
        self.colliderRadius.fill_(sphere.radius)

        wp.launch(kernel=Cloth.rest_distances,
                  dim=(self.distConstraints.count,),
                  inputs=[self.pos, self.distConstraints.indices, self.distConstraints.rests])

    def init_headless(self):
        # GL-free counterpart to init(): allocate pos/normals/triIds as plain Warp
        # arrays (no OpenGL interop) so the simulation can run headless for testing
        # and benchmarking. Simulation state and kernels are otherwise identical.
        self.pos = wp.clone(self.restPos)
        self.normals = wp.zeros(self.numParticles, dtype=wp.vec3)
        self.triIds = wp.array(self.hostTriIds, dtype=wp.int32)  # (numTris, 3)

        # Warm up the hash grid (first build allocates; matches init()).
        self.grid.build(self.pos, gridCellSize)

        self.colliderCenter.fill_(sphere.center)
        self.colliderRadius.fill_(sphere.radius)

        wp.launch(kernel=Cloth.rest_distances,
                  dim=(self.distConstraints.count,),
                  inputs=[self.pos, self.distConstraints.indices, self.distConstraints.rests])

    def pre_render(self, **kwargs):
        now = time.perf_counter()
        if _perf.t_last is not None:
            _perf.total += now - _perf.t_last
        _perf.t_last = now

        self.update_anchors()

        t0 = time.perf_counter()
        if state & (State.RUN | State.STEP):
            self.simulate(steps=numSubsteps if State.FRAME_STEP in state else 1,
                          integrate=State.SOLVER_STEP not in state,
                          self_collision=State.SELF_COLLISION in state and State.SOLVER_STEP not in state,
                          solve_constraints=State.CONTACT_STEP not in state)
        _perf.sim += time.perf_counter() - t0  # simulate() ends with a stream sync

        self.update_mesh()

    def render(self, **kwargs):
        t0 = time.perf_counter()
        # Upload the sim results to the GL buffers: brief map -> device copy ->
        # unmap (unmap orders the copy against subsequent GL reads).
        dst = self._pos_gl.map(dtype=wp.vec3, shape=(self.numParticles,))
        wp.copy(dst, self.pos)
        self._pos_gl.unmap()
        dst = self._normals_gl.map(dtype=wp.vec3, shape=(self.numParticles,))
        wp.copy(dst, self.normals)
        self._normals_gl.unmap()
        # Make sure all the CUDA operations have completed before calling OpenGL
        wp.synchronize_stream()

        glColor3f(1.0, 0.0, 0.0)
        glNormal3f(0.0, 0.0, -1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if State.WIREFRAME in state else GL_FILL)
        glLineWidth(1.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.pos_gl_buffer)
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, self.normals_gl_buffer)
        glNormalPointer(GL_FLOAT, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.triIds_gl_buffer)
        if State.CULL_FACE in state:
            glCullFace(GL_FRONT)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElements(GL_TRIANGLES, 3 * self.numTris, GL_UNSIGNED_INT, None)
            glCullFace(GL_BACK)
            glColor3f(1.0, 1.0, 0.0)
            glDrawElements(GL_TRIANGLES, 3 * self.numTris, GL_UNSIGNED_INT, None)
        else:
            glDisable(GL_CULL_FACE)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElements(GL_TRIANGLES, 3 * self.numTris, GL_UNSIGNED_INT, None)
            glEnable(GL_CULL_FACE)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        # kinematic particles / anchors
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        host_pos = self.hostPos.numpy()
        for anchor in filter(lambda a: a.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED), self.anchors):
            if anchor.flags & AnchorFlag.SELECTED:
                glColor3f(0 / 255, 145 / 255, 255 / 255)
            else:
                glColor3f(1.0, 1.0, 1.0)
            glPushMatrix()
            pos = host_pos[anchor.id]
            glTranslatef(pos[0], pos[1], pos[2])
            gluSphere(self._quad, 0.02, 40, 40)
            glPopMatrix()

        # Frame-time breakdown, printed once per _perf.PERIOD frames: `sim` is the
        # captured-graph replay (wall, incl. its stream sync), `draw` this method
        # (buffer upload + GL command issue), `other` = everything else in the
        # frame (update_mesh kernels, GL rasterization + page flip, input stack).
        _perf.draw += time.perf_counter() - t0
        _perf.frames += 1
        if _perf.frames == _perf.PERIOD:
            f = float(_perf.frames)
            total = _perf.total / f * 1e3
            sim = _perf.sim / f * 1e3
            draw = _perf.draw / f * 1e3
            ovf = int(self.selfCollisionOverflow.numpy()[0])  # last frame's drops
            print(f"[perf] fps={1e3 / total:5.1f}  frame={total:6.1f} ms  "
                  f"sim={sim:6.1f}  draw={draw:5.1f}  other={total - sim - draw:6.1f}"
                  + (f"  <<< candidate overflow={ovf} (contacts dropped)" if ovf else ""),
                  flush=True)
            _perf.frames = 0
            _perf.total = _perf.sim = _perf.draw = 0.0

    def reset(self):
        self.vel.zero_()
        wp.copy(self.pos, self.restPos)
        wp.copy(self.prevPos, self.restPos)
        wp.copy(self.hostPos, self.restPos)
        for anchor in self.anchors:
            anchor.flags &= ~(AnchorFlag.ACTIVE | AnchorFlag.LOCKED)
        # self.update_anchors()


class Ground(Input):

    NORMAL = wp.vec3(0.0, 1.0, 0.0)

    def __init__(self):
        super().__init__("ground")
        num_tiles = 30
        tile_size = 0.5
        vertices = np.zeros(3 * 4 * num_tiles * num_tiles, dtype=float)
        colors = np.zeros(3 * 4 * num_tiles * num_tiles, dtype=float)
        square = [[0, 0], [0, 1], [1, 1], [1, 0]]
        r = num_tiles / 2.0 * tile_size
        for xi in range(num_tiles):
            for zi in range(num_tiles):
                x = (-num_tiles / 2.0 + xi) * tile_size
                z = (-num_tiles / 2.0 + zi) * tile_size
                p = xi * num_tiles + zi
                for i in range(4):
                    q = 4 * p + i
                    px = x + square[i][0] * tile_size
                    pz = z + square[i][1] * tile_size
                    vertices[3 * q] = px
                    vertices[3 * q + 2] = pz
                    col = 0.4
                    if (xi + zi) % 2 == 1:
                        col = 0.8
                    pr = math.sqrt(px * px + pz * pz)
                    d = max(0.0, 1.0 - pr / r)
                    col = col * d
                    for j in range(3):
                        colors[3 * q + j] = col
        self.colors = colors
        self.vertices = vertices

    def render(self, frame, time):
        glColor3f(1.0, 1.0, 1.0)
        glNormal3f(0.0, 1.0, 0.0)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glColorPointer(3, GL_FLOAT, 0, self.colors)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDrawArrays(GL_QUADS, 0, math.floor(len(self.vertices) / 3))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)


class Sphere(Input):

    def __init__(self, center: wp.vec3, radius: float):
        super().__init__("sphere")
        # TODO: use a wp.transformation
        self.center = center
        self.quat = quat_from_unit_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 1.0, 0.0))
        self.radius = radius
        self.dc = wp.vec3()
        self.dq = wp.quat_identity()
        self.dr = 0.0
        self._quad = None

    def init(self, **kwargs):
        self._quad = gluNewQuadric()

    def translate(self, dc: wp.vec3):
        c = self.center + dc
        if c[1] < 0.0:
            c[1] = 0.0
            dc = c - self.center
        self.dc += dc

    def rotate(self, dq: wp.quat):
        self.dq = dq * self.dq

    def resize(self, dr: float):
        self.dr += dr

    def render(self, **kwargs):
        if (not state & (State.RUN | State.STEP)
                and (wp.length(self.dc) > 0.0 or self.dr != 0.0)):
            self.draw(self.center, self.radius, self.quat, fill=False)
        self.draw(self.center + self.dc, self.radius + self.dr, self.dq * self.quat)

    def draw(self, pos: wp.vec3, rad: float, quat: wp.quat, fill=True, line=True):
        rot = wp.quat_to_matrix(quat)
        glPushMatrix()
        glMultMatrixf(wp.mat44(
            rot[0, 0], rot[1, 0], rot[2, 0], 0.0,
            rot[0, 1], rot[1, 1], rot[2, 1], 0.0,
            rot[0, 2], rot[1, 2], rot[2, 2], 0.0,
            pos[0],    pos[1],    pos[2],    1.0,
        ))
        if fill:
            glColor3f(0.8, 0.8, 0.8)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            gluSphere(self._quad, rad, 40, 40)
        if line:
            glColor3f(0.75, 0.75, 0.75)
            glLineWidth(2.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            gluSphere(self._quad, rad, 40, 40)
        glPopMatrix()

    def post_render(self, **kwargs):
        if state & (State.RUN | State.STEP):
            # One simulation step has run,
            # so changes have been integrated and can be reset
            self.center += self.dc
            self.quat = self.dq * self.quat
            self.radius += self.dr
            self.dc = wp.vec3()
            self.dq = wp.quat_identity()
            self.dr = 0.0


class Camera(Input):

    UP = wp.vec3(0.0, 1.0, 0.0)
    RIGHT = wp.vec3(1.0, 0.0, 0.0)
    EPS = 0.000001
    MIN_DISTANCE = 2.0
    MAX_DISTANCE = 100.0

    def __init__(self):
        super().__init__("camera")
        self.pos = wp.vec3(0.0, 1.0, 5.0)
        self.forward = wp.vec3(0.0, 0.0, -1.0)
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.right = wp.cross(self.forward, self.up)
        self.target = wp.vec3(0.0, 1.0, 0.0)
        self.quat = quat_from_unit_vectors(self.up, Camera.UP)

    def init(self, width, height):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, float(width) / float(height), 0.01, 1000.0)

    def pre_render(self, **kwargs):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.pos[0], self.pos[1], self.pos[2],
            self.pos[0] + self.forward[0], self.pos[1] + self.forward[1], self.pos[2] + self.forward[2],
            self.up[0], self.up[1], self.up[2])

    def orbit(self, dx, dy, gain=0.001):
        self.rotate(-wp.TAU * dx, -wp.TAU * dy, gain)

    def rotate(self, dth, dph, gain=1.0):
        vec = self.pos - self.target
        vec = wp.quat_rotate(self.quat, vec)

        radius = wp.length(vec)
        theta = wp.atan2(vec[0], vec[2])
        theta += dth * gain

        phi = wp.acos(wp.clamp(vec[1] / radius, - 1.0, 1.0))
        phi += dph * gain
        phi = wp.max(Camera.EPS, wp.min(wp.pi - Camera.EPS, phi))

        sin_phi_radius = wp.sin(phi) * radius
        vec = wp.vec3(sin_phi_radius * wp.sin(theta), wp.cos(phi) * radius, sin_phi_radius * wp.cos(theta))
        vec = wp.quat_rotate_inv(self.quat, vec)

        self.pos = self.target + vec
        self.forward = self.target - self.pos
        self.forward = wp.normalize(self.forward)
        self.right = wp.cross(self.forward, Camera.UP)
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)
        self.up = wp.normalize(self.up)

    def dolly(self, delta, gain=0.1):
        dist = wp.length(self.target - self.pos)
        delta *= gain * dist / Camera.MIN_DISTANCE
        delta = dist - wp.clamp(dist - delta, Camera.MIN_DISTANCE, Camera.MAX_DISTANCE)
        self.pos += delta * self.forward

    def dolly_scale(self, scale):
        dist = wp.length(self.target - self.pos) / scale
        dist = wp.clamp(dist, Camera.MIN_DISTANCE, Camera.MAX_DISTANCE)
        self.pos = self.target - dist * self.forward

    def track(self, dx, dy, gain=0.001):
        gain *= wp.length(self.target - self.pos) / Camera.MIN_DISTANCE
        track_x = gain * dx * self.right
        track_y = gain * dy * self.up
        self.pos -= track_x
        self.pos += track_y
        self.target -= track_x
        self.target += track_y


class Mouse(shaderbang.input.Mouse):

    def __init__(self):
        super().__init__("mouse")
        self.particle: Optional[Particle] = None

    def pre_render(self, **kwargs):
        if self.deltaW != 0:
            camera.dolly(self.deltaW, gain=0.1)
        if self.click and not self.particle:
            if self.button == EV_KEY.BTN_LEFT:
                self.particle = cloth.drag_anchor(self.mouseX, self.mouseY)
        elif self.drag:
            if self.particle:
                self.particle.screen = wp.vec2(self.mouseX, self.mouseY)
            elif self.button == EV_KEY.BTN_LEFT:
                camera.orbit(self.deltaX, self.deltaY, 0.5 / self.resolution[1])
            elif self.button == EV_KEY.BTN_RIGHT:
                camera.track(self.deltaX, self.deltaY, gain=0.001)
        elif self.particle:
            self.particle.flags &= ~AnchorFlag.ACTIVE
            if keyboard.down(any, EV_KEY.KEY_LEFTCTRL, EV_KEY.KEY_RIGHTCTRL):
                self.particle.flags |= AnchorFlag.LOCKED
            if self.particle.click():
                self.particle.flags ^= AnchorFlag.SELECTED
            self.particle = None


class ParticleSlot(TouchSlot):

    def __init__(self):
        super().__init__()
        self.particle: Optional[Particle] = None


class Touchscreen(shaderbang.input.MultiTouch[ParticleSlot]):

    def __init__(self):
        super().__init__("touchscreen", ParticleSlot)

    def holroyd_trackball(self, screen_x, screen_y) -> wp.vec3:
        """
        https://www.khronos.org/opengl/wiki/Object_Mouse_Trackball
        :param screen_x: image plan abscissa
        :param screen_y: image plan ordinate
        :return: the Holroyd's trackball 3D projection
        """
        width, height = self.resolution
        vec = wp.vec3(screen_x / width * 2 - 1.0, - screen_y / height * 2 + 1.0, 0.0)
        len2 = wp.length_sq(vec)
        vec[2] = 0.5 / wp.sqrt(len2) if len2 > 0.5 else wp.sqrt(1.0 - len2)
        return vec

    def pre_render(self, **kwargs):
        slots: list[ParticleSlot] = []
        for slot in self.slots:
            if slot.touch:
                if slot.particle:
                    slot.particle.drop()
                slot.particle = cloth.drag_anchor(slot.touchX, slot.touchY)
            elif slot.drag:
                if slot.particle:
                    slot.particle.screen = wp.vec2(slot.touchX, slot.touchY)
                else:
                    slots.append(slot)
            elif slot.particle:
                slot.particle.flags &= ~AnchorFlag.ACTIVE
                if keyboard.down(any, EV_KEY.KEY_LEFTCTRL, EV_KEY.KEY_RIGHTCTRL):
                    slot.particle.flags |= AnchorFlag.LOCKED
                if slot.particle.click():
                    slot.particle.flags ^= AnchorFlag.SELECTED
                slot.particle = None

        n = len(slots)
        if n == 1:
            slot = slots[0]

            u = quat_from_unit_vectors(Camera.UP, camera.up)
            v = quat_from_unit_vectors(Camera.RIGHT, camera.right)
            quat = wp.mul(u, v)

            vec1 = self.holroyd_trackball(slot.prevX, slot.prevY)
            vec2 = self.holroyd_trackball(slot.touchX, slot.touchY)

            vec1 = wp.quat_rotate(quat, vec1)
            vec2 = wp.quat_rotate(quat, vec2)

            theta = wp.atan2(wp.dot(wp.cross(vec2, vec1), camera.UP), wp.dot(vec2, vec1))
            camera.rotate(wp.PI * theta, - wp.TAU * slots[0].deltaY * 0.5 / self.resolution[1])

        elif n > 1:
            cx = cy = dx = dy = 0.0
            for slot in slots:
                cx += slot.touchX
                dx += slot.deltaX
                cy += slot.touchY
                dy += slot.deltaY
            cx /= n
            cy /= n
            dx /= n
            dy /= n

            for slot in slots:
                slot.prevX += dx
                slot.prevY += dy

            scale, theta, tx, ty = homothety_and_rotation(slots, center=(cx, cy))

            if n < 4:
                camera.track(dx, dy)
                camera.dolly_scale(scale)
                camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)
            elif n == 4:
                dcx = 0.002 * dx * camera.right
                dcy = -0.002 * dy * camera.up
                dcz = 1.5 * (1.0 - scale) * camera.forward
                sphere.translate(dcx + dcy + dcz)
            else:
                qr = wp.quat_from_axis_angle(camera.forward, wp.PI * theta)
                qx = wp.quat_from_axis_angle(camera.up, wp.TAU * dx / self.resolution[1])
                qy = wp.quat_from_axis_angle(camera.right, wp.TAU * dy / self.resolution[1])
                sphere.rotate(qx * qy * qr)
                dr = wp.clamp(sphere.radius * scale, 0.25, 1.25) - sphere.radius
                sphere.resize(dr)


class Trackpad(shaderbang.input.MultiTouch[TouchSlot]):

    def __init__(self):
        super().__init__("trackpad")

    def pre_render(self, **kwargs):
        slots = [slot for slot in self.slots if slot.drag]
        n = len(slots)
        if n > 1:
            cx = cy = dx = dy = 0.0
            for slot in slots:
                cx += slot.touchX
                dx += slot.deltaX
                cy += slot.touchY
                dy += slot.deltaY
            cx /= n
            cy /= n
            dx /= n
            dy /= n

            for slot in slots:
                slot.prevX += dx
                slot.prevY += dy

            scale, theta, tx, ty = homothety_and_rotation(slots, center=(cx, cy))

            if n == 2:
                camera.orbit(dx, dy, 0.5 / self.resolution[1])
                camera.dolly_scale(scale)
                camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)
            elif n == 3:
                camera.track(dx, dy)
                camera.dolly_scale(scale)
                camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)
            elif n == 4:
                dcx = 0.002 * dx * camera.right
                dcy = -0.002 * dy * camera.up
                dcz = 1.5 * (1.0 - scale) * camera.forward
                sphere.translate(dcx + dcy + dcz)
            else:
                qr = wp.quat_from_axis_angle(camera.forward, wp.PI * theta)
                qx = wp.quat_from_axis_angle(camera.up, wp.TAU * dx / self.resolution[1])
                qy = wp.quat_from_axis_angle(camera.right, wp.TAU * dy / self.resolution[1])
                sphere.rotate(qx * qy * qr)
                dr = wp.clamp(sphere.radius * scale, 0.25, 1.25) - sphere.radius
                sphere.resize(dr)


class Keyboard(shaderbang.input.Keyboard):

    def __init__(self):
        super().__init__("keyboard")

    def pre_render(self, **kwargs):
        global state
        if self.pressed(EV_KEY.KEY_C):
            state ^= State.SELF_COLLISION
        if self.pressed(EV_KEY.KEY_P):
            state ^= State.RUN
        if self.pressed(EV_KEY.KEY_R):
            wp.synchronize_stream()
            cloth.reset()
        if self.pressed(EV_KEY.KEY_S):
            match state & STEPS:
                case State.FRAME_STEP:
                    step = State.SMALL_STEP
                case State.SMALL_STEP:
                    step = State.CONTACT_STEP
                case State.CONTACT_STEP | State.SOLVER_STEP:
                    step = State.FRAME_STEP
            state &= ~STEPS
            state |= step
        if self.pressed(EV_KEY.KEY_F):
            state ^= State.CULL_FACE
        if self.pressed(EV_KEY.KEY_W):
            state ^= State.WIREFRAME
        if self.down(any, EV_KEY.KEY_RIGHT, EV_KEY.KEY_SPACE):
            state |= State.STEP
        if self.down(any, EV_KEY.KEY_LEFTCTRL, EV_KEY.KEY_RIGHTCTRL) and self.pressed(EV_KEY.KEY_A):
            for anchor in cloth.anchors:
                anchor.flags |= AnchorFlag.SELECTED
        if self.pressed(any, EV_KEY.KEY_DELETE, EV_KEY.KEY_BACKSPACE):
            for anchor in filter(lambda a: AnchorFlag.SELECTED in a.flags, cloth.anchors):
                anchor.flags &= ~(AnchorFlag.LOCKED | AnchorFlag.SELECTED)

    def post_render(self, **kwargs):
        global state
        if state & (State.RUN | State.STEP) and state & (State.CONTACT_STEP | State.SOLVER_STEP):
            state ^= State.CONTACT_STEP | State.SOLVER_STEP
        state &= ~State.STEP


class Scene(Input):

    def __init__(self):
        super().__init__("scene")

    def init(self, width, height):
        glViewport(0, 0, width, height)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_CULL_FACE)
        glShadeModel(GL_SMOOTH)
        glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 10.0, 10.0, 0.0])

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

        glEnable(GL_NORMALIZE)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

    def render(self, frame, time):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


DistIndex = Callable[[int, int], tuple[int, int] | list[tuple[int, int]]]
BendIndex = Callable[[int, int], tuple[int, int, int, int] | list[tuple[int, int, int, int]]]

T = TypeVar('T', DistIndex, BendIndex)

class Constraint(Generic[T]):

    def __init__(self, *ranges: tuple[range, range, T], ke: float, kd: float, parallel: bool, relaxation = 1.0):
        self.ranges = ranges
        self.ke = ke
        self.kd = kd
        self.parallel = parallel
        self.relaxation = relaxation

        sizes = []
        for xi, yi, index in ranges:
            size = 1
            idx = index(0, 0)
            if isinstance(idx, list):
                size = len(idx)
            sizes.append(len(xi) * len(yi) * size)
        self.sizes = sizes
        self.count = sum(sizes)


class Constraints(Generic[T]):

    class Chain:

        def __init__(self, *constraints):
            self.constraints = constraints

        def __iter__(self):
            for constraints in self.constraints:
                for constraint in constraints:
                    yield constraint

    def __init__(self, *constraints: Constraint[T], dim: int, kernel: Callable):
        self.constraints = constraints
        self.dim = dim
        self.kernel = kernel

        count = 0
        for constraint in constraints:
            count += constraint.count
        self.count = count

        indices = np.zeros((self.count, self.dim), dtype=wp.int32)
        i = 0
        for constraint in constraints:
            for rx, ry, index in constraint.ranges:
                for xi in rx:
                    for yi in ry:
                        idx = index(xi, yi)
                        if isinstance(idx, list):
                            for e in idx:
                                indices[i] = e
                                i += 1
                        else:
                            indices[i] = idx
                            i += 1

        self.indices = wp.array2d(indices, dtype=wp.int32)
        self.rests = wp.zeros((self.count,), dtype=float)
        self.lambdas = wp.zeros((self.count,), dtype=float)

    def __iter__(self):
        i = 0
        for c in self.constraints:
            if c.parallel:
                yield i, c.count, self.kernel, self.indices, self.rests, self.lambdas, c.ke, c.kd, True, c.relaxation
                i += c.count
            else:
                for size in c.sizes:
                    yield i, size, self.kernel, self.indices, self.rests, self.lambdas, c.ke, c.kd, False, c.relaxation
                    i += size


class DistConstraints(Constraints[DistIndex]):

    def __init__(self, *constraints: Constraint[DistIndex]):
        super().__init__(*constraints, dim=2, kernel=Cloth.distance_constraints)


class BendConstraints(Constraints[BendIndex]):

    def __init__(self, *constraints: Constraint[BendIndex]):
        super().__init__(*constraints, dim=4, kernel=Cloth.bending_constraints)


def ray_from_screen(screen_x, screen_y) -> tuple[wp.vec3f, wp.vec3f]:
    viewport = glGetIntegerv(GL_VIEWPORT)
    model_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    proj_matrix = glGetDoublev(GL_PROJECTION_MATRIX)

    screen_y = viewport[3] - screen_y - 1
    p0 = gluUnProject(screen_x, screen_y, 0.0, model_matrix, proj_matrix, viewport)
    p1 = gluUnProject(screen_x, screen_y, 1.0, model_matrix, proj_matrix, viewport)
    origin = wp.vec3(p0[0], p0[1], p0[2])
    direction = wp.vec3(p1[0], p1[1], p1[2]) - origin
    direction = wp.normalize(direction)
    return origin, direction


def ray_to_sphere(origin: wp.vec3, direction: wp.vec3, center: wp.vec3, radius: float) -> Optional[tuple[float, float]]:
    m = origin - center
    b = wp.dot(m, direction)
    c = wp.dot(m, m) - radius * radius
    d = b * b - c
    if d < 0.0:
        return None
    d = wp.sqrt(d)
    return -b - d, -b + d


def quat_fraction(q: wp.quat, f: float) -> wp.quat:
    # q^f: the fraction f of the rotation encoded by (normalized) q, i.e. slerp from
    # identity. Used to split a per-frame sphere rotation into per-substep steps so
    # the collider's surface velocity is consistent with the per-substep translation.
    w = max(-1.0, min(1.0, q[3]))
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1.0e-6:
        return wp.quat_identity()
    phi = math.acos(w) * f          # per-substep half-angle
    sn = math.sin(phi) / s
    return wp.quat(q[0] * sn, q[1] * sn, q[2] * sn, math.cos(phi))


def quat_from_unit_vectors(from_vec: wp.vec3, to_vec: wp.vec3) -> wp.quat:
    rot = wp.dot(from_vec, to_vec) + 1.0
    if rot < epsilon:
        rot = 0.0
        if abs(from_vec[0]) > abs(from_vec[2]):
            quat = wp.quat(-from_vec[1], from_vec[0], 0.0, rot)
        else:
            quat = wp.quat(0.0, -from_vec[2], from_vec[1], rot)
    else:
        vec = wp.cross(from_vec, to_vec)
        # quat = wp.quaternion(wp.cross(from_vec, to_vec), rot)
        quat = wp.quat(vec[0], vec[1], vec[2], rot)
    return wp.normalize(quat)


def input_from_device(dev: Device):
    if dev.has(EV_REL) and dev.has(EV_KEY.BTN_LEFT):
        # Mouse
        shaderbang.input.ButtonMouse(dev.name, dev, mouse)
    elif dev.has(EV_KEY) and dev.has(EV_KEY.KEY_A):
        # Keyboard
        shaderbang.input.AsciiKeyboard(dev.name, dev, keyboard)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_DIRECT):
        # Touchscreen
        # Only consider direct input devices, like touchscreens and drawing tablets, see:
        # https://www.kernel.org/doc/Documentation/input/event-codes.txt
        shaderbang.input.Touchscreen(dev.name, dev, touchscreen)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_POINTER):
        # Trackpad
        # https://www.kernel.org/doc/Documentation/input/multi-touch-protocol.txt
        shaderbang.input.Trackpad(dev.name, dev, trackpad, mouse)
    else:
        dev.fd.close()


def hot_plug_devices(devices: ExitStack, inotify: INotify):
    with devices:
        while True:
            for ev in inotify.read():
                p = os.path.join("/dev/input", ev.name)
                if (str.startswith(ev.name, "event")
                        and os.path.exists(p)
                        and os.access(p, os.R_OK)
                        and stat.S_ISCHR(os.stat(p)[stat.ST_MODE])):
                    input_from_device(Device(devices.enter_context(open(p, "rb"))))


if __name__ == "__main__":
    args = parser.parse_args()

    scene = Scene()
    camera = Camera()
    ground = Ground()
    sphere = Sphere(center=wp.vec3(0.0, 1.5, 0.0), radius=0.5)
    cloth = Cloth(y_offset=2.2, num_x=400, num_y=400, spacing=0.015)

    keyboard = Keyboard()
    mouse = Mouse()
    touchscreen = Touchscreen()
    trackpad = Trackpad()

    devices = ExitStack()
    with devices:
        for path in list(filter(lambda p: os.path.exists(p) and stat.S_ISCHR(os.stat(p)[stat.ST_MODE]),
                                glob.glob("{}/event*".format("/dev/input")))):
            input_from_device(Device(devices.enter_context(open(path, "rb"))))
        devices = devices.pop_all()

    inotify = INotify()
    inotify.add_watch("/dev/input", IN_CREATE | IN_ATTRIB)
    Thread(target=hot_plug_devices, args=[devices, inotify], daemon=True).start()

    ret = sb.init(ctypes.byref(options(args)))
    if ret != 0:
        devices.close()
        exit(ret)

    ret = sb.run()
    if ret != 0:
        devices.close()
        exit(ret)

    stopped = threading.Event()
    pthread_sigmask(signal.SIG_BLOCK, [signal.SIGCONT])

    def join():
        sb.join()
        stopped.set()
        pthread_kill(main_thread().ident, signal.SIGCONT)

    Thread(target=join, daemon=True).start()

    if sigwait({signal.SIGINT, signal.SIGCONT}) == signal.SIGINT:
        sb.stop()
        ret = stopped.wait(timeout=5.0)

    inotify.close()
    devices.close()
