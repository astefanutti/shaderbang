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
from shaderbang.pathtracer.renderer import PathTracer
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
args = parser.parse_args()

gravity = wp.vec3(0.0, -9.80665, 0.0)

thickness = 0.001
particleRadius = 0.0045
maxVelocity = 1e2

# Planar Divide-and-Truncate (PDT) collision parameters
d_offset = 2.0 * particleRadius  # cloth self-collision separation (matches old 2*particleRadius)
gamma_r = 0.9                    # conservative truncation safety ratio (Newton uses 0.85-0.95)

numIterations = 2
numSubsteps = 60
timeStep = 1.0 / 30.0
epsilon = sys.float_info.epsilon

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

    def drag(self) -> Self:
        self.origin = wp.vec2(self.screen)
        self.time = time.time()
        return self

    def click(self) -> bool:
        return wp.length(self.screen - self.origin) < 1.0 and time.time() - self.time < 0.5

    def drop(self):
        self.flags &= ~(AnchorFlag.ACTIVE | AnchorFlag.LOCKED)

@wp.struct
class MeshQueryRay:
    hit: bool
    face: wp.int32
    dist: wp.float32


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

        # Geometry device arrays (allocated in init()); path traced, not GL-bound.
        self.pos = None
        self.normals = None
        self.triIds = None

        self.numCols = num_y + 1  # grid stride, for topological-neighbor exclusion
        self.truncation_ts = wp.zeros(self.numParticles, dtype=float)  # per-vertex PDT scale (atomic_min)
        self.push = wp.zeros_like(self.restPos)  # C<0 feasibility-recovery separation (atomic_add)
        self.mesh = None

        # Unique mesh edges (sorted vertex pairs) for edge-edge self-collision.
        edge_set = set()
        for tri in self.hostTriIds:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, w in ((a, b), (b, c), (c, a)):
                edge_set.add((u, w) if u < w else (w, u))
        edge_arr = np.array(sorted(edge_set), dtype=np.int32)
        self.numEdges = len(edge_arr)
        self.edgeIds = wp.array(edge_arr, dtype=wp.int32)  # [numEdges, 2], va < vb
        print(str(self.numEdges) + " edges created")

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
        lc = wp.length(dc)
        if lc < epsilon:
            n = wp.normalize(r)
            if dr >= 0.0:
                return True, o + (radius + dr) * n
            return True, p

        nz = dc / lc
        z = wp.dot(r, nz)
        if z <= 0.0:
            n = wp.normalize(r)
            if dr >= 0.0:
                return True, o + (radius + dr) * n
            return True, p

        nx = wp.cross(r, nz)
        lnx = wp.length(nx)
        if lnx < epsilon:
            # r is parallel to the motion axis (particle dead ahead of the sphere):
            # the swept frame is degenerate, so pick an arbitrary perpendicular. The
            # reconstruction below still pushes the particle clear of the swept tube.
            up = wp.vec3(0.0, 1.0, 0.0)
            if wp.abs(nz[1]) > 0.9:
                up = wp.vec3(1.0, 0.0, 0.0)
            nx = wp.normalize(wp.cross(nz, up))
        else:
            nx = nx / lnx
        ny = wp.cross(nz, nx)

        dl = (dt - t) * lc / dt
        radius += dr
        z *= (radius + dl) / radius
        if z <= dl:
            y = radius
        else:
            dz = z - dl
            y = wp.sqrt(wp.max(0.0, radius * radius - dz * dz))
        return True, o + y * ny + z * nz

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
    def self_collision_truncate(
            mesh: wp.uint64,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),  # frozen reference state X
            pos: wp.array(dtype=wp.vec3),       # X + accumulated displacement
            num_cols: wp.int32,
            truncation_ts: wp.array(dtype=float),  # pre-filled with 1.0 (atomic_min)
            push: wp.array(dtype=wp.vec3)):        # pre-zeroed C<0 recovery (atomic_add)
        v = wp.tid()
        if inv_mass[v] == 0.0:
            return

        xv = prev_pos[v]
        dxv = pos[v] - xv

        # Broadphase: query the LBVH with the swept AABB of this vertex, inflated
        # by the contact offset (the reference geometry is frozen at prev_pos).
        r = d_offset + wp.length(dxv)
        lo = wp.min(xv, pos[v]) - wp.vec3(r, r, r)
        hi = wp.max(xv, pos[v]) + wp.vec3(r, r, r)
        query = wp.mesh_query_aabb(mesh, lo, hi)
        face = wp.int32(0)
        while wp.mesh_query_aabb_next(query, face):
            i0 = wp.mesh_get_index(mesh, 3 * face + 0)
            i1 = wp.mesh_get_index(mesh, 3 * face + 1)
            i2 = wp.mesh_get_index(mesh, 3 * face + 2)
            # Skip incident and topological-neighbor triangles (2-ring): their
            # closest point is naturally within the offset and would freeze bending.
            if (Cloth.within_ring(v, i0, num_cols)
                    or Cloth.within_ring(v, i1, num_cols)
                    or Cloth.within_ring(v, i2, num_cols)):
                continue

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
    @wp.kernel
    def self_collision_truncate_edges(
            mesh: wp.uint64,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),  # frozen reference state X
            pos: wp.array(dtype=wp.vec3),       # X + accumulated displacement
            num_cols: wp.int32,
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2], va < vb
            truncation_ts: wp.array(dtype=float),  # pre-filled with 1.0 (atomic_min)
            push: wp.array(dtype=wp.vec3)):        # pre-zeroed C<0 recovery (atomic_add)
        # Edge-edge complement to the vertex-triangle pass: catches folds where two
        # edges cross with no vertex near either face (the classic "X" configuration).
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        if wa == 0.0 and wb == 0.0:
            return  # both endpoints host-driven

        xa = prev_pos[va]
        xb = prev_pos[vb]

        # Broadphase: swept AABB of the edge over the frozen + current state,
        # inflated by the contact offset (the LBVH is frozen at prev_pos).
        off = wp.vec3(d_offset, d_offset, d_offset)
        lo = wp.min(wp.min(xa, xb), wp.min(pos[va], pos[vb])) - off
        hi = wp.max(wp.max(xa, xb), wp.max(pos[va], pos[vb])) + off
        query = wp.mesh_query_aabb(mesh, lo, hi)
        face = wp.int32(0)
        while wp.mesh_query_aabb_next(query, face):
            i0 = wp.mesh_get_index(mesh, 3 * face + 0)
            i1 = wp.mesh_get_index(mesh, 3 * face + 1)
            i2 = wp.mesh_get_index(mesh, 3 * face + 2)
            for k in range(3):
                vc = i0
                vd = i1
                if k == 1:
                    vc = i1
                    vd = i2
                if k == 2:
                    vc = i2
                    vd = i0
                if vc > vd:  # canonical order of the candidate edge
                    swap = vc
                    vc = vd
                    vd = swap
                # Skip pairs that share a vertex (adjacent edges never separate).
                if va == vc or va == vd or vb == vc or vb == vd:
                    continue
                # No key dedup: each edge processes every pair it discovers so it
                # strongly constrains ITSELF as the query. The broadphase is asymmetric
                # (a fast edge's swept AABB finds a slow edge's frozen triangle, but not
                # the reverse), so keying off a lower-index "owner" is not guaranteed to
                # see the pair -- and skipping the discovering thread would let a fast
                # edge tunnel. Both edges get the strong constraint from their own query;
                # atomic_min makes the redundant weak constraint from the other thread
                # (and any double-processing via a shared triangle) order-independent.
                # Skip topological neighbors (2-ring on the grid) to keep bending free.
                if (Cloth.within_ring(va, vc, num_cols)
                        or Cloth.within_ring(va, vd, num_cols)
                        or Cloth.within_ring(vb, vc, num_cols)
                        or Cloth.within_ring(vb, vd, num_cols)):
                    continue

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

                # Displacement of each edge's contact point (barycentric blend).
                dca = (1.0 - se) * (pos[va] - xa) + se * (pos[vb] - xb)
                dcf = (1.0 - sf) * (pos[vc] - yc) + sf * (pos[vd] - yd)

                delta_e_n = wp.max(-wp.dot(n, dca), 0.0)
                delta_f_n = wp.max(wp.dot(n, dcf), 0.0)
                ssum = delta_e_n + delta_f_n
                if ssum == 0.0:
                    lmbd = 0.5
                else:
                    lmbd = wp.clamp(delta_f_n / ssum, 0.05, 0.95)

                if c < 0.0:
                    # Feasibility recovery for an already-overlapping pair. Each edge
                    # pushes only its OWN endpoints away from the other (weighted along
                    # the edge by the reference barycentric coordinate). The candidate
                    # edge separates itself in its own thread -- an overlap means the
                    # pair is within d_offset, so both edges discover each other -- and
                    # the two threads' (1 - lmbd) weights are complementary, so the pair
                    # still separates by ~depth without an atomic_add cross-push
                    # double-count between the query and candidate sides.
                    depth = -c
                    if wa != 0.0:
                        wp.atomic_add(push, va, (1.0 - lmbd) * depth * (1.0 - se) * n)
                    if wb != 0.0:
                        wp.atomic_add(push, vb, (1.0 - lmbd) * depth * se * n)
                    continue

                # TRUNCATE: scaling both endpoints of an edge by t scales its contact
                # point's displacement by t (linearity), so the plane is respected.
                p_plane = cb + (d_offset + lmbd * c) * n
                t_e = Cloth.planar_truncation_t(ca, dca, n, p_plane)
                if wa != 0.0:
                    wp.atomic_min(truncation_ts, va, t_e)
                if wb != 0.0:
                    wp.atomic_min(truncation_ts, vb, t_e)
                t_f = Cloth.planar_truncation_t(cb, dcf, -n, p_plane)
                if inv_mass[vc] != 0.0:
                    wp.atomic_min(truncation_ts, vc, t_f)
                if inv_mass[vd] != 0.0:
                    wp.atomic_min(truncation_ts, vd, t_f)

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
        pos[i] = prev_pos[i] + (pos[i] - prev_pos[i]) * truncation_ts[i] + push[i]

    @staticmethod
    @wp.kernel
    def collider_project(
            dt: float,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            center: wp.vec3,
            radius: float,
            dc: wp.vec3,
            dr: float,
            dq: wp.quat):
        # Swept CCD against the moving / growing sphere: the analytic time-of-impact
        # catches approaching particles on the swept surface (no tunneling at any
        # drag speed); a static end-pose contact then resolves already-inside /
        # resting particles and applies depth-based Coulomb friction. Runs after the
        # constraint solve so its penetration-free result has the final say.
        i = wp.tid()
        if inv_mass[i] == 0.0:
            return

        x = pos[i]
        vel_eff = (x - prev_pos[i]) / dt  # cloth substep velocity, before collision

        # Pass 1: swept sphere -> snap approaching particles onto the swept surface.
        hit, c = Cloth.swept_sphere_ccd(prev_pos[i], vel_eff, center, radius, dc, dr, dt)
        if hit:
            n = wp.normalize(c - center - dc)
            x = c + thickness * n

        # Pass 2: static contact at the end pose. The solved position's penetration
        # depth is the normal force signal for friction (rest contacts, where the
        # swept test does not fire because the per-substep travel is sub-margin).
        c_end = center + dc
        r_end = radius + dr + thickness
        dv = x - c_end
        d = wp.length(dv)
        if d > epsilon and d < r_end:
            n = dv / d
            lambda_n = d - r_end  # < 0 when penetrating
            arm = r_end * n
            frame_dt = float(numSubsteps) * dt
            v_surf = (dc + wp.quat_rotate(dq, arm) - arm) / frame_dt
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
        vij = pi - pj

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

        cos_theta = wp.dot(n1, n2)

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

        grad_dot_v = wp.dot(grad_x1, p1) + wp.dot(grad_x2, p2) + wp.dot(grad_x3, p3) + wp.dot(grad_x4, p4)
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
        tid = wp.tid()
        pos[tid] += deltas[tid]

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
        if mag > maxVelocity:
            v *= maxVelocity / mag
        vel[tid] = v

    @staticmethod
    @wp.kernel
    def cast_ray(origin: wp.vec3,
                 direction: wp.vec3,
                 mesh: wp.uint64,
                 rays: wp.array(dtype=MeshQueryRay)):
        tid = wp.tid()
        query = wp.mesh_query_ray(mesh, origin, direction, 1.0e6)
        rays[tid] = MeshQueryRay(query.result, query.face, dist=query.t)

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

        # Find the closest triangle hit by the ray
        rays = wp.empty((1,), dtype=MeshQueryRay)
        wp.launch(kernel=Cloth.cast_ray,
                  dim=1,
                  inputs=[origin, direction, self.mesh.id],
                  outputs=[rays])
        rays = rays.numpy()
        hit, tri_id, min_tri_dist = rays[0]
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

        # Otherwise return a new anchor for the triangle
        particle_id = self.hostTriIds[tri_id, 0].item()
        inv_mass = self.hostInvMass.numpy()
        anchor = Particle(id=particle_id,
                          screen=wp.vec2(screen_x, screen_y),
                          mass=inv_mass[particle_id].item(),
                          depth=min_tri_dist)

        self.anchors.append(anchor)
        return anchor.drag()

    def update_anchors(self):
        inv_mass = self.hostInvMass.numpy()
        pos = self.hostPos.numpy()

        for particle in self.anchors:
            if not particle.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED):
                inv_mass[particle.id] = particle.mass
                continue

            if not particle.flags & AnchorFlag.ACTIVE:
                # host-side sphere push-out for released anchors
                r = wp.vec3f(pos[particle.id]) - sphere.center
                d = wp.length(r) - sphere.radius - thickness - particleRadius
                if d < 0:
                    pos[particle.id] -= d * r
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

        self.anchors[:] = [anchor for anchor in self.anchors if anchor.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED)]

    def simulate(self, steps=numSubsteps, iterations=numIterations, integrate=True, self_collision=True, solve_constraints=True):
        dt = timeStep / numSubsteps

        wp.copy(self.pos, self.hostPos)
        wp.copy(self.invMass, self.hostInvMass)

        graph = None
        for step in range(steps):
            if graph:
                wp.capture_launch(graph)
            else:
                with wp.ScopedCapture() as capture:
                    self.step(dt, iterations, integrate, self_collision, solve_constraints)
                graph = capture.graph

        wp.copy(self.hostPos, self.pos)

    def step(self, dt: float, iterations=numIterations, integrate=True, self_collision=True, solve_constraints=True):
        # Phase 0: refit the LBVH against the start-of-substep geometry. Before
        # integrate runs, self.pos still holds the previous substep's end position,
        # which integrate is about to freeze into self.prevPos -> the broadphase
        # bounds stay consistent with the reference state used to build the
        # division planes. (refit is warmed up once in init() to avoid a capture
        # allocation.)
        self.mesh.refit()

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
            for iteration in range(iterations):
                for offset, count, kernel, indices, rests, lambdas, ke, kd, parallel, relaxation in self.constraints:
                    if parallel:
                        self.deltas.zero_()
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

        # Planar Divide-and-Truncate: clamp the net per-vertex displacement so no
        # vertex crosses a plane that separated it from a nearby triangle (vertex-
        # triangle) or a nearby edge (edge-edge) at the frozen reference state,
        # recovering feasibility where a pair already overlaps. Both passes write
        # the same truncation_ts (atomic_min) and push (atomic_add) buffers.
        if self_collision:
            self.truncation_ts.fill_(1.0)
            self.push.zero_()
            wp.launch(kernel=Cloth.self_collision_truncate,
                      dim=self.numParticles,
                      inputs=[
                          self.mesh.id,
                          self.invMass,
                          self.prevPos,
                          self.pos,
                          self.numCols,
                      ],
                      outputs=[
                          self.truncation_ts,
                          self.push,
                      ])
            wp.launch(kernel=Cloth.self_collision_truncate_edges,
                      dim=self.numEdges,
                      inputs=[
                          self.mesh.id,
                          self.invMass,
                          self.prevPos,
                          self.pos,
                          self.numCols,
                          self.edgeIds,
                      ],
                      outputs=[
                          self.truncation_ts,
                          self.push,
                      ])
            wp.launch(kernel=Cloth.apply_truncation,
                      dim=self.numParticles,
                      inputs=[self.invMass, self.prevPos, self.pos, self.truncation_ts, self.push])

        # Swept-CCD sphere projection + ground contact run last so their
        # penetration-free result has final say over the substep.
        wp.launch(kernel=Cloth.collider_project,
                  dim=self.numParticles,
                  inputs=[
                      dt,
                      self.invMass,
                      self.prevPos,
                      self.pos,
                      sphere.center,
                      sphere.radius,
                      sphere.dc,
                      sphere.dr,
                      sphere.dq,
                  ])

        wp.launch(kernel=Cloth.update_velocity,
                  dim=self.numParticles,
                  inputs=[
                      dt,
                      self.pos,
                      self.prevPos,
                  ],
                  outputs=[self.vel])

    def update_mesh(self):
        self.mesh.refit()
        self.normals.zero_()
        wp.launch(kernel=Cloth.add_normals,
                  dim=self.numTris,
                  inputs=[self.pos, self.triIds, self.normals])
        wp.launch(kernel=Cloth.normalize_normals,
                  dim=self.numParticles,
                  inputs=[self.normals])

    def init(self, **kwargs):
        self._quad = gluNewQuadric()

        # Pure path tracing: the geometry lives in plain device arrays, not GL
        # vertex buffers. The OptiX renderer builds its GAS straight from these
        # pointers and refits them in place each frame, so the *same* arrays feed
        # both the Warp physics / LBVH self-collision and the ray tracer -- no GL
        # interop and no CPU copy on the render hot path.
        self.pos = wp.clone(self.restPos)
        self.normals = wp.zeros(self.numParticles, dtype=wp.vec3)
        self.triIds = wp.array(self.hostTriIds, dtype=wp.int32)

        self.mesh = wp.Mesh(self.pos, self.triIds.flatten(), self.vel, bvh_constructor="lbvh")
        # Warm up refit() outside CUDA-graph capture so any scratch allocation
        # happens now, not during the first captured step().
        self.mesh.refit()

        wp.launch(kernel=Cloth.rest_distances,
                  dim=(self.distConstraints.count,),
                  inputs=[self.pos, self.distConstraints.indices, self.distConstraints.rests])

    def pre_render(self, **kwargs):
        self.update_anchors()

        if state & (State.RUN | State.STEP):
            self.simulate(steps=numSubsteps if State.FRAME_STEP in state else 1,
                          integrate=State.SOLVER_STEP not in state,
                          self_collision=State.SELF_COLLISION in state and State.SOLVER_STEP not in state,
                          solve_constraints=State.CONTACT_STEP not in state)

        self.update_mesh()

    def render(self, **kwargs):
        # The cloth mesh is path traced (see PathTracerView), which presents a
        # full-screen frame over the whole viewport. The only fixed-function
        # overlay left is the anchor gizmos, drawn *after* that frame -- see
        # draw_anchors(), which PathTracerView calls once the traced image is up.
        pass

    def draw_anchors(self):
        # Kinematic particles / anchors, drawn as small spheres on top of the
        # traced frame. Positions come from the host mirror the physics keeps in
        # sync; make sure any in-flight copy has landed before reading it.
        wp.synchronize_stream()
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
        # The ground plane is intersected analytically by the path tracer
        # (PathTracer.set_ground); there is nothing to rasterize here.
        pass


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
        # The sphere collider is intersected analytically by the path tracer;
        # PathTracerView pushes its live centre/radius each frame, so there is
        # no GL sphere to draw here.
        pass

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


class PathTracerView(Input):
    """Presents the scene with the OptiX path tracer instead of GL rasterization.

    Constructed last so it renders after every other Input: by the time its
    render() runs, the cloth physics has stepped and refit its mesh (in
    Cloth.pre_render), so the GAS can be refit from the very same device arrays --
    no CPU copy, no GL interop for the geometry. The traced + denoised frame is
    tone-mapped straight into a PBO and drawn full-screen; the only remaining
    fixed-function draw is the anchor gizmos, layered on top afterwards.
    """

    FOV_Y = 40.0        # matches Camera.init's gluPerspective vertical FOV
    EXPOSURE = 1.2
    # Samples per animating frame. While the sim runs there is no cross-frame
    # HDR accumulation (every frame resets), so this burst is what feeds the
    # temporal denoiser a cleaner-than-1-spp image and keeps fast cloth motion
    # from ghosting. Rendering at half res (upscale=2) leaves the 5090 ample
    # headroom for a few samples; dial up for less noise, down for more speed.
    # Still frames ignore it and refine progressively (reset=False, 1 spp/frame).
    SPP = 4

    def __init__(self, camera, sphere, cloth):
        super().__init__("pathtracer")
        self.camera = camera
        self.sphere = sphere
        self.cloth = cloth
        self.pt = None
        self._width = None
        self._height = None
        self._sig = None

    def init(self, width, height):
        # Defer the (GPU-heavy) tracer construction to the first render, when the
        # cloth device geometry exists and a GL context is current; just capture
        # the framebuffer size here.
        self._width = width
        self._height = height

    def _build(self):
        cloth = self.cloth
        # Render at half the framebuffer resolution and let the OptiX temporal
        # denoiser upscale 2x back to native (1080p -> 4K on the target display),
        # which both denoises and antialiases the 1-spp trace far cheaper than
        # rendering natively. Requires the patched otk-pyoptix binding (see
        # shaderbang/pathtracer/patches/); with the stock binding drop upscale to
        # get the single-frame HDR denoiser at native resolution.
        # interop="auto" (M6): present straight into the GL texture via a CUDA
        # surface when the raw-CUDA/texture interop is available, dropping the
        # per-frame PBO->texture copy; it falls back to the portable PBO upload
        # path automatically if that interop cannot be set up.
        self.pt = PathTracer(self._width // 2, self._height // 2, upscale=2,
                             exposure=PathTracerView.EXPOSURE, interop="auto")
        # Pass the cloth's per-vertex smooth normals (updated in place by
        # Cloth.update_mesh each frame) so the Disney BSDF shades a smooth
        # surface instead of faceted triangles.
        self.pt.set_geometry(cloth.pos, cloth.triIds, normals=cloth.normals)
        self.pt.set_ground(y=0.0, albedo=(0.55, 0.55, 0.6))
        self.pt.set_light(direction=(0.5, 1.0, 0.4), color=(1.1, 1.05, 0.95))
        self.pt.set_cloth_albedo(front=(0.2, 0.45, 0.85), back=(0.85, 0.6, 0.2))
        self.pt.set_sphere(center=self.sphere.center, radius=self.sphere.radius,
                           albedo=(0.8, 0.8, 0.8))
        # Disney materials: matte fabric, a semi-glossy sphere, a diffuse floor.
        self.pt.set_cloth_material(roughness=0.6)
        self.pt.set_sphere_material(roughness=0.3)
        self.pt.set_ground_material(roughness=0.9)
        self.pt.set_path_depth(max_depth=4)
        self.pt.init_gl()

    def render(self, **kwargs):
        if self.pt is None:
            self._build()

        cam = self.camera
        eye = (cam.pos[0], cam.pos[1], cam.pos[2])
        target = (cam.pos[0] + cam.forward[0],
                  cam.pos[1] + cam.forward[1],
                  cam.pos[2] + cam.forward[2])
        up = (cam.up[0], cam.up[1], cam.up[2])
        self.pt.set_camera_lookat(eye=eye, target=target, up=up,
                                  fov_y_deg=PathTracerView.FOV_Y,
                                  aspect=self._width / float(self._height))

        # Apply the interactive centre/radius deltas so the traced sphere matches
        # what the old GL draw showed (post_render integrates them once a step
        # runs).
        centre = self.sphere.center + self.sphere.dc
        radius = self.sphere.radius + self.sphere.dr
        self.pt.set_sphere(center=(centre[0], centre[1], centre[2]), radius=radius)

        running = bool(state & (State.RUN | State.STEP))
        if running:
            # The cloth deformed this frame; track it with an in-place GAS refit.
            self.pt.refit()

        # Restart accumulation whenever anything moves; otherwise keep refining
        # the denoised image while the scene is paused and still.
        sig = (running, eye, target, up,
               (centre[0], centre[1], centre[2]), radius)
        reset = running or sig != self._sig
        self._sig = sig

        # Burst several samples on any frame that restarts accumulation (every
        # animating frame, plus the first still frame after motion stops); once
        # the scene is paused and unchanged, refine with 1 spp/frame on top of
        # the retained accumulation.
        self.pt.render(reset=reset, spp=PathTracerView.SPP if reset else 1)
        self.pt.present()

        # Anchor gizmos, layered on top of the traced frame.
        self.cloth.draw_anchors()


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


scene = Scene()
camera = Camera()
ground = Ground()
sphere = Sphere(center=wp.vec3(0.0, 1.5, 0.0), radius=0.5)
cloth = Cloth(y_offset=2.2, num_x=400, num_y=400, spacing=0.015)
# Constructed last so it renders after the physics has stepped and the scene
# Inputs have updated (camera basis, sphere transform): the path-traced frame is
# presented over the whole viewport in place of the fixed-function draws.
pathtracer = PathTracerView(camera=camera, sphere=sphere, cloth=cloth)

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
