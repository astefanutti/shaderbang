#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# Copyright (c) 2022 NVIDIA
# www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import glob
import os
import math
import stat
import sys
import signal
import threading

import numpy as np
import warp as wp

from dataclasses import dataclass
from enum import auto, Flag
from typing import Callable, Generic, Optional, TypeVar

from contextlib import ExitStack
from pathlib import Path
from signal import pthread_sigmask, pthread_kill, sigwait
from threading import main_thread, Thread

from libevdev import *

import shaderbang.input
from shaderbang.inotify import INotify, IN_CREATE, IN_ATTRIB
from shaderbang.input import Input, TouchSlot
from shaderbang.gesture import homothety_and_rotation
from lib import glsl, options

# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["USE_ACCELERATE"] = "False"
# os.environ["ERROR_CHECKING"] = "False"
from OpenGL import setPlatform
setPlatform("egl")

from OpenGL.GL import *
from OpenGL.GLU import *

wp.init()
wp.set_device("cuda")

gravity = wp.vec3(0.0, -9.80665, 0.0)

thickness = 0.001
particleRadius = 0.0045
maxVelocity = 1e2

numIterations = 2
numSubsteps = 50
timeStep = 1.0 / 30.0
epsilon = sys.float_info.epsilon


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

state: State = State.RUN | State.FRAME_STEP | State.SELF_COLLISION | State.WIREFRAME


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
                ke=1.0e9,
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
                scale=0.7,
                ke=1.0e9,
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
                scale=0.5,
                ke=1.0e4,
                kd=100.0,
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
        self.gl_buffers = []

        self.grid = wp.HashGrid(128, 128, 128)
        self.mesh = None

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
    def swept_sphere_ccd(pos: wp.vec3,
                         vel: wp.vec3,
                         center: wp.vec3,
                         radius: float,
                         dc: wp.vec3,
                         dr: float,
                         dt: float):
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
        nx = wp.normalize(nx)
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
            vel: wp.array(dtype=wp.vec3),
            center: wp.vec3,
            radius: float,
            dc: wp.vec3,
            dr: float,
            dq: wp.quat):
        tid = wp.tid()

        if inv_mass[tid] == 0.0:
            prev_pos[tid] = pos[tid]
            return

        hit, c = Cloth.swept_sphere_ccd(prev_pos[tid], vel[tid], center, radius, dc, dr, dt)
        if hit:
            r = c - center - dc
            n = wp.normalize(r)
            c += thickness * n
            pos[tid] = c

        prev_pos[tid] = pos[tid]
        vp = vel[tid] + gravity * dt
        pos[tid] += vp * dt
        center += dc
        radius += dr

        hit, c = Cloth.swept_sphere_ccd(prev_pos[tid], vp, center, radius, wp.vec3(), 0.0, dt)
        if hit:
            r = c - center
            n = wp.normalize(r)
            c += thickness * n
            r += thickness * n
            lambda_n = wp.dot(n, pos[tid] - c)
            delta_n = n * lambda_n
            vs = (wp.quat_rotate(dq, r) - r) / dt / float(numSubsteps)
            # v = vp - vs
            v = (c - prev_pos[tid]) / dt - vs
            vt = v - wp.dot(n, v) * n
            mu = 0.4
            lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
            delta_f = wp.normalize(vt) * lambda_f
            pos[tid] += (delta_f - delta_n) * 1.0

        elif pos[tid][1] < thickness:
            n = Ground.NORMAL
            lambda_n = pos[tid][1] - thickness
            delta_n = n * lambda_n
            vt = vp - wp.dot(n, vp) * n
            mu = 0.65
            lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
            delta_f = wp.normalize(vt) * lambda_f
            pos[tid] += (delta_f - delta_n) * 1.0

    @staticmethod
    @wp.kernel
    def particle_particle_contacts(
            dt: float,
            grid: wp.uint64,
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3),
            deltas: wp.array(dtype=wp.vec3),
    ):
        tid, _ = wp.tid()

        # order threads by cell
        i = wp.hash_grid_point_id(grid, tid)
        p = pos[i]
        v = vel[i]
        w1 = inv_mass[i]

        if w1 == 0.0:
            return

        cohesion = particleRadius / 1000.0
        query = wp.hash_grid_query(grid, p, 2.0 * particleRadius + cohesion)
        index = wp.int32(0)
        delta = wp.vec3(0.0)

        while wp.hash_grid_query_next(query, index):
            if inv_mass[index] == 0.0 or index == i:
                continue

            n = p - pos[index]
            d = wp.length(n)
            c = d - 2.0 * particleRadius
            w2 = inv_mass[index]
            w = w1 + w2

            if c > cohesion or w == 0.0 or d < epsilon:
                continue

            n = n / d
            lambda_n = c
            delta_n = n * lambda_n
            vr = v - vel[index]
            vn = wp.dot(n, vr)
            vt = vr - n * vn

            mu = 0.2
            lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
            delta_f = wp.normalize(vt) * lambda_f
            delta += (delta_f - delta_n) / w

        wp.atomic_add(deltas, i, delta * w1)

    @staticmethod
    @wp.kernel
    def distance_constraints(
            dt: float,
            ke: float,
            kd: float,
            scale: float,
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

        wp.atomic_add(deltas, i, dxi * scale)
        wp.atomic_add(deltas, j, dxj * scale)

    @staticmethod
    @wp.kernel
    def bending_constraints(
            dt: float,
            ke: float,
            kd: float,
            scale: float,
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

        wp.atomic_add(deltas, i, delta0 * scale)
        wp.atomic_add(deltas, j, delta1 * scale)
        wp.atomic_add(deltas, k, delta2 * scale)
        wp.atomic_add(deltas, l, delta3 * scale)

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
            center: wp.vec3,
            radius: float,
            vel: wp.array(dtype=wp.vec3)):
        tid = wp.tid()

        v = (pos[tid] - prev_pos[tid]) / dt
        mag = wp.length(v)
        if mag > maxVelocity:
            v *= maxVelocity / mag
        vel[tid] = v

        hit, c = Cloth.swept_sphere_ccd(prev_pos[tid], v, center, radius, wp.vec3(), thickness, dt)
        if hit:
            pos[tid] = c
        elif pos[tid][1] < 0.0:
            r = radius + thickness
            rr = r * r
            if wp.length_sq(pos[tid] - center) <= rr:
                x = pos[tid][0] - center[0]
                z = pos[tid][2] - center[2]
                pos[tid][1] = center[1] - wp.sqrt(wp.max(0.0, rr - x * x - z * z))

    @staticmethod
    @wp.kernel
    def cast_ray(origin: wp.vec3,
                 direction: wp.vec3,
                 mesh: wp.uint64,
                 dist: wp.array(dtype=MeshQueryRay)):
        tid = wp.tid()
        query = wp.mesh_query_ray(mesh, origin, direction, 1.0e6)
        dist[tid] = MeshQueryRay(query.result, query.face, dist=query.t)

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
            return near_anchor

        # Otherwise return a new anchor for the triangle
        particle_id = self.hostTriIds[tri_id, 0].item()
        inv_mass = self.hostInvMass.numpy()
        anchor = Particle(id=particle_id,
                          screen=wp.vec2(screen_x, screen_y),
                          mass=inv_mass[particle_id].item(),
                          depth=min_tri_dist)

        self.anchors.append(anchor)
        return anchor

    def update_anchors(self):
        inv_mass = self.hostInvMass.numpy()
        pos = self.hostPos.numpy()

        for particle in self.anchors:
            if not particle.flags & AnchorFlag.ACTIVE:
                if not particle.flags & AnchorFlag.LOCKED:
                    inv_mass[particle.id] = particle.mass
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

        for step in range(steps):
            if integrate:
                wp.launch(kernel=Cloth.integrate,
                          dim=self.numParticles,
                          inputs=[
                              dt,
                              self.invMass,
                              self.prevPos,
                              self.pos,
                              self.vel,
                              sphere.center,
                              sphere.radius,
                              sphere.dc,
                              sphere.dr,
                              sphere.dq,
                          ])

            if self_collision:
                self.grid.build(self.pos, 2.0 * particleRadius)

                self.deltas.zero_()
                wp.launch(kernel=Cloth.particle_particle_contacts,
                          dim=self.numParticles,
                          inputs=[
                              dt,
                              self.grid.id,
                              self.invMass,
                              self.pos,
                              self.vel,
                          ],
                          outputs=[
                              self.deltas,
                          ])
                wp.launch(kernel=Cloth.add_deltas,
                          dim=self.numParticles,
                          inputs=[self.pos, self.deltas])

            if not solve_constraints:
                continue

            self.distConstraints.lambdas.zero_()
            self.bendConstraints.lambdas.zero_()

            for iteration in range(iterations):
                for offset, count, kernel, indices, rests, lambdas, ke, kd, parallel, scale in self.constraints:
                    if parallel:
                        self.deltas.zero_()
                        wp.launch(kernel=kernel,
                                  dim=count,
                                  inputs=[
                                      dt,
                                      ke,
                                      kd,
                                      scale,
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
                                      scale,
                                      offset,
                                      self.pos,
                                      self.prevPos,
                                      self.invMass,
                                      indices,
                                      rests,
                                      lambdas,
                                      self.pos,
                                  ])

            wp.launch(kernel=Cloth.update_velocity,
                      dim=self.numParticles,
                      inputs=[
                          dt,
                          self.pos,
                          self.prevPos,
                          sphere.center + sphere.dc,
                          sphere.radius + sphere.dr,
                      ],
                      outputs=[self.vel])

        self.mesh.refit()

        wp.copy(self.hostPos, self.pos)

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

        host_pos = self.hostPos.numpy()
        glGenBuffers(1, ctypes.pointer(self.pos_gl_buffer))
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_gl_buffer)
        glBufferData(GL_ARRAY_BUFFER, host_pos.nbytes, host_pos, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        buffer = wp.RegisteredGLBuffer(int(self.pos_gl_buffer.value),
                                       flags=wp.RegisteredGLBuffer.NONE,
                                       fallback_to_copy=False)
        self.gl_buffers.append(buffer)
        self.pos = buffer.map(dtype=wp.vec3, shape=(self.numParticles,))

        normals = wp.zeros(self.numParticles, dtype=wp.vec3).numpy()
        glGenBuffers(1, ctypes.pointer(self.normals_gl_buffer))
        glBindBuffer(GL_ARRAY_BUFFER, self.normals_gl_buffer)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        buffer = wp.RegisteredGLBuffer(int(self.normals_gl_buffer.value),
                                       flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                       fallback_to_copy=False)
        self.gl_buffers.append(buffer)
        self.normals = buffer.map(dtype=wp.vec3, shape=(self.numParticles,))

        tri_ids = self.hostTriIds
        glGenBuffers(1, ctypes.pointer(self.triIds_gl_buffer))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.triIds_gl_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, tri_ids.nbytes, tri_ids, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        buffer = wp.RegisteredGLBuffer(int(self.triIds_gl_buffer.value),
                                       flags=wp.RegisteredGLBuffer.READ_ONLY,
                                       fallback_to_copy=False)
        self.gl_buffers.append(buffer)
        self.triIds = buffer.map(dtype=wp.int32, shape=(self.numTris, 3))

        self.mesh = wp.Mesh(self.pos, self.triIds.flatten(), self.vel, bvh_constructor="lbvh")

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
        glColor3f(1.0, 1.0, 1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        host_pos = self.hostPos.numpy()
        for anchor in filter(lambda a: a.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED), self.anchors):
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
        self.update_anchors()


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
            if keyboard.down(EV_KEY.KEY_LEFTCTRL) or keyboard.down(EV_KEY.KEY_RIGHTCTRL):
                self.particle.flags |= AnchorFlag.LOCKED
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
                    slot.particle.flags &= ~AnchorFlag.ACTIVE
                slot.particle = cloth.drag_anchor(slot.touchX, slot.touchY)
            elif slot.drag:
                if slot.particle:
                    slot.particle.screen = wp.vec2(slot.touchX, slot.touchY)
                else:
                    slots.append(slot)
            elif slot.particle:
                slot.particle.flags &= ~AnchorFlag.ACTIVE
                if keyboard.down(EV_KEY.KEY_LEFTCTRL) or keyboard.down(EV_KEY.KEY_RIGHTCTRL):
                    slot.particle.flags |= AnchorFlag.LOCKED
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
            # elif n == 3:
            #     camera.orbit(dx, dy, 0.5 / self.resolution[1])
            #     camera.dolly_scale(scale)
            #     camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)
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
        if self.down(EV_KEY.KEY_RIGHT) or self.pressed(EV_KEY.KEY_SPACE):
            state |= State.STEP

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


DistIndex = Callable[[int, int], list(tuple[int, int])]
BendIndex = Callable[[int, int], list(tuple[int, int, int, int])]

T = TypeVar('T', bound=DistIndex|BendIndex)

class Constraint(Generic[T]):

    def __init__(self, *ranges: tuple[range, range, T], ke: float, kd: float, parallel: bool, scale = 1.0):
        self.ranges = ranges
        self.ke = ke
        self.kd = kd
        self.parallel = parallel
        self.scale = scale

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
                yield i, c.count, self.kernel, self.indices, self.rests, self.lambdas, c.ke, c.kd, True, c.scale
                i += c.count
            else:
                for size in c.sizes:
                    yield i, size, self.kernel, self.indices, self.rests, self.lambdas, c.ke, c.kd, False, c.scale
                    i += size

class DistConstraints(Constraints[DistIndex]):

    def __init__(self, *constraints: Constraint):
        super().__init__(*constraints, dim=2, kernel=Cloth.distance_constraints)

class BendConstraints(Constraints[BendIndex]):

    def __init__(self, *constraints: Constraint):
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


parser = argparse.ArgumentParser(description="Run cloth simulation")
parser.add_argument("--async-page-flip", action=argparse.BooleanOptionalAction,
                    help="use async page flipping")
parser.add_argument("--atomic-drm-mode", action=argparse.BooleanOptionalAction,
                    help="use atomic mode setting and fencing")
parser.add_argument("-C", "--connector", metavar="CONNECTOR", type=int,
                    help="the DRM connector")
parser.add_argument("-D", "--device", metavar="DEVICE", type=Path,
                    help="the DRM device")
parser.add_argument("--mode", metavar="MODE", type=str,
                    help="specify the video mode in the format <resolution>[-<vrefresh>]")
parser.add_argument("-n", "--frames", metavar="N", type=int,
                    help="run for the given number of frames and exit")
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

ret = glsl.init(bytes("", "utf-8"), ctypes.byref(options(args)))
if ret != 0:
    devices.close()
    exit(ret)

ret = glsl.run()
if ret != 0:
    devices.close()
    exit(ret)

stopped = threading.Event()
pthread_sigmask(signal.SIG_BLOCK, [signal.SIGCONT])


def join():
    glsl.join()
    stopped.set()
    pthread_kill(main_thread().ident, signal.SIGCONT)


Thread(target=join, daemon=True).start()

if sigwait({signal.SIGINT, signal.SIGCONT}) == signal.SIGINT:
    glsl.stop()
    ret = stopped.wait(timeout=5.0)

inotify.close()
devices.close()
