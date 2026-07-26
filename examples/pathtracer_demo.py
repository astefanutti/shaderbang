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
Path tracer demo
================

A minimal second consumer of ``shaderbang.pathtracer`` (besides cloth.py),
showing that the tracer is scene-agnostic: the whole scene is built from the
generic ``add_*`` API and animated per frame through the same ``Input`` render
callback -- no cloth, no Warp physics.

The scene is a couple of rigid instanced meshes (two spinning cubes) plus an
analytic sphere and ground plane, lit by a directional key light and a warm
point light, viewed from a slowly orbiting camera. It exercises:

* ``add_material``            -- a device material table (diffuse / glossy / metal)
* ``add_mesh(deformable=False)`` -- rigid triangle meshes as IAS instances
* ``add_sphere`` / ``add_plane`` -- analytic primitives
* ``add_light`` (directional + point)
* ``set_instance_transform`` -- per-frame rigid transforms (rigid motion vectors)
* ``set_camera_lookat``      -- per-frame camera (camera motion vectors)

Everything animates every frame, so each frame restarts accumulation and bursts
a few samples into the temporal denoiser (the same pattern cloth.py uses while
the sim runs).

Run headless for N frames with ``-n N``; otherwise it renders until Ctrl+C.
"""


import argparse
import ctypes
import math
import signal
import threading

import numpy as np
import warp as wp

from pathlib import Path
from signal import pthread_sigmask, pthread_kill, sigwait
from threading import main_thread, Thread

from shaderbang.input import Input
from shaderbang.pathtracer.renderer import PathTracer
from shaderbang import lib as sb, options

from OpenGL import setPlatform
setPlatform("egl")


parser = argparse.ArgumentParser(description="Run the path tracer demo")
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

wp.init()
wp.set_device("cuda")


# --------------------------------------------------------------------------- #
# Geometry + small transform helpers (host side; the mesh buffers live on the
# GPU as wp.arrays so the tracer references them with zero copies).
# --------------------------------------------------------------------------- #
def make_cube(half=0.5):
    """A flat-shaded axis-aligned cube centered at the origin: 24 vertices (4 per
    face, so each face carries its own outward normal) and 12 triangles. Returns
    (vertices, normals, indices) as wp.arrays on the current CUDA device."""
    faces = [
        ((1.0, 0.0, 0.0),  [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
        ((-1.0, 0.0, 0.0), [(-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1)]),
        ((0.0, 1.0, 0.0),  [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)]),
        ((0.0, -1.0, 0.0), [(-1, -1, 1), (-1, -1, -1), (1, -1, -1), (1, -1, 1)]),
        ((0.0, 0.0, 1.0),  [(1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)]),
        ((0.0, 0.0, -1.0), [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]),
    ]
    verts, norms, tris = [], [], []
    for normal, corners in faces:
        base = len(verts)
        for cx, cy, cz in corners:
            verts.append((cx * half, cy * half, cz * half))
            norms.append(normal)
        tris.append((base + 0, base + 1, base + 2))
        tris.append((base + 0, base + 2, base + 3))
    v = wp.array(np.asarray(verts, dtype=np.float32), dtype=wp.vec3)
    n = wp.array(np.asarray(norms, dtype=np.float32), dtype=wp.vec3)
    idx = wp.array(np.asarray(tris, dtype=np.int32), dtype=wp.int32)
    return v, n, idx


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                    dtype=np.float64)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
                    dtype=np.float64)


def xform(rot, translation):
    """A 3x4 object->world transform from a 3x3 rotation and a translation (the
    OptixInstance layout expected by set_instance_transform)."""
    m = np.zeros((3, 4), dtype=np.float64)
    m[:3, :3] = rot
    m[:3, 3] = translation
    return m


class Demo(Input):
    """Builds the scene once on the first frame, then animates it every frame.

    Constructed as a shaderbang ``Input``: the driver calls ``init`` with the
    framebuffer size and ``render`` once per frame with the elapsed ``time``.
    """

    FOV_Y = 45.0
    EXPOSURE = 1.1
    SPP = 4          # samples burst per (always-animating) frame

    def __init__(self):
        super().__init__("pathtracer-demo")
        self.pt = None
        self._width = None
        self._height = None
        self._cube_a = None
        self._cube_b = None

    def init(self, width, height):
        # Defer the GPU-heavy tracer construction to the first render (a GL
        # context must be current); just capture the framebuffer size here.
        self._width = width
        self._height = height

    def _build(self):
        self.pt = PathTracer(self._width // 2, self._height // 2, upscale=2,
                             exposure=Demo.EXPOSURE, interop="auto")
        # A cool-grey sky so the metal cube has something to reflect.
        self.pt.set_sky(top=(0.30, 0.42, 0.60), bottom=(0.75, 0.78, 0.82))
        # Materials: a diffuse floor, a glossy dielectric ball, a matte red cube
        # and a gold metallic cube.
        ground_mat = self.pt.add_material(base_color=(0.45, 0.47, 0.5),
                                          roughness=0.9)
        ball_mat = self.pt.add_material(base_color=(0.85, 0.87, 0.9),
                                        roughness=0.12)
        red_mat = self.pt.add_material(base_color=(0.8, 0.22, 0.18),
                                       roughness=0.4)
        gold_mat = self.pt.add_material(base_color=(1.0, 0.78, 0.34),
                                        roughness=0.22, metallic=1.0)
        # Two rigid cube instances (one shared cube mesh per instance) + an
        # analytic sphere sitting on an analytic ground plane.
        va, na, ia = make_cube(half=0.5)
        vb, nb, ib = make_cube(half=0.5)
        self._cube_a = self.pt.add_mesh(va, ia, normals=na,
                                        material_id=red_mat, deformable=False)
        self._cube_b = self.pt.add_mesh(vb, ib, normals=nb,
                                        material_id=gold_mat, deformable=False)
        self.pt.add_sphere(center=(0.0, 0.6, 0.0), radius=0.6,
                           material_id=ball_mat)
        self.pt.add_plane(normal=(0.0, 1.0, 0.0), offset=0.0,
                          material_id=ground_mat)
        # A directional key light + a warm point light for local falloff/shadows.
        self.pt.add_light("directional", direction=(0.5, 1.0, 0.4),
                          color=(1.05, 1.0, 0.9))
        self.pt.add_light("point", position=(-1.5, 3.0, 2.0),
                          color=(6.0, 3.5, 1.5))
        self.pt.set_path_depth(max_depth=6)
        self.pt.init_gl()

    def render(self, frame, time):
        if self.pt is None:
            self._build()
        t = float(time)

        # Slowly orbit the camera around the scene centre.
        angle = 0.2 * t
        radius = 5.0
        eye = (radius * math.sin(angle), 2.0, radius * math.cos(angle))
        target = (0.0, 0.6, 0.0)
        self.pt.set_camera_lookat(eye=eye, target=target, up=(0.0, 1.0, 0.0),
                                  fov_y_deg=Demo.FOV_Y,
                                  aspect=self._width / float(self._height))

        # Spin the cubes in place (rigid transforms -> rigid motion vectors). The
        # gold cube also bobs vertically to exercise a translating rigid instance.
        self.pt.set_instance_transform(
            self._cube_a, xform(rot_y(1.1 * t), (-1.6, 0.5, 0.2)))
        # Lifted clear of the floor: a tumbling half-extent-0.5 cube swings its
        # corners ~0.87 below its centre, so keep the centre above ~0.87.
        bob = 1.3 + 0.15 * math.sin(1.5 * t)
        self.pt.set_instance_transform(
            self._cube_b, xform(rot_y(-0.8 * t) @ rot_x(0.4 * t),
                                (1.6, bob, -0.2)))

        # Everything animates every frame: restart accumulation and burst a few
        # samples into the temporal denoiser.
        self.pt.render(reset=True, spp=Demo.SPP)
        self.pt.present()


demo = Demo()

ret = sb.init(ctypes.byref(options(args)))
if ret != 0:
    exit(ret)

ret = sb.run()
if ret != 0:
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
    stopped.wait(timeout=5.0)
