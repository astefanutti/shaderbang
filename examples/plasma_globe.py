#!/usr/bin/env -S uv run --script

# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "shaderbang",
#     "pyopengl",
#     "warp-lang==1.17.0",
#     "numpy",
# ]
#
# [tool.uv.sources]
# shaderbang = { path = "..", editable = true }   # this checkout (branch plasma-globe): the key table and the ABI fix live here
# ///

"""
Plasma Globe
============

A physically-based plasma globe: dielectric-breakdown discharges grown by NVIDIA Warp inside a
glass sphere, carried by the buoyant convection of the gas they heat, and rendered by a GLSL
path tracer, adapted to run with Shaderbang.

Physics (see docs/plasma_globe.md for provenance and measurements)
------------------------------------------------------------------
* A commercial plasma globe is a near-atmospheric dielectric-barrier discharge: Ne + few % Xe at
  740 Torr, ~26 kHz, ~5 kV, ~1 mA total (PPPL-4485, Campanell et al. 2010). Filaments re-strike
  every half-cycle along the hot, density-depleted channel left by the previous one; that channel
  rises by buoyancy at ~1 cm/s, which is why filaments drift upward.
* Growth: dielectric breakdown model (Niemeyer et al.; Kim & Lin 2004/2007; Kim, Sewall, Sud,
  Lin 2007) with the harmonic-split potential phi = u_BC + sum_j q_j / |x - x_j| -- the exact
  concentric-sphere Dirichlet solution plus conductor charges on the filaments. Filaments repel
  each other through this shared field; nothing artificial keeps them apart.
* Memory + convection: filaments heat the gas (Boussinesq stable fluids, Bickel/Wicke/Gross 2006
  mechanism); the hot channel lowers the breakdown field (E/N) so the next strike follows it, and
  the filament nodes are advected by the flow (hybrid model: persistent channels + re-strikes).
* Current budget: a small series capacitance limits the total current; each attached filament
  draws a share through its glass termination; a finger on the glass multiplies that share ~215x,
  so the touched filament brightens while the others starve and retract. The filament count is
  emergent (0 below ~3 kV, ~27 at 5 kV, saturating near 30).

Operating point (MEASURED = PPPL-4485; DERIVED = neon at 740 Torr / 300 K; CHOSEN = design)
    electrode R1 = 1.5 cm (MEASURED)       glass R2 = 7.5 cm inner, 2.5 mm thick (MEASURED/CHOSEN)
    gas Ne + 2 % Xe, 740 Torr, 300 K       drive 26 kHz, 5 kV, 1 mA (MEASURED)
    node spacing 1.5 mm, a = h/4 (CHOSEN)  strike/sustain currents 40 / 32 uA (CHOSEN, N_max ~ 30)
    P_gas 0.75 W top-down (CHOSEN)         gas grid 64^3 over 16 cm, render grid 96^3 (CHOSEN)

Keyboard Controls
-----------------
    P                       Pause / resume the simulation
    Space / Right           Advance one frame (works while paused)
    R                       Reset the simulation
    G                       Invert gravity (filaments must still rise in the world frame)
    I                       Toggle an ice cap on top of the globe (filaments bend away from it)
    - / =  or V / Shift+V   Drive voltage -/+ 250 V (2-8 kV: the filament count law)
    [ / ]  or N / Shift+N   Drive frequency -/+ 2 kHz (10-40 kHz)
    E / Shift+E             Growth exponent eta x0.8 / x1.25 (120: smooth ropes, 3: lightning trees)
    Y / Shift+Y             Thermal memory exponent gamma -/+ 0.5
    F / Shift+F             Finger charge -/+ 0.01
    T                       Cycle the gas preset (tyrian / Ne+Xe / Ne / Ar / Kr / coral)
    X                       Toggle quincunx surface-charge mode
    H                       Print these keyboard controls
    J                       Toggle the hybrid re-strike model (off = persistent channels only)
    L / A                   Toggle line lights / glow
    Up / Down               Glow width x1.25 / x0.8
    , / .  or O / Shift+O   Exposure bias -/+ 0.5 EV
    U                       Toggle the temporal upscale (off = native full-resolution reference path)
    W                       Toggle the tree wireframe overlay
    0-9                     Debug views (0 beauty, 1 emissive, 2 layer id, 3 motion vectors, 4 history
                            weight, 5 candidates per ray, 6 grid occupancy, 7 temperature slice,
                            8 surface charge, 9 native reference)
    B                       Print per-pass timings and simulation counters
    Ctrl+S                  Dump the simulation state for the headless validator
    Every change prints '[keys] <setting> <new value>'.

Mouse Controls
--------------
    Left drag               Orbit the camera
    Ctrl + left drag        Finger on the glass
    Right drag              Track (pan) the camera
    Scroll wheel            Dolly (zoom) the camera

Touchscreen Controls
--------------------
    Finger on the globe     Finger on the glass (up to 10)
    1 finger elsewhere      Orbit the camera (trackball)
    2-3 fingers elsewhere   Track, dolly, and rotate the camera

Trackpad Controls
-----------------
    2 fingers               Orbit, dolly, and rotate the camera
    3 fingers               Track, dolly, and rotate the camera

Running
-------
From a bare VT (the display server must not hold the DRM device)::

    python examples/plasma_globe.py --mode 3840x2160 --triple-buffer
    python examples/plasma_globe.py --mode 3840x2160 --test -n 3600     # invariants every 60 frames
    python examples/plasma_globe.py --mode 3840x2160 --profile          # per-pass GPU timings

The internal render resolution is the mode divided by ``--internal-scale`` (default 2) with a 2x
temporal upscale; ``--gas-res 96`` trades ~1 ms of graph time for a finer plume. Everything the
app does can be exercised without a display: ``python -m plasma.headless_render --sim`` (from
``examples/``) runs the same Globe and Renderer into an offscreen 4K framebuffer, and
``python examples/plasma_globe_headless.py check`` runs the growth-engine invariants.
"""

import argparse
import ctypes
import glob
import math
import os
import stat
import sys
import signal
import threading
import time

import numpy as np
import warp as wp

from enum import auto, Flag
from typing import Optional

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plasma import params as P
from plasma.params import ParamsBuffer, MAX_FINGERS


parser = argparse.ArgumentParser(description="Run the plasma globe")
parser.add_argument("-D", "--device", metavar="DEVICE", type=Path,
                    help="the DRM device")
parser.add_argument("-C", "--connector", metavar="CONNECTOR", type=int,
                    help="the DRM connector")
parser.add_argument("--mode", metavar="MODE", type=str,
                    help="the name of the video mode, e.g., 3840x2160")
parser.add_argument("--refresh", metavar="FREQ", type=int,
                    help="the vertical refresh rate in Hz")
parser.add_argument("--async-page-flip", action=argparse.BooleanOptionalAction,
                    help="use async page flipping")
parser.add_argument("--atomic-drm-mode", action=argparse.BooleanOptionalAction,
                    help="use atomic mode setting")
parser.add_argument("--triple-buffer", action=argparse.BooleanOptionalAction,
                    help="use triple buffering")
parser.add_argument("-n", "--frames", metavar="N", type=int,
                    help="run for N frames and exit")
parser.add_argument("--internal-scale", metavar="S", type=int, default=2,
                    help="render at (display / S) and upscale temporally (default 2)")
parser.add_argument("--gas-res", metavar="N", type=int, default=64,
                    help="gas grid resolution (default 64)")
parser.add_argument("--seed", metavar="SEED", type=int, default=1,
                    help="random seed of the discharge growth")
parser.add_argument("--preset", metavar="GAS", type=str, default="tyrian",
                    help="gas preset: tyrian (Ne/Kr/Xe, the pink globe of the recordings), ne_xe, ne, ar, kr, "
                         "coral (the green forking globes); 'video' is an alias of tyrian")
parser.add_argument("--profile", action="store_true",
                    help="print per-pass GPU timings every 120 frames")
parser.add_argument("--test", action="store_true",
                    help="check simulation invariants every 60 frames and fail on violation")
args = parser.parse_args()

wp.init()
wp.set_device("cuda")

FRAME_DT = 1.0 / 60.0


class State(Flag):
    RUN = auto()
    STEP = auto()
    TAAU = auto()
    GLOW = auto()
    LIGHTS = auto()
    WIREFRAME = auto()
    HYBRID = auto()
    QUINCUNX = auto()
    ICE = auto()
    INVERT = auto()


state = State.RUN | State.TAAU | State.GLOW | State.LIGHTS | State.HYBRID
debug_view = 0
gas_presets = ["tyrian", "ne_xe", "ne", "ar", "kr", "coral"]


def quat_from_unit_vectors(from_vec: wp.vec3, to_vec: wp.vec3) -> wp.quat:
    """Copied from examples/cloth.py."""
    r = wp.dot(from_vec, to_vec) + 1.0
    if r < 1e-6:
        r = 0.0
        if abs(from_vec[0]) > abs(from_vec[2]):
            quat = wp.quat(-from_vec[1], from_vec[0], 0.0, r)
        else:
            quat = wp.quat(0.0, -from_vec[2], from_vec[1], r)
    else:
        cross = wp.cross(from_vec, to_vec)
        quat = wp.quat(cross[0], cross[1], cross[2], r)
    return wp.normalize(quat)


def ray_to_sphere(origin, direction, center, radius) -> Optional[tuple[float, float]]:
    """Both intersection distances of a ray with a sphere (copied from examples/cloth.py)."""
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc)) - radius * radius
    h = b * b - c
    if h < 0.0:
        return None
    h = math.sqrt(h)
    return -b - h, -b + h


class Camera(Input):
    """Orbit camera (orbit / rotate / dolly / track copied from examples/cloth.py) exposing a
    numpy basis, view-projection matrices and a display-space pick ray for the GLSL tracer."""

    UP = wp.vec3(0.0, 1.0, 0.0)
    RIGHT = wp.vec3(1.0, 0.0, 0.0)
    EPS = 0.000001
    MIN_DISTANCE = 0.15
    MAX_DISTANCE = 3.0
    FOV_Y = 40.0
    # starting pose (from a Ctrl+S dump of the view the user settled on, 2026-09-15): 29 cm from the
    # globe, 30 deg to the right of the front, level with the bulb
    DEFAULT_EYE = (0.1551, -0.0306, 0.2672)
    DEFAULT_TARGET = (0.0105, -0.0219, 0.0202)

    def __init__(self):
        super().__init__("camera")
        self.pos = wp.vec3(*Camera.DEFAULT_EYE)
        self.target = wp.vec3(*Camera.DEFAULT_TARGET)
        self.forward = wp.normalize(self.target - self.pos)
        self.right = wp.normalize(wp.cross(self.forward, Camera.UP))
        self.up = wp.normalize(wp.cross(self.right, self.forward))
        self.quat = quat_from_unit_vectors(self.up, Camera.UP)
        self.width = 1
        self.height = 1
        self.view = np.eye(4, dtype=np.float32)
        self.proj = np.eye(4, dtype=np.float32)
        self.vp = np.eye(4, dtype=np.float32)
        self.vp_prev = np.eye(4, dtype=np.float32)
        self.moved = True

    def init(self, width, height):
        self.width, self.height = width, height
        self.update_matrices()
        self.vp_prev = self.vp.copy()

    def pre_render(self, **kwargs):
        self.vp_prev = self.vp.copy()
        self.update_matrices()

    # -- basis / matrices -----------------------------------------------------------------------
    def basis(self):
        """(eye, u, v, w): ray = eye + normalize(u * ndc_x + v * ndc_y + w), as the tracer expects."""
        aspect = self.width / float(self.height)
        t = math.tan(math.radians(Camera.FOV_Y) * 0.5)
        f = np.array(self.forward, dtype=np.float64)
        r = np.array(self.right, dtype=np.float64)
        u = np.array(self.up, dtype=np.float64)
        return (np.array(self.pos, dtype=np.float64), r * t * aspect, u * t, f)

    def update_matrices(self):
        eye, u, v, w = self.basis()
        r = np.array(self.right, dtype=np.float64)
        up = np.array(self.up, dtype=np.float64)
        f = np.array(self.forward, dtype=np.float64)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = r, up, -f
        view[:3, 3] = -view[:3, :3] @ eye
        aspect = self.width / float(self.height)
        near, far = 0.01, 100.0
        t = 1.0 / math.tan(math.radians(Camera.FOV_Y) * 0.5)
        proj = np.zeros((4, 4))
        proj[0, 0] = t / aspect
        proj[1, 1] = t
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2.0 * far * near / (near - far)
        proj[3, 2] = -1.0
        new_vp = (proj @ view).astype(np.float32)
        self.moved = not np.array_equal(new_vp, self.vp)
        self.view, self.proj, self.vp = view.astype(np.float32), proj.astype(np.float32), new_vp

    def pick_ray(self, screen_x, screen_y):
        """World-space ray through a display pixel (origin, unit direction)."""
        eye, u, v, w = self.basis()
        ndc_x = 2.0 * screen_x / self.width - 1.0
        ndc_y = 1.0 - 2.0 * screen_y / self.height
        d = u * ndc_x + v * ndc_y + w
        return eye, d / np.linalg.norm(d)

    def pick_globe(self, screen_x, screen_y) -> Optional[np.ndarray]:
        """Unit direction of the point where a display pixel hits the outer glass, or None."""
        o, d = self.pick_ray(screen_x, screen_y)
        hit = ray_to_sphere(o, d, np.zeros(3), P.R2O)
        if hit is None or hit[0] < 0.0:
            return None
        p = o + d * hit[0]
        return (p / np.linalg.norm(p)).astype(np.float32)

    # -- gestures (examples/cloth.py:1213-1258) -------------------------------------------------
    def orbit(self, dx, dy, gain=0.001):
        self.rotate(-wp.TAU * dx, -wp.TAU * dy, gain)

    def rotate(self, dth, dph, gain=1.0):
        vec = self.pos - self.target
        vec = wp.quat_rotate(self.quat, vec)
        radius = wp.length(vec)
        theta = wp.atan2(vec[0], vec[2])
        theta += dth * gain
        phi = wp.acos(wp.clamp(vec[1] / radius, -1.0, 1.0))
        phi += dph * gain
        phi = wp.max(Camera.EPS, wp.min(wp.pi - Camera.EPS, phi))
        sin_phi_radius = wp.sin(phi) * radius
        vec = wp.vec3(sin_phi_radius * wp.sin(theta), wp.cos(phi) * radius, sin_phi_radius * wp.cos(theta))
        vec = wp.quat_rotate_inv(self.quat, vec)
        self.pos = self.target + vec
        self.forward = wp.normalize(self.target - self.pos)
        self.right = wp.normalize(wp.cross(self.forward, Camera.UP))
        self.up = wp.normalize(wp.cross(self.right, self.forward))

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


class Fingers:
    """Up to MAX_FINGERS contact directions on the glass, keyed by input slot."""

    MOUSE_SLOT = MAX_FINGERS - 1

    def __init__(self):
        self.dirs: dict[int, np.ndarray] = {}

    def press(self, slot, direction) -> bool:
        if direction is None or (len(self.dirs) >= MAX_FINGERS and slot not in self.dirs):
            return False
        self.dirs[slot] = direction
        return True

    def move(self, slot, direction):
        if slot in self.dirs and direction is not None:
            self.dirs[slot] = direction

    def release(self, slot):
        self.dirs.pop(slot, None)

    def active(self):
        return list(self.dirs.values())


TRACK_SENSITIVITY = 0.15      # pan (track) drags: mouse right drag, 2-3 fingers on the touchscreen, 3 fingers on the trackpad
TRACKPAD_ORBIT_GAIN = 0.25    # two-finger orbit on the trackpad: a full swipe turns a quarter circle (was a full circle; the
                              # mouse turns a half circle over the screen); the rotations themselves are otherwise unchanged


class Mouse(shaderbang.input.Mouse):

    def __init__(self):
        super().__init__("mouse")
        self.finger = False

    def pre_render(self, **kwargs):
        if self.deltaW != 0:
            camera.dolly(self.deltaW, gain=0.1)
        if self.click and not self.finger:
            if self.button == EV_KEY.BTN_LEFT and keyboard.down(any, EV_KEY.KEY_LEFTCTRL, EV_KEY.KEY_RIGHTCTRL):
                self.finger = fingers.press(Fingers.MOUSE_SLOT, camera.pick_globe(self.mouseX, self.mouseY))
        elif self.drag:
            if self.finger:
                fingers.move(Fingers.MOUSE_SLOT, camera.pick_globe(self.mouseX, self.mouseY))
            elif self.button == EV_KEY.BTN_LEFT:
                camera.orbit(self.deltaX, self.deltaY, 0.5 / self.resolution[1])
            elif self.button == EV_KEY.BTN_RIGHT:
                camera.track(self.deltaX, self.deltaY, gain=TRACK_SENSITIVITY * 0.001)
        elif self.finger:
            fingers.release(Fingers.MOUSE_SLOT)
            self.finger = False


class FingerSlot(TouchSlot):

    def __init__(self):
        super().__init__()
        self.finger = False


class Touchscreen(shaderbang.input.MultiTouch[FingerSlot]):

    def __init__(self):
        super().__init__("touchscreen", FingerSlot)

    def holroyd_trackball(self, screen_x, screen_y) -> wp.vec3:
        width, height = self.resolution
        vec = wp.vec3(screen_x / width * 2 - 1.0, - screen_y / height * 2 + 1.0, 0.0)
        len2 = wp.length_sq(vec)
        vec[2] = 0.5 / wp.sqrt(len2) if len2 > 0.5 else wp.sqrt(1.0 - len2)
        return vec

    def pre_render(self, **kwargs):
        slots: list[FingerSlot] = []
        for i, slot in enumerate(self.slots):
            if slot.touch:
                slot.finger = fingers.press(i, camera.pick_globe(slot.touchX, slot.touchY))
            elif slot.drag:
                if slot.finger:
                    fingers.move(i, camera.pick_globe(slot.touchX, slot.touchY))
                else:
                    slots.append(slot)
            elif slot.finger:
                fingers.release(i)
                slot.finger = False

        n = len(slots)
        if n == 1:
            slot = slots[0]
            u = quat_from_unit_vectors(Camera.UP, camera.up)
            v = quat_from_unit_vectors(Camera.RIGHT, camera.right)
            quat = wp.mul(u, v)
            vec1 = wp.quat_rotate(quat, self.holroyd_trackball(slot.prevX, slot.prevY))
            vec2 = wp.quat_rotate(quat, self.holroyd_trackball(slot.touchX, slot.touchY))
            theta = wp.atan2(wp.dot(wp.cross(vec2, vec1), camera.UP), wp.dot(vec2, vec1))
            camera.rotate(wp.PI * theta, - wp.TAU * slot.deltaY * 0.5 / self.resolution[1])
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
            camera.track(dx, dy, gain=TRACK_SENSITIVITY * 0.001)
            camera.dolly_scale(scale)
            camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)


class Trackpad(shaderbang.input.MultiTouch[TouchSlot]):

    def __init__(self):
        super().__init__("trackpad")

    def pre_render(self, **kwargs):
        slots = [slot for slot in self.slots if slot.drag]
        n = len(slots)
        if n < 2:
            return
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
            camera.orbit(dx, dy, TRACKPAD_ORBIT_GAIN / self.resolution[1])
        else:
            camera.track(dx, dy, gain=TRACK_SENSITIVITY * 0.002)
        camera.dolly_scale(scale)
        camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)


def camera_pose():
    """(eye, target, up) of the current camera as plain tuples, for logs and dumps."""
    return (tuple(float(v) for v in camera.pos), tuple(float(v) for v in camera.target), tuple(float(v) for v in camera.up))


def log_key(msg):
    """Every key that changes a setting prints its new value (the effect is otherwise hard to see)."""
    print(f"[keys] {msg}", flush=True)


def print_controls():
    """Key H: print the Keyboard Controls section of the module docstring."""
    doc = __doc__ or ""
    start = doc.find("Keyboard Controls")
    end = doc.find("Mouse Controls", start)
    section = doc[start:end].rstrip() if start >= 0 else "(no docstring)"
    print("\n" + section + "\n", flush=True)


class Keyboard(shaderbang.input.Keyboard):

    def __init__(self):
        super().__init__("keyboard")

    def pre_render(self, **kwargs):
        global state, debug_view
        shift = self.down(any, EV_KEY.KEY_LEFTSHIFT, EV_KEY.KEY_RIGHTSHIFT)
        ctrl = self.down(any, EV_KEY.KEY_LEFTCTRL, EV_KEY.KEY_RIGHTCTRL)
        if self.pressed(EV_KEY.KEY_P):
            state ^= State.RUN
            log_key("run" if state & State.RUN else "paused")
        if self.down(any, EV_KEY.KEY_RIGHT, EV_KEY.KEY_SPACE):
            state |= State.STEP
        if self.pressed(EV_KEY.KEY_R):
            globe.request_reset()
            log_key("reset")
        if self.pressed(EV_KEY.KEY_G):
            state ^= State.INVERT
            log_key(f"gravity {'inverted' if state & State.INVERT else 'normal'}")
        if self.pressed(EV_KEY.KEY_I):
            state ^= State.ICE
            log_key(f"ice cap {'on' if state & State.ICE else 'off'}")
        if self.pressed(EV_KEY.KEY_MINUS) or (self.pressed(EV_KEY.KEY_V) and not shift):
            globe.knobs["voltage"] = max(2000.0, globe.knobs["voltage"] - 250.0)
            log_key(f"voltage {globe.knobs['voltage']:.0f} V")
        if self.pressed(EV_KEY.KEY_EQUAL) or (self.pressed(EV_KEY.KEY_V) and shift):
            globe.knobs["voltage"] = min(8000.0, globe.knobs["voltage"] + 250.0)
            log_key(f"voltage {globe.knobs['voltage']:.0f} V")
        if self.pressed(EV_KEY.KEY_LEFTBRACE) or (self.pressed(EV_KEY.KEY_N) and not shift):
            globe.knobs["frequency"] = max(10.0e3, globe.knobs["frequency"] - 2.0e3)
            log_key(f"frequency {globe.knobs['frequency'] / 1e3:.0f} kHz")
        if self.pressed(EV_KEY.KEY_RIGHTBRACE) or (self.pressed(EV_KEY.KEY_N) and shift):
            globe.knobs["frequency"] = min(40.0e3, globe.knobs["frequency"] + 2.0e3)
            log_key(f"frequency {globe.knobs['frequency'] / 1e3:.0f} kHz")
        if self.pressed(EV_KEY.KEY_E):
            globe.knobs["eta"] = max(1.0, min(1000.0, globe.knobs["eta"] * (1.25 if shift else 0.8)))
            log_key(f"eta {globe.knobs['eta']:.1f}")
        if self.pressed(EV_KEY.KEY_Y):
            globe.knobs["gamma"] = max(0.0, min(6.0, globe.knobs["gamma"] + (0.5 if shift else -0.5)))
            log_key(f"gamma {globe.knobs['gamma']:.1f}")
        if self.pressed(EV_KEY.KEY_F):
            globe.knobs["q_finger"] = max(0.0, min(1.0, globe.knobs["q_finger"] + (0.01 if shift else -0.01)))
            log_key(f"finger charge {globe.knobs['q_finger']:.2f}")
        if self.pressed(EV_KEY.KEY_T):
            globe.cycle_preset()
            log_key(f"gas preset {globe.presets[globe.preset_index]}")
        if self.pressed(EV_KEY.KEY_X):
            state ^= State.QUINCUNX
            log_key(f"quincunx sigma {'on' if state & State.QUINCUNX else 'off'}")
        if self.pressed(EV_KEY.KEY_H):
            print_controls()
        if self.pressed(EV_KEY.KEY_J):
            state ^= State.HYBRID
            log_key(f"hybrid re-strikes {'on' if state & State.HYBRID else 'off (persistent only)'}")
        if self.pressed(EV_KEY.KEY_L):
            state ^= State.LIGHTS
            log_key(f"line lights {'on' if state & State.LIGHTS else 'off'}")
        if self.pressed(EV_KEY.KEY_A):
            state ^= State.GLOW
            log_key(f"glow {'on' if state & State.GLOW else 'off'}")
        if self.pressed(EV_KEY.KEY_UP):
            renderer.set_glow_width(renderer.knobs["glow_width"] * 1.25)
            log_key(f"glow width {renderer.knobs['glow_width']:.0f} px")
        if self.pressed(EV_KEY.KEY_DOWN):
            renderer.set_glow_width(renderer.knobs["glow_width"] / 1.25)
            log_key(f"glow width {renderer.knobs['glow_width']:.0f} px")
        if self.pressed(EV_KEY.KEY_COMMA) or (self.pressed(EV_KEY.KEY_O) and not shift):
            renderer.knobs["exposure_bias"] -= 0.5
            log_key(f"exposure bias {renderer.knobs['exposure_bias']:+.1f} EV")
        if self.pressed(EV_KEY.KEY_DOT) or (self.pressed(EV_KEY.KEY_O) and shift):
            renderer.knobs["exposure_bias"] += 0.5
            log_key(f"exposure bias {renderer.knobs['exposure_bias']:+.1f} EV")
        if self.pressed(EV_KEY.KEY_U):
            state ^= State.TAAU
            log_key(f"temporal upscale {'on' if state & State.TAAU else 'off (native)'}")
        if self.pressed(EV_KEY.KEY_W):
            state ^= State.WIREFRAME
            log_key(f"wireframe {'on' if state & State.WIREFRAME else 'off'}")
        for i, key in enumerate((EV_KEY.KEY_0, EV_KEY.KEY_1, EV_KEY.KEY_2, EV_KEY.KEY_3, EV_KEY.KEY_4,
                                 EV_KEY.KEY_5, EV_KEY.KEY_6, EV_KEY.KEY_7, EV_KEY.KEY_8, EV_KEY.KEY_9)):
            if self.pressed(key):
                debug_view = i
                log_key(f"debug view {i}")
        if self.pressed(EV_KEY.KEY_B):
            renderer.print_timings()
            globe.print_counters()
        if ctrl and self.pressed(EV_KEY.KEY_S):
            pose = camera_pose()
            globe.dump_state(camera_pos=np.array(pose[0]), camera_target=np.array(pose[1]), camera_up=np.array(pose[2]))
            log_key("state dumped; camera eye ({:.3f}, {:.3f}, {:.3f}) target ({:.3f}, {:.3f}, {:.3f}) up ({:.3f}, {:.3f}, {:.3f})".format(*pose[0], *pose[1], *pose[2]))

    def post_render(self, **kwargs):
        global state
        state &= ~State.STEP


def input_from_device(dev: Device):
    if dev.has(EV_REL) and dev.has(EV_KEY.BTN_LEFT):
        shaderbang.input.ButtonMouse(dev.name, dev, mouse)
    elif dev.has(EV_KEY) and dev.has(EV_KEY.KEY_A):
        shaderbang.input.AsciiKeyboard(dev.name, dev, keyboard)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_DIRECT):
        shaderbang.input.Touchscreen(dev.name, dev, touchscreen)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_POINTER):
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


# The simulation (Globe) and the GLSL renderer (Renderer) Inputs are assembled from the
# plasma package modules: see plasma/globe.py and plasma/renderer.py.
from plasma.globe import Globe, SimFlags            # noqa: E402
from plasma.renderer import Renderer, RenderFlags   # noqa: E402


def sim_flags():
    return SimFlags(running=bool(state & (State.RUN | State.STEP)), invert=State.INVERT in state,
                    ice=State.ICE in state, hybrid=State.HYBRID in state, quincunx=State.QUINCUNX in state)


def render_flags():
    return RenderFlags(taau=State.TAAU in state, glow=State.GLOW in state, lights=State.LIGHTS in state,
                       invert=State.INVERT in state)


class Test(Input):
    """``--test``: simulation invariants every 60 frames, read one frame late from the device
    (a stream sync here costs a stall only in test runs). Fails the run on the first violation."""

    def __init__(self):
        super().__init__("Test")
        self.frame = 0
        self.low_count_frames = 0

    def post_render(self, **kwargs):
        self.frame += 1
        if self.frame % 60:
            return
        wp.synchronize()
        e = globe.engine
        c = globe.counters_now()
        flags = e.flags.numpy()
        alive = (flags & P.NODE_ALIVE) != 0
        pos = e.pos.numpy()[alive]
        parent = e.parent.numpy()
        has_parent = alive & (parent >= 0)
        dangling = int((has_parent & ~alive[np.where(parent >= 0, parent, 0)]).sum())
        segments = int(globe.pub.seg_valid.numpy().sum())
        hist = renderer.read_histogram()
        rays = max(int(hist[:64].sum()), 1)
        capped = int(hist[64]) / rays
        st = e.t_state.numpy()
        problems = []
        if not np.isfinite(pos).all():
            problems.append("non-finite node positions")
        if c["live"] >= e.n_max:
            problems.append(f"node pool exhausted ({c['live']} >= {e.n_max})")
        if segments > globe.pub.seg_valid.shape[0]:
            problems.append(f"segments {segments} > SEG_MAX")
        if dangling:
            problems.append(f"{dangling} live nodes with a dead parent")
        if capped > 1e-3:
            problems.append(f"{capped * 100:.2f} % of rays hit the candidate cap")
        if int((st == P.TREE_ATTACHED).sum()) != c["attached"]:
            problems.append("attached count disagrees between engine and counters")
        if c["attached"] < 4 and globe.knobs["voltage"] >= 4000.0 and State.RUN in state:
            self.low_count_frames += 60
            if self.low_count_frames > 120:
                print(f"[test] warning: {c['attached']} filaments for {self.low_count_frames / 60:.0f} s at "
                      f"{globe.knobs['voltage']:.0f} V")
        else:
            self.low_count_frames = 0
        if problems:
            print(f"[test] frame {self.frame}: FAIL: " + "; ".join(problems), file=sys.stderr)
            globe.dump_state()
            os._exit(2)
        print(f"[test] frame {self.frame}: ok (live {c['live']}, attached {c['attached']}, segments {segments}, "
              f"capped {capped * 100:.3f} %)")


camera = Camera()
fingers = Fingers()
globe = Globe(camera, fingers, sim_flags, args)
_preset = {"video": "tyrian"}.get(args.preset, args.preset)
if _preset in globe.presets:
    globe.preset_index = globe.presets.index(_preset)         # applied when the globe builds (init)
else:
    print(f"[globe] unknown preset {args.preset!r}, using {globe.presets[0]}")
renderer = Renderer(camera, globe, render_flags, lambda: debug_view, args)
test = Test() if args.test else None

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
