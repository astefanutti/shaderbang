# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Headless render of the plasma globe tracer (no DRM, no display): feeds the publish-stage
buffers of a filament tree (a synthetic star by default, or a dumped simulation state) to
``shaders/trace.comp`` through a GL 4.6 context on the EGL device platform, times the pass with
a GL timer query, and writes tone-mapped PNGs for eyeballing plus the candidates-per-ray
histogram the render budget is built on.

    ../.venv/bin/python -m plasma.headless_render --out /tmp/plasma_trace.png
"""

import argparse
import ctypes
import math
import os
import re
import sys
import time

import numpy as np
import warp as wp

from OpenGL import setPlatform
setPlatform("egl")
from OpenGL.EGL import *                                   # noqa: E402,F401
from OpenGL.EGL.EXT.platform_device import *               # noqa: E402,F401
from OpenGL.EGL.EXT.device_base import *                   # noqa: E402,F401
from OpenGL.GL import *                                    # noqa: E402,F401

from plasma import publish
from plasma.params import R2O
from plasma.circuit import SIG_MAX

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")


def make_context():
    n = EGLint()
    eglQueryDevicesEXT(0, None, n)
    devs = (EGLDeviceEXT * n.value)()
    eglQueryDevicesEXT(n.value, devs, n)
    dpy = None
    for i in range(n.value):
        d = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devs[i], None)
        if d == EGL_NO_DISPLAY or not eglInitialize(d, EGLint(), EGLint()):
            continue
        if b"NVIDIA" in eglQueryString(d, EGL_VENDOR):
            dpy = d
            break
    assert dpy is not None, "no NVIDIA EGL device"
    assert eglBindAPI(EGL_OPENGL_API)
    ca = [EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
          EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE]
    cfgs = (EGLConfig * 1)()
    nc = EGLint()
    assert eglChooseConfig(dpy, (EGLint * len(ca))(*ca), cfgs, 1, nc) and nc.value
    xa = [EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6,
          EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, EGL_NONE]
    ctx = eglCreateContext(dpy, cfgs[0], EGL_NO_CONTEXT, (EGLint * len(xa))(*xa))
    assert ctx != EGL_NO_CONTEXT
    assert eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)
    return dpy, ctx


def load_source(path, defines=()):
    src = open(path).read()
    src = re.sub(r'#include\s+"([^"]+)"', lambda m: open(os.path.join(os.path.dirname(path), m.group(1))).read(), src)
    lines = src.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("#version"):
            lines[i] = l + "\n" + "\n".join(f"#define {d}" for d in defines)
            break
    return "\n".join(lines)


def compile_compute(path, defines=()):
    sh = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(sh, load_source(path, defines))
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(f"{path}: {glGetShaderInfoLog(sh)}")
    prog = glCreateProgram()
    glAttachShader(prog, sh)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(f"{path}: {glGetProgramInfoLog(prog)}")
    return prog


def image2d(w, h, fmt):
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexStorage2D(GL_TEXTURE_2D, 1, fmt, w, h)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def ssbo(data, binding):
    buf = int(glGenBuffers(1))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    data = np.ascontiguousarray(data)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf)
    return buf


def camera_basis(eye, target, up, fov_y_deg, aspect):
    eye = np.asarray(eye, np.float64); target = np.asarray(target, np.float64)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    t = math.tan(math.radians(fov_y_deg) * 0.5)
    view = np.eye(4); view[0, :3], view[1, :3], view[2, :3] = r, u, -f; view[:3, 3] = -view[:3, :3] @ eye
    tt = 1.0 / t; near, far = 0.01, 100.0
    proj = np.zeros((4, 4)); proj[0, 0] = tt / aspect; proj[1, 1] = tt
    proj[2, 2] = (far + near) / (near - far); proj[2, 3] = 2 * far * near / (near - far); proj[3, 2] = -1.0
    return eye, r * t * aspect, u * t, f, (proj @ view)


def params_ubo(eye, cu, cv, cw, vp, vp_prev, jitter, res, rad_scale, debug, lights_on, frame, volume_on):
    blob = np.zeros(64, np.float32)
    blob[0:3] = eye; blob[4:7] = cu; blob[8:11] = cv; blob[12:15] = cw
    blob[16:32] = np.asarray(vp, np.float32).T.reshape(-1)        # column-major
    blob[32:48] = np.asarray(vp_prev, np.float32).T.reshape(-1)
    blob[48:52] = (jitter[0], jitter[1], res[0], res[1])
    blob[52:56] = (rad_scale, 1.0, 0.0, 1.0)
    blob[56:60] = np.array([debug, lights_on, frame, volume_on], np.int32).view(np.float32)
    blob[60:64] = (0.0, 0.0, 0.0, 1.0)
    return blob


def tonemap_png(hdr, path, exposure=1.0):
    from PIL import Image
    c = np.clip(hdr[..., :3] * exposure, 0.0, None)
    c = c / (1.0 + c)                                    # Reinhard for the preview
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.clip(c, 1e-6, None), 1 / 2.4) - 0.055)
    Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)[::-1]).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/plasma_trace.png")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--rad-scale", type=float, default=60.0)
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-lights", action="store_true")
    ap.add_argument("--no-volume", action="store_true")
    ap.add_argument("--state", default=None, help=".npz simulation dump instead of the synthetic star")
    args = ap.parse_args()
    w, h = (int(v) for v in args.res.split("x"))

    wp.init()
    dev = "cuda:0"
    nodes, trees, n_live = publish.synthetic_star(device=dev)
    pub = publish.Publisher(dev)
    pub.launch(nodes, trees)
    wp.synchronize()
    segments = pub.segments.numpy()
    cell_start = pub.cell_start.numpy(); cell_count = pub.cell_count.numpy(); cell_items = pub.cell_items.numpy()
    lights = pub.lights.numpy()
    volume = pub.volume.numpy()                             # (z, y, x) vec4h
    seg_valid = pub.seg_valid.numpy()

    dpy, ctx = make_context()
    print("GL", glGetString(GL_VERSION).decode())
    prog = compile_compute(os.path.join(SHADER_DIR, "trace.comp"))
    tex_hdr = image2d(w, h, GL_RGBA16F); tex_em = image2d(w, h, GL_RGBA16F)
    tex_mv = image2d(w, h, GL_RG16F); tex_layer = image2d(w, h, GL_RG32F)
    glBindImageTexture(0, tex_hdr, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
    glBindImageTexture(1, tex_em, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
    glBindImageTexture(2, tex_mv, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F)
    glBindImageTexture(3, tex_layer, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F)
    # volume texture
    vol = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_3D, vol)
    n3 = publish.GRID_N
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA16F, n3, n3, n3)
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, n3, n3, n3, GL_RGBA, GL_HALF_FLOAT, np.ascontiguousarray(volume))  # float16 array: PyOpenGL converts integer views numerically
    for pn in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
        glTexParameteri(GL_TEXTURE_3D, pn, GL_LINEAR)
    for pn in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R):
        glTexParameteri(GL_TEXTURE_3D, pn, GL_CLAMP_TO_EDGE)
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, vol)
    # buffers
    ssbo(segments.astype(np.float32), 1)
    ssbo(cell_start.astype(np.int32), 2)
    ssbo(cell_count.astype(np.int32), 3)
    ssbo(cell_items.astype(np.int32), 4)
    ssbo(lights.astype(np.float32), 5)
    sigma = np.zeros((SIG_MAX * 2, 4), np.float32)
    # two saturated feet where trees 0 and 1 end, for the foot-disc look
    for k in (0, 1):
        foot = int(trees.foot.numpy()[k]); d = nodes.pos.numpy()[foot]; d = d / np.linalg.norm(d)
        sigma[2 * k] = (d[0], d[1], d[2], 1.0); sigma[2 * k + 1] = (0.012, 1.0, 0.0, 0.0)
    ssbo(sigma, 6)
    hist_buf = ssbo(np.zeros(68, np.int32), 7)
    # camera / params
    eye, cu, cv, cw, vp = camera_basis((0.0, 0.06, 0.32), (0.0, 0.03, 0.0), (0, 1, 0), 40.0, w / h)
    blob = params_ubo(eye, cu, cv, cw, vp, vp, (0.0, 0.0), (w, h), args.rad_scale, args.debug,
                      0 if args.no_lights else 1, 0, 0 if args.no_volume else 1)
    ubo = int(glGenBuffers(1)); glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferData(GL_UNIFORM_BUFFER, blob.nbytes, blob, GL_DYNAMIC_DRAW); glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo)

    glUseProgram(prog)
    gx, gy = (w + 7) // 8, (h + 7) // 8
    glDispatchCompute(gx, gy, 1); glMemoryBarrier(GL_ALL_BARRIER_BITS); glFinish()
    q = int(glGenQueries(1)[0])
    times = []
    for _ in range(args.iters):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, hist_buf)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, np.zeros(68, np.int32))
        glBeginQuery(GL_TIME_ELAPSED, q)
        glDispatchCompute(gx, gy, 1)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)
        glEndQuery(GL_TIME_ELAPSED)
        ns = ctypes.c_uint64(); glGetQueryObjectui64v(q, GL_QUERY_RESULT, ns)
        times.append(ns.value / 1e6)
    glFinish()
    hist = np.zeros(68, np.int32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, hist_buf)
    hist[:] = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, hist.nbytes), np.int32)
    # read back
    glBindTexture(GL_TEXTURE_2D, tex_hdr)
    hdr = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT).reshape(h, w, 4)
    glBindTexture(GL_TEXTURE_2D, tex_em)
    em = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT).reshape(h, w, 4)
    glBindTexture(GL_TEXTURE_2D, tex_layer)
    layer = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT).reshape(h, w, 4)[..., :2]
    glBindTexture(GL_TEXTURE_2D, tex_mv)
    mv = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT).reshape(h, w, 4)[..., :2]

    tonemap_png(hdr, args.out, args.exposure)
    tonemap_png(em, args.out.replace(".png", "_emissive.png"), args.exposure)
    counts = hist[:64]; centers = np.arange(64) * 8 + 4
    total = counts.sum()
    mean = float((counts * centers).sum() / max(total, 1))
    cum = np.cumsum(counts) / max(total, 1); p99 = int(centers[np.searchsorted(cum, 0.99)] if total else 0)
    lid = np.floor(layer[..., 0] + 0.25).astype(int)
    print(f"segments {int(seg_valid.sum())}, live nodes {n_live}; {w}x{h}")
    print(f"trace.comp: median {np.median(times):.3f} ms, min {min(times):.3f} ms over {args.iters} dispatches")
    print(f"candidates/ray (interior rays): mean {mean:.1f}, p99 {p99}, rays {total}, capped {hist[64]}")
    print("layers: " + ", ".join(f"L{i}={int((lid == i).sum())}" for i in range(4)))
    luma = hdr[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    eluma = em[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    print(f"HDR max {hdr[..., :3].max():.3f}, mean {hdr[..., :3].mean():.4f}; emissive max {em[..., :3].max():.3f}; |mv| max {np.abs(mv).max():.2f} px")
    print("luma percentiles 50/90/99/99.9: " + ", ".join(f"{np.percentile(luma, q):.3f}" for q in (50, 90, 99, 99.9)))
    print("mean luma / emissive luma per layer: " + ", ".join(f"L{i}={luma[lid == i].mean():.3f}/{eluma[lid == i].mean():.3f}" for i in range(4) if (lid == i).any()))
    tonemap_png(np.repeat((layer[..., :1] / 3.5), 4, axis=2), args.out.replace(".png", "_layers.png"), 1.0)
    nan = int(np.isnan(hdr).sum() + np.isnan(em).sum())
    print(f"NaNs: {nan}; wrote {args.out}")
    return 0 if nan == 0 else 1


if __name__ == "__main__" and "--full" not in sys.argv and "--sim" not in sys.argv:
    raise SystemExit(main())


# ---- full pipeline (renderer.py) on a fake globe -----------------------------------------------------

class _FakeCamera:
    def __init__(self, w, h, eye=None, target=None):
        self.width, self.height = w, h
        self.angle = 0.0
        self.eye0 = tuple(eye) if eye is not None else None          # fixed pose (--eye/--target), else the orbit
        self.target = tuple(target) if target is not None else (0.0, 0.03, 0.0)
        self.vp = np.eye(4, dtype=np.float32)
        self.vp_prev = np.eye(4, dtype=np.float32)
        self.update()

    def update(self):
        if self.eye0 is not None:
            c, s_ = math.cos(self.angle), math.sin(self.angle)
            eye = (c * self.eye0[0] + s_ * self.eye0[2], self.eye0[1], -s_ * self.eye0[0] + c * self.eye0[2])
        else:
            eye = (0.32 * math.sin(self.angle), 0.06, 0.32 * math.cos(self.angle))
        e, cu, cv, cw, vp = camera_basis(eye, self.target, (0, 1, 0), 40.0, self.width / self.height)
        self._basis = (e, cu, cv, cw)
        self.vp_prev = self.vp
        self.vp = vp.astype(np.float32)

    def basis(self):
        return self._basis


class _FakeGlobe:
    def __init__(self, device):
        from plasma.circuit import CircuitState
        self.nodes, self.trees, self.n_live = publish.synthetic_star(device=device)
        self.pub = publish.Publisher(device)
        self.circuit = CircuitState(device)
        # two saturated feet for the look
        sig_dir = np.zeros((SIG_MAX, 3), np.float32); amp = np.zeros(SIG_MAX, np.float32); rad = np.zeros(SIG_MAX, np.float32); alive = np.zeros(SIG_MAX, np.int32)
        for k in (0, 1):
            foot = int(self.trees.foot.numpy()[k]); d = self.nodes.pos.numpy()[foot]; sig_dir[k] = d / np.linalg.norm(d)
            amp[k] = 1.0; rad[k] = 0.012; alive[k] = 1
        self.circuit.sig_dir.assign(sig_dir); self.circuit.sig_amp.assign(amp); self.circuit.sig_radius.assign(rad); self.circuit.sig_alive.assign(alive)
        self.pub.launch(self.nodes, self.trees)
        wp.synchronize()


def run_full(args):
    """Exercise plasma.renderer end to end into an offscreen 'display' FBO and save the last frame."""
    import types
    from plasma.renderer import Renderer, RenderFlags
    dw, dh = (int(v) for v in args.display.split("x"))
    make_context()
    wp.init()
    dev = "cuda:0"
    globe = _FakeGlobe(dev)
    cam = _FakeCamera(dw, dh)
    rargs = types.SimpleNamespace(internal_scale=2, profile=False)
    flags = RenderFlags(taau=not args.no_taau, glow=not args.no_glow, lights=not args.no_lights, volume=not args.no_volume)
    renderer = Renderer(cam, globe, lambda: flags, lambda: args.debug, rargs)
    # offscreen "default framebuffer"
    fbo = int(glGenFramebuffers(1)); tex = image2d(dw, dh, GL_RGBA8)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
    renderer.init(dw, dh)
    renderer.knobs["rad_scale"] = args.rad_scale
    times = []
    for f in range(args.frames):
        cam.angle += math.radians(args.orbit_deg_per_frame)
        cam.update()
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        t0 = time.perf_counter()
        renderer.render()
        renderer.post_render()
        glFinish()
        times.append((time.perf_counter() - t0) * 1e3)
    renderer.print_timings()
    print(f"full frame (CPU wall incl. glFinish): median {np.median(times[5:]):.2f} ms over {len(times)-5} frames at {dw}x{dh} (internal {renderer.iw}x{renderer.ih})")
    hist = renderer.read_histogram()
    print(f"candidates histogram: capped {hist[64]}, rays {hist[:64].sum()}")
    glBindTexture(GL_TEXTURE_2D, tex)
    img = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    from PIL import Image
    Image.fromarray(np.frombuffer(img, np.uint8).reshape(dh, dw, 4)[::-1, :, :3]).save(args.out)
    print("wrote", args.out)


if __name__ == "__main__" and "--full" in sys.argv:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--display", default="3840x2160")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--orbit-deg-per-frame", type=float, default=0.5)
    ap.add_argument("--rad-scale", type=float, default=60.0)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--out", default="/tmp/plasma_full.png")
    ap.add_argument("--no-taau", action="store_true")
    ap.add_argument("--no-glow", action="store_true")
    ap.add_argument("--no-lights", action="store_true")
    ap.add_argument("--no-volume", action="store_true")
    run_full(ap.parse_args())
    raise SystemExit(0)


# ---- coupled simulation + renderer, headless ------------------------------------------------------------

class _NoFingers:
    def __init__(self):
        self.dirs = []

    def active(self):
        return self.dirs


def run_sim(args):
    """The real Globe + Renderer, headless: N frames, PNG snapshots, counters and timings."""
    import types
    from plasma.globe import Globe, SimFlags
    from plasma.renderer import Renderer, RenderFlags
    dw, dh = (int(v) for v in args.display.split("x"))
    make_context()
    wp.init()
    sargs = types.SimpleNamespace(seed=args.seed, gas_res=args.gas_res, internal_scale=2, profile=False)
    vec = lambda spec: tuple(float(v) for v in spec.split(",")) if spec else None
    cam = _FakeCamera(dw, dh, vec(args.eye), vec(args.target))
    fingers = _NoFingers()
    simflags = SimFlags(hybrid=not args.no_hybrid, invert=args.invert, ice=args.ice)
    rflags = RenderFlags(taau=not args.no_taau, glow=not args.no_glow, lights=not args.no_lights)
    globe = Globe(cam, fingers, lambda: simflags, sargs)
    renderer = Renderer(cam, globe, lambda: rflags, lambda: args.debug, sargs)
    fbo = int(glGenFramebuffers(1)); tex = image2d(dw, dh, GL_RGBA8)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    t0 = time.perf_counter()
    globe.init(dw, dh)
    renderer.init(dw, dh)
    renderer.knobs["rad_scale"] = args.rad_scale
    print(f"init {time.perf_counter() - t0:.1f} s")
    globe.knobs["voltage"] = args.voltage
    if args.preset:
        globe.preset_index = globe.presets.index(args.preset)
        globe.apply_preset()
    snaps = set(int(v) for v in args.snapshots.split(",")) if args.snapshots else set()
    sim_ms, frame_ms = [], []
    track_err = []
    root_track = []          # (--track-roots) per frame: root / foot directions and states, for the drift statistics
    from PIL import Image
    for f in range(args.frames):
        if args.touch_frame >= 0 and f == args.touch_frame:
            dirs = []
            for spec in args.fingers.split(";"):
                v = np.array([float(c) for c in spec.split(",")], np.float32)
                dirs.append(v / np.linalg.norm(v))
            fingers.dirs = dirs
            base_dirs = [d.copy() for d in dirs]
        if args.touch_frame >= 0 and f > args.touch_frame and args.finger_speed != 0.0:
            a = args.finger_speed * (f - args.touch_frame) / 60.0
            c, s_ = math.cos(a), math.sin(a)
            rot = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]], np.float32)
            fingers.dirs = [rot @ d for d in base_dirs]
            if f % 10 == 0 and f > args.touch_frame + 60:
                wp.synchronize()
                e_ = globe.engine; st_ = e_.t_state.numpy(); fd_ = e_.t_foot_dir.numpy()
                f2_ = globe.tree_foot2_dir.numpy().reshape(-1, 3)
                feet = np.concatenate([fd_[st_ == 3], f2_[np.linalg.norm(f2_, axis=1) > 0.5]]) if (st_ == 3).any() else np.zeros((0, 3))
                d0 = fingers.dirs[0]
                err = float(np.degrees(np.arccos(np.clip((feet @ d0).max(), -1.0, 1.0)))) if len(feet) else 90.0
                track_err.append(err)
        cam.angle += math.radians(args.orbit_deg_per_frame)
        cam.update()
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        ta = time.perf_counter()
        globe.pre_render()
        wp.synchronize()
        if args.track_roots and f >= args.track_roots:
            e_ = globe.engine
            root_track.append((e_.t_root_dir.numpy().copy(), e_.t_foot_dir.numpy().copy(), e_.t_state.numpy().copy(),
                               e_.t_birth.numpy().copy()))
        tb = time.perf_counter()
        renderer.render()
        renderer.post_render()
        glFinish()
        tc = time.perf_counter()
        sim_ms.append((tb - ta) * 1e3); frame_ms.append((tc - ta) * 1e3)
        hist = renderer.read_histogram()        # per-frame statistics (the read clears the buffer)
        if (f + 1) % args.log_every == 0 or f == 0:
            globe.print_counters()
            if args.verbose:
                e = globe.engine
                st = e.t_state.numpy(); stretch = e.t_stretch.numpy(); tl = e.t_L.numpy()
                att = st == 3
                tip = e.t_tip.numpy(); pos = e.pos.numpy(); nn = e.t_nodes.numpy()
                grow = np.where((st != 0) & (st != 3))[0]
                tipr = [round(float(np.linalg.norm(pos[tip[t]]) / 0.075), 2) if tip[t] >= 0 else -1 for t in grow[:10]]
                fl = e.flags.numpy(); par = e.parent.numpy(); alive = (fl & 1) != 0
                has_par = alive & (par >= 0)
                dangling = int((has_par & ~alive[np.where(par >= 0, par, 0)]).sum())
                seg = np.linalg.norm(pos[has_par] - pos[par[has_par]], axis=1)
                foot = e.t_foot.numpy(); main = []
                for t in np.where(att)[0][:12]:
                    n = foot[t]; c = 0
                    while n >= 0 and c < 4096:
                        c += 1; n = par[n]
                    main.append(f"{c}/{nn[t]}")
                rg = e.t_regrow.numpy()
                print(f"        brush {np.bincount(e.t_brush.numpy(), minlength=3).tolist()}  brush_req {np.bincount(e.t_brush_req.numpy(), minlength=3).tolist()}  "
                      f"foot2 {int((e.t_foot2.numpy() >= 0).sum())}  coral {int(((fl & 512) != 0).sum())}  cs_brush_req {np.bincount(globe.circuit.tree_brush_req.numpy(), minlength=3).tolist()}")
                print(f"        states {np.bincount(st, minlength=7).tolist()}  regrow {np.bincount(rg, minlength=3).tolist()}  "
                      f"stretch(att) {np.round(stretch[att], 2).tolist()[:12]}")
                print(f"        main/total(att) {main}  growing: nodes {nn[grow][:10].tolist()} tip r/R2 {tipr}  "
                      f"dangling {dangling}  seg max {seg.max() * 1e3 if seg.size else 0:.1f} mm  >3h {int((seg > 4.5e-3).sum())}")
        if (f + 1) in snaps:
            glBindTexture(GL_TEXTURE_2D, tex)
            img = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
            path = args.out.replace(".png", f"_{f + 1:04d}.png")
            Image.fromarray(np.frombuffer(img, np.uint8).reshape(dh, dw, 4)[::-1, :, :3]).save(path)
            print("wrote", path)
    renderer.print_timings()
    counts = hist[:64]; centers = np.arange(64) * 8 + 4; tot = max(int(counts.sum()), 1)
    cum = np.cumsum(counts) / tot
    print(f"candidates/ray last frame: mean {(counts * centers).sum() / tot:.1f}, p99 {int(centers[min(np.searchsorted(cum, 0.99), 63)])}, "
          f"rays {tot}, capped {hist[64]}; bins(0..7 of 8) {counts[:8].tolist()} ... top {counts[-4:].tolist()}")
    print(f"sim (graph replay, synced) median {np.median(sim_ms[5:]):.2f} ms; whole frame median {np.median(frame_ms[5:]):.2f} ms, p99 {np.percentile(frame_ms[5:], 99):.2f} ms")
    c = globe.counters_now()
    nan = int(np.isnan(globe.engine.pos.numpy()).sum())
    print(f"final: {c}; NaN positions {nan}")
    e = globe.engine; fl = e.flags.numpy(); alive = (fl & 1) != 0
    pos = e.pos.numpy()[alive]
    if len(root_track) > 1:
        # drift of the roots on the electrode and of the feet on the glass (attached trees, same tree
        # both frames, no re-strike in between): vertical speed in mm/s, + up
        vr, vf = [], []
        for (r0, f0, s0_, b0), (r1, f1, s1_, b1) in zip(root_track, root_track[1:]):
            same = (s0_ == 3) & (s1_ == 3) & (b0 == b1)
            vr.extend(((r1[same, 1] - r0[same, 1]) * 0.011 * 60.0 * 1e3).tolist())
            moved = same & (np.linalg.norm(f1 - f0, axis=1) < 0.05)           # a foot jump (re-route) is not a walk
            vf.extend(((f1[moved, 1] - f0[moved, 1]) * 0.075 * 60.0 * 1e3).tolist())
        vr, vf = np.array(vr), np.array(vf)
        for name, v in (("roots", vr), ("feet", vf)):
            if v.size:
                print(f"drift {name}: n {v.size}, |v| median {np.median(np.abs(v)):.1f} mm/s, mean v {v.mean():+.1f} mm/s, "
                      f"fraction up {(v > 0).mean():.2f}, p10 {np.percentile(v, 10):+.1f} p90 {np.percentile(v, 90):+.1f}")
    st = e.t_state.numpy(); f2 = e.t_foot2.numpy().reshape(-1, 4)
    print(f"morphology: attached {int((st == 3).sum())}, secondary feet {int((f2 >= 0).sum())} on "
          f"{int(((f2 >= 0).any(axis=1)).sum())} trees, coral nodes {int((alive & ((fl & 512) != 0)).sum())}, "
          f"stalled {int(e.t_stalled.numpy().sum())}")
    s0 = globe.gas_state[0]
    T = getattr(s0, "T", None)
    line = f"orientation: mean node y {pos[:, 1].mean() * 100:+.2f} cm (n {len(pos)})"
    if T is not None:
        t = T.numpy().astype(np.float64); dT = np.clip(t - 300.0, 0.0, None)
        n = t.shape[1]; ys = (np.arange(n) + 0.5) / n - 0.5
        w = dT.sum()
        if w > 0:
            cy = (dT.sum(axis=(0, 2)) * ys).sum() / w      # array layout (z, y, x)
            line += f"; heat centroid y {cy:+.3f} (fraction of the box, +up), max dT {dT.max():.1f} K"
    print(line)
    if track_err:
        te = np.array(track_err)
        print(f"moving finger: nearest foot angle mean {te.mean():.1f} deg, p90 {np.percentile(te, 90):.1f} deg, max {te.max():.1f} deg, "
              f"frames > 10 deg: {int((te > 10).sum())}/{len(te)}")
    if fingers.dirs:
        st = e.t_state.numpy(); fd = e.t_foot_dir.numpy(); cur = globe.circuit.tree_current.numpy()
        f2 = globe.tree_foot2_dir.numpy().reshape(-1, 4, 3); touch = globe.circuit.tree_touch.numpy()
        att = np.where(st == 3)[0]
        for n, d in enumerate(fingers.dirs):
            if att.size:
                ang = np.degrees(np.arccos(np.clip(fd[att] @ d, -1.0, 1.0)))
                j = int(np.argmin(ang))
                brush = [(int(t), k, round(float(np.degrees(np.arccos(np.clip(f2[t, k] @ d, -1, 1)))), 1))
                         for t in att for k in range(4) if np.linalg.norm(f2[t, k]) > 0.5
                         and np.degrees(np.arccos(np.clip(f2[t, k] @ d, -1, 1))) < 10.0]
                owners = [int(t) for t in att if touch[t] > 0.0]
                print(f"finger {n} {np.round(d, 2).tolist()}: nearest main foot {ang[j]:.1f} deg (tree {att[j]}, "
                      f"{cur[att[j]] * 1e6:.0f} uA); main feet within 10 deg: {int((ang < 10.0).sum())}; "
                      f"brush feet within 10 deg: {brush}; touched trees: {owners}")
            else:
                print(f"finger {n}: no attached filament")
    return 0 if nan == 0 else 1


if __name__ == "__main__" and "--sim" in sys.argv:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--display", default="3840x2160")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--snapshots", default="60,180,300")
    ap.add_argument("--orbit-deg-per-frame", type=float, default=0.0)
    ap.add_argument("--rad-scale", type=float, default=60.0)
    ap.add_argument("--voltage", type=float, default=5000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gas-res", type=int, default=64)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--touch-frame", type=int, default=-1)
    ap.add_argument("--fingers", default="0,0,1", help="finger directions at --touch-frame, 'x,y,z;x,y,z;...'")
    ap.add_argument("--finger-speed", type=float, default=0.0, help="rotate the fingers about +y at this rate (rad/s) after --touch-frame")
    ap.add_argument("--out", default="/tmp/plasma_sim.png")
    ap.add_argument("--no-taau", action="store_true")
    ap.add_argument("--no-glow", action="store_true")
    ap.add_argument("--no-lights", action="store_true")
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--invert", action="store_true", help="globe upside down (gravity flipped in globe space)")
    ap.add_argument("--ice", action="store_true", help="ice cap on top of the globe")
    ap.add_argument("--log-every", type=int, default=60)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--preset", default=None, help="gas preset name (video, ne_xe, ne, ar, kr)")
    ap.add_argument("--track-roots", type=int, default=0, help="from this frame on, record root / foot directions and print their drift")
    ap.add_argument("--eye", default=None, help="camera position 'x,y,z' (m); default: the 0.32 m orbit")
    ap.add_argument("--target", default=None, help="camera target 'x,y,z' (m); default 0,0.03,0")
    raise SystemExit(run_sim(ap.parse_args()))
