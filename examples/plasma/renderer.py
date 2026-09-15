# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
The GLSL renderer Input of the plasma globe (plan section 5): owns every GL object, the CUDA
registrations of the render buffers, and the per-frame pass sequence

    P0b  Warp -> GL   map the registered buffers, copy the publish-stage arrays, unmap
    P1   trace.comp   internal resolution: HDR, emissive, motion vectors, layer ids
    P2   glow_*.comp  6-level pyramid on the emissive image shaped like the APSF profile
    P4   exposure     log-luminance reduce + asymmetric adaptation
    P3   taau.comp    2x temporal upscale with per-layer motion vectors (or a plain upscale)
    P5   present      AgX + blue-noise dither to the default framebuffer

It runs both inside shaderbang (the C loop makes the DRM/GBM context current on the render
thread and page-flips after ``render``) and headlessly for tests (any current GL 4.6 context).
All GL objects are created in ``init``; nothing is allocated per frame.
"""

import ctypes
import math
import os
import re
import time

import numpy as np
import warp as wp

from OpenGL.GL import *  # noqa: F401,F403

from shaderbang.input import Input

from plasma import publish, interop
from plasma.circuit import SIG_MAX, F_MAX

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PARAMS_BYTES = 320                               # Params UBO: 16 vec4 (camera, matrices, knobs) + 4 preset colours
VIDEO_ANODE_COLOURS = ((0.52, 0.20, 1.00),        # electrode glow discharge (footage face (125,82,168) sRGB)
                       (0.20, 0.08, 1.00),        # far haze around the bulb (deep violet)
                       (0.55, 0.12, 1.00),        # near haze / sheath layer (violet-magenta)
                       (1.00, 0.34, 0.72))        # feet and root pools (magenta)
GLOW_LEVELS = 6
GLOW_TABLE_PX = (24, 32, 48, 64, 96, 128, 192)   # glow kernels fitted once; interpolated with the zoom
R2O_M = 0.0775                                   # plasma.params.R2O: the globe's outer radius (m)
GLOBE_REF_FRACTION = 0.347                       # projected globe radius / internal height at the reference framing
HALTON_PHASES = 32
DEFAULT_GLOW_WEIGHTS = (0.55, 0.40, 0.28, 0.20, 0.14, 0.10)


class RenderFlags:
    """Per-frame toggles the application derives from its State flags."""

    def __init__(self, taau=True, glow=True, lights=True, invert=False, volume=True):
        self.taau = taau
        self.glow = glow
        self.lights = lights
        self.invert = invert
        self.volume = volume


# ---- small GL helpers (self-contained so the renderer has no dependency on the bench code) ----

def _load_source(path, defines=()):
    src = open(path).read()
    src = re.sub(r'#include\s+"([^"]+)"',
                 lambda m: open(os.path.join(os.path.dirname(path), m.group(1))).read(), src)
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("#version"):
            lines[i] = line + "\n" + "\n".join(f"#define {d}" for d in defines)
            break
    return "\n".join(lines)


def _compile(kind, path, defines=()):
    sh = glCreateShader(kind)
    glShaderSource(sh, _load_source(path, defines))
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(f"{os.path.basename(path)}: {glGetShaderInfoLog(sh)}")
    return sh


def _link(*shaders):
    prog = glCreateProgram()
    for sh in shaders:
        glAttachShader(prog, sh)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog))
    for sh in shaders:
        glDeleteShader(sh)
    return prog


def compute_program(name, defines=()):
    return _link(_compile(GL_COMPUTE_SHADER, os.path.join(SHADER_DIR, name), defines))


def raster_program(vert, frag):
    return _link(_compile(GL_VERTEX_SHADER, os.path.join(SHADER_DIR, vert)),
                 _compile(GL_FRAGMENT_SHADER, os.path.join(SHADER_DIR, frag)))


def texture2d(w, h, fmt, filt=GL_LINEAR):
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexStorage2D(GL_TEXTURE_2D, 1, fmt, w, h)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def texture3d(n, fmt):
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_3D, tex)
    glTexStorage3D(GL_TEXTURE_3D, 1, fmt, n, n, n)
    for pn in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
        glTexParameteri(GL_TEXTURE_3D, pn, GL_LINEAR)
    for pn in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R):
        glTexParameteri(GL_TEXTURE_3D, pn, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_3D, 0)
    return tex


class TimerQuery:
    """Double-buffered GL_TIME_ELAPSED query: reading never stalls on the current frame."""

    def __init__(self, name):
        self.name = name
        self.q = [int(v) for v in glGenQueries(2)]
        self.frame = 0
        self.ms = 0.0

    def begin(self):
        glBeginQuery(GL_TIME_ELAPSED, self.q[self.frame & 1])

    def end(self):
        glEndQuery(GL_TIME_ELAPSED)
        prev = self.q[(self.frame + 1) & 1]
        if self.frame > 0:
            ns = ctypes.c_uint64()
            glGetQueryObjectui64v(prev, GL_QUERY_RESULT, ns)
            self.ms = ns.value / 1e6
        self.frame += 1


class GLBuffer:
    """A GL buffer (immutable storage) aliased by a Warp array through CUDA graphics interop.
    All buffers are mapped together once per frame by ``interop.BatchedMap`` (see Renderer.upload)."""

    def __init__(self, target, nbytes, binding, dtype, shape):
        self.target = target
        self.binding = binding
        self.dtype = dtype
        self.shape = shape
        self.id = int(glGenBuffers(1))
        glBindBuffer(target, self.id)
        glBufferStorage(target, nbytes, None, GL_DYNAMIC_STORAGE_BIT)
        glBindBuffer(target, 0)
        self.registered = interop.register_buffer(self.id, flags=wp.RegisteredGLBuffer.WRITE_DISCARD)

    def bind(self):
        glBindBufferBase(self.target, self.binding, self.id)


def halton(index, base):
    f, r = 1.0, 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


class Renderer(Input):
    """See the module docstring. ``globe`` exposes ``pub`` (plasma.publish.Publisher),
    ``circuit`` (plasma.circuit.CircuitState), ``temperature``/``speed`` staging grids and
    ``camera``; ``state_fn``/``debug_fn`` return the app's State flags / debug view index."""

    def __init__(self, camera, globe, state_fn, debug_fn, args):
        super().__init__("renderer")
        self.camera = camera
        self.globe = globe
        self.state_fn = state_fn
        self.debug_fn = debug_fn
        self.args = args
        self._glow_table = None
        self._glow_kernel_now = 40.0
        self._glow_gain_scale = 1.0
        self.ref_area = 1.0
        self.knobs = {"glow_width": 40.0, "exposure_bias": 0.0, "glow_gain": 0.4, "rad_scale": 60.0,
                      "ambient_gain": 0.0, "tonemap": 1.0, "exposure": 0.0}
        # tonemap 1 = camera clip (0 = AgX); exposure > 0 = fixed (0 = metered)   # glow: a tight camera PSF; the halo is the physical sheath   # glow width = APSF kernel radius (internal px); ambient = volume glow gain
        self.timers = {}
        self.frame = 0
        self.first_frame = True
        self.width = self.height = 0
        self.iw = self.ih = 0
        self.spectra = None

    # ---- setup ---------------------------------------------------------------------------------
    def init(self, width, height):
        self.width, self.height = int(width), int(height)
        scale = max(1, int(getattr(self.args, "internal_scale", 2)))
        self.iw, self.ih = self.width // scale, self.height // scale
        print(f"[renderer] display {self.width}x{self.height}, internal {self.iw}x{self.ih}, "
              f"GL {glGetString(GL_VERSION).decode()}")
        # programs
        self.prog_trace = compute_program("trace.comp")
        self.prog_glow_down = compute_program("glow_down.comp")
        self.prog_glow_up = compute_program("glow_up.comp")
        self.prog_exposure_reduce = compute_program("exposure.comp", ("REDUCE",))
        self.prog_exposure_resolve = compute_program("exposure.comp", ("RESOLVE",))
        self.prog_taau = compute_program("taau.comp")
        self.prog_present = raster_program("present.vert", "present.frag")
        # internal-resolution targets
        self.tex_hdr = texture2d(self.iw, self.ih, GL_RGBA16F)
        self.tex_emissive = texture2d(self.iw, self.ih, GL_RGBA16F)
        self.tex_mv = texture2d(self.iw, self.ih, GL_RG16F, GL_NEAREST)
        self.tex_layer = texture2d(self.iw, self.ih, GL_RG32F, GL_NEAREST)
        # glow pyramid: level 0 = emissive itself, levels 1..GLOW_LEVELS at halving resolution
        self.glow_down = []
        self.glow_up = []
        w, h = self.iw, self.ih
        for _ in range(GLOW_LEVELS):
            w, h = max(1, w // 2), max(1, h // 2)
            self.glow_down.append((texture2d(w, h, GL_RGBA16F), w, h))
            self.glow_up.append((texture2d(w, h, GL_RGBA16F), w, h))
        self.set_glow_width(self.knobs["glow_width"])
        # history (display resolution), double-buffered
        self.history = [(texture2d(self.width, self.height, GL_RGBA16F),
                         texture2d(self.width, self.height, GL_R32F, GL_NEAREST)) for _ in range(2)]
        # exposure SSBO (8 x 4 bytes header + 64 int bins), params UBO (256 bytes), histogram SSBO
        self.ubo = int(glGenBuffers(1))
        glBindBuffer(GL_UNIFORM_BUFFER, self.ubo)
        glBufferData(GL_UNIFORM_BUFFER, PARAMS_BYTES, None, GL_DYNAMIC_DRAW)
        self.ssbo_exposure = int(glGenBuffers(1))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_exposure)
        glBufferData(GL_SHADER_STORAGE_BUFFER, (8 + 64) * 4, np.zeros(8 + 64, np.float32), GL_DYNAMIC_DRAW)   # header + 64 histogram bins
        self.ssbo_hist = int(glGenBuffers(1))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_hist)
        glBufferData(GL_SHADER_STORAGE_BUFFER, 68 * 4, np.zeros(68, np.int32), GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        # CUDA-registered render buffers fed by the publish stage
        pub = self.globe.pub
        self.buf_segments = GLBuffer(GL_SHADER_STORAGE_BUFFER, publish.SEG_MAX * publish.SEG_STRIDE * 16, 1, wp.vec4,
                                     (publish.SEG_MAX * publish.SEG_STRIDE,))
        self.buf_cell_start = GLBuffer(GL_SHADER_STORAGE_BUFFER, publish.GRID_CELLS * 4, 2, wp.int32, (publish.GRID_CELLS,))
        self.buf_cell_count = GLBuffer(GL_SHADER_STORAGE_BUFFER, publish.GRID_CELLS * 4, 3, wp.int32, (publish.GRID_CELLS,))
        self.buf_cell_items = GLBuffer(GL_SHADER_STORAGE_BUFFER, publish.ITEMS_MAX * 4, 4, wp.int32, (publish.ITEMS_MAX,))
        self.buf_lights = GLBuffer(GL_SHADER_STORAGE_BUFFER, publish.LIGHT_MAX * 3 * 16, 5, wp.vec4, (publish.LIGHT_MAX * 3,))
        # SIG_MAX foot records + F_MAX root records, 2 vec4 each
        self.buf_sigma = GLBuffer(GL_SHADER_STORAGE_BUFFER, (SIG_MAX + F_MAX) * 2 * 16, 6, wp.vec4, ((SIG_MAX + F_MAX) * 2,))
        self.sigma_pack = wp.zeros((SIG_MAX + F_MAX) * 2, dtype=wp.vec4, device=pub.device)
        # volume: the float16 RGBA staging grid is copied straight into the registered 3D texture
        n3 = publish.GRID_N
        self.tex_volume = texture3d(n3, GL_RGBA16F)
        self.volume_resource = interop.register_image(self.tex_volume, interop.GL_TEXTURE_3D)
        self.gl_buffers = [self.buf_segments, self.buf_cell_start, self.buf_cell_count, self.buf_cell_items,
                           self.buf_lights, self.buf_sigma]
        self.batch = interop.BatchedMap([b.registered for b in self.gl_buffers] + [self.volume_resource], device=pub.device)
        self.volume_index = len(self.gl_buffers)
        # blue noise
        self.tex_noise = self._load_noise()
        self.fbo_default = 0
        self.vao = int(glGenVertexArrays(1))
        for name in ("trace", "glow", "exposure", "taau", "present", "upload"):
            self.timers[name] = TimerQuery(name)
        self._sigma_pack_kernel = _k_pack_sigma

    def _glow_weights_for(self, kernel_px):
        """Pyramid weights for a glow kernel, interpolated in a table fitted once (a fit costs ~15 ms)."""
        if self._glow_table is None:
            try:
                from plasma.apsf import pyramid_weights_default
                self._glow_table = [(k, list(pyramid_weights_default(GLOW_LEVELS, k))) for k in GLOW_TABLE_PX]
            except Exception:
                self._glow_table = []
        if not self._glow_table:
            return list(self.glow_weights)
        ks = [k for k, _ in self._glow_table]
        if kernel_px <= ks[0]:
            return list(self._glow_table[0][1])
        for (k0, w0), (k1, w1) in zip(self._glow_table, self._glow_table[1:]):
            if kernel_px <= k1:
                f = (kernel_px - k0) / (k1 - k0)
                return [a + (b - a) * f for a, b in zip(w0, w1)]
        return list(self._glow_table[-1][1])

    def set_glow_width(self, kernel_px):
        """Glow width knob = APSF kernel radius (internal pixels, 24-195): refit the pyramid weights."""
        self.knobs["glow_width"] = float(min(max(kernel_px, 24.0), 192.0))
        try:
            from plasma.apsf import pyramid_weights_default
            self.glow_weights = list(pyramid_weights_default(GLOW_LEVELS, int(self.knobs["glow_width"])))
        except Exception:
            self.glow_weights = list(DEFAULT_GLOW_WEIGHTS)

    def _load_noise(self):
        path = os.path.join(DATA_DIR, "blue_noise_64.png")
        try:
            from PIL import Image
            arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        except Exception:
            arr = (np.random.default_rng(7).random((64, 64)) * 255).astype(np.uint8)
        tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, 64, 64)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 64, 64, GL_RED, GL_UNSIGNED_BYTE, np.ascontiguousarray(arr))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex

    # ---- per frame -----------------------------------------------------------------------------
    def _params_blob(self, jitter):
        cam = self.camera
        eye, cu, cv, cw = cam.basis()
        # the tracer's camera basis is built for the INTERNAL aspect (same as display)
        blob = np.zeros(PARAMS_BYTES // 4, np.float32)
        blob[0:3] = eye; blob[4:7] = cu; blob[8:11] = cv; blob[12:15] = cw
        blob[16:32] = cam.vp.T.reshape(-1)
        blob[32:48] = cam.vp_prev.T.reshape(-1)
        blob[48:52] = (jitter[0], jitter[1], self.iw, self.ih)
        flags = self.state_fn()
        blob[52:56] = (self.knobs["rad_scale"], self.knobs["ambient_gain"], time.time() % 1000.0, 0.0)
        blob[56:60] = np.array([self.debug_fn(), 1 if flags.lights else 0, self.frame, 1 if flags.volume else 0], np.int32).view(np.float32)
        # gas: x total current (mA, previous frame's counters; drives the electrode halo and the
        # background discharge), w gravity sign
        counters = getattr(self.globe, "counters_now", None)
        i_tot_ma = float(counters()["i_tot"]) * 1e3 if counters is not None else 1.0
        blob[60:64] = (i_tot_ma, 0.0, 0.0, -1.0 if flags.invert else 1.0)
        # the electrode glow, the haze around it, the sheath layer and the foot / root-pool colours
        # follow the gas preset (the 'video' preset keeps the values calibrated on the footage)
        for k, c in enumerate(self._preset_colours()):
            blob[64 + 4 * k:67 + 4 * k] = c
        return blob

    def _preset_colours(self):
        """(electrode, halo far, halo near / sheath, foot) linear rgb for the current gas preset."""
        name = getattr(self.globe, "preset_name", "video")
        if name == "video" or not hasattr(self.globe, "preset_rgb"):
            return VIDEO_ANODE_COLOURS
        neutral, ion = (np.asarray(c, np.float64) for c in self.globe.preset_rgb)
        def unit(c):
            c = np.maximum(c, 0.0); return c / max(float(c.max()), 1e-6)
        n, i = unit(neutral), unit(ion)
        return (unit(0.35 * n + 0.65 * i), i, unit(0.5 * n + 0.5 * i), n)

    def upload(self):
        """P0b: publish-stage arrays -> GL buffers / 3D texture (device-to-device)."""
        pub, cs = self.globe.pub, self.globe.circuit
        t = self.timers["upload"]; t.begin()
        e = self.globe.engine
        wp.launch(self._sigma_pack_kernel, dim=SIG_MAX + F_MAX,
                  inputs=[cs.sig_dir, cs.sig_amp, cs.sig_radius, cs.sig_age, cs.sig_alive, cs.sig_I,
                          e.t_state, e.t_root_dir, cs.tree_current, self.sigma_pack],
                  device=pub.device)
        sources = (pub.segments, pub.cell_start, pub.cell_count, pub.cell_items, pub.lights, self.sigma_pack)
        m = self.batch.map()
        try:
            for i, (buf, src) in enumerate(zip(self.gl_buffers, sources)):
                wp.copy(m.array(i, buf.dtype, buf.shape), src)
            m.texture(self.volume_index).copy_from(pub.volume)
        finally:
            m.unmap()
        t.end()

    def render(self, **kwargs):
        flags = self.state_fn()
        prev_fb = glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)
        taau_on = flags.taau
        phase = self.frame % HALTON_PHASES
        jitter = ((halton(phase + 1, 2) - 0.5), (halton(phase + 1, 3) - 0.5)) if taau_on else (0.0, 0.0)

        self.upload()

        # ---- P1 trace
        glBindBuffer(GL_UNIFORM_BUFFER, self.ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, self._params_blob(jitter))
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.ubo)
        for b in (self.buf_segments, self.buf_cell_start, self.buf_cell_count, self.buf_cell_items, self.buf_lights, self.buf_sigma):
            b.bind()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self.ssbo_hist)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, self.tex_volume)
        glBindImageTexture(0, self.tex_hdr, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
        glBindImageTexture(1, self.tex_emissive, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
        glBindImageTexture(2, self.tex_mv, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F)
        glBindImageTexture(3, self.tex_layer, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F)
        t = self.timers["trace"]; t.begin()
        glUseProgram(self.prog_trace)
        glDispatchCompute((self.iw + 7) // 8, (self.ih + 7) // 8, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)
        t.end()

        # ---- zoom: the globe's projected radius in internal pixels drives the meter's reference
        # area and the glow kernel/gain, so widths and brightness scale with the camera distance
        # (a fixed-pixel glow and a frame-wide meter made a touched filament and the bulb's halo
        # look the same at any zoom)
        cam_basis = self.camera.basis()
        eye_p = np.asarray(cam_basis[0], np.float64); cv_p = np.asarray(cam_basis[2], np.float64)
        dist = max(float(np.linalg.norm(eye_p)), 1e-3)
        r_px = (R2O_M / dist) / max(float(np.linalg.norm(cv_p)), 1e-6) * (self.ih / 2.0)
        self.ref_area = float(np.pi * r_px * r_px)
        zoom = r_px / (GLOBE_REF_FRACTION * self.ih)
        self._glow_gain_scale = min(1.0, zoom)
        kernel = float(min(max(self.knobs["glow_width"] * zoom, 24.0), 192.0))
        if abs(kernel - self._glow_kernel_now) > 0.06 * self._glow_kernel_now:
            self.glow_weights = self._glow_weights_for(kernel)
            self._glow_kernel_now = kernel

        # ---- P2 glow pyramid
        t = self.timers["glow"]; t.begin()
        if flags.glow:
            glUseProgram(self.prog_glow_down)
            src = self.tex_emissive
            for i, (tex, w, h) in enumerate(self.glow_down):
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, src)
                glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
                glUniform1i(glGetUniformLocation(self.prog_glow_down, "firstLevel"), 1 if i == 0 else 0)
                glDispatchCompute((w + 15) // 16, (h + 15) // 16, 1)
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT)
                src = tex
            glUseProgram(self.prog_glow_up)
            loc_w = glGetUniformLocation(self.prog_glow_up, "levelWeight")
            loc_r = glGetUniformLocation(self.prog_glow_up, "radius")
            loc_c = glGetUniformLocation(self.prog_glow_up, "hasCoarse")
            coarse = None
            for i in range(GLOW_LEVELS - 1, -1, -1):
                tex_level, w, h = self.glow_down[i]
                tex_out = self.glow_up[i][0]
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, coarse if coarse is not None else tex_level)
                glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_level)
                glBindImageTexture(0, tex_out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
                glUniform1f(loc_w, float(self.glow_weights[i]))
                glUniform1f(loc_r, 1.0)                     # PYRAMID_SIGMA is measured for a 1-texel tent
                glUniform1i(loc_c, 1 if coarse is not None else 0)
                glDispatchCompute((w + 15) // 16, (h + 15) // 16, 1)
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT)
                coarse = tex_out
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0)
        t.end()

        # ---- P4 exposure
        t = self.timers["exposure"]; t.begin()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ssbo_exposure)
        glUseProgram(self.prog_exposure_reduce)
        glBindImageTexture(0, self.tex_hdr, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F)
        glDispatchCompute((self.iw + 15) // 16, (self.ih + 15) // 16, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glUseProgram(self.prog_exposure_resolve)
        glUniform1f(glGetUniformLocation(self.prog_exposure_resolve, "dt"), 1.0 / 60.0)
        glUniform1f(glGetUniformLocation(self.prog_exposure_resolve, "bias"), float(self.knobs["exposure_bias"]))
        glUniform1f(glGetUniformLocation(self.prog_exposure_resolve, "refArea"), float(self.ref_area))
        glUniform1i(glGetUniformLocation(self.prog_exposure_resolve, "firstFrame"), 1 if self.first_frame else 0)
        glDispatchCompute(1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        exposure_ratio = 1.0
        t.end()

        # ---- P3 temporal upscale
        t = self.timers["taau"]; t.begin()
        cur, prev = self.history[self.frame & 1], self.history[(self.frame + 1) & 1]
        glUseProgram(self.prog_taau)
        for unit, tex in enumerate((self.tex_hdr, self.tex_mv, self.tex_layer, prev[0], prev[1])):
            glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, tex)
        glBindImageTexture(0, cur[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
        glBindImageTexture(1, cur[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F)
        glUniform2f(glGetUniformLocation(self.prog_taau, "jitter"), jitter[0], jitter[1])
        glUniform1f(glGetUniformLocation(self.prog_taau, "exposureRatio"), exposure_ratio)
        glUniform1i(glGetUniformLocation(self.prog_taau, "firstFrame"), 1 if self.first_frame else 0)
        glUniform1i(glGetUniformLocation(self.prog_taau, "enabled"), 1 if taau_on else 0)
        glUniform1f(glGetUniformLocation(self.prog_taau, "clampGamma"), 1.0)
        glDispatchCompute((self.width + 15) // 16, (self.height + 15) // 16, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT)
        for unit in range(5):
            glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, 0)
        t.end()

        # ---- P5 present
        t = self.timers["present"]; t.begin()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_fb)
        glViewport(0, 0, self.width, self.height)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.prog_present)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, cur[0])
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, self.glow_up[0][0])
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, self.tex_noise)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ssbo_exposure)
        glUniform1f(glGetUniformLocation(self.prog_present, "glowGain"),
                    float(self.knobs["glow_gain"]) * self._glow_gain_scale if flags.glow else 0.0)
        glUniform1i(glGetUniformLocation(self.prog_present, "debugView"), 1 if self.debug_fn() == 1 else 0)
        glUniform1i(glGetUniformLocation(self.prog_present, "tonemapMode"), int(self.knobs["tonemap"]))
        glUniform1f(glGetUniformLocation(self.prog_present, "fixedExposure"), float(self.knobs["exposure"]))
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
        for unit in range(3):
            glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, 0)
        t.end()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_fb)

    def post_render(self, **kwargs):
        self.frame += 1
        self.first_frame = False
        if getattr(self.args, "profile", False) and self.frame % 120 == 0:
            self.print_timings()

    def print_timings(self):
        print("[renderer] " + "  ".join(f"{k} {v.ms:.2f} ms" for k, v in self.timers.items())
              + f"  total {sum(v.ms for v in self.timers.values()):.2f} ms")

    def read_histogram(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_hist)
        data = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 68 * 4), np.int32).copy()
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, np.zeros(68, np.int32))
        return data


@wp.kernel
def _k_pack_sigma(sig_dir: wp.array(dtype=wp.vec3), sig_amp: wp.array(dtype=wp.float32),
                  sig_radius: wp.array(dtype=wp.float32), sig_age: wp.array(dtype=wp.float32),
                  sig_alive: wp.array(dtype=wp.int32), sig_I: wp.array(dtype=wp.float32),
                  tree_state: wp.array(dtype=wp.int32), tree_root_dir: wp.array(dtype=wp.vec3),
                  tree_current: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    if i < SIG_MAX:
        d = sig_dir[i]
        out[2 * i] = wp.vec4(d[0], d[1], d[2], sig_amp[i])
        out[2 * i + 1] = wp.vec4(sig_radius[i], float(sig_alive[i]), sig_age[i], sig_I[i] * 1.0e6)   # w: uA
    else:
        # root records: [root dir, I (uA)] [0, attached, 0, 0]
        t = i - SIG_MAX
        d = tree_root_dir[t]
        out[2 * i] = wp.vec4(d[0], d[1], d[2], tree_current[t] * 1.0e6)
        out[2 * i + 1] = wp.vec4(0.0, wp.where(tree_state[t] == 3, 1.0, 0.0), 0.0, 0.0)
