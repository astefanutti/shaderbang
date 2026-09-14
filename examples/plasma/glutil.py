# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Small OpenGL 4.6 helpers shared by the plasma-globe renderer, its benchmarks and its headless tests.

Everything uses direct-state-access entry points (GL 4.5) so no helper leaves a stray binding behind,
which matters inside shaderbang's render loop (``render()`` must return with the default draw
framebuffer bound).

Contents
--------
load_shader_source   ``#include "file"`` preprocessor (next to the including file, then ``shaders/``; never the
                     working directory) + ``#define`` injection
compile_compute / compile_program / Program   linking with full info logs, typed uniform setters, dispatch
Image2D / Image3D    immutable textures (RGBA16F, RG16F, RG32F, R32F, RGBA8, R8, R32UI), image or sampler binding
Buffer               SSBO / UBO with ``glNamedBufferStorage`` (registerable with CUDA) or ``glNamedBufferData``
Framebuffer          colour-only FBO over ``Image2D`` attachments, blits, viewport handling
TimerQuery           ring of ``GL_TIME_ELAPSED`` queries: ``.ms()`` never stalls, ``.collect()`` for benchmarks
FullscreenTriangle   3-vertex draw driven by ``gl_VertexID`` (``fullscreen.vert``)
save_png             dump an ``Image2D`` to disk with Pillow (linear -> sRGB for float formats)
"""

import ctypes
import math
import re
from pathlib import Path

import numpy as np
from OpenGL.GL import *  # noqa: F403
from OpenGL.raw.GL.VERSION.GL_1_5 import glGetQueryObjectuiv as _glGetQueryObjectuiv
from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v as _glGetQueryObjectui64v

SHADER_DIR = Path(__file__).resolve().parent / "shaders"

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$', re.MULTILINE)


# ----------------------------------------------------------------------------------------------------
# Shader sources and programs
# ----------------------------------------------------------------------------------------------------

def resolve_shader_path(path) -> Path:
    """Absolute path of a shader file: an absolute path is taken as is, anything else is looked up under
    ``SHADER_DIR`` first; a relative path with a directory component (``some/dir/x.comp``) may also name a file
    outside ``SHADER_DIR``.  A bare file name is never resolved against the working directory, so a stray
    ``common.glsl`` or ``bench_*.comp`` in the CWD cannot shadow the package shaders."""
    path = Path(path)
    if path.is_absolute():
        return path
    candidate = SHADER_DIR / path
    if candidate.exists():
        return candidate.resolve()
    if len(path.parts) > 1 and path.exists():
        return path.resolve()
    raise FileNotFoundError(f"shader {path} not found under {SHADER_DIR}")


def load_shader_source(path, defines: dict | None = None) -> str:
    """Read a GLSL file (see :func:`resolve_shader_path`), expand ``#include "name"`` (looked up next to the
    including file, then in ``SHADER_DIR``; never in the working directory) and inject ``#define KEY VALUE``
    lines right after the ``#version`` directive.

    ``#line`` directives are emitted so the driver's info log keeps the original line numbers:
    source string 0 is the main file, string ``k`` the k-th distinct include (in order of first use).
    """
    path = resolve_shader_path(path)
    text = path.read_text()
    includes: list[Path] = []

    def expand(src: str, base: Path, string_index: int) -> str:
        out = []
        line_no = 1
        pos = 0
        for m in _INCLUDE_RE.finditer(src):
            chunk = src[pos:m.start()]
            out.append(chunk)
            line_no += chunk.count("\n")
            name = m.group(1)
            inc = base / name
            if not inc.exists():
                inc = SHADER_DIR / name
            if not inc.exists():
                raise FileNotFoundError(f"{path}: cannot resolve #include \"{name}\"")
            inc = inc.resolve()
            if inc in includes:
                idx = includes.index(inc) + 1
                out.append(f"// #include \"{name}\" (already included as string {idx})\n")
            else:
                includes.append(inc)
                idx = len(includes)
                body = expand(inc.read_text(), inc.parent, idx)
                out.append(f"#line 1 {idx}\n{body}\n")
            # the include directive itself occupied one line; restore numbering after it
            line_no += 1
            out.append(f"#line {line_no} {string_index}\n")
            pos = m.end()
            if pos < len(src) and src[pos] == "\n":
                pos += 1
        out.append(src[pos:])
        return "".join(out)

    expanded = expand(text, path.parent, 0)
    if defines:
        lines = expanded.split("\n", 1)
        head, rest = (lines[0], lines[1]) if len(lines) == 2 else (lines[0], "")
        if not head.lstrip().startswith("#version"):
            raise ValueError(f"{path}: first line must be a #version directive to inject defines")
        block = "".join(f"#define {k} {v}\n" for k, v in defines.items())
        expanded = f"{head}\n{block}#line 2 0\n{rest}"
    return expanded


def _numbered(src: str) -> str:
    return "\n".join(f"{i + 1:4d}  {line}" for i, line in enumerate(src.split("\n")))


def compile_shader(stage, source: str, name: str = "<shader>") -> int:
    shader = glCreateShader(stage)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(shader).decode(errors="replace")
        glDeleteShader(shader)
        raise RuntimeError(f"{name}: shader compilation failed\n{log}\n--- source ---\n{_numbered(source)}")
    return shader


def link_program(shaders: list[int], name: str = "<program>") -> int:
    program = glCreateProgram()
    for shader in shaders:
        glAttachShader(program, shader)
    glLinkProgram(program)
    for shader in shaders:
        glDetachShader(program, shader)
        glDeleteShader(shader)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        log = glGetProgramInfoLog(program).decode(errors="replace")
        glDeleteProgram(program)
        raise RuntimeError(f"{name}: program link failed\n{log}")
    return program


_UNIFORM_FLOAT = {GL_FLOAT: (1, glProgramUniform1fv), GL_FLOAT_VEC2: (2, glProgramUniform2fv),
                  GL_FLOAT_VEC3: (3, glProgramUniform3fv), GL_FLOAT_VEC4: (4, glProgramUniform4fv)}
_UNIFORM_INT = {GL_INT: (1, glProgramUniform1iv), GL_INT_VEC2: (2, glProgramUniform2iv),
                GL_INT_VEC3: (3, glProgramUniform3iv), GL_INT_VEC4: (4, glProgramUniform4iv),
                GL_BOOL: (1, glProgramUniform1iv), GL_SAMPLER_2D: (1, glProgramUniform1iv),
                GL_SAMPLER_3D: (1, glProgramUniform1iv), GL_SAMPLER_CUBE: (1, glProgramUniform1iv),
                GL_IMAGE_2D: (1, glProgramUniform1iv), GL_IMAGE_3D: (1, glProgramUniform1iv),
                GL_UNSIGNED_INT_IMAGE_2D: (1, glProgramUniform1iv)}
_UNIFORM_UINT = {GL_UNSIGNED_INT: (1, glProgramUniform1uiv), GL_UNSIGNED_INT_VEC2: (2, glProgramUniform2uiv),
                 GL_UNSIGNED_INT_VEC3: (3, glProgramUniform3uiv), GL_UNSIGNED_INT_VEC4: (4, glProgramUniform4uiv)}
_UNIFORM_MAT = {GL_FLOAT_MAT3: (9, glProgramUniformMatrix3fv), GL_FLOAT_MAT4: (16, glProgramUniformMatrix4fv)}


class Program:
    """A linked program with typed uniform setters (types come from ``glGetActiveUniform`` so passing an
    ``int`` to a ``float`` uniform, or a Python float to an ``int`` uniform, is converted, not corrupted).
    """

    def __init__(self, program: int, name: str, compute: bool):
        self.id = program
        self.name = name
        self.compute = compute
        self.uniforms: dict[str, tuple[int, int, int]] = {}
        count = glGetProgramiv(program, GL_ACTIVE_UNIFORMS)
        for i in range(count):
            uname, size, utype = glGetActiveUniform(program, i)
            uname = uname.decode() if isinstance(uname, bytes) else uname
            base = uname[:-3] if uname.endswith("[0]") else uname
            location = glGetUniformLocation(program, base)
            if location >= 0:
                self.uniforms[base] = (location, int(utype), int(size))
        if compute:
            size = np.zeros(3, dtype=np.int32)
            glGetProgramiv(program, GL_COMPUTE_WORK_GROUP_SIZE, size)
            self.local_size = (int(size[0]), int(size[1]), int(size[2]))
        else:
            self.local_size = None

    def use(self):
        glUseProgram(self.id)

    def set(self, name: str, value):
        """Set a uniform by name; silently ignores names the compiler optimised away."""
        entry = self.uniforms.get(name)
        if entry is None:
            return
        location, utype, size = entry
        if utype in _UNIFORM_MAT:
            n, fn = _UNIFORM_MAT[utype]
            data = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
            fn(self.id, location, data.size // n, GL_FALSE, data)
            return
        arr = np.asarray(value)
        if utype in _UNIFORM_FLOAT:
            n, fn = _UNIFORM_FLOAT[utype]
            data = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        elif utype in _UNIFORM_INT:
            n, fn = _UNIFORM_INT[utype]
            data = np.ascontiguousarray(arr, dtype=np.int32).reshape(-1)
        elif utype in _UNIFORM_UINT:
            n, fn = _UNIFORM_UINT[utype]
            data = np.ascontiguousarray(arr, dtype=np.uint32).reshape(-1)
        else:
            raise TypeError(f"{self.name}: unsupported uniform type 0x{utype:x} for '{name}'")
        if data.size % n != 0:
            raise ValueError(f"{self.name}: uniform '{name}' expects multiples of {n} components, got {data.size}")
        fn(self.id, location, data.size // n, data)

    def groups(self, nx: int, ny: int = 1, nz: int = 1) -> tuple[int, int, int]:
        lx, ly, lz = self.local_size
        return (-(-nx // lx), -(-ny // ly), -(-nz // lz))

    def dispatch(self, nx: int, ny: int = 1, nz: int = 1):
        """Dispatch enough work groups to cover ``nx * ny * nz`` invocations (the shader guards the edge)."""
        gx, gy, gz = self.groups(nx, ny, nz)
        glUseProgram(self.id)
        glDispatchCompute(gx, gy, gz)

    def dispatch_groups(self, gx: int, gy: int = 1, gz: int = 1):
        glUseProgram(self.id)
        glDispatchCompute(gx, gy, gz)

    def delete(self):
        if self.id:
            glDeleteProgram(self.id)
            self.id = 0


def compile_compute(path, defines: dict | None = None) -> Program:
    source = load_shader_source(path, defines)
    name = str(Path(path).name)
    shader = compile_shader(GL_COMPUTE_SHADER, source, name)
    return Program(link_program([shader], name), name, compute=True)


def compile_program(vert_path, frag_path, defines: dict | None = None) -> Program:
    vs = compile_shader(GL_VERTEX_SHADER, load_shader_source(vert_path, defines), str(Path(vert_path).name))
    fs = compile_shader(GL_FRAGMENT_SHADER, load_shader_source(frag_path, defines), str(Path(frag_path).name))
    name = f"{Path(vert_path).name}+{Path(frag_path).name}"
    return Program(link_program([vs, fs], name), name, compute=False)


def barrier_image():
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT)


def barrier_ssbo():
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)


def barrier_all():
    glMemoryBarrier(GL_ALL_BARRIER_BITS)


# ----------------------------------------------------------------------------------------------------
# Textures
# ----------------------------------------------------------------------------------------------------

# name -> (internal format, pixel format, pixel type, channels, numpy dtype for upload/readback)
FORMATS = {
    "RGBA16F": (GL_RGBA16F, GL_RGBA, GL_FLOAT, 4, np.float32),
    "RG16F": (GL_RG16F, GL_RG, GL_FLOAT, 2, np.float32),
    "RG32F": (GL_RG32F, GL_RG, GL_FLOAT, 2, np.float32),
    "R32F": (GL_R32F, GL_RED, GL_FLOAT, 1, np.float32),
    "R16F": (GL_R16F, GL_RED, GL_FLOAT, 1, np.float32),
    "RGBA8": (GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, np.uint8),
    "R8": (GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, np.uint8),
    "R32UI": (GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1, np.uint32),
}


class _Image:
    target = None

    def __init__(self, fmt: str, linear: bool = True, wrap=GL_CLAMP_TO_EDGE):
        if fmt not in FORMATS:
            raise ValueError(f"unsupported format {fmt}; use one of {list(FORMATS)}")
        self.fmt = fmt
        self.internal, self.pixel_format, self.pixel_type, self.channels, self.np_dtype = FORMATS[fmt]
        ids = np.zeros(1, dtype=np.uint32)
        glCreateTextures(self.target, 1, ids)
        self.id = int(ids[0])
        integer = fmt.endswith("UI")
        filt = GL_LINEAR if (linear and not integer) else GL_NEAREST
        glTextureParameteri(self.id, GL_TEXTURE_MIN_FILTER, filt)
        glTextureParameteri(self.id, GL_TEXTURE_MAG_FILTER, filt)
        glTextureParameteri(self.id, GL_TEXTURE_WRAP_S, wrap)
        glTextureParameteri(self.id, GL_TEXTURE_WRAP_T, wrap)
        glTextureParameteri(self.id, GL_TEXTURE_WRAP_R, wrap)

    def bind_image(self, unit: int, access=GL_READ_WRITE, level: int = 0):
        glBindImageTexture(unit, self.id, level, GL_TRUE, 0, access, self.internal)

    def bind_sampler(self, unit: int):
        glBindTextureUnit(unit, self.id)

    def delete(self):
        if self.id:
            glDeleteTextures(1, [self.id])
            self.id = 0


class Image2D(_Image):
    target = GL_TEXTURE_2D

    def __init__(self, width: int, height: int, fmt: str = "RGBA16F", linear: bool = True, wrap=GL_CLAMP_TO_EDGE,
                 levels: int = 1):
        super().__init__(fmt, linear, wrap)
        self.width, self.height, self.levels = int(width), int(height), int(levels)
        glTextureStorage2D(self.id, self.levels, self.internal, self.width, self.height)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def upload(self, data: np.ndarray, level: int = 0):
        """Upload a ``(height, width[, channels])`` array (converted to the format's upload dtype)."""
        data = np.ascontiguousarray(data, dtype=self.np_dtype)
        h, w = data.shape[0], data.shape[1]
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTextureSubImage2D(self.id, level, 0, 0, w, h, self.pixel_format, self.pixel_type, data)

    def read(self, level: int = 0) -> np.ndarray:
        """Read back as ``(height, width, channels)`` (``float32`` for float formats, ``uint8`` / ``uint32`` otherwise)."""
        w = max(1, self.width >> level)
        h = max(1, self.height >> level)
        buf = np.empty((h, w, self.channels), dtype=self.np_dtype)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glGetTextureImage(self.id, level, self.pixel_format, self.pixel_type, buf.nbytes, buf)
        return buf

    def clear(self, value=(0.0, 0.0, 0.0, 0.0), level: int = 0):
        data = np.ascontiguousarray(np.asarray(value, dtype=self.np_dtype).reshape(-1)[:self.channels])
        glClearTexImage(self.id, level, self.pixel_format, self.pixel_type, data)


class Image3D(_Image):
    target = GL_TEXTURE_3D

    def __init__(self, width: int, height: int, depth: int, fmt: str = "RGBA16F", linear: bool = True,
                 wrap=GL_CLAMP_TO_EDGE):
        super().__init__(fmt, linear, wrap)
        self.width, self.height, self.depth = int(width), int(height), int(depth)
        glTextureStorage3D(self.id, 1, self.internal, self.width, self.height, self.depth)

    def upload(self, data: np.ndarray):
        """Upload a ``(depth, height, width[, channels])`` array."""
        data = np.ascontiguousarray(data, dtype=self.np_dtype)
        d, h, w = data.shape[0], data.shape[1], data.shape[2]
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTextureSubImage3D(self.id, 0, 0, 0, 0, w, h, d, self.pixel_format, self.pixel_type, data)

    def read(self) -> np.ndarray:
        buf = np.empty((self.depth, self.height, self.width, self.channels), dtype=self.np_dtype)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glGetTextureImage(self.id, 0, self.pixel_format, self.pixel_type, buf.nbytes, buf)
        return buf

    def clear(self, value=(0.0, 0.0, 0.0, 0.0)):
        data = np.ascontiguousarray(np.asarray(value, dtype=self.np_dtype).reshape(-1)[:self.channels])
        glClearTexImage(self.id, 0, self.pixel_format, self.pixel_type, data)


# ----------------------------------------------------------------------------------------------------
# Buffers
# ----------------------------------------------------------------------------------------------------

class Buffer:
    """A GL buffer object.

    ``immutable=True`` uses ``glNamedBufferStorage`` (``GL_DYNAMIC_STORAGE_BIT`` so ``upload`` works); this is
    the form to use for buffers registered with CUDA (their storage must never be re-specified).
    """

    def __init__(self, size_or_data, target=GL_SHADER_STORAGE_BUFFER, immutable: bool = True,
                 usage=GL_DYNAMIC_DRAW, flags=GL_DYNAMIC_STORAGE_BIT):
        self.target = target
        ids = np.zeros(1, dtype=np.uint32)
        glCreateBuffers(1, ids)
        self.id = int(ids[0])
        if isinstance(size_or_data, (int, np.integer)):
            self.nbytes = int(size_or_data)
            data = None
        else:
            data = np.ascontiguousarray(size_or_data)
            self.nbytes = int(data.nbytes)
        if immutable:
            glNamedBufferStorage(self.id, self.nbytes, data, flags)
        else:
            glNamedBufferData(self.id, self.nbytes, data, usage)

    def bind_base(self, index: int, target=None):
        glBindBufferBase(self.target if target is None else target, index, self.id)

    def upload(self, data: np.ndarray, offset: int = 0):
        data = np.ascontiguousarray(data)
        if offset + data.nbytes > self.nbytes:
            raise ValueError(f"upload of {data.nbytes} B at {offset} exceeds buffer size {self.nbytes}")
        glNamedBufferSubData(self.id, offset, data.nbytes, data)

    def read(self, dtype=np.uint8, count: int | None = None, offset: int = 0) -> np.ndarray:
        dtype = np.dtype(dtype)
        if count is None:
            count = (self.nbytes - offset) // dtype.itemsize
        out = np.empty(count, dtype=dtype)
        glGetNamedBufferSubData(self.id, offset, out.nbytes, out)
        return out

    def clear(self, value: int = 0):
        glClearNamedBufferData(self.id, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, np.array([value], dtype=np.uint32))

    def delete(self):
        if self.id:
            glDeleteBuffers(1, [self.id])
            self.id = 0


# ----------------------------------------------------------------------------------------------------
# Framebuffers, timers, fullscreen draws, PNG dumps
# ----------------------------------------------------------------------------------------------------

class Framebuffer:
    """Colour-only framebuffer over one or more ``Image2D`` attachments (all the same size)."""

    def __init__(self, colors: list[Image2D]):
        self.colors = list(colors)
        self.width, self.height = self.colors[0].width, self.colors[0].height
        ids = np.zeros(1, dtype=np.uint32)
        glCreateFramebuffers(1, ids)
        self.id = int(ids[0])
        attachments = []
        for i, image in enumerate(self.colors):
            glNamedFramebufferTexture(self.id, GL_COLOR_ATTACHMENT0 + i, image.id, 0)
            attachments.append(GL_COLOR_ATTACHMENT0 + i)
        glNamedFramebufferDrawBuffers(self.id, len(attachments), np.array(attachments, dtype=np.uint32))
        status = glCheckNamedFramebufferStatus(self.id, GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"framebuffer incomplete: 0x{int(status):x}")

    def bind(self, viewport: bool = True):
        glBindFramebuffer(GL_FRAMEBUFFER, self.id)
        if viewport:
            glViewport(0, 0, self.width, self.height)

    @staticmethod
    def unbind():
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def blit_to(self, dst, dst_width: int, dst_height: int, filter=GL_LINEAR):
        """Blit colour attachment 0 to ``dst`` (a ``Framebuffer`` or ``0`` for the default framebuffer)."""
        dst_id = dst.id if isinstance(dst, Framebuffer) else int(dst)
        glBlitNamedFramebuffer(self.id, dst_id, 0, 0, self.width, self.height, 0, 0, dst_width, dst_height,
                               GL_COLOR_BUFFER_BIT, filter)

    def delete(self):
        if self.id:
            glDeleteFramebuffers(1, [self.id])
            self.id = 0


class TimerQuery:
    """Ring of ``GL_TIME_ELAPSED`` queries.

    App path: ``begin()/end()`` around a pass each frame and ``ms()`` for the newest *available* result.
    Nothing in that path blocks: ``ms()`` lags by one to ``depth`` frames, and if the GPU falls more than
    ``depth`` frames behind (deep pipelining, ``--triple-buffer``) ``begin()`` drops the oldest pending sample
    instead of waiting for it, so the profile shows a stale value rather than stalling the frame.  The
    benchmarks use a deep ring (``depth = iters``) and ``collect(wait=True)`` after ``glFinish`` to obtain
    every sample exactly.
    """

    def __init__(self, depth: int = 2):
        self.depth = int(depth)
        self.ids = [int(q) for q in np.atleast_1d(glGenQueries(self.depth))]
        self.head = 0
        self.pending: list[int] = []
        self.last_ms = 0.0
        self.active = False

    def begin(self):
        if len(self.pending) >= self.depth:
            # harvest what is ready; if the ring is still full, the oldest sample is dropped (never block):
            # ids[head] is that oldest query and glBeginQuery on it discards its result anyway
            self.collect(wait=False)
            if len(self.pending) >= self.depth:
                self.pending.pop(0)
        glBeginQuery(GL_TIME_ELAPSED, self.ids[self.head])
        self.active = True

    def end(self):
        glEndQuery(GL_TIME_ELAPSED)
        self.active = False
        self.pending.append(self.ids[self.head])
        self.head = (self.head + 1) % self.depth

    def collect(self, wait: bool = False) -> list[float]:
        """Return the results (ms) of the pending queries that are available (all of them if ``wait``)."""
        out = []
        available = ctypes.c_uint32(0)
        ns = ctypes.c_uint64(0)
        while self.pending:
            q = self.pending[0]
            if not wait:
                # raw entry points: PyOpenGL's wrappers cannot allocate the 64-bit result type
                _glGetQueryObjectuiv(q, GL_QUERY_RESULT_AVAILABLE, ctypes.byref(available))
                if available.value == 0:
                    break
            _glGetQueryObjectui64v(q, GL_QUERY_RESULT, ctypes.byref(ns))
            out.append(float(ns.value) * 1e-6)
            self.pending.pop(0)
        if out:
            self.last_ms = out[-1]
        return out

    def ms(self) -> float:
        """Newest available elapsed time in milliseconds (non-blocking; repeats the last value otherwise)."""
        self.collect(wait=False)
        return self.last_ms

    def delete(self):
        if self.ids:
            glDeleteQueries(len(self.ids), np.array(self.ids, dtype=np.uint32))
            self.ids = []


class FullscreenTriangle:
    """Draws one triangle covering the viewport; positions come from ``gl_VertexID`` in ``fullscreen.vert``.

    A VAO is not required in the compatibility profile, but one is created and bound anyway so the draw is
    valid under a core profile too.
    """

    def __init__(self):
        self.vao = int(glGenVertexArrays(1))

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

    def delete(self):
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = 0


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def save_png(image: Image2D, path, exposure: float = 1.0, srgb: bool = True, flip: bool = True) -> Path:
    """Write ``image`` to ``path``.  Float formats are scaled by ``exposure``, clamped and sRGB-encoded
    (``srgb=False`` writes the clamped linear values); 8-bit formats are written as they are.
    """
    from PIL import Image as PILImage

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = image.read()
    if data.dtype == np.uint8:
        rgb = data
    elif data.dtype == np.uint32:
        m = float(data.max()) if data.max() > 0 else 1.0
        rgb = (255.0 * data.astype(np.float32) / m).astype(np.uint8)
    else:
        rgb = data.astype(np.float32) * float(exposure)
        rgb = linear_to_srgb(rgb) if srgb else np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] == 2:
        rgb = np.concatenate([rgb, np.zeros_like(rgb[..., :1])], axis=2)
    elif rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    if flip:
        rgb = rgb[::-1]
    PILImage.fromarray(np.ascontiguousarray(rgb), "RGB").save(path)
    return path


def save_heatmap_png(values: np.ndarray, path, vmax: float | None = None, flip: bool = True) -> Path:
    """Write a scalar ``(height, width)`` array as a viridis-like heat map (used for candidates per ray)."""
    from PIL import Image as PILImage

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values, dtype=np.float32)
    vmax = float(v.max()) if vmax is None else float(vmax)
    t = np.clip(v / max(vmax, 1e-6), 0.0, 1.0)
    # 5-stop viridis approximation
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=np.float32)
    pos = t * (len(stops) - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    f = (pos - i0)[..., None]
    rgb = stops[i0] * (1.0 - f) + stops[i0 + 1] * f
    rgb = np.where(v[..., None] < 0, 0.0, rgb)
    rgb = rgb.astype(np.uint8)
    if flip:
        rgb = rgb[::-1]
    PILImage.fromarray(np.ascontiguousarray(rgb), "RGB").save(path)
    return path


def perspective_camera(eye, target, up, fov_y_deg: float, width: int, height: int) -> dict:
    """Pinhole camera basis shared by the shaders (``rayDir = normalize(fwd + x*right*tanHalf*aspect + y*up*tanHalf)``)
    and their numpy references."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    cup = np.cross(right, fwd)
    tan_half = math.tan(math.radians(fov_y_deg) * 0.5)
    return {"eye": eye, "fwd": fwd, "right": right, "up": cup, "tan_half": tan_half,
            "aspect": width / height, "width": width, "height": height}


def camera_rays(cam: dict, jitter=(0.5, 0.5)) -> tuple[np.ndarray, np.ndarray]:
    """Ray origins/directions ``(height, width, 3)`` matching the shader's ``cameraRay`` for pixel centres."""
    w, h = cam["width"], cam["height"]
    px = (np.arange(w, dtype=np.float64) + jitter[0]) / w * 2.0 - 1.0
    py = (np.arange(h, dtype=np.float64) + jitter[1]) / h * 2.0 - 1.0
    X, Y = np.meshgrid(px, py)
    d = (cam["fwd"][None, None, :]
         + X[..., None] * cam["right"][None, None, :] * cam["tan_half"] * cam["aspect"]
         + Y[..., None] * cam["up"][None, None, :] * cam["tan_half"])
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    o = np.broadcast_to(cam["eye"], d.shape)
    return o, d

