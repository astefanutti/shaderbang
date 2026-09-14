# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Headless desktop-GL context via the EGL device platform.

Shaderbang's own runtime creates the GL context on the DRM/GBM surface (``shaderbang/lib/common.c``),
which needs DRM master.  The benchmarks and the headless render tests instead create a surfaceless
GL 4.6 *compatibility* context (the same profile the runtime requests) on the NVIDIA EGL device, so
the exact same shaders, image formats and Warp interop path can be exercised while a compositor
owns the display.

Usage::

    from plasma.glctx import make_headless_context, gl_limits

    with make_headless_context() as ctx:
        limits = gl_limits()
        ...

``PyOpenGL`` must be switched to the EGL platform before ``OpenGL.GL`` is imported anywhere in the
process; importing this module does that.
"""

from OpenGL import setPlatform

setPlatform("egl")

from OpenGL import arrays  # noqa: E402
from OpenGL.EGL import (  # noqa: E402
    EGL_BLUE_SIZE, EGL_CONTEXT_MAJOR_VERSION, EGL_CONTEXT_MINOR_VERSION, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_GREEN_SIZE, EGL_NO_CONTEXT, EGL_NO_SURFACE, EGL_NONE, EGL_OPENGL_API,
    EGL_OPENGL_BIT, EGL_PBUFFER_BIT, EGL_RED_SIZE, EGL_RENDERABLE_TYPE, EGL_SURFACE_TYPE, EGL_VENDOR, EGLConfig, EGLint,
    eglBindAPI, eglChooseConfig, eglCreateContext, eglDestroyContext, eglInitialize, eglMakeCurrent, eglQueryString,
    eglTerminate,
)
from OpenGL.EGL.EXT.device_base import EGLDeviceEXT, eglQueryDevicesEXT  # noqa: E402
from OpenGL.EGL.EXT.platform_base import eglGetPlatformDisplayEXT  # noqa: E402
from OpenGL.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT  # noqa: E402
from OpenGL.GL import (  # noqa: E402
    GL_MAX_3D_TEXTURE_SIZE, GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES, GL_MAX_COMPUTE_IMAGE_UNIFORMS,
    GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, GL_MAX_COMPUTE_WORK_GROUP_COUNT, GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
    GL_MAX_COMPUTE_WORK_GROUP_SIZE, GL_MAX_FRAGMENT_IMAGE_UNIFORMS, GL_MAX_IMAGE_UNITS,
    GL_MAX_SHADER_STORAGE_BLOCK_SIZE, GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, GL_MAX_TEXTURE_BUFFER_SIZE,
    GL_MAX_UNIFORM_BLOCK_SIZE, GL_RENDERER, GL_SHADING_LANGUAGE_VERSION, GL_VERSION, glGetIntegeri_v, glGetIntegerv,
    glGetString,
)
from OpenGL.raw.EGL._errors import EGLError  # noqa: E402


class HeadlessContext:
    """A current, surfaceless GL 4.6 compatibility context on an EGL device.

    Attributes:
        dpy: the ``EGLDisplay`` of the selected device
        ctx: the ``EGLContext``
        vendor: the EGL vendor string of the device (``b"NVIDIA"``)
        version, glsl, renderer: the ``GL_VERSION`` / ``GL_SHADING_LANGUAGE_VERSION`` / ``GL_RENDERER`` strings
    """

    def __init__(self, vendor: bytes = b"NVIDIA", major: int = 4, minor: int = 6, verbose: bool = True):
        count = EGLint()
        eglQueryDevicesEXT(0, None, count)
        devices = (EGLDeviceEXT * count.value)()
        eglQueryDevicesEXT(count.value, devices, count)

        self.dpy = None
        self.ctx = None
        for i in range(count.value):
            dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[i], None)
            try:
                if not eglInitialize(dpy, EGLint(), EGLint()):
                    continue
            except EGLError:
                # Mesa answers for the render nodes it cannot drive; skip them.
                continue
            dev_vendor = eglQueryString(dpy, EGL_VENDOR)
            if vendor in dev_vendor:
                self.dpy = dpy
                self.vendor = dev_vendor
                break
            eglTerminate(dpy)
        if self.dpy is None:
            raise RuntimeError(f"no EGL device with vendor {vendor!r} among {count.value} device(s)")

        eglBindAPI(EGL_OPENGL_API)
        config_attribs = arrays.GLintArray.asArray([
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
            EGL_NONE,
        ])
        configs = (EGLConfig * 1)()
        num_configs = EGLint()
        eglChooseConfig(self.dpy, config_attribs, configs, 1, num_configs)
        if num_configs.value < 1:
            raise RuntimeError("eglChooseConfig returned no OpenGL-capable config")

        context_attribs = arrays.GLintArray.asArray([
            EGL_CONTEXT_MAJOR_VERSION, major,
            EGL_CONTEXT_MINOR_VERSION, minor,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
            EGL_NONE,
        ])
        self.ctx = eglCreateContext(self.dpy, configs[0], EGL_NO_CONTEXT, context_attribs)
        if not self.ctx:
            raise RuntimeError(f"eglCreateContext failed for GL {major}.{minor} compatibility")
        if not eglMakeCurrent(self.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, self.ctx):
            raise RuntimeError("eglMakeCurrent failed (surfaceless)")

        self.version = glGetString(GL_VERSION).decode()
        self.glsl = glGetString(GL_SHADING_LANGUAGE_VERSION).decode()
        self.renderer = glGetString(GL_RENDERER).decode()
        if verbose:
            print(f"GL {self.version} | GLSL {self.glsl} | {self.renderer} | EGL {self.vendor.decode()} device")

    def close(self):
        if self.dpy is not None:
            eglMakeCurrent(self.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
            if self.ctx is not None:
                eglDestroyContext(self.dpy, self.ctx)
                self.ctx = None
            eglTerminate(self.dpy)
            self.dpy = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_headless_context(verbose: bool = True) -> HeadlessContext:
    """Create and make current a headless GL 4.6 compatibility context on the NVIDIA EGL device."""
    return HeadlessContext(verbose=verbose)


def _indexed(pname, n: int = 3) -> tuple:
    return tuple(int(glGetIntegeri_v(pname, i)[0]) for i in range(n))


def gl_limits() -> dict:
    """Return the implementation limits that constrain the renderer design (see ``docs/plasma_globe.md``).

    ``GL_MAX_IMAGE_UNITS`` (8 on NVIDIA) bounds the number of ``image*`` bindings per pass;
    ``GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES`` (16) bounds images + SSBOs + draw buffers together.
    """
    return {
        "GL_MAX_IMAGE_UNITS": int(glGetIntegerv(GL_MAX_IMAGE_UNITS)),
        "GL_MAX_COMPUTE_IMAGE_UNIFORMS": int(glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS)),
        "GL_MAX_FRAGMENT_IMAGE_UNIFORMS": int(glGetIntegerv(GL_MAX_FRAGMENT_IMAGE_UNIFORMS)),
        "GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES": int(glGetIntegerv(GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES)),
        "GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS": int(glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)),
        "GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS": int(glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)),
        "GL_MAX_COMPUTE_WORK_GROUP_SIZE": _indexed(GL_MAX_COMPUTE_WORK_GROUP_SIZE),
        "GL_MAX_COMPUTE_WORK_GROUP_COUNT": _indexed(GL_MAX_COMPUTE_WORK_GROUP_COUNT),
        "GL_MAX_COMPUTE_SHARED_MEMORY_SIZE": int(glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE)),
        "GL_MAX_SHADER_STORAGE_BLOCK_SIZE": int(glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)),
        "GL_MAX_UNIFORM_BLOCK_SIZE": int(glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE)),
        "GL_MAX_TEXTURE_BUFFER_SIZE": int(glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE)),
        "GL_MAX_3D_TEXTURE_SIZE": int(glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE)),
    }


def print_gl_limits():
    for name, value in gl_limits().items():
        print(f"  {name:44s} {value}")


__all__ = ["HeadlessContext", "make_headless_context", "gl_limits", "print_gl_limits"]
