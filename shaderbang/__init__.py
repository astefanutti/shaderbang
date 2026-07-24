# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

from ctypes import *
import os
import warnings


class _MissingLibrary:
    """No-op stand-in for the native ``_shaderbang.so`` extension.

    The extension provides the EGL/DRM/KMS render loop and GL entry points and
    only builds on Linux. On a non-Linux / CUDA-less development box it may be
    absent; falling back to this stub keeps the pure-Python and Warp parts of
    the package importable (e.g. the cloth physics kernels and the path-tracer
    smoke test) for off-target development and CPU-side testing.

    Every attribute resolves to a callable that returns ``0`` and does nothing:
    ``shaderbang.input`` registers ``onInit``/``onRender`` callbacks at import
    time, so the stub must be callable to let the import succeed. The native
    render loop never runs through the stub, so this is only useful for
    headless, non-rendering use — any real GL/render work is silently skipped.
    """

    def __init__(self, error):
        self._error = error

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return 0
        return _noop


try:
    lib = CDLL(os.path.join(os.path.dirname(__file__), "_shaderbang.so"))
except OSError as e:
    warnings.warn(
        f"could not load the native _shaderbang.so extension ({e}); falling "
        "back to a no-op stub. EGL/DRM/KMS rendering is unavailable — only "
        "headless / CPU use (e.g. Warp physics, off-target development) works.",
        RuntimeWarning,
        stacklevel=2,
    )
    lib = _MissingLibrary(e)


class OPTIONS(Structure):
    _fields_ = [
        ("device",          c_char_p),
        ("connector",       c_int),
        ("mode",            c_ubyte * 32),
        ("refresh",         c_int),
        ("format",          c_uint32),
        ("modifier",        c_uint64),
        ("async_page_flip", c_bool),
        ("atomic_drm_mode", c_bool),
        ("surfaceless",     c_bool),
        ("frames",          c_uint),
    ]


def options(args):
    c_opts = OPTIONS()
    if args.device:
        c_opts.device = bytes(args.device.as_posix(), 'utf-8')
    if args.connector:
        c_opts.connector = c_int(args.connector)
    else:
        c_opts.connector = -1
    if args.mode:
        c_opts.mode = (c_ubyte * 32)(*bytes(args.mode, 'utf-8'))
    if args.refresh:
        c_opts.refresh = c_int(args.refresh)
    if args.async_page_flip:
        c_opts.async_page_flip = c_bool(True)
    if args.atomic_drm_mode:
        c_opts.atomic_drm_mode = c_bool(True)
    if args.frames:
        c_opts.frames = c_uint(args.frames)
    return c_opts
