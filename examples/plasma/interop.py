# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Warp <-> OpenGL data path (plan 6.4).

The CUDA graph writes Warp-owned staging arrays only; every frame the render thread maps the GL resources,
copies staging -> GL and unmaps.  The fixed cost is the GL <-> CUDA handoff of a map/unmap round trip: about
0.14 ms of stream latency on this box (measured as map + unmap + ``wp.synchronize()``; the CPU submission
itself is ~0.013 ms), independent of the number and size of the resources mapped together.  Nested
per-resource maps (map all, unmap all -- Warp's own path, and this module's fallback) share ONE round trip;
sequential pairs (map A, unmap A, map B, unmap B) pay one each (0.25 ms for two).  ``BatchedMap`` maps the N
resources of a frame with one ``cuGraphicsMapResources`` / ``cuGraphicsUnmapResources`` pair (ctypes on
``libcuda``): it makes the sequential pattern impossible and saves ~0.01-0.03 ms of CPU per frame over the
nested Warp calls, nothing more.  The mapped addresses are re-read after every map and never baked into a
capture.

Teardown: a registered resource that outlives its GL context makes Warp's ``__del__`` print ``CUDA error 219:
invalid OpenGL or DirectX context``.  Call :meth:`BatchedMap.close` (or :func:`unregister` on each resource)
and drop every reference to the registered objects *before* the GL objects / context are destroyed --
interpreter exit tears module globals down in arbitrary order.

Warp 1.17 internals relied upon (``warp/_src/context.py`` and ``warp/_src/texture.py`` of the installed wheel):

* ``wp.RegisteredGLBuffer(gl_id, device, flags, fallback_to_copy=False)`` keeps in its public attribute
  ``.resource`` the ``void*`` returned by the native ``wp_cuda_graphics_register_gl_buffer``
  (``warp/native/warp.cu``): that is a pointer to a heap-allocated ``CUgraphicsResource`` slot
  (``new CUgraphicsResource``), NOT the resource handle itself -- the raw ``CUgraphicsResource`` is read by
  dereferencing it once (:func:`resource_handle`).  ``.device`` / ``.context`` hold the Warp device and its
  CUDA context.
* ``wp.GLTextureResource(gl_tex, gl_target, device, flags)`` keeps the same kind of slot pointer (from
  ``wp_cuda_graphics_register_gl_image``) in the private attribute ``._resource`` and the Warp device in
  ``._device``.  Its constructor does ``from pyglet import gl`` only to compare ``gl_target`` with
  ``gl.GL_TEXTURE_1D/2D/3D``; :func:`install_pyglet_stub` provides exactly those three constants when pyglet
  is not installed.
* Warp's own ``wp_cuda_graphics_map`` / ``unmap`` call ``cuGraphicsMapResources(1, slot, get_current_stream())``
  per resource, i.e. they are ordered on Warp's current stream exactly like the batched call below.
* ``wp.Texture3D(cuda_array=<CUarray>, device=...)`` wraps a mapped array (descriptor read with
  ``cuArrayGetDescriptor``) and ``.copy_from(wp.array)`` issues ``cuMemcpy3DAsync`` on the device stream.
* ``warp._src.context.runtime.core.wp_cuda_graphics_map / wp_cuda_graphics_unmap /
  wp_cuda_graphics_device_ptr_and_size / wp_cuda_graphics_sub_resource_get_mapped_array`` are the per-resource
  fallback used when ``libcuda`` cannot be driven through ctypes.
* ``device.context_guard`` makes the device's CUDA context current around raw driver calls;
  ``wp.get_stream(device).cuda_stream`` is the ``CUstream`` handle the batched map is ordered on.

Formats verified on this box: std430 SSBOs (any size), ``GL_TEXTURE_3D`` ``GL_RGBA16F`` (``wp.vec4h`` staging,
``(depth, height, width)``), ``GL_R32F``.  ``GL_RGB*`` and ``GL_R11F_G11F_B10F`` are not CUDA-registerable.
"""

import ctypes
import sys
import types

import warp as wp

GL_TEXTURE_1D = 0x0DE0
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_3D = 0x806F

CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_ARRAY = 3


def install_pyglet_stub():
    """Insert a minimal ``pyglet.gl`` into ``sys.modules`` (the three texture-target enums Warp reads)."""
    if "pyglet.gl" in sys.modules:
        return
    try:
        import pyglet.gl  # noqa: F401
        return
    except ImportError:
        pass
    pyglet = types.ModuleType("pyglet")
    gl = types.ModuleType("pyglet.gl")
    gl.GL_TEXTURE_1D = GL_TEXTURE_1D
    gl.GL_TEXTURE_2D = GL_TEXTURE_2D
    gl.GL_TEXTURE_3D = GL_TEXTURE_3D
    pyglet.gl = gl
    sys.modules["pyglet"] = pyglet
    sys.modules["pyglet.gl"] = gl


def register_buffer(gl_id: int, flags: int = wp.RegisteredGLBuffer.NONE, device=None) -> wp.RegisteredGLBuffer:
    """Register a GL buffer object with CUDA (``fallback_to_copy=False``: a failure raises instead of hiding)."""
    return wp.RegisteredGLBuffer(int(gl_id), device=device, flags=flags, fallback_to_copy=False)


def register_image(gl_tex: int, target: int = GL_TEXTURE_3D,
                   flags: int = wp.TextureResourceFlags.SURFACE_LDST, device=None) -> wp.GLTextureResource:
    """Register a GL texture with CUDA.  ``SURFACE_LDST`` is required if a Warp kernel writes it through a
    surface; plain ``cuMemcpy3D`` uploads work with any flag."""
    install_pyglet_stub()
    return wp.GLTextureResource(int(gl_tex), int(target), device=device, flags=flags)


def unregister(obj):
    """Unregister a Warp GL resource now (idempotent) and neuter its ``__del__``; the resource must be unmapped."""
    if isinstance(obj, wp.RegisteredGLBuffer):
        if obj.resource:
            with obj.device.context_guard:
                wp._src.context.runtime.core.wp_cuda_graphics_unregister_resource(obj.context, obj.resource)
            obj.resource = None
    elif isinstance(obj, wp.GLTextureResource):
        if obj._resource:
            with obj._device.context_guard:
                obj.unmap()
                obj._runtime.core.wp_cuda_graphics_unregister_resource(obj._device.context, obj._resource)
            obj._resource = None
    else:
        raise TypeError(f"not a Warp GL resource: {type(obj)}")


def resource_slot(obj) -> int:
    """Warp's ``void*`` for a registered resource: the address of its heap ``CUgraphicsResource`` slot."""
    if isinstance(obj, wp.RegisteredGLBuffer):
        slot = obj.resource
    elif isinstance(obj, wp.GLTextureResource):
        slot = obj._resource
    else:
        raise TypeError(f"not a Warp GL resource: {type(obj)}")
    if not slot:
        raise RuntimeError("resource is not registered (CUDA/GL interop unavailable?)")
    return int(slot)


def resource_handle(obj) -> int:
    """The raw ``CUgraphicsResource`` of a Warp registered resource (``*slot``, see the module docstring)."""
    return int(ctypes.cast(resource_slot(obj), ctypes.POINTER(ctypes.c_void_p)).contents.value or 0)


# ----------------------------------------------------------------------------------------------------
# libcuda through ctypes
# ----------------------------------------------------------------------------------------------------

class CUDA_MEMCPY3D(ctypes.Structure):
    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t), ("srcZ", ctypes.c_size_t),
        ("srcLOD", ctypes.c_size_t), ("srcMemoryType", ctypes.c_int),
        ("srcHost", ctypes.c_void_p), ("srcDevice", ctypes.c_void_p), ("srcArray", ctypes.c_void_p),
        ("reserved0", ctypes.c_void_p), ("srcPitch", ctypes.c_size_t), ("srcHeight", ctypes.c_size_t),
        ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t), ("dstZ", ctypes.c_size_t),
        ("dstLOD", ctypes.c_size_t), ("dstMemoryType", ctypes.c_int),
        ("dstHost", ctypes.c_void_p), ("dstDevice", ctypes.c_void_p), ("dstArray", ctypes.c_void_p),
        ("reserved1", ctypes.c_void_p), ("dstPitch", ctypes.c_size_t), ("dstHeight", ctypes.c_size_t),
        ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t), ("Depth", ctypes.c_size_t),
    ]


class _Driver:
    """The handful of CUDA driver entry points the interop needs, or ``None`` if libcuda is unavailable."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        lib = self.lib
        lib.cuGraphicsMapResources.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.cuGraphicsMapResources.restype = ctypes.c_int
        lib.cuGraphicsUnmapResources.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.cuGraphicsUnmapResources.restype = ctypes.c_int
        lib.cuGraphicsResourceGetMappedPointer_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                                              ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
        lib.cuGraphicsResourceGetMappedPointer_v2.restype = ctypes.c_int
        lib.cuGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                                            ctypes.c_uint, ctypes.c_uint]
        lib.cuGraphicsSubResourceGetMappedArray.restype = ctypes.c_int
        lib.cuMemcpy3DAsync_v2.argtypes = [ctypes.POINTER(CUDA_MEMCPY3D), ctypes.c_void_p]
        lib.cuMemcpy3DAsync_v2.restype = ctypes.c_int
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.cuGetErrorString.restype = ctypes.c_int

    def error_string(self, code: int) -> str:
        msg = ctypes.c_char_p()
        self.lib.cuGetErrorString(code, ctypes.byref(msg))
        return msg.value.decode() if msg.value else f"CUDA error {code}"

    def check(self, code: int, what: str):
        if code != 0:
            raise RuntimeError(f"{what} failed: {self.error_string(code)}")


def _load_driver():
    try:
        return _Driver()
    except (OSError, AttributeError) as e:
        print(f"interop: libcuda unavailable through ctypes ({e}); using Warp's per-resource map/unmap")
        return None


_driver = _load_driver()


def driver_available() -> bool:
    return _driver is not None


# ----------------------------------------------------------------------------------------------------
# Batched map / unmap
# ----------------------------------------------------------------------------------------------------

class BatchedMap:
    """Map N registered GL resources with one ``cuGraphicsMapResources`` call on Warp's current stream.

    After :meth:`map`, :meth:`array` returns a ``wp.array`` view of a mapped buffer, :meth:`cuarray` the mapped
    ``CUarray`` of a texture and :meth:`texture` a ``wp.Texture3D`` wrapper around it (created once per map).
    All of them are invalid after :meth:`unmap`; re-read them after every map.

    If the driver cannot be called through ctypes, or the batched call fails, the object falls back to Warp's
    own per-resource ``wp_cuda_graphics_map`` / ``unmap`` in nested order (map all, unmap all): same single
    GL <-> CUDA handoff per frame, ~0.01-0.03 ms more CPU time.

    :meth:`close` unmaps, drops the wrappers and (by default) unregisters the resources; call it from the
    owning ``Input``'s teardown before the GL objects go away.
    """

    def __init__(self, resources, device=None):
        self.resources = list(resources)
        self.device = wp.get_device(device)
        self.count = len(self.resources)
        self.handles = (ctypes.c_void_p * self.count)(*[resource_handle(r) for r in self.resources])
        self.slots = [resource_slot(r) for r in self.resources]
        self.is_buffer = [isinstance(r, wp.RegisteredGLBuffer) for r in self.resources]
        self.batched = _driver is not None
        self.ptrs = [0] * self.count
        self.sizes = [0] * self.count
        self.arrays = [0] * self.count
        self._textures: dict[int, wp.Texture3D] = {}
        self.mapped = False
        self._core = wp._src.context.runtime.core

    @property
    def stream(self) -> int:
        return int(wp.get_stream(self.device).cuda_stream)

    def map(self) -> "BatchedMap":
        if self.mapped:
            return self
        if not self.resources:
            raise RuntimeError("BatchedMap is closed")
        with self.device.context_guard:
            if self.batched:
                code = _driver.lib.cuGraphicsMapResources(self.count, self.handles, ctypes.c_void_p(self.stream))
                if code != 0:
                    print(f"interop: cuGraphicsMapResources failed ({_driver.error_string(code)}); "
                          f"falling back to per-resource mapping")
                    self.batched = False
            if not self.batched:
                for slot in self.slots:
                    if not self._core.wp_cuda_graphics_map(self.device.context, slot):
                        raise RuntimeError("wp_cuda_graphics_map failed")
            for i, h in enumerate(self.handles):
                slot = self.slots[i]
                if self.is_buffer[i]:
                    ptr = ctypes.c_void_p(0)
                    size = ctypes.c_size_t(0)
                    if self.batched:
                        _driver.check(_driver.lib.cuGraphicsResourceGetMappedPointer_v2(ctypes.byref(ptr),
                                                                                         ctypes.byref(size), h),
                                      "cuGraphicsResourceGetMappedPointer")
                        self.ptrs[i], self.sizes[i] = int(ptr.value or 0), int(size.value)
                    else:
                        ptr64 = ctypes.c_uint64(0)
                        self._core.wp_cuda_graphics_device_ptr_and_size(self.device.context, slot,
                                                                        ctypes.byref(ptr64), ctypes.byref(size))
                        self.ptrs[i], self.sizes[i] = int(ptr64.value), int(size.value)
                else:
                    if self.batched:
                        arr = ctypes.c_void_p(0)
                        _driver.check(_driver.lib.cuGraphicsSubResourceGetMappedArray(ctypes.byref(arr), h, 0, 0),
                                      "cuGraphicsSubResourceGetMappedArray")
                        self.arrays[i] = int(arr.value or 0)
                    else:
                        self.arrays[i] = int(self._core.wp_cuda_graphics_sub_resource_get_mapped_array(
                            self.device.context, slot, 0, 0))
        self.mapped = True
        return self

    def unmap(self):
        if not self.mapped:
            return
        self._textures.clear()
        with self.device.context_guard:
            if self.batched:
                _driver.check(_driver.lib.cuGraphicsUnmapResources(self.count, self.handles,
                                                                    ctypes.c_void_p(self.stream)),
                              "cuGraphicsUnmapResources")
            else:
                for slot in self.slots:
                    self._core.wp_cuda_graphics_unmap(self.device.context, slot)
        self.mapped = False

    def array(self, i: int, dtype, shape) -> wp.array:
        """``wp.array`` view of mapped buffer ``i`` (valid until :meth:`unmap`)."""
        if not self.mapped or not self.is_buffer[i]:
            raise RuntimeError("resource is not a mapped buffer")
        return wp.array(ptr=self.ptrs[i], dtype=dtype, shape=shape, device=self.device)

    def cuarray(self, i: int) -> int:
        if not self.mapped or self.is_buffer[i]:
            raise RuntimeError("resource is not a mapped image")
        return self.arrays[i]

    def texture(self, i: int) -> wp.Texture3D:
        """``wp.Texture3D`` aliasing mapped image ``i`` (supports ``.copy_from(staging)``).

        A per-map wrapper: creating it costs a ``cuTexObjectCreate`` (~0.012 ms) and dropping it destroys the
        texture object from whichever thread releases the last reference, so do NOT keep the returned object
        across :meth:`unmap`.  :func:`upload_volume` (plain ``cuMemcpy3DAsync`` into :meth:`cuarray`, no texture
        object) is the cheaper default for uploads."""
        tex = self._textures.get(i)
        if tex is None:
            tex = wp.Texture3D(cuda_array=self.cuarray(i), device=self.device)
            self._textures[i] = tex
        return tex

    def close(self, unregister_resources: bool = True):
        """Unmap if mapped, drop the texture wrappers and the resource list; with ``unregister_resources`` also
        unregister every resource from CUDA (Warp's ``__del__`` then has nothing left to do at interpreter
        exit, whatever the teardown order).  Idempotent."""
        if not self.resources:
            return
        self.unmap()
        self._textures.clear()
        if unregister_resources:
            for r in self.resources:
                unregister(r)
        self.resources = []
        self.count = 0

    def __del__(self):
        try:
            self.close(unregister_resources=False)
        except Exception:
            pass

    def __enter__(self) -> "BatchedMap":
        return self.map()

    def __exit__(self, *exc):
        self.unmap()
        return False


# ----------------------------------------------------------------------------------------------------
# Volume upload
# ----------------------------------------------------------------------------------------------------

def upload_volume(dst_cuarray: int, src: wp.array, width: int, height: int, depth: int, texel_bytes: int,
                  device=None, stream: int | None = None):
    """``cuMemcpy3DAsync`` from a contiguous device array (``depth * height * width * texel_bytes`` bytes, x fastest)
    into a mapped ``CUarray`` (e.g. a ``GL_RGBA16F`` ``GL_TEXTURE_3D``: ``texel_bytes = 8``).

    Stream-ordered on Warp's current stream (or ``stream``); nothing is synchronised.
    """
    if _driver is None:
        raise RuntimeError("upload_volume needs libcuda through ctypes; use BatchedMap.texture(i).copy_from(src)")
    device = wp.get_device(device)
    if not src.is_contiguous or src.device != device:
        raise ValueError("staging array must be contiguous and on the texture's device")
    if src.size * wp.types.type_size_in_bytes(src.dtype) < width * height * depth * texel_bytes:
        raise ValueError("staging array is smaller than the volume")
    desc = CUDA_MEMCPY3D()
    desc.srcMemoryType = CU_MEMORYTYPE_DEVICE
    desc.srcDevice = ctypes.c_void_p(int(src.ptr))
    desc.srcPitch = width * texel_bytes
    desc.srcHeight = height
    desc.dstMemoryType = CU_MEMORYTYPE_ARRAY
    desc.dstArray = ctypes.c_void_p(int(dst_cuarray))
    desc.WidthInBytes = width * texel_bytes
    desc.Height = height
    desc.Depth = depth
    if stream is None:
        stream = int(wp.get_stream(device).cuda_stream)
    with device.context_guard:
        _driver.check(_driver.lib.cuMemcpy3DAsync_v2(ctypes.byref(desc), ctypes.c_void_p(stream)), "cuMemcpy3DAsync")


def pointer_stability(batch: BatchedMap, cycles: int = 1000) -> dict:
    """Map/unmap ``cycles`` times and count how often each mapped address / array handle changed."""
    previous = None
    changes = [0] * batch.count
    for _ in range(cycles):
        batch.map()
        current = [batch.ptrs[i] if batch.is_buffer[i] else batch.arrays[i] for i in range(batch.count)]
        batch.unmap()
        if previous is not None:
            for i in range(batch.count):
                if current[i] != previous[i]:
                    changes[i] += 1
        previous = current
    return {"cycles": cycles, "changes": changes, "stable": all(c == 0 for c in changes)}
