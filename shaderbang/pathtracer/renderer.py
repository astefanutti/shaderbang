# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""Reusable real-time OptiX path tracer (milestone M1: the first live frame).

:class:`PathTracer` owns a complete OptiX pipeline and renders the *same*
geometry the Warp physics writes -- plain ``wp.array``s -- with no CPU copy on
the hot path:

  * a triangle GAS is built once from the geometry device pointers and *refit*
    in place every frame as the vertices deform (``refit()``);
  * one primary ray per pixel is traced against that GAS, with the sphere and
    ground intersected analytically in the raygen program (see ``programs.cu``);
  * radiance is accumulated in an HDR buffer, run through the OptiX AI denoiser
    (single-frame HDR by default; a temporal 2x-upscale model with motion vectors
    when constructed with ``upscale=2``), then tone-mapped (ACES) straight into
    the display surface -- either a Warp kernel into an OpenGL PBO followed by a
    ``glTexSubImage2D`` upload (the portable default), or, with
    ``interop="texture"``/``"auto"`` (M6), a CUDA surface-write kernel straight
    into the GL texture, eliminating the PBO->texture copy.

Everything -- OptiX, Warp, CuPy -- shares the CUDA *primary* context (OptiX is
created with ``cu_ctx = 0``) so device memory and the stream are shared. The
host-side API shapes here mirror ``smoke.py``, which is the load-bearing M0
de-risk test; anything that could not be checked on the CUDA-less dev box is
flagged ``# VERIFY-ON-TARGET``.

RTX-on-target only. Importing this module on a CUDA-less box is fine (warp/numpy
are the only import-time deps); constructing :class:`PathTracer` is what needs
the GPU stack (cupy, optix, cuda.bindings.nvrtc, the OptiX + CUDA headers).
"""

import ctypes
import math
import os

import numpy as np
import warp as wp


# --------------------------------------------------------------------------- #
# Launch parameters -- MUST match the Params struct in programs.cu field-for-
# field. align=True reproduces C struct padding: the 8-byte members come first
# (twelve pointers + the IAS handle), then the 4-byte scalars, then the tightly
# packed float3s (each an ('f4', (3,)) subarray == float3; float2 == ('f4',
# (2,))). itemsize rounds up to a multiple of 8. Mesh geometry moved off Params
# into per-instance SBT hitgroup records in M7d (see _hitgroup_record_dtype).
# --------------------------------------------------------------------------- #
_PARAMS_NAMES = [
    "accum", "output", "albedo", "normal",
    "flow", "handle",
    "env_data", "env_cdf", "materials", "lights", "spheres", "planes",
    "width", "height", "subframe", "max_depth", "rr_depth", "exposure",
    "env_width", "env_height", "env_enabled",
    "num_materials", "num_lights",
    "num_spheres", "num_planes",
    "cam_eye", "cam_u", "cam_v", "cam_w",
    "prev_cam_eye", "prev_cam_u", "prev_cam_v", "prev_cam_w",
    "sky_top", "sky_bottom",
    "env_total_sum", "env_intensity", "env_rotation",
]
_PARAMS_FORMATS = [
    "u8", "u8", "u8", "u8",
    "u8", "u8",
    "u8", "u8", "u8", "u8", "u8", "u8",
    "u4", "u4", "u4", "u4", "u4", "f4",
    "u4", "u4", "u4",
    "u4", "u4",
    "u4", "u4",
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)),
    "f4", "f4", "f4",
]
# Build with align=True, then pin itemsize up to the 8-byte struct alignment so
# it equals the CUDA sizeof(Params) exactly (numpy stops at the last field; C
# rounds the struct up to a multiple of alignof == 8). A -D PARAMS_EXPECTED_SIZE
# static_assert in programs.cu catches any host/device layout drift at compile.
_PARAMS_BASE = np.dtype({"names": _PARAMS_NAMES, "formats": _PARAMS_FORMATS,
                         "align": True})
_PARAMS_ITEMSIZE = (_PARAMS_BASE.itemsize + 7) & ~7
PARAMS_DTYPE = np.dtype({
    "names": _PARAMS_NAMES,
    "formats": [_PARAMS_BASE.fields[n][0] for n in _PARAMS_NAMES],
    "offsets": [_PARAMS_BASE.fields[n][1] for n in _PARAMS_NAMES],
    "itemsize": _PARAMS_ITEMSIZE,
    "align": True,
})


# --------------------------------------------------------------------------- #
# ACES tone-map, run on device by a Warp kernel post-denoise (keeps the GL side
# to just "upload texture + draw quad", robust across the target's GL version).
# --------------------------------------------------------------------------- #
@wp.func
def _aces(x: float) -> float:
    # Narkowicz 2015 ACES filmic approximation.
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return wp.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


@wp.func
def _to_u8(x: float) -> wp.uint8:
    return wp.uint8(int(wp.clamp(x, 0.0, 1.0) * 255.0 + 0.5))


@wp.kernel
def _tonemap_kernel(hdr: wp.array(dtype=wp.vec4),
                    out: wp.array(dtype=wp.uint8),
                    exposure: float):
    i = wp.tid()
    c = hdr[i]
    inv_gamma = 1.0 / 2.2
    r = wp.pow(_aces(c[0] * exposure), inv_gamma)
    g = wp.pow(_aces(c[1] * exposure), inv_gamma)
    b = wp.pow(_aces(c[2] * exposure), inv_gamma)
    o = i * 4
    out[o + 0] = _to_u8(r)
    out[o + 1] = _to_u8(g)
    out[o + 2] = _to_u8(b)
    out[o + 3] = wp.uint8(255)


# --------------------------------------------------------------------------- #
# M6 direct-to-texture present: the same ACES tone-map, but written straight into
# a GL texture bound as a CUDA surface (surf2Dwrite) instead of into a PBO. This
# is the raw-CUDA kernel that Warp cannot express (Warp's GL interop is
# buffer-only and it has no surface-object write), compiled at runtime with CuPy's
# RawModule and launched over the mapped GL-texture surface. It must stay
# pixel-identical to _tonemap_kernel above so switching present paths does not
# change the image: same ACES curve, same 1/2.2 gamma, same round-to-nearest u8.
# Surface writes are BYTE-addressed on x, so the column offset is x*4 (uchar4).
# --------------------------------------------------------------------------- #
_SURF_TONEMAP_SRC = r"""
__device__ __forceinline__ float _aces(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    float v = (x * (a * x + b)) / (x * (c * x + d) + e);
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ __forceinline__ unsigned char _to_u8(float x) {
    return (unsigned char)((int)(fminf(fmaxf(x, 0.0f), 1.0f) * 255.0f + 0.5f));
}

extern "C" __global__ void tonemap_surface(
        unsigned long long hdr_addr,   // const float4* (wp.vec4) denoised buffer
        cudaSurfaceObject_t surf,      // the mapped GL texture
        float exposure,
        int width,
        int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const float4* hdr = reinterpret_cast<const float4*>(hdr_addr);
    float4 c = hdr[y * width + x];
    const float inv_gamma = 1.0f / 2.2f;
    uchar4 o;
    o.x = _to_u8(powf(_aces(c.x * exposure), inv_gamma));
    o.y = _to_u8(powf(_aces(c.y * exposure), inv_gamma));
    o.z = _to_u8(powf(_aces(c.z * exposure), inv_gamma));
    o.w = (unsigned char)255;
    // Byte-addressed x (uchar4 == 4 bytes); row 0 == bottom, matching the ray-gen
    // and the PBO path's glTexSubImage2D upload, so the quad's texcoords are the
    // same either way.
    surf2Dwrite(o, surf, x * 4, y);
}
"""


# --------------------------------------------------------------------------- #
# Toolchain include-path discovery (NVRTC needs the OptiX + CUDA headers).
# Same env vars as smoke.py.
# --------------------------------------------------------------------------- #
def _optix_include_dir():
    direct = os.environ.get("OPTIX_INCLUDE_DIR")
    if direct and os.path.isdir(direct):
        return direct
    for var in ("OPTIX_PATH", "OPTIX_ROOT", "OptiX_INSTALL_DIR", "OPTIX_SDK_PATH"):
        root = os.environ.get(var)
        if root:
            inc = os.path.join(root, "include")
            if os.path.isdir(inc):
                return inc
    return None


def _cuda_include_dir():
    direct = os.environ.get("CUDA_INCLUDE_DIR")
    if direct and os.path.isdir(direct):
        return direct
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root:
            inc = os.path.join(root, "include")
            if os.path.isdir(inc):
                return inc
    if os.path.isdir("/usr/local/cuda/include"):
        return "/usr/local/cuda/include"
    return None


def _round_up(val, mult_of):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def _device_ptr(arr):
    """Device address (int) of a wp.array or cupy ndarray."""
    ptr = getattr(arr, "ptr", None)
    if ptr is not None:
        return int(ptr)
    cai = getattr(arr, "__cuda_array_interface__", None)
    if cai is not None:
        return int(cai["data"][0])
    data = getattr(arr, "data", None)
    if data is not None and hasattr(data, "ptr"):
        return int(data.ptr)
    raise TypeError(f"cannot obtain a device pointer from {type(arr)!r}")


def _vec3(x, y=None, z=None):
    """Coerce a scalar/sequence/wp.vec3 into a plain (x, y, z) float tuple."""
    if y is None and z is None:
        if hasattr(x, "__len__"):
            return (float(x[0]), float(x[1]), float(x[2]))
        return (float(x), float(x), float(x))
    return (float(x), float(y), float(z))


def _count_triangles(indices):
    """Triangle count from a (num_tris, 3) or flat length-3N index array."""
    shape = getattr(indices, "shape", None)
    if shape is not None and len(shape) == 2:
        if shape[1] != 3:
            raise ValueError(
                f"expected a (num_tris, 3) index array, got shape {tuple(shape)}")
        n = int(shape[0])
    else:
        n_idx = int(len(indices))
        if n_idx % 3 != 0:
            raise ValueError(
                f"flat index count must be a multiple of 3, got {n_idx}")
        n = n_idx // 3
    if n < 1:
        raise ValueError("geometry needs at least 1 triangle")
    return n


def _xform12(transform):
    """Coerce a transform into a 12-float object->world 3x4 row-major list (the
    OptixInstance::transform layout). Accepts a length-12 sequence, a (3, 4), or a
    (4, 4) matrix (bottom row dropped)."""
    arr = np.asarray(transform, dtype=np.float64)
    if arr.shape == (12,):
        flat = arr
    elif arr.shape == (3, 4):
        flat = arr.reshape(-1)
    elif arr.shape == (4, 4):
        flat = arr[:3, :].reshape(-1)
    else:
        raise ValueError(
            "transform must be 12 floats, (3, 4) or (4, 4), got shape "
            f"{tuple(arr.shape)}")
    return [float(x) for x in flat]


class PathTracer:
    """OptiX path tracer over Warp-owned geometry, presented via an OpenGL PBO.

    Typical per-frame use from a shaderbang ``Input``::

        pt.set_camera_lookat(eye, target, up=(0, 1, 0), fov_y_deg=40, aspect=w/h)
        pt.set_sphere(center, radius)
        pt.refit()          # GAS in-place update after the physics moved verts
        pt.render()         # trace + accumulate + denoise
        pt.present()        # tone-map -> PBO -> textured full-screen quad

    Construction compiles the pipeline and allocates all fixed-size buffers, so
    it must run on the target GPU with a current CUDA primary context (call
    ``wp.init()`` / select ``cuda:0`` first, as cloth.py already does).
    """

    def __init__(self, width, height, device="cuda:0", exposure=1.0,
                 upscale=1, log_level=0, denoiser="optix", interop="pbo"):
        width = int(width)
        height = int(height)
        upscale = int(upscale)
        exposure = float(exposure)
        # Validate up front: bad extents/upscale corrupt buffer sizing, and the
        # denoiser only ships a 1x (HDR) and a 2x (temporal-upscale) model.
        # Native 4K is just upscale=1 at a 4K extent -- e.g.
        # ``PathTracer(3840, 2160, upscale=1)`` -- no special path required.
        if width < 1 or height < 1:
            raise ValueError(
                f"width and height must be >= 1, got {width}x{height}")
        if upscale not in (1, 2):
            raise ValueError(
                "upscale must be 1 (single-frame HDR denoiser) or 2 (temporal "
                f"2x-upscale denoiser), got {upscale}")
        if not (exposure > 0.0):
            raise ValueError(f"exposure must be > 0, got {exposure}")
        # Denoiser backend: "optix" (the on-GPU OptiX AI denoiser, default) or
        # "oidn" (Intel Open Image Denoise, a non-temporal CPU-roundtrip fallback
        # for still frames / portability -- see _create_denoiser_oidn). OIDN
        # cannot upscale, so it is only valid at upscale=1.
        if denoiser not in ("optix", "oidn"):
            raise ValueError(
                f"denoiser must be 'optix' or 'oidn', got {denoiser!r}")
        if denoiser == "oidn" and upscale != 1:
            raise ValueError(
                "the OIDN backend is non-temporal and cannot upscale; use "
                "upscale=1 (or denoiser='optix' for the 2x temporal upscaler)")
        # Present interop (M6): how the tone-mapped frame reaches the GL texture.
        # "pbo" (default) tone-maps into a GL PBO then uploads with
        # glTexSubImage2D (portable, the only path with no raw-CUDA dependency);
        # "texture" registers the GL texture as a CUDA surface and tone-maps
        # straight into it (one device->device copy fewer per frame), raising if
        # that cannot be set up; "auto" prefers "texture" and silently falls back
        # to "pbo" if the raw-CUDA/texture interop is unavailable. Both texture
        # paths are VERIFY-ON-TARGET (buffer-only Warp interop cannot express the
        # surface write, so they drop to the cuda-python driver API).
        if interop not in ("pbo", "texture", "auto"):
            raise ValueError(
                f"interop must be 'pbo', 'texture' or 'auto', got {interop!r}")
        self._interop = interop
        self._backend = denoiser
        self.width = width
        self.height = height
        # Output extent. The temporal-upscale denoiser (M3b) produces a 2x-larger
        # image than it is fed, so the beauty/guide/flow buffers stay at the
        # render res (self.width/height) while the denoised result, tone-map
        # target, present texture and PBO use the output res (self._out_*).
        self.upscale = upscale
        self._out_width = self.width * self.upscale
        self._out_height = self.height * self.upscale
        self.exposure = exposure
        self._device = device
        self._subframe = 0
        self._has_history = False   # temporal denoiser: no valid prev frame yet

        # Lazy GPU-stack imports (kept out of module import so the dev box can
        # import this file for byte-compile / partial use).
        import cupy as cp
        import optix
        try:
            from cuda.bindings import nvrtc
        except Exception:  # noqa: BLE001 -- older cuda-python layout
            from cuda import nvrtc
        self._cp = cp
        self._optix = optix
        self._nvrtc = nvrtc

        optix.init()

        # OptiX bound to the CUDA primary context (shared with Warp/CuPy).
        self._log_level = log_level

        def _logger(level, tag, msg):
            if level <= log_level:
                print(f"[optix][{level}][{tag}] {msg}")

        ctx_options = optix.DeviceContextOptions(
            logCallbackFunction=_logger, logCallbackLevel=log_level)
        self._ctx = optix.deviceContextCreate(0, ctx_options)

        # Warp stream shared with OptiX so physics/refit/launch/denoise/tonemap
        # all order implicitly without cross-stream syncs.
        self._wp_stream = wp.get_stream(self._device)
        try:
            self._stream_ptr = int(self._wp_stream.cuda_stream)  # VERIFY-ON-TARGET
        except Exception:  # noqa: BLE001
            self._stream_ptr = 0

        self._build_pipeline()
        self._build_sbt()
        self._alloc_image_buffers()
        self._create_denoiser()
        self._init_params()

        # Scene geometry state (M7d): a list of triangle meshes, each with its own
        # GAS, assembled into a single-level IAS. Each mesh emits one SBT hitgroup
        # record (its indices/normals/prev_vertices/material/transform); the IAS is
        # params.handle. Populated by add_mesh / set_instance_transform / refit.
        self._meshes = []            # list of per-mesh dicts (see add_mesh)
        self._cloth_mesh_id = None   # set_geometry back-compat: the one deformable mesh
        self._d_instances = None     # packed OptixInstance array on device (IAS input)
        self._d_instances_nbytes = 0
        self._instances_dirty = False  # a rigid transform changed -> re-upload instances
        self._d_ias = None           # IAS output buffer
        self._d_ias_temp = None      # IAS build/update scratch
        self._ias_temp_size = 0
        self._ias_output_size = 0
        self._ias_handle = 0
        self._d_ch = None            # hitgroup SBT records (one per instance)
        self._ch_stride = 0
        self._sbt_dirty = False      # a prev_xform / material changed -> re-pack records
        self._env_data = None        # HDR env-map device buffer (optional, M4c)
        self._env_cdf = None         # env-map sin(theta)-weighted CDF (optional)

        # GL present state (created lazily on the first present, when a GL
        # context is current).
        self._gl_ready = False
        self._tex = None
        self._pbo = None
        self._pbo_reg = None
        # M6 direct-to-texture interop state (only populated when the "texture"/
        # "auto" path successfully registers the GL texture with CUDA).
        self._tex_interop_ready = False
        self._cuda_driver = None       # cuda.bindings.driver module
        self._cuda_gl_resource = None  # CUgraphicsResource for the GL texture
        self._cuda_surf = None         # CUsurfObject over the mapped texture
        self._cuda_array_handle = None # int handle of the currently-bound array
        self._surf_module = None       # CuPy RawModule (surf tone-map)
        self._surf_kernel = None       # its tonemap_surface Function

    # ------------------------------------------------------------------ #
    # Pipeline / SBT
    # ------------------------------------------------------------------ #
    def _compile_ptx(self):
        nvrtc = self._nvrtc
        optix_inc = _optix_include_dir()
        cuda_inc = _cuda_include_dir()
        if not optix_inc:
            raise RuntimeError(
                "OptiX headers not found; set OPTIX_INCLUDE_DIR (or "
                "OPTIX_PATH/OPTIX_ROOT to the SDK root)")
        if not cuda_inc:
            raise RuntimeError(
                "CUDA headers not found; set CUDA_INCLUDE_DIR (or CUDA_HOME)")

        src_path = os.path.join(os.path.dirname(__file__), "programs.cu")
        with open(src_path, "rb") as f:
            src = f.read()

        def check(result, prog=None):
            if result[0].value:
                if prog is not None:
                    res, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
                    if not res.value:
                        log = b" " * logsize
                        nvrtc.nvrtcGetProgramLog(prog, log)
                        print(log.decode(errors="replace"))
                raise RuntimeError(
                    f"NVRTC error {result[0].value} "
                    f"({nvrtc.nvrtcGetErrorString(result[0])[1]})")
            if len(result) == 1:
                return None
            if len(result) == 2:
                return result[1]
            return result[1:]

        options = [
            b"-use_fast_math",
            b"-lineinfo",
            b"-default-device",
            b"-std=c++11",
            b"-rdc", b"true",
            # ABI guard: fail the compile if the device struct Params lays out to
            # a different size than the host PARAMS_DTYPE (see the static_assert
            # in programs.cu). Silent drift here is memory corruption at launch.
            f"-DPARAMS_EXPECTED_SIZE={PARAMS_DTYPE.itemsize}".encode(),
            # Same guard for the per-instance HitGroupData: the device struct must
            # match the SBT record's data span (record itemsize - header).
            f"-DHITGROUP_DATA_EXPECTED_SIZE={self._hitgroup_data_size()}".encode(),
            f"-I{optix_inc}".encode(),
            f"-I{cuda_inc}".encode(),
        ]
        prog = check(nvrtc.nvrtcCreateProgram(src, b"programs.cu", 0, [], []))
        check(nvrtc.nvrtcCompileProgram(prog, len(options), options), prog)
        ptx_size = check(nvrtc.nvrtcGetPTXSize(prog))
        ptx = b" " * ptx_size
        check(nvrtc.nvrtcGetPTX(prog, ptx))
        return ptx

    def _build_pipeline(self):
        optix = self._optix
        ptx = self._compile_ptx()

        self._pipeline_options = optix.PipelineCompileOptions(
            usesMotionBlur=False,
            traversableGraphFlags=int(
                optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING),
            numPayloadValues=11,    # t + Ng(xyz) + Ns(xyz) + prev-pos(xyz) + material_id
            numAttributeValues=2,   # built-in triangle barycentrics
            exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
            pipelineLaunchParamsVariableName="params",
            usesPrimitiveTypeFlags=optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE,
        )
        module_options = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        module, _ = self._ctx.moduleCreate(module_options,
                                           self._pipeline_options, ptx)

        rg_desc = optix.ProgramGroupDesc()
        rg_desc.raygenModule = module
        rg_desc.raygenEntryFunctionName = "__raygen__rg"
        rg_group, _ = self._ctx.programGroupCreate([rg_desc])

        ms_desc = optix.ProgramGroupDesc()
        ms_desc.missModule = module
        ms_desc.missEntryFunctionName = "__miss__ms"
        ms_group, _ = self._ctx.programGroupCreate([ms_desc])

        # Second miss program for shadow/occlusion rays (M4b NEE): it only clears
        # the "occluded" payload, so its SBT record (miss index 1) is what
        # sceneOcclude's optixTrace targets.
        ms_shadow_desc = optix.ProgramGroupDesc()
        ms_shadow_desc.missModule = module
        ms_shadow_desc.missEntryFunctionName = "__miss__shadow"
        ms_shadow_group, _ = self._ctx.programGroupCreate([ms_shadow_desc])

        ch_desc = optix.ProgramGroupDesc()
        ch_desc.hitgroupModuleCH = module
        ch_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
        ch_group, _ = self._ctx.programGroupCreate([ch_desc])

        self._rg_group = rg_group[0]
        self._ms_group = ms_group[0]
        self._ms_shadow_group = ms_shadow_group[0]
        self._ch_group = ch_group[0]
        program_groups = [self._rg_group, self._ms_group,
                          self._ms_shadow_group, self._ch_group]

        max_trace_depth = 1  # primary rays only in M1 (no shadow/GI recursion)
        link_options = optix.PipelineLinkOptions()
        link_options.maxTraceDepth = max_trace_depth
        self._pipeline = self._ctx.pipelineCreate(
            self._pipeline_options, link_options, program_groups, "")

        stack_sizes = optix.StackSizes()
        for pg in program_groups:
            try:
                optix.util.accumulateStackSizes(pg, stack_sizes, self._pipeline)
            except TypeError:
                optix.util.accumulateStackSizes(pg, stack_sizes)
        dc_trav, dc_state, cc_size = optix.util.computeStackSizes(
            stack_sizes, max_trace_depth, 0, 0)
        self._pipeline.setStackSize(dc_trav, dc_state, cc_size, 1)
        self._module = module

    def _build_sbt(self):
        optix = self._optix
        cp = self._cp
        header_fmt = f"{optix.SBT_RECORD_HEADER_SIZE}B"
        align = optix.SBT_RECORD_ALIGNMENT

        def _record_dtype():
            itemsize = _round_up(
                np.dtype({"names": ["header"], "formats": [header_fmt],
                          "align": True}).itemsize, align)
            return np.dtype({"names": ["header"], "formats": [header_fmt],
                             "itemsize": itemsize, "align": True})

        def header_record(group):
            dt = _record_dtype()
            h = np.array([0], dtype=dt)
            optix.sbtRecordPackHeader(group, h)
            d = cp.cuda.alloc(dt.itemsize)
            d.copy_from_host(ctypes.c_void_p(h.ctypes.data), dt.itemsize)
            return d, dt.itemsize

        def header_records(groups):
            # Pack several program-group headers into one contiguous device buffer
            # so they can be addressed as base + stride * index (used for the two
            # miss records: index 0 = primary, index 1 = shadow).
            dt = _record_dtype()
            combined = np.zeros(len(groups), dtype=dt)
            for i, group in enumerate(groups):
                one = np.zeros(1, dtype=dt)
                optix.sbtRecordPackHeader(group, one)
                combined[i] = one[0]
            total = dt.itemsize * len(groups)
            d = cp.cuda.alloc(total)
            d.copy_from_host(ctypes.c_void_p(combined.ctypes.data), total)
            return d, dt.itemsize

        # Raygen + the two miss records are scene-independent and built once. The
        # hitgroup records are one-per-instance and (re)built by
        # _rebuild_hitgroup_sbt as meshes are added / their transforms advance, so
        # self._sbt stays None until the first add_mesh (render() requires a mesh).
        self._d_rg, self._rg_stride = header_record(self._rg_group)
        self._d_ms, self._ms_stride = header_records(
            [self._ms_group, self._ms_shadow_group])
        self._sbt = None

    def _hitgroup_data_dtype(self):
        # Device HitGroupData (programs.cu): three 8-byte pointers, two uints, then
        # a 12-float transform. align=True reproduces the C padding; itemsize == the
        # device sizeof(HitGroupData) (80 bytes, guarded by the static_assert).
        return np.dtype({
            "names": ["indices", "normals", "prev_vertices",
                      "material_id", "flags", "prev_xform"],
            "formats": ["u8", "u8", "u8", "u4", "u4", ("f4", (12,))],
            "align": True,
        })

    def _hitgroup_data_size(self):
        return int(self._hitgroup_data_dtype().itemsize)

    def _hitgroup_record_dtype(self):
        # A full SBT hitgroup record: the opaque header followed by HitGroupData.
        # The header is 32 bytes and the pointers need 8-byte alignment, so the data
        # lands at offset 32 (== the header size) -- byte-identical to the data-only
        # dtype above, which the device reads via optixGetSbtDataPointer(). itemsize
        # is rounded up to SBT_RECORD_ALIGNMENT.
        optix = self._optix
        header_fmt = f"{optix.SBT_RECORD_HEADER_SIZE}B"
        names = ["header", "indices", "normals", "prev_vertices",
                 "material_id", "flags", "prev_xform"]
        formats = [header_fmt, "u8", "u8", "u8", "u4", "u4", ("f4", (12,))]
        base = np.dtype({"names": names, "formats": formats, "align": True})
        itemsize = _round_up(base.itemsize, optix.SBT_RECORD_ALIGNMENT)
        return np.dtype({
            "names": names,
            "formats": [base.fields[n][0] for n in names],
            "offsets": [base.fields[n][1] for n in names],
            "itemsize": itemsize,
            "align": True,
        })

    def _rebuild_hitgroup_sbt(self):
        # Pack one hitgroup record per instance (same order as the OptixInstance
        # array, so instance i's sbtOffset=i selects record i) and (re)create the
        # ShaderBindingTable. Called when a mesh is added or when a rigid instance's
        # previous-frame transform / material advances (self._sbt_dirty).
        optix = self._optix
        cp = self._cp
        n = len(self._meshes)
        if n == 0:
            self._sbt = None
            self._sbt_dirty = False
            return
        dt = self._hitgroup_record_dtype()
        recs = np.zeros(n, dtype=dt)
        for i, m in enumerate(self._meshes):
            one = np.zeros(1, dtype=dt)
            optix.sbtRecordPackHeader(self._ch_group, one)
            recs[i] = one[0]
            recs[i]["indices"] = int(m["idx_ptr"])
            recs[i]["normals"] = int(m["nrm_ptr"])
            recs[i]["prev_vertices"] = int(m["prev_ptr"])
            recs[i]["material_id"] = int(m["material_id"])
            recs[i]["flags"] = int(m["flags"])
            recs[i]["prev_xform"] = np.asarray(m["prev_transform"],
                                               dtype=np.float32)
        total = dt.itemsize * n
        self._d_ch = cp.cuda.alloc(total)
        self._d_ch.copy_from_host(ctypes.c_void_p(recs.ctypes.data), total)
        self._ch_stride = dt.itemsize
        self._sbt = optix.ShaderBindingTable(
            raygenRecord=self._d_rg.ptr,
            missRecordBase=self._d_ms.ptr,
            missRecordStrideInBytes=self._ms_stride,
            missRecordCount=2,
            hitgroupRecordBase=self._d_ch.ptr,
            hitgroupRecordStrideInBytes=self._ch_stride,
            hitgroupRecordCount=n,
        )
        self._sbt_dirty = False

    # ------------------------------------------------------------------ #
    # Buffers / denoiser / params
    # ------------------------------------------------------------------ #
    def _alloc_image_buffers(self):
        n = self.width * self.height
        self.d_accum = wp.zeros(n, dtype=wp.vec4, device=self._device)
        self.d_output = wp.zeros(n, dtype=wp.vec4, device=self._device)
        # Denoised result is at the output extent (2x under temporal upscale).
        self.d_denoised = wp.zeros(self._out_width * self._out_height,
                                   dtype=wp.vec4, device=self._device)
        # The buffer the tone-map reads. Under the temporal-upscale model the
        # output double-buffers and this is re-pointed at the live one each frame.
        self._d_denoised_cur = self.d_denoised
        # Denoiser guide AOVs (written by the raygen program every frame).
        self.d_albedo = wp.zeros(n, dtype=wp.vec4, device=self._device)
        self.d_normal = wp.zeros(n, dtype=wp.vec4, device=self._device)
        # Motion-vector (flow) AOV: input-res 2D vector, current -> previous
        # frame, in pixels; written by the raygen program every frame and fed to
        # the temporal denoiser as guideLayer.flow.
        self.d_flow = wp.zeros(n, dtype=wp.vec2, device=self._device)
        # Offscreen LDR target (present() writes into a mapped PBO instead). Sized
        # for the *output* extent, which is 2x the render res under the upscaling
        # temporal model (see _out_width/_out_height).
        self.d_ldr = wp.zeros(self._out_width * self._out_height * 4,
                              dtype=wp.uint8, device=self._device)

    def _image2d(self, wp_arr, width=None, height=None, fmt=None,
                 pixel_stride=16):
        """Wrap a device buffer as an OptixImage2D. Defaults to a render-res
        FLOAT4 image (beauty/albedo/normal); pass width/height/fmt/pixel_stride
        for the output-res denoised image (2x) or the FLOAT2 flow AOV (8B)."""
        optix = self._optix
        oi = optix.Image2D()
        oi.data = _device_ptr(wp_arr)   # wp.array or cupy (internal guide layers)
        oi.width = self.width if width is None else width
        oi.height = self.height if height is None else height
        oi.rowStrideInBytes = oi.width * pixel_stride
        oi.pixelStrideInBytes = pixel_stride
        oi.format = optix.PIXEL_FORMAT_FLOAT4 if fmt is None else fmt
        return oi

    def _create_denoiser(self):
        if self._backend == "oidn":
            self._create_denoiser_oidn()
            return
        optix = self._optix
        dn_options = optix.DenoiserOptions()
        # Albedo + normal guides sharpen edges the beauty alone smears; the
        # raygen program fills d_albedo/d_normal every frame (see programs.cu).
        dn_options.guideAlbedo = 1
        dn_options.guideNormal = 1
        if self.upscale > 1:
            self._create_denoiser_temporal(dn_options)
        else:
            self._create_denoiser_hdr(dn_options)

    def _create_denoiser_hdr(self, dn_options):
        """Single-frame HDR denoiser (M1/M3a): no motion vectors, no upscale, so
        the output extent equals the render extent. Works against the stock
        otk-pyoptix binding (selected when ``upscale == 1``)."""
        optix = self._optix
        cp = self._cp
        self._temporal = False
        self._denoiser = self._ctx.denoiserCreate(
            optix.DENOISER_MODEL_KIND_HDR, dn_options)

        mem = self._denoiser.computeMemoryResources(self.width, self.height)
        self._d_state = cp.empty((mem.stateSizeInBytes,), dtype=cp.uint8)
        self._d_scratch = cp.empty(
            (mem.withoutOverlapScratchSizeInBytes,), dtype=cp.uint8)
        self._d_intensity = cp.empty((1,), dtype=cp.float32)
        self._denoiser.setup(
            self._stream_ptr, self.width, self.height,
            self._d_state.data.ptr, self._d_state.nbytes,
            self._d_scratch.data.ptr, self._d_scratch.nbytes)

        self._dn_input = self._image2d(self.d_output)
        self._dn_output = self._image2d(self.d_denoised)
        self._dn_layer = optix.DenoiserLayer()
        self._dn_layer.input = self._dn_input
        self._dn_layer.output = self._dn_output
        # Guide layer: albedo + view-space normal AOVs (both FLOAT4, same extent
        # as the beauty).
        self._dn_albedo_img = self._image2d(self.d_albedo)
        self._dn_normal_img = self._image2d(self.d_normal)
        self._dn_guide = optix.DenoiserGuideLayer()
        self._dn_guide.albedo = self._dn_albedo_img
        self._dn_guide.normal = self._dn_normal_img
        self._dn_params = optix.DenoiserParams()
        # denoiseAlpha moved from DenoiserParams to DenoiserOptions in OptiX 8.0;
        # the patched binding gates it out of Params for >= 8.0, so only touch it
        # where it still exists (the field defaults to 0 either way).
        if hasattr(self._dn_params, "denoiseAlpha"):
            self._dn_params.denoiseAlpha = 0
        self._dn_params.hdrIntensity = int(self._d_intensity.data.ptr)
        self._dn_params.hdrAverageColor = 0
        self._dn_params.blendFactor = 0.0
        self._d_denoised_cur = self.d_denoised

    def _create_denoiser_temporal(self, dn_options):
        """Temporal + 2x-upscale denoiser (M3b): renders at self.width x
        self.height and produces a 2x image. Recurrent state is carried across
        frames through the double-buffered internal guide layers and the previous
        denoised output, reprojected with the flow AOV. Requires the patched
        otk-pyoptix binding (see pathtracer/patches/); selected by ``upscale=2``.
        """
        optix = self._optix
        cp = self._cp
        self._temporal = True
        self._dn_has_history = False   # no valid previous output/guide on frame 0
        self._out_flip = 0
        self._denoiser = self._ctx.denoiserCreate(
            optix.DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X, dn_options)

        # computeMemoryResources / setup take the INPUT (render) extent; the model
        # knows its output is 2x. Full-frame (no tiling) -> withoutOverlap scratch.
        mem = self._denoiser.computeMemoryResources(self.width, self.height)
        self._d_state = cp.empty((mem.stateSizeInBytes,), dtype=cp.uint8)
        self._d_scratch = cp.empty(
            (mem.withoutOverlapScratchSizeInBytes,), dtype=cp.uint8)
        # The upscale family normalizes with the average colour (a 3-float device
        # buffer filled by computeAverageColor), not the scalar hdrIntensity.
        self._d_avg_color = cp.zeros((3,), dtype=cp.float32)
        self._denoiser.setup(
            self._stream_ptr, self.width, self.height,
            self._d_state.data.ptr, self._d_state.nbytes,
            self._d_scratch.data.ptr, self._d_scratch.nbytes)

        # Double-buffered denoised output (2x): d_denoised is buffer A (from
        # _alloc_image_buffers); allocate buffer B here.
        self.d_denoised2 = wp.zeros(self._out_width * self._out_height,
                                    dtype=wp.vec4, device=self._device)
        self._out_bufs = (self.d_denoised, self.d_denoised2)
        self._d_denoised_cur = self.d_denoised

        # Double-buffered internal guide layers (the model's recurrent hidden
        # state), at the OUTPUT extent, opaque pixels of the model's size. Zeroed
        # so the previous-guide read on frame 0 is well-defined.
        self._ig_pixel = int(mem.internalGuideLayerPixelSizeInBytes)
        ig_bytes = self._out_width * self._out_height * self._ig_pixel
        self._d_ig = (cp.zeros((ig_bytes,), dtype=cp.uint8),
                      cp.zeros((ig_bytes,), dtype=cp.uint8))

        # Persistent OptiX image/layer/guide objects. The 1x inputs are fixed; the
        # 2x output + internal-guide pointers are rebound per frame as the double
        # buffers swap (see _denoise_temporal).
        self._dn_input = self._image2d(self.d_output)             # beauty (1x)
        self._dn_albedo_img = self._image2d(self.d_albedo)        # 1x
        self._dn_normal_img = self._image2d(self.d_normal)        # 1x
        self._dn_flow_img = self._image2d(                        # 1x, FLOAT2
            self.d_flow, fmt=optix.PIXEL_FORMAT_FLOAT2, pixel_stride=8)
        self._dn_out_img = (
            self._image2d(self.d_denoised, self._out_width, self._out_height),
            self._image2d(self.d_denoised2, self._out_width, self._out_height),
        )
        self._dn_ig_img = (
            self._image2d(self._d_ig[0], self._out_width, self._out_height,
                          optix.PIXEL_FORMAT_INTERNAL_GUIDE_LAYER, self._ig_pixel),
            self._image2d(self._d_ig[1], self._out_width, self._out_height,
                          optix.PIXEL_FORMAT_INTERNAL_GUIDE_LAYER, self._ig_pixel),
        )
        self._dn_layer = optix.DenoiserLayer()
        self._dn_layer.input = self._dn_input
        self._dn_guide = optix.DenoiserGuideLayer()
        self._dn_guide.albedo = self._dn_albedo_img
        self._dn_guide.normal = self._dn_normal_img
        self._dn_guide.flow = self._dn_flow_img
        self._dn_params = optix.DenoiserParams()
        if hasattr(self._dn_params, "denoiseAlpha"):   # OptiX < 8 only (see above)
            self._dn_params.denoiseAlpha = 0
        self._dn_params.hdrIntensity = 0
        self._dn_params.hdrAverageColor = int(self._d_avg_color.data.ptr)
        self._dn_params.blendFactor = 0.0

    def _init_params(self):
        self._h_params = np.zeros(1, PARAMS_DTYPE)
        self._d_params = self._cp.cuda.alloc(PARAMS_DTYPE.itemsize)
        p = self._h_params[0]
        p["accum"] = int(self.d_accum.ptr)
        p["output"] = int(self.d_output.ptr)
        p["albedo"] = int(self.d_albedo.ptr)
        p["normal"] = int(self.d_normal.ptr)
        p["flow"] = int(self.d_flow.ptr)
        # params.handle (the scene IAS) is wired by add_mesh; 0 until the first
        # mesh is added (render() requires one).
        p["handle"] = 0
        p["width"] = self.width
        p["height"] = self.height
        p["max_depth"] = 4          # bounces (set_path_depth to override)
        p["rr_depth"] = 2           # Russian roulette starts here
        p["exposure"] = self.exposure
        # Sensible scene defaults; callers override via the setters below.
        p["cam_eye"] = (0.0, 1.0, 5.0)
        p["cam_u"] = (1.0, 0.0, 0.0)
        p["cam_v"] = (0.0, 1.0, 0.0)
        p["cam_w"] = (0.0, 0.0, -1.0)
        # Previous-frame camera starts equal to the current one (zero motion).
        p["prev_cam_eye"] = tuple(p["cam_eye"])
        p["prev_cam_u"] = tuple(p["cam_u"])
        p["prev_cam_v"] = tuple(p["cam_v"])
        p["prev_cam_w"] = tuple(p["cam_w"])
        p["sky_top"] = (0.35, 0.55, 0.9)
        p["sky_bottom"] = (0.9, 0.9, 0.95)
        # Material table (M7a): the scene's materials live in a device buffer
        # indexed by material_id (see add_material / _materials_to_device). The
        # three default slots reproduce the M6 look; consumers add more slots via
        # add_material(). self._cloth_material is the slot the set_geometry /
        # set_cloth_* back-compat shims assign to the deformable cloth mesh (M7d:
        # a mesh carries its material_id in its SBT record, not in Params); the
        # sphere/ground slots are referenced by the analytic primitives below (M7c).
        self._materials_host = []
        self._d_materials = None
        self._cloth_material = self.add_material(
            (0.2, 0.45, 0.85), roughness=0.6, metallic=0.0,
            base_color_back=(0.85, 0.6, 0.2))
        self._sphere_material = self.add_material(
            (0.75, 0.2, 0.2), roughness=0.1, metallic=0.0)
        self._ground_material = self.add_material(
            (0.6, 0.6, 0.6), roughness=0.9, metallic=0.0)
        # Analytic primitive tables (M7c): spheres and planes live in device
        # buffers looped over on the device, each carrying its own material_id
        # (a sphere also carries its previous-frame center for rigid motion
        # vectors). One default sphere + one ground plane reproduce the M6 scene;
        # consumers add more via add_sphere / add_plane. The tables upload lazily
        # from render() when dirty (they mutate every frame, unlike the material/
        # light tables), so no redundant per-frame transfer. The set_sphere /
        # set_ground live setters delegate to these ids (removed with the shims
        # in M7e).
        self._spheres_host = []
        self._d_spheres = None
        self._spheres_dirty = False
        self._planes_host = []
        self._d_planes = None
        self._planes_dirty = False
        self._sphere_prim = self.add_sphere(
            (0.0, 1.5, 0.0), 0.5, self._sphere_material)
        self._ground_prim = self.add_plane(
            (0.0, 1.0, 0.0), 0.0, self._ground_material)
        # Light table (M7b): analytic delta lights live in a device buffer
        # (see add_light / _lights_to_device). One default directional "sun"
        # reproduces the M4b default; set_light updates it, consumers add more
        # (directional or point) via add_light().
        self._lights_host = []
        self._d_lights = None
        self._sun_light = self.add_light(
            "directional", direction=(0.4, 1.0, 0.3), color=(1.0, 1.0, 1.0))
        # Environment map (optional; wired by set_environment). 0 pointers +
        # env_enabled=0 => the analytic gradient sky above is used on a miss.
        p["env_data"] = 0
        p["env_cdf"] = 0
        p["env_width"] = 0
        p["env_height"] = 0
        p["env_enabled"] = 0
        p["env_total_sum"] = 0.0
        p["env_intensity"] = 1.0
        p["env_rotation"] = 0.0

    # ------------------------------------------------------------------ #
    # Scene setters
    # ------------------------------------------------------------------ #
    def set_camera(self, eye, u, v, w):
        p = self._h_params[0]
        p["cam_eye"] = _vec3(eye)
        p["cam_u"] = _vec3(u)
        p["cam_v"] = _vec3(v)
        p["cam_w"] = _vec3(w)

    def set_camera_lookat(self, eye, target, up=(0.0, 1.0, 0.0),
                          fov_y_deg=40.0, aspect=None):
        """Set the camera basis from a look-at + vertical FOV (matches the
        gluPerspective/gluLookAt convention cloth.py uses)."""
        if aspect is None:
            aspect = self.width / float(self.height)
        eye = _vec3(eye)
        target = _vec3(target)
        up = _vec3(up)
        if not (aspect > 0.0):
            raise ValueError(f"set_camera_lookat: aspect must be > 0, got {aspect}")
        if not (0.0 < fov_y_deg < 180.0):
            raise ValueError(
                f"set_camera_lookat: fov_y_deg must be in (0, 180), got "
                f"{fov_y_deg}")
        w = _sub3(target, eye)          # not normalized: |W| is the focal length
        wlen = _len3(w)
        if not (wlen > 0.0):
            raise ValueError(
                "set_camera_lookat: eye and target coincide (zero view "
                f"direction): eye={eye}, target={target}")
        u_raw = _cross3(w, up)
        if not (_len3(u_raw) > 0.0):
            raise ValueError(
                "set_camera_lookat: up is parallel to the view direction; pick "
                f"a different up vector (view={w}, up={up})")
        u = _norm3(u_raw)
        v = _norm3(_cross3(u, w))
        vlen = wlen * math.tan(0.5 * math.radians(fov_y_deg))
        ulen = vlen * aspect
        self.set_camera(eye, _mul3(u, ulen), _mul3(v, vlen), w)

    def set_sphere(self, center, radius, albedo=None):
        # Back-compat shim (M7c): drive the default analytic sphere. No radius
        # guard -- this is a per-frame live setter (cloth.py pushes the
        # interactive radius every frame), and the analytic intersector uses the
        # radius only as r*r, so a stray negative value renders as |r| rather
        # than corrupting anything -- not worth crashing a running session over.
        self.update_sphere(self._sphere_prim, center=center, radius=radius)
        if albedo is not None:
            self.update_material(self._sphere_material, base_color=albedo)

    def set_ground(self, y, albedo=None):
        # Back-compat shim (M7c): the ground is the default analytic plane, a
        # horizontal plane y = const (normal (0,1,0), offset == y).
        self.update_plane(self._ground_prim, offset=float(y))
        if albedo is not None:
            self.update_material(self._ground_material, base_color=albedo)

    # ------------------------------------------------------------------ #
    # Analytic primitive tables (M7c). Spheres and planes live in device buffers
    # looped over on the device (sceneIntersect / sceneOcclude), each carrying
    # its own material_id. A sphere also stores its previous-frame center so the
    # rigid motion-vector reprojection works for an animated ball. Unlike the
    # material/light tables (which change at setup / on rare tweaks and upload
    # eagerly), these mutate every frame, so add_/update_ only mark the table
    # dirty and render() uploads once per frame before the launch -- keeping the
    # tiny transfer off the meaningful hot path while guaranteeing the device
    # buffer matches host state at every launch. A sphere is 8 float32 slots
    # (center.xyz, radius, center_prev.xyz, material_id); a plane 5 slots
    # (normal.xyz, offset, material_id). The trailing uint material_id is written
    # into its float slot bit-for-bit (np.uint32 view), matching the device
    # struct's trailing ``unsigned int`` -- the Light.type packing trick.
    # ------------------------------------------------------------------ #
    def add_sphere(self, center, radius, material_id):
        """Append an analytic sphere and return its integer id. ``center`` is a
        world point, ``radius`` a scalar, ``material_id`` a slot from
        add_material(). center_prev is seeded to center (zero initial motion)."""
        c = _vec3(center)
        self._spheres_host.append(
            [c[0], c[1], c[2], float(radius),
             c[0], c[1], c[2], int(material_id)])
        self._spheres_dirty = True
        return len(self._spheres_host) - 1

    def update_sphere(self, sphere_id, center=None, radius=None,
                      material_id=None):
        """Modify fields of an existing sphere in place; unspecified fields keep
        their current value. ``center_prev`` is untouched here -- it advances in
        _snapshot_prev after each render, so motion vectors reproject against the
        position actually rendered last frame."""
        e = self._spheres_host[int(sphere_id)]
        if center is not None:
            e[0], e[1], e[2] = _vec3(center)
        if radius is not None:
            e[3] = float(radius)
        if material_id is not None:
            e[7] = int(material_id)
        self._spheres_dirty = True

    def _spheres_to_device(self):
        """Upload the current sphere table and repoint Params.spheres."""
        self._spheres_dirty = False
        if not self._spheres_host:
            self._h_params[0]["num_spheres"] = 0
            return
        n = len(self._spheres_host)
        arr = np.asarray(self._spheres_host, dtype=np.float32)   # (n, 8)
        arr.view(np.uint32)[:, 7] = arr[:, 7].astype(np.uint32)  # material_id bits
        flat = np.ascontiguousarray(arr.reshape(-1))
        self._d_spheres = wp.array(flat, dtype=wp.float32, device=self._device)
        p = self._h_params[0]
        p["spheres"] = int(self._d_spheres.ptr)
        p["num_spheres"] = n

    def add_plane(self, normal, offset, material_id):
        """Append an analytic infinite plane { p : dot(normal, p) == offset } and
        return its integer id. ``normal`` is normalized here; ``material_id`` is a
        slot from add_material(). Planes are static (no motion vectors)."""
        nrm = _norm3(_vec3(normal))
        self._planes_host.append(
            [nrm[0], nrm[1], nrm[2], float(offset), int(material_id)])
        self._planes_dirty = True
        return len(self._planes_host) - 1

    def update_plane(self, plane_id, normal=None, offset=None, material_id=None):
        """Modify fields of an existing plane in place; unspecified fields keep
        their current value. ``normal`` is re-normalized."""
        e = self._planes_host[int(plane_id)]
        if normal is not None:
            e[0], e[1], e[2] = _norm3(_vec3(normal))
        if offset is not None:
            e[3] = float(offset)
        if material_id is not None:
            e[4] = int(material_id)
        self._planes_dirty = True

    def _planes_to_device(self):
        """Upload the current plane table and repoint Params.planes."""
        self._planes_dirty = False
        if not self._planes_host:
            self._h_params[0]["num_planes"] = 0
            return
        n = len(self._planes_host)
        arr = np.asarray(self._planes_host, dtype=np.float32)   # (n, 5)
        arr.view(np.uint32)[:, 4] = arr[:, 4].astype(np.uint32)  # material_id bits
        flat = np.ascontiguousarray(arr.reshape(-1))
        self._d_planes = wp.array(flat, dtype=wp.float32, device=self._device)
        p = self._h_params[0]
        p["planes"] = int(self._d_planes.ptr)
        p["num_planes"] = n

    # ------------------------------------------------------------------ #
    # Light table (M7b). Analytic delta lights (directional + point) live in a
    # device buffer; add_light appends one, update_light edits one, and both
    # re-upload the (tiny) table. Both types are next-event-estimated only, so
    # the device gathers them with misWeight = 1 and no /pdf (see directLight).
    # ------------------------------------------------------------------ #
    def add_light(self, kind, direction=None, position=None,
                  color=(1.0, 1.0, 1.0), radius=0.0):
        """Append a light to the device table and return its integer id.

        ``kind`` is ``"directional"`` (needs ``direction`` -- a vector *toward*
        the light, normalized here) or ``"point"`` (needs ``position`` -- a world
        point; radiance attenuates as 1/dist^2 on the device). ``color`` is the
        radiance (directional) or pre-attenuation intensity (point).
        """
        k = str(kind).lower()
        if k in ("directional", "dir", "distant", "sun"):
            t = 0
            if direction is None:
                raise ValueError("add_light(directional) needs a direction")
            vec = _norm3(_vec3(direction))
        elif k == "point":
            t = 1
            if position is None:
                raise ValueError("add_light(point) needs a position")
            vec = _vec3(position)
        else:
            raise ValueError(f"add_light: unknown kind {kind!r} "
                             "(expected 'directional' or 'point')")
        col = _vec3(color)
        self._lights_host.append(
            (t, vec[0], vec[1], vec[2], col[0], col[1], col[2], float(radius)))
        self._lights_to_device()
        return len(self._lights_host) - 1

    def update_light(self, light_id, direction=None, position=None,
                     color=None, radius=None):
        """Modify fields of an existing light in place; unspecified fields keep
        their current value. ``direction`` (directional) is re-normalized;
        ``position`` (point) is stored as-is -- do not mix the two for one id."""
        e = list(self._lights_host[int(light_id)])
        if direction is not None:
            e[1], e[2], e[3] = _norm3(_vec3(direction))
        if position is not None:
            e[1], e[2], e[3] = _vec3(position)
        if color is not None:
            e[4], e[5], e[6] = _vec3(color)
        if radius is not None:
            e[7] = float(radius)
        self._lights_host[int(light_id)] = tuple(e)
        self._lights_to_device()

    def _lights_to_device(self):
        """Upload the current light table and repoint Params.lights. Each entry
        is 8 float32 slots; slot 0 holds the uint ``type`` bit-for-bit (matching
        the device ``Light`` struct's leading ``unsigned int``), slots 1..7 the
        float direction/position, color and radius."""
        if not self._lights_host:
            return
        n = len(self._lights_host)
        arr = np.zeros((n, 8), dtype=np.float32)
        types = np.empty(n, dtype=np.uint32)
        for i, e in enumerate(self._lights_host):
            types[i] = int(e[0])
            arr[i, 1:] = e[1:]
        arr.view(np.uint32)[:, 0] = types      # reinterpret slot 0 as the uint type
        flat = np.ascontiguousarray(arr.reshape(-1))
        self._d_lights = wp.array(flat, dtype=wp.float32, device=self._device)
        p = self._h_params[0]
        p["lights"] = int(self._d_lights.ptr)
        p["num_lights"] = n

    def set_light(self, direction, color=(1.0, 1.0, 1.0)):
        """Update the default directional 'sun' light (back-compat shim; prefer
        add_light / update_light)."""
        self.update_light(self._sun_light, direction=direction, color=color)

    def set_sky(self, top, bottom):
        p = self._h_params[0]
        p["sky_top"] = _vec3(top)
        p["sky_bottom"] = _vec3(bottom)

    def set_environment(self, image, intensity=1.0, rotation=0.0):
        """Bind an HDR lat-long (equirectangular) environment map for image-based
        lighting with importance sampling + MIS (M4c).

        ``image`` is an ``(H, W, 3)`` or ``(H, W, 4)`` float image -- a numpy
        array, a ``wp.array``, or any object exposing
        ``__cuda_array_interface__``. It is read once (a host copy drives the CDF
        build) and uploaded to two device buffers this instance then owns; the
        render hot path touches only those device buffers. Row 0 is the +Y (top)
        pole, matching the ``theta = acos(dir.y)`` convention in programs.cu.
        ``intensity`` scales the env radiance; ``rotation`` offsets the azimuth
        in uv units (``1.0`` == a full turn). A weightless (all-black) map is
        rejected -- it falls back to the analytic sky. Call
        ``clear_environment()`` to unbind and return to the analytic sky.
        """
        cp = self._cp
        # Bring the image to a contiguous host float32 (H, W, C) for the CDF
        # build (a one-time setup cost, never on the render hot path).
        if hasattr(image, "numpy"):            # wp.array
            host = image.numpy()
        elif isinstance(image, cp.ndarray):    # cupy device array
            host = cp.asnumpy(image)
        else:
            host = np.asarray(image)
        host = np.ascontiguousarray(host, dtype=np.float32)
        if host.ndim != 3 or host.shape[2] not in (3, 4):
            raise ValueError(
                "set_environment expects an (H, W, 3) or (H, W, 4) image, got "
                f"shape {tuple(host.shape)}")
        h, w = int(host.shape[0]), int(host.shape[1])
        rgb = host[:, :, :3]

        # Per-texel weight = Rec.709 luminance * sin(theta) at the row center,
        # the lat-long solid-angle row weighting the GLSL-PathTracer reference
        # omits (baking sin(theta) here concentrates samples away from the
        # oversampled poles). This sin(thetaCenter) is the texel-selection
        # weight; the device pdf keeps a separate 1/sin(thetaActual) Jacobian
        # term (they do not cancel -- see envTexelPdf in programs.cu).
        lum = (0.212671 * rgb[:, :, 0] + 0.715160 * rgb[:, :, 1]
               + 0.072169 * rgb[:, :, 2]).astype(np.float64)
        v = (np.arange(h, dtype=np.float64) + 0.5) / h
        sin_theta = np.sin(v * math.pi)
        weights = lum * sin_theta[:, None]
        # Single flat, unnormalized, row-major running-sum CDF over all texels
        # (index y*W + x), matching envmap.glsl BinarySearch / EnvironmentMap.cpp.
        cdf = np.cumsum(weights.ravel(order="C"))
        total = float(cdf[-1]) if cdf.size else 0.0
        if not (total > 0.0):
            self.clear_environment()
            return

        # float4 device buffer (RGB + 1), row-major to match env_cdf and the
        # device index y*W + x.
        rgba = np.ones((h * w, 4), dtype=np.float32)
        rgba[:, :3] = rgb.reshape(h * w, 3)
        self._env_data = wp.array(rgba, dtype=wp.vec4, device=self._device)
        self._env_cdf = wp.array(cdf.astype(np.float32), dtype=wp.float32,
                                 device=self._device)

        p = self._h_params[0]
        p["env_data"] = int(self._env_data.ptr)
        p["env_cdf"] = int(self._env_cdf.ptr)
        p["env_width"] = w
        p["env_height"] = h
        p["env_enabled"] = 1
        p["env_total_sum"] = total
        p["env_intensity"] = float(intensity)
        p["env_rotation"] = float(rotation)

    def clear_environment(self):
        """Unbind the environment map: the analytic gradient sky (``set_sky``) is
        used on a miss again. The device buffers are kept alive but disabled."""
        self._h_params[0]["env_enabled"] = 0

    # ------------------------------------------------------------------ #
    # Material table (M7a). Materials live in a device buffer indexed by
    # material_id; add_material appends a slot, update_material edits one, and
    # both re-upload the (tiny) table. This is never on the render hot path in
    # the meaningful sense -- the table is a handful of 32-byte entries and
    # changes at setup / on rare live tweaks, not per vertex or per pixel.
    # ------------------------------------------------------------------ #
    def add_material(self, base_color, roughness=0.5, metallic=0.0,
                     base_color_back=None):
        """Append a material to the device table and return its integer id.

        ``base_color`` is the front-face albedo; ``base_color_back`` the
        back-face albedo for two-sided surfaces (defaults to ``base_color``).
        ``roughness`` is used directly as the GGX alpha (Disney linear
        roughness). The returned id is passed to geometry as its ``material_id``.
        """
        fc = _vec3(base_color)
        bc = _vec3(base_color_back) if base_color_back is not None else fc
        self._materials_host.append(
            (fc[0], fc[1], fc[2], bc[0], bc[1], bc[2],
             float(roughness), float(metallic)))
        self._materials_to_device()
        return len(self._materials_host) - 1

    def update_material(self, material_id, base_color=None, roughness=None,
                        metallic=None, base_color_back=None):
        """Modify fields of an existing material in place; unspecified fields
        keep their current value."""
        e = list(self._materials_host[int(material_id)])
        if base_color is not None:
            e[0], e[1], e[2] = _vec3(base_color)
        if base_color_back is not None:
            e[3], e[4], e[5] = _vec3(base_color_back)
        if roughness is not None:
            e[6] = float(roughness)
        if metallic is not None:
            e[7] = float(metallic)
        self._materials_host[int(material_id)] = tuple(e)
        self._materials_to_device()

    def _materials_to_device(self):
        """Upload the current material table and repoint Params.materials."""
        if not self._materials_host:
            return
        arr = np.asarray(self._materials_host, dtype=np.float32).reshape(-1)
        self._d_materials = wp.array(arr, dtype=wp.float32, device=self._device)
        p = self._h_params[0]
        p["materials"] = int(self._d_materials.ptr)
        p["num_materials"] = len(self._materials_host)

    def set_cloth_albedo(self, front, back=None):
        self.update_material(self._cloth_material,
                             base_color=front,
                             base_color_back=back if back is not None else front)

    def set_cloth_material(self, roughness, metallic=0.0):
        """Cloth Disney material. ``roughness`` is used directly as GGX alpha."""
        self.update_material(self._cloth_material,
                             roughness=roughness, metallic=metallic)

    def set_sphere_material(self, roughness, metallic=0.0):
        self.update_material(self._sphere_material,
                             roughness=roughness, metallic=metallic)

    def set_ground_material(self, roughness, metallic=0.0):
        self.update_material(self._ground_material,
                             roughness=roughness, metallic=metallic)

    def set_path_depth(self, max_depth, rr_depth=None):
        """Set the maximum number of bounces and where Russian roulette begins.

        ``max_depth`` counts bounces after the primary ray (max_depth=0 is a
        direct-lighting-only image). ``rr_depth`` defaults to min(max_depth, 2)."""
        p = self._h_params[0]
        md = max(0, int(max_depth))
        p["max_depth"] = md
        p["rr_depth"] = min(md, 2) if rr_depth is None else max(0, int(rr_depth))

    # ------------------------------------------------------------------ #
    # Geometry / acceleration structure (M7d: multi-mesh IAS)
    # ------------------------------------------------------------------ #
    def _check_instancing_support(self):
        """Fail loudly (not with a cryptic AttributeError mid-build) if the pinned
        otk-pyoptix lacks the single-level-instancing surface M7d needs. This is
        the plan's on-target binding gate, enforced at the first add_mesh."""
        optix = self._optix
        required = ["Instance", "getDeviceRepresentation",
                    "BuildInputInstanceArray", "INSTANCE_FLAG_NONE",
                    "TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING"]
        missing = [name for name in required if not hasattr(optix, name)]
        if missing:
            raise RuntimeError(
                "otk-pyoptix is missing the instancing bindings M7d needs: "
                f"{', '.join(missing)}. Rebuild the pinned otk-pyoptix with the "
                "instance-array build input exposed.")

    def add_mesh(self, vertices, indices, normals=None, material_id=0,
                 deformable=False, transform=None):
        """Register a triangle mesh as one instance in the scene IAS; return its
        ``mesh_id``.

        ``vertices`` is a ``wp.array(dtype=wp.vec3)``; ``indices`` a
        ``wp.array(dtype=wp.int32)`` of shape ``(num_tris, 3)`` (or a flat
        length-``3*num_tris`` array). ``normals``, if given, is a per-vertex
        ``wp.array(dtype=wp.vec3)`` (one per vertex) used for smooth shading;
        omitted => geometric normal. ``material_id`` indexes the material table
        (see add_material). ``transform`` is the object->world placement (12 floats
        3x4 row-major, a (3,4) or a (4,4) matrix; default identity). ``deformable``
        meshes have their vertices rewritten in place each frame (refit() rebuilds
        the GAS + IAS, motion vectors come from a per-vertex previous snapshot);
        rigid meshes are placed/animated via set_instance_transform (motion vectors
        come from the previous frame's transform). The mesh keeps references to the
        vertex / index / normal arrays so their device memory stays alive for as long
        as the mesh does: the per-instance SBT record stores raw pointers into them
        and closesthit dereferences those every frame."""
        self._check_instancing_support()

        num_vertices = int(len(vertices))
        if num_vertices < 3:
            raise ValueError(f"add_mesh needs at least 3 vertices, got "
                             f"{num_vertices}")
        num_triangles = _count_triangles(indices)
        if normals is not None and int(len(normals)) != num_vertices:
            raise ValueError(
                f"normals length ({int(len(normals))}) must match vertex count "
                f"({num_vertices})")

        # Previous-frame vertex snapshot for deformable motion vectors. The
        # closesthit reads it via the SBT record's prev_vertices pointer; rigid
        # meshes leave it null and reconstruct the previous position from the
        # instance transform instead.
        if deformable:
            prev_vertices = wp.zeros(num_vertices, dtype=wp.vec3,
                                     device=self._device)
            wp.copy(prev_vertices, vertices)     # frame 0: prev == current
            prev_ptr = int(prev_vertices.ptr)
        else:
            prev_vertices = None
            prev_ptr = 0

        xform = (_xform12(transform) if transform is not None
                 else [1.0, 0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0])
        mesh = {
            "vertices": vertices,
            "vtx_ptr": int(_device_ptr(vertices)),
            "num_vertices": num_vertices,
            "indices": indices,   # retained: the SBT record points into it (below)
            "idx_ptr": int(_device_ptr(indices)),
            "num_triangles": num_triangles,
            "normals": normals,
            "nrm_ptr": int(_device_ptr(normals)) if normals is not None else 0,
            "deformable": bool(deformable),
            "prev_vertices": prev_vertices,
            "prev_ptr": prev_ptr,
            "material_id": int(material_id),
            "flags": 0,
            "transform": list(xform),
            "prev_transform": list(xform),   # frame 0: prev == current
            # per-mesh GAS state
            "d_gas": None, "d_gas_temp": None,
            "gas_temp_size": 0, "gas_output_size": 0, "gas_handle": 0,
        }
        self._meshes.append(mesh)
        mesh_id = len(self._meshes) - 1

        self._build_mesh_gas(mesh, update=False)
        self._instances_dirty = True
        self._build_ias(update=False)          # (re)build the IAS with the new instance
        self._rebuild_hitgroup_sbt()           # one record per instance
        return mesh_id

    def set_geometry(self, vertices, indices, normals=None):
        """Back-compat shim (M7d): the cloth is a single deformable mesh in the
        IAS. First call registers it; subsequent calls re-point its buffers."""
        if self._cloth_mesh_id is None:
            self._cloth_mesh_id = self.add_mesh(
                vertices, indices, normals=normals,
                material_id=self._cloth_material, deformable=True)
        else:
            self.update_mesh(self._cloth_mesh_id, vertices=vertices,
                             normals=normals)

    def update_mesh(self, mesh_id, vertices=None, normals=None):
        """Re-point a mesh's vertex / normal buffers (rare: deformable meshes are
        normally updated in place, so refit() suffices). A changed vertex buffer
        forces a GAS rebuild; a changed normal buffer updates the SBT record."""
        m = self._meshes[int(mesh_id)]
        if vertices is not None:
            if int(len(vertices)) != m["num_vertices"]:
                raise ValueError(
                    "update_mesh cannot change the vertex count "
                    f"({int(len(vertices))} != {m['num_vertices']})")
            m["vertices"] = vertices
            m["vtx_ptr"] = int(_device_ptr(vertices))
            if m["deformable"]:
                wp.copy(m["prev_vertices"], vertices)
            self._build_mesh_gas(m, update=False)
            self._instances_dirty = True       # GAS handle may have changed
            self._build_ias(update=False)
        if normals is not None:
            if int(len(normals)) != m["num_vertices"]:
                raise ValueError(
                    f"normals length ({int(len(normals))}) must match vertex "
                    f"count ({m['num_vertices']})")
            m["normals"] = normals
            m["nrm_ptr"] = int(_device_ptr(normals))
            self._sbt_dirty = True

    def set_instance_transform(self, mesh_id, transform):
        """Set a rigid mesh's object->world transform (12 floats 3x4 row-major, a
        (3,4) or a (4,4) matrix). Takes effect on the next refit()/render()."""
        m = self._meshes[int(mesh_id)]
        if m["deformable"]:
            raise ValueError(
                "set_instance_transform on a deformable mesh; deformable meshes "
                "move via their vertex buffer + refit(), not a transform")
        m["transform"] = _xform12(transform)
        self._instances_dirty = True

    def _triangle_input(self, mesh):
        optix = self._optix
        tri = optix.BuildInputTriangleArray()
        tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
        tri.numVertices = mesh["num_vertices"]
        tri.vertexBuffers = [mesh["vtx_ptr"]]
        tri.vertexStrideInBytes = 12       # wp.vec3 is tightly packed (3x f32)
        tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
        tri.numIndexTriplets = mesh["num_triangles"]
        tri.indexBuffer = mesh["idx_ptr"]
        tri.indexStrideInBytes = 12
        tri.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
        tri.numSbtRecords = 1
        return tri

    def _build_flags(self, deformable):
        # RANDOM_VERTEX_ACCESS: closesthit reads triangle vertices for the
        # geometric normal. ALLOW_UPDATE only for deformable meshes (refit).
        optix = self._optix
        flags = int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
                    | optix.BUILD_FLAG_PREFER_FAST_TRACE)
        if deformable:
            flags |= int(optix.BUILD_FLAG_ALLOW_UPDATE)
        return flags

    def _build_mesh_gas(self, mesh, update):
        optix = self._optix
        cp = self._cp
        tri = self._triangle_input(mesh)
        opts = optix.AccelBuildOptions(
            buildFlags=self._build_flags(mesh["deformable"]),
            operation=(optix.BUILD_OPERATION_UPDATE if update
                       else optix.BUILD_OPERATION_BUILD),
        )
        if not update:
            sizes = self._ctx.accelComputeMemoryUsage([opts], [tri])
            mesh["gas_temp_size"] = max(sizes.tempSizeInBytes,
                                        sizes.tempUpdateSizeInBytes)
            mesh["gas_output_size"] = sizes.outputSizeInBytes
            mesh["d_gas_temp"] = cp.cuda.alloc(mesh["gas_temp_size"])
            mesh["d_gas"] = cp.cuda.alloc(mesh["gas_output_size"])
        handle = self._ctx.accelBuild(
            self._stream_ptr, [opts], [tri],
            mesh["d_gas_temp"].ptr, mesh["gas_temp_size"],
            mesh["d_gas"].ptr, mesh["gas_output_size"], [])
        if int(handle) != int(mesh["gas_handle"]):
            mesh["gas_handle"] = int(handle)
            self._instances_dirty = True       # IAS instance data references it

    def _upload_instances(self):
        # Pack the OptixInstance array (one per mesh) and upload it as the IAS
        # build input. instance i gets instanceId=i and sbtOffset=i so its
        # hitgroup record (index i) is selected on a hit.
        optix = self._optix
        cp = self._cp
        instances = [
            optix.Instance(
                transform=[float(x) for x in m["transform"]],
                instanceId=i,
                sbtOffset=i,
                visibilityMask=255,
                flags=int(optix.INSTANCE_FLAG_NONE),
                traversableHandle=int(m["gas_handle"]),
            )
            for i, m in enumerate(self._meshes)
        ]
        data = optix.getDeviceRepresentation(instances)   # packed OptixInstance bytes
        buf = np.frombuffer(data, dtype=np.uint8)
        if self._d_instances is None or self._d_instances_nbytes != buf.nbytes:
            self._d_instances = cp.cuda.alloc(int(buf.nbytes))
            self._d_instances_nbytes = int(buf.nbytes)
        self._d_instances.copy_from_host(
            ctypes.c_void_p(buf.ctypes.data), int(buf.nbytes))
        self._instances_dirty = False

    def _build_ias(self, update):
        optix = self._optix
        cp = self._cp
        n = len(self._meshes)
        if n == 0:
            return
        # A full (re)build always re-reads freshly packed instances; a refit only
        # needs a re-upload if a rigid transform / GAS handle changed since.
        if (not update) or self._instances_dirty:
            self._upload_instances()
        inst = optix.BuildInputInstanceArray(int(self._d_instances.ptr), 0, n)
        opts = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_ALLOW_UPDATE
                           | optix.BUILD_FLAG_PREFER_FAST_TRACE),
            operation=(optix.BUILD_OPERATION_UPDATE if update
                       else optix.BUILD_OPERATION_BUILD),
        )
        if not update:
            sizes = self._ctx.accelComputeMemoryUsage([opts], [inst])
            self._ias_temp_size = max(sizes.tempSizeInBytes,
                                      sizes.tempUpdateSizeInBytes)
            self._ias_output_size = sizes.outputSizeInBytes
            self._d_ias_temp = cp.cuda.alloc(self._ias_temp_size)
            self._d_ias = cp.cuda.alloc(self._ias_output_size)
        self._ias_handle = self._ctx.accelBuild(
            self._stream_ptr, [opts], [inst],
            self._d_ias_temp.ptr, self._ias_temp_size,
            self._d_ias.ptr, self._ias_output_size, [])
        self._h_params[0]["handle"] = int(self._ias_handle)

    def refit(self):
        """Per-frame acceleration-structure update: refit each deformable mesh's
        GAS (vertices moved in place) then refit the IAS (child AABBs / rigid
        transforms changed). Cheap relative to a full rebuild; topology/counts must
        be unchanged."""
        if not self._meshes:
            raise RuntimeError("refit() called before add_mesh()/set_geometry()")
        for m in self._meshes:
            if m["deformable"]:
                self._build_mesh_gas(m, update=True)
        self._build_ias(update=True)

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #
    def render(self, reset=True, spp=1):
        """Trace + accumulate ``spp`` sample(s), then denoise into ``d_denoised``.

        ``reset=True`` restarts accumulation (use it every frame while the scene
        animates); ``reset=False`` keeps accumulating (progressive refinement
        while paused).

        ``spp`` traces that many samples into the HDR accumulator *before* the
        single denoise -- a per-frame burst. It is the main quality lever for
        animating frames: with ``reset=True`` there is no cross-frame HDR
        accumulation (each frame starts fresh), so ``spp`` is the only way to
        hand the denoiser a cleaner input than a lone 1-spp trace, cutting the
        temporal ghosting/lag it would otherwise have to invent. The denoiser,
        the ``subframe`` history and the motion-vector snapshot advance exactly
        once per call regardless of ``spp`` (the burst is a single displayed
        frame at one camera/geometry pose). ``spp=1`` reproduces the old
        one-sample-per-call behaviour. Call ``present()`` or ``download_ldr()``
        afterwards.
        """
        if not self._meshes:
            raise RuntimeError(
                "render() called before add_mesh()/set_geometry()")
        spp = max(1, int(spp))
        if reset:
            self._subframe = 0

        p = self._h_params[0]
        p["exposure"] = self.exposure
        # Pick up a rigid transform change (set_instance_transform) even when the
        # consumer did not call refit() this frame, and re-pack the hitgroup SBT
        # when a previous-frame transform / material advanced (_snapshot_prev,
        # update_mesh). refit() clears _instances_dirty, so the IAS update here is
        # a no-op when refit() already ran this frame.
        if self._instances_dirty:
            self._build_ias(update=True)
        if self._sbt_dirty:
            self._rebuild_hitgroup_sbt()
        p["handle"] = int(self._ias_handle)
        # Upload the analytic primitive tables if they changed since the last
        # frame (add_/update_sphere/plane and _snapshot_prev only mark them
        # dirty), so the device buffers match host state at this launch.
        if self._spheres_dirty:
            self._spheres_to_device()
        if self._planes_dirty:
            self._planes_to_device()
        # Accumulate the burst: each launch reads the running subframe (so
        # programs.cu folds it into the mean accum/(subframe+1)) and advances it.
        for _ in range(spp):
            p["subframe"] = self._subframe
            self._d_params.copy_from_host(
                ctypes.c_void_p(self._h_params.ctypes.data),
                PARAMS_DTYPE.itemsize)
            self._optix.launch(
                self._pipeline, self._stream_ptr, self._d_params.ptr,
                PARAMS_DTYPE.itemsize, self._sbt, self.width, self.height, 1)
            self._subframe += 1

        self._denoise()
        self._snapshot_prev()

    def _snapshot_prev(self):
        """Freeze this frame's camera, spheres and mesh state as the ``previous``
        state the next frame's motion vectors reproject against.

        Runs after the launch has consumed the *current* prev_* fields. The
        deformable vertex copy is issued on the render stream (the Warp default
        stream, shared with the physics), so it is ordered strictly after this
        frame's trace and strictly before next frame's physics overwrites the
        vertices in place -- no extra sync, no race on the shared buffer.
        """
        p = self._h_params[0]
        p["prev_cam_eye"] = tuple(p["cam_eye"])
        p["prev_cam_u"] = tuple(p["cam_u"])
        p["prev_cam_v"] = tuple(p["cam_v"])
        p["prev_cam_w"] = tuple(p["cam_w"])
        # Advance each sphere's center_prev (slots 4..6) to the center actually
        # rendered this frame (slots 0..2). Only re-dirty when it moved, so a
        # static scene does not re-upload the table every refinement frame.
        for e in self._spheres_host:
            if e[4:7] != e[0:3]:
                e[4], e[5], e[6] = e[0], e[1], e[2]
                self._spheres_dirty = True
        # Advance mesh motion-vector state: deformable meshes snapshot their
        # vertices; rigid meshes snapshot the transform actually rendered (which
        # re-packs their hitgroup record's prev_xform before the next launch).
        for m in self._meshes:
            if m["deformable"]:
                if m["prev_vertices"] is not None and m["vertices"] is not None:
                    wp.copy(m["prev_vertices"], m["vertices"])
            elif m["prev_transform"] != m["transform"]:
                m["prev_transform"] = list(m["transform"])
                self._sbt_dirty = True
        self._has_history = True

    def _denoise(self):
        if self._backend == "oidn":
            self._denoise_oidn()
        elif self._temporal:
            self._denoise_temporal()
        else:
            self._denoise_hdr()

    def _denoise_hdr(self):
        self._denoiser.computeIntensity(
            self._stream_ptr, self._dn_input, self._d_intensity.data.ptr,
            self._d_scratch.data.ptr, self._d_scratch.nbytes)
        self._denoiser.invokeTiled(
            self._stream_ptr, self._dn_params,
            self._d_state.data.ptr, self._d_state.nbytes,
            self._dn_guide, [self._dn_layer],
            self._d_scratch.data.ptr, self._d_scratch.nbytes,
            0, self.width, self.height)

    def _denoise_temporal(self):
        # Average colour drives the HDR normalization for the upscale family (it
        # does not use the scalar intensity computeIntensity produces).
        self._denoiser.computeAverageColor(
            self._stream_ptr, self._dn_input, self._d_avg_color.data.ptr,
            self._d_scratch.data.ptr, self._d_scratch.nbytes)

        cur = self._out_flip
        prev = 1 - cur
        # This frame writes buffer `cur`; the previous frame's denoised output and
        # internal guide (buffer `prev`) feed the temporal reprojection. On frame 0
        # the `prev` buffers are zeroed and ignored (temporalModeUsePreviousLayers=0).
        self._dn_layer.output = self._dn_out_img[cur]
        self._dn_layer.previousOutput = self._dn_out_img[prev]
        self._dn_guide.outputInternalGuideLayer = self._dn_ig_img[cur]
        self._dn_guide.previousOutputInternalGuideLayer = self._dn_ig_img[prev]
        self._dn_params.temporalModeUsePreviousLayers = (
            1 if self._dn_has_history else 0)

        # Non-tiled full-frame invoke: a single layer (numLayers = 1). The setup /
        # scratch were sized for the whole frame, so no tiling/overlap is needed.
        self._denoiser.invoke(
            self._stream_ptr, self._dn_params,
            self._d_state.data.ptr, self._d_state.nbytes,
            self._dn_guide, self._dn_layer, 1,
            0, 0,
            self._d_scratch.data.ptr, self._d_scratch.nbytes)

        self._d_denoised_cur = self._out_bufs[cur]
        self._out_flip = prev
        self._dn_has_history = True

    # ------------------------------------------------------------------ #
    # OIDN backend (M5b) -- non-temporal fallback for still frames /
    # portability. This is the ONE place the pipeline takes a CPU roundtrip; it
    # is deliberately off the live hot path (selected only with
    # ``denoiser="oidn"``, which forbids upscale). Intel OIDN denoises the beauty
    # with the albedo + normal guide AOVs on the host, then the result is
    # uploaded back for the usual device tone-map. The whole backend is soft: the
    # ``oidn`` package is imported lazily and only when this backend is chosen.
    # The binding targeted is the ctypes ``oidn`` wrapper whose functions mirror
    # the C API with the ``oidn``/``OIDN_`` prefixes stripped and numpy arrays as
    # shared buffers (oidn.NewDevice / SetSharedFilterImage / ExecuteFilter).
    # VERIFY-ON-TARGET: exact binding surface + performance.
    # ------------------------------------------------------------------ #
    def _create_denoiser_oidn(self):
        try:
            import oidn
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "denoiser='oidn' needs the Intel Open Image Denoise Python "
                "binding; install it (e.g. `pip install oidn`) or use "
                "denoiser='optix'") from e
        self._oidn = oidn
        self._temporal = False
        h, w, n = self.height, self.width, self.width * self.height

        # Persistent host staging bound once as OIDN "shared" images: OIDN reads
        # /writes these numpy buffers in place every ExecuteFilter, so they must
        # outlive the filter and be refilled (not reallocated) per frame. RGB
        # only (the AOVs are FLOAT4 on device; we drop the padding alpha).
        self._oidn_color = np.zeros((h, w, 3), dtype=np.float32)   # HDR beauty
        self._oidn_albedo = np.zeros((h, w, 3), dtype=np.float32)  # [0,1]
        self._oidn_normal = np.zeros((h, w, 3), dtype=np.float32)  # view-space
        self._oidn_out = np.zeros((h, w, 3), dtype=np.float32)     # denoised
        # Reused (n, 4) host buffer for the upload (alpha pinned to 1).
        self._oidn_rgba = np.ones((n, 4), dtype=np.float32)

        self._oidn_device = oidn.NewDevice()
        oidn.CommitDevice(self._oidn_device)
        f = oidn.NewFilter(self._oidn_device, "RT")
        fmt = oidn.FORMAT_FLOAT3
        oidn.SetSharedFilterImage(f, "color", self._oidn_color, fmt, w, h)
        oidn.SetSharedFilterImage(f, "albedo", self._oidn_albedo, fmt, w, h)
        oidn.SetSharedFilterImage(f, "normal", self._oidn_normal, fmt, w, h)
        oidn.SetSharedFilterImage(f, "output", self._oidn_out, fmt, w, h)
        # The beauty is linear HDR (tone-mapped later on device), so the filter
        # must run in HDR mode. The bool-setter name differs across binding
        # versions (OIDN 2.x SetFilterBool vs 1.x SetFilter1b), so probe both.
        self._oidn_set_bool(f, "hdr", True)
        oidn.CommitFilter(f)
        self._oidn_filter = f
        self._d_denoised_cur = self.d_denoised   # valid target before frame 0

    def _oidn_set_bool(self, filt, name, value):
        for fn in ("SetFilterBool", "SetFilter1b", "SetFilter1i"):
            setter = getattr(self._oidn, fn, None)
            if setter is None:
                continue
            try:
                setter(filt, name, bool(value) if fn != "SetFilter1i"
                       else int(value))
                return True
            except Exception:  # noqa: BLE001 -- try the next spelling
                continue
        return False

    def _denoise_oidn(self):
        # Make sure the OptiX launch's writes to the beauty/guide AOVs are
        # visible on the host before the download (this backend is synchronous by
        # design -- it is the still-frame fallback, not the live hot path).
        wp.synchronize_stream(self._wp_stream)
        h, w, n = self.height, self.width, self.width * self.height
        # Download device FLOAT4 AOVs -> refill the bound FLOAT3 host buffers in
        # place (drop the padding alpha). ``[:] =`` copies into the pre-bound
        # arrays OIDN holds pointers to.
        self._oidn_color[:] = self.d_output.numpy().reshape(h, w, 4)[:, :, :3]
        self._oidn_albedo[:] = self.d_albedo.numpy().reshape(h, w, 4)[:, :, :3]
        self._oidn_normal[:] = self.d_normal.numpy().reshape(h, w, 4)[:, :, :3]

        self._oidn.ExecuteFilter(self._oidn_filter)

        # Upload the denoised RGB back to a device FLOAT4 buffer for the tone-map.
        self._oidn_rgba[:, :3] = self._oidn_out.reshape(n, 3)
        self._d_denoised_cur = wp.array(self._oidn_rgba, dtype=wp.vec4,
                                        device=self._device)

    def _release_oidn(self):
        oidn = getattr(self, "_oidn", None)
        if oidn is None:
            return
        f = getattr(self, "_oidn_filter", None)
        if f is not None:
            try:
                oidn.ReleaseFilter(f)
            except Exception:  # noqa: BLE001
                pass
            self._oidn_filter = None
        dev = getattr(self, "_oidn_device", None)
        if dev is not None:
            try:
                oidn.ReleaseDevice(dev)
            except Exception:  # noqa: BLE001
                pass
            self._oidn_device = None

    def _tonemap_into(self, out_u8):
        # The denoised buffer is at the output extent (2x under temporal upscale);
        # _d_denoised_cur tracks the live one of the double buffer. One thread per
        # output pixel.
        wp.launch(_tonemap_kernel, dim=self._out_width * self._out_height,
                  inputs=[self._d_denoised_cur, out_u8, self.exposure],
                  device=self._device, stream=self._wp_stream)

    def download_ldr(self):
        """Tone-map into ``d_ldr`` and return an (H, W, 4) uint8 numpy image
        (row 0 = bottom, matching the ray-gen convention) at the *output*
        extent. For offscreen use."""
        self._tonemap_into(self.d_ldr)
        wp.synchronize_stream(self._wp_stream)
        flat = self.d_ldr.numpy()
        return flat.reshape(self._out_height, self._out_width, 4)

    # ------------------------------------------------------------------ #
    # OpenGL present (fixed-function; robust across the target's GL version)
    # ------------------------------------------------------------------ #
    def init_gl(self):
        """Create the present texture (and, unless the M6 texture interop takes
        over, the PBO). Must be called with a GL context current (i.e. from
        inside the shaderbang render loop)."""
        from OpenGL.GL import (
            glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
            GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
            GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
            GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
        )
        # The present texture is shared by both paths (the PBO uploads into it;
        # the surface path renders directly into it).
        self._tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     self._out_width, self._out_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        # M6: try the direct-to-texture surface path first when requested. On
        # success no PBO is needed; on failure "auto" falls through to the PBO
        # path below and "texture" re-raises.
        if self._interop in ("texture", "auto"):
            try:
                self._init_texture_interop()
                self._tex_interop_ready = True
            except Exception as e:  # noqa: BLE001
                if self._interop == "texture":
                    raise
                self._tex_interop_ready = False
                if self._log_level:
                    print(f"[pathtracer] texture interop unavailable, falling "
                          f"back to the PBO present path: {e}")

        if not self._tex_interop_ready:
            self._init_pbo()
        self._gl_ready = True

    def _init_pbo(self):
        """Allocate the present PBO and register it with Warp so the tone-map
        kernel writes into it with no CPU copy (the portable present path)."""
        from OpenGL.GL import (
            glGenBuffers, glBindBuffer, glBufferData,
            GL_PIXEL_UNPACK_BUFFER, GL_STREAM_DRAW,
        )
        self._pbo = int(glGenBuffers(1))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     self._out_width * self._out_height * 4,
                     None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._pbo_reg = wp.RegisteredGLBuffer(
            self._pbo, self._device,
            wp.RegisteredGLBuffer.WRITE_DISCARD)  # VERIFY-ON-TARGET (ctor args)

    # ------------------------------------------------------------------ #
    # M6 direct-to-texture present (raw CUDA: cuGraphicsGLRegisterImage +
    # a CUDA surface object, written by the CuPy-compiled surf2Dwrite kernel).
    # Warp's GL interop is buffer-only, so this leg drops to the cuda-python
    # driver API. Everything here is VERIFY-ON-TARGET.
    # ------------------------------------------------------------------ #
    def _init_texture_interop(self):
        from OpenGL.GL import GL_TEXTURE_2D
        try:
            from cuda.bindings import driver as cuda
        except Exception:  # noqa: BLE001 -- older cuda-python layout
            from cuda import cuda
        self._cuda_driver = cuda

        # Compile the surface-write tone-map with CuPy's NVRTC (RawModule forces
        # the compile now, so a compile failure is caught here and lets "auto"
        # fall back before any frame is presented).
        module = self._cp.RawModule(code=_SURF_TONEMAP_SRC,
                                    options=("--use_fast_math",))
        self._surf_kernel = module.get_function("tonemap_surface")
        self._surf_module = module

        # Register the GL texture as a CUDA graphics resource with surface
        # load/store so a surface object can be bound over its mapped array.
        flags = cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST
        self._cuda_gl_resource = self._cuda_check(
            cuda.cuGraphicsGLRegisterImage(int(self._tex), int(GL_TEXTURE_2D),
                                           flags))

    def _cuda_check(self, result):
        """Unpack a cuda-python ``(CUresult, *payload)`` tuple, raising on error
        and returning the payload (mirrors the NVRTC ``check`` helper above)."""
        if not isinstance(result, (tuple, list)):
            result = (result,)
        err = result[0]
        cuda = self._cuda_driver
        if int(err) != int(cuda.CUresult.CUDA_SUCCESS):
            msg = str(err)
            try:
                _, name = cuda.cuGetErrorName(err)
                _, desc = cuda.cuGetErrorString(err)
                msg = f"{name.decode(errors='replace')}: {desc.decode(errors='replace')}"
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"CUDA driver error: {msg}")
        rest = result[1:]
        if not rest:
            return None
        return rest[0] if len(rest) == 1 else rest

    def _bind_surface(self, array):
        """(Re)create the CUDA surface object over the mapped texture array.

        The mapped array handle is stable for a fixed-size texture, so the
        surface is built once and reused every frame; it is only rebuilt if the
        driver hands back a different array (the docs permit it to change across
        maps). A rebuild syncs first so no in-flight tone-map kernel is still
        referencing the surface being destroyed."""
        cuda = self._cuda_driver
        if self._cuda_surf is not None:
            wp.synchronize_stream(self._wp_stream)
            try:
                cuda.cuSurfObjectDestroy(self._cuda_surf)
            except Exception:  # noqa: BLE001
                pass
            self._cuda_surf = None
        desc = cuda.CUDA_RESOURCE_DESC()
        desc.resType = cuda.CUresourcetype.CU_RESOURCE_TYPE_ARRAY
        desc.res.array.hArray = array
        desc.flags = 0
        self._cuda_surf = self._cuda_check(cuda.cuSurfObjectCreate(desc))

    def present(self):
        """Tone-map the denoised frame into the GL texture and draw it
        full-screen. Uses the M6 surface path when it was set up, else the PBO
        upload. Must be called with a GL context current."""
        if not self._gl_ready:
            self.init_gl()
        if self._tex_interop_ready:
            self._present_texture()
        else:
            self._present_pbo()
        self._draw_fullscreen_quad()

    def _present_texture(self):
        """M6: map the GL texture, tone-map straight into its surface, unmap. No
        PBO, no glTexSubImage2D copy."""
        cuda = self._cuda_driver
        res = self._cuda_gl_resource
        stream = self._stream_ptr
        # Map on the render stream (implicit GL->CUDA sync) and fetch the array.
        self._cuda_check(cuda.cuGraphicsMapResources(1, res, stream))
        try:
            array = self._cuda_check(
                cuda.cuGraphicsSubResourceGetMappedArray(res, 0, 0))
            handle = self._array_handle(array)
            if self._cuda_surf is None or handle != self._cuda_array_handle:
                self._bind_surface(array)
                self._cuda_array_handle = handle

            bx, by = 16, 16
            gx = (self._out_width + bx - 1) // bx
            gy = (self._out_height + by - 1) // by
            args = (np.uint64(_device_ptr(self._d_denoised_cur)),
                    np.uint64(int(self._cuda_surf)),
                    np.float32(self.exposure),
                    np.int32(self._out_width),
                    np.int32(self._out_height))
            # Launch on the shared render stream so the write orders after the
            # denoise and before the unmap hands the texture back to GL.
            with self._cp.cuda.ExternalStream(stream):
                self._surf_kernel((gx, gy), (bx, by), args)
        finally:
            # Unmap on the same stream: stream-ordered after the kernel, so GL
            # sees a complete texture. Always runs, even if the launch raised, so
            # the resource is never left mapped.
            self._cuda_check(cuda.cuGraphicsUnmapResources(1, res, stream))

    # Sentinel handle used when a mapped array is not int-convertible: it
    # compares equal to itself across frames, so the surface is built once and
    # reused rather than rebuilt every frame (id() would differ per wrapper).
    _REUSE_SURFACE = "reuse-surface"

    @classmethod
    def _array_handle(cls, array):
        """Best-effort int handle for a cuda-python CUarray, used to detect a
        remap returning a *different* array. Falls back to a constant sentinel if
        the object is not int-convertible, so the surface is simply reused."""
        try:
            return int(array)
        except Exception:  # noqa: BLE001
            return cls._REUSE_SURFACE

    def _present_pbo(self):
        """Portable present: tone-map into the mapped PBO, then upload it into the
        texture with glTexSubImage2D."""
        from OpenGL.GL import (
            glBindBuffer, glBindTexture, glTexSubImage2D,
            GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_PIXEL_UNPACK_BUFFER,
        )
        # The presented image is at the output extent (2x under temporal upscale).
        n = self._out_width * self._out_height
        mapped = self._pbo_reg.map(dtype=wp.uint8, shape=(n * 4,))
        self._tonemap_into(mapped)
        self._pbo_reg.unmap()   # ensures the CUDA write is complete for GL

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self._out_width, self._out_height,
                        GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def _draw_fullscreen_quad(self):
        from OpenGL.GL import (
            glMatrixMode, glLoadIdentity, glPushMatrix, glPopMatrix,
            glOrtho, glViewport, glDisable, glEnable, glBindTexture, glColor3f,
            glBegin, glEnd, glVertex2f, glTexCoord2f, glDepthMask,
            GL_PROJECTION, GL_MODELVIEW, GL_TEXTURE_2D, GL_DEPTH_TEST,
            GL_LIGHTING, GL_QUADS, GL_TRUE,
        )
        # Fill the full framebuffer (== output extent by construction).
        glViewport(0, 0, self._out_width, self._out_height)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glColor3f(1.0, 1.0, 1.0)
        # texcoord (0,0) at bottom-left matches row 0 = bottom in the ray-gen.
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 1.0)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #
    def close(self):
        """Release the OptiX pipeline, denoiser, device context and every device
        / GL resource this instance owns.

        Idempotent, and safe to call after a partially-failed construction:
        every step is guarded, so one bad handle cannot leak the rest. If
        ``init_gl()`` was used, call ``close()`` with the same GL context current
        so the texture/PBO deletes land on the right context. Also usable as a
        context manager (``with PathTracer(...) as pt: ...``).
        """
        # OIDN backend (M5b), if it was the selected denoiser.
        if getattr(self, "_backend", None) == "oidn":
            self._release_oidn()

        # OptiX objects, children before the context that owns them. otk-pyoptix
        # exposes a .destroy() method on each handle.        # VERIFY-ON-TARGET
        for attr in ("_pipeline", "_rg_group", "_ms_group", "_ms_shadow_group",
                     "_ch_group", "_module", "_denoiser", "_ctx"):
            self._destroy_optix(attr)

        # GL present resources (only if init_gl ran; needs a current context).
        if getattr(self, "_gl_ready", False):
            self._release_gl()
        else:
            # Partially-constructed GL state: still unregister any CUDA graphics
            # resource that got created (these are CUDA calls, not GL, so they
            # run without a current GL context). Idempotent with _release_gl.
            self._release_texture_interop()

        # Drop references to device buffers / OptiX host structs so CuPy and Warp
        # free the backing memory (these are plain allocations with no explicit
        # destroy; refcount/GC reclaims them once unreferenced).
        for attr in (
            "_pbo_reg", "_sbt", "_surf_kernel", "_surf_module", "_cuda_driver",
            "_dn_input", "_dn_output", "_dn_layer", "_dn_guide", "_dn_params",
            "_dn_albedo_img", "_dn_normal_img", "_dn_flow_img",
            "_dn_out_img", "_dn_ig_img", "_out_bufs", "_d_ig",
            "_oidn_color", "_oidn_albedo", "_oidn_normal", "_oidn_out",
            "_oidn_rgba",
            "d_accum", "d_output", "d_denoised", "d_denoised2", "d_albedo",
            "d_normal", "d_flow", "d_ldr", "_d_denoised_cur",
            "_env_data", "_env_cdf",
            "_d_params", "_d_rg", "_d_ms", "_d_ch",
            "_d_state", "_d_scratch", "_d_intensity", "_d_avg_color",
            "_d_ias", "_d_ias_temp", "_d_instances",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)
        # Per-mesh GAS buffers (device allocations held in the mesh dicts) go with
        # the list; the previous-vertex snapshots (wp.arrays) drop with them too.
        self._meshes = []
        self._ias_handle = 0

    def _destroy_optix(self, attr):
        obj = getattr(self, attr, None)
        if obj is None:
            return
        try:
            destroy = getattr(obj, "destroy", None)
            if callable(destroy):
                destroy()
        except Exception:  # noqa: BLE001 -- best-effort teardown
            pass
        setattr(self, attr, None)

    def _release_texture_interop(self):
        """Destroy the CUDA surface object and unregister the GL texture (M6),
        before the texture itself is deleted. Best-effort and idempotent."""
        cuda = getattr(self, "_cuda_driver", None)
        if cuda is None:
            self._tex_interop_ready = False
            return
        # A tone-map kernel may still be in flight against the surface; drain the
        # render stream before tearing the surface/resource down.
        try:
            wp.synchronize_stream(self._wp_stream)
        except Exception:  # noqa: BLE001
            pass
        if self._cuda_surf is not None:
            try:
                cuda.cuSurfObjectDestroy(self._cuda_surf)
            except Exception:  # noqa: BLE001
                pass
            self._cuda_surf = None
        if self._cuda_gl_resource is not None:
            try:
                cuda.cuGraphicsUnregisterResource(self._cuda_gl_resource)
            except Exception:  # noqa: BLE001
                pass
            self._cuda_gl_resource = None
        self._cuda_array_handle = None
        self._tex_interop_ready = False

    def _release_gl(self):
        try:
            from OpenGL.GL import glDeleteTextures, glDeleteBuffers
        except Exception:  # noqa: BLE001
            self._release_texture_interop()
            self._gl_ready = False
            return
        # Unregister the CUDA graphics resource / surface (M6) before the texture
        # is deleted, and drop the Warp<->GL PBO registration before the PBO is
        # deleted, so CUDA lets go of each object first.         # VERIFY-ON-TARGET
        self._release_texture_interop()
        self._pbo_reg = None
        if self._tex:
            try:
                glDeleteTextures(1, [int(self._tex)])
            except Exception:  # noqa: BLE001
                pass
        if self._pbo:
            try:
                glDeleteBuffers(1, [int(self._pbo)])
            except Exception:  # noqa: BLE001
                pass
        self._tex = None
        self._pbo = None
        self._gl_ready = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


# --------------------------------------------------------------------------- #
# Tiny host-side vec3 helpers (kept dependency-free).
# --------------------------------------------------------------------------- #
def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul3(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _len3(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _norm3(a):
    a = _vec3(a)
    n = _len3(a)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)
