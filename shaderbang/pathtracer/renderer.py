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
# field. align=True reproduces C struct padding: the eight 8-byte members come
# first (offsets 0/8/.../56), then the 4-byte scalars, then the tightly packed
# float3s (each an ('f4', (3,)) subarray == float3; float2 == ('f4', (2,))).
# itemsize rounds up to a multiple of 8.
# --------------------------------------------------------------------------- #
_PARAMS_NAMES = [
    "accum", "output", "albedo", "normal",
    "prev_vertices", "tri_indices", "flow", "handle", "cloth_normals",
    "env_data", "env_cdf", "materials",
    "width", "height", "subframe", "max_depth", "rr_depth", "exposure",
    "env_width", "env_height", "env_enabled",
    "num_materials", "cloth_material", "sphere_material", "ground_material",
    "cam_eye", "cam_u", "cam_v", "cam_w",
    "prev_cam_eye", "prev_cam_u", "prev_cam_v", "prev_cam_w",
    "light_dir", "light_color", "sky_top", "sky_bottom",
    "sphere_center", "sphere_radius",
    "sphere_center_prev", "ground_y",
    "env_total_sum", "env_intensity", "env_rotation",
]
_PARAMS_FORMATS = [
    "u8", "u8", "u8", "u8",
    "u8", "u8", "u8", "u8", "u8",
    "u8", "u8", "u8",
    "u4", "u4", "u4", "u4", "u4", "f4",
    "u4", "u4", "u4",
    "u4", "u4", "u4", "u4",
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), "f4",
    ("f4", (3,)), "f4",
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

        # GAS state (populated by set_geometry / refit).
        self._gas_handle = 0
        self._d_gas = None
        self._d_temp = None
        self._d_temp_size = 0
        self._gas_output_size = 0
        self._vertices = None
        self._vtx_ptr = 0
        self._num_vertices = 0
        self._idx_ptr = 0
        self._num_triangles = 0
        self._normals = None         # per-vertex smooth normals (optional)
        self._prev_vertices = None   # previous-frame vertex snapshot (motion vec)
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
                optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
            numPayloadValues=10,    # t + Ng(xyz) + Ns(xyz) + prev-pos(xyz)
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

        self._d_rg, rg_stride = header_record(self._rg_group)
        self._d_ms, ms_stride = header_records(
            [self._ms_group, self._ms_shadow_group])
        self._d_ch, ch_stride = header_record(self._ch_group)
        self._sbt = optix.ShaderBindingTable(
            raygenRecord=self._d_rg.ptr,
            missRecordBase=self._d_ms.ptr,
            missRecordStrideInBytes=ms_stride,
            missRecordCount=2,
            hitgroupRecordBase=self._d_ch.ptr,
            hitgroupRecordStrideInBytes=ch_stride,
            hitgroupRecordCount=1,
        )

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
        # prev_vertices / tri_indices / cloth_normals are wired by set_geometry
        # (0 until then; a 0 cloth_normals makes the device fall back to the
        # geometric normal for shading).
        p["prev_vertices"] = 0
        p["tri_indices"] = 0
        p["cloth_normals"] = 0
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
        p["light_dir"] = _norm3((0.4, 1.0, 0.3))
        p["light_color"] = (1.0, 1.0, 1.0)
        p["sky_top"] = (0.35, 0.55, 0.9)
        p["sky_bottom"] = (0.9, 0.9, 0.95)
        p["sphere_center"] = (0.0, 1.5, 0.0)
        p["sphere_radius"] = 0.5
        p["sphere_center_prev"] = tuple(p["sphere_center"])
        p["ground_y"] = 0.0
        # Material table (M7a): the scene's materials live in a device buffer
        # indexed by material_id (see add_material / _materials_to_device). The
        # three default slots reproduce the M6 look; consumers add more slots via
        # add_material(). cloth_material / sphere_material / ground_material map
        # the still-hardcoded geometry (cloth GAS, analytic sphere/ground) to
        # their slots -- these three ids are transient and go away as the
        # geometry itself gains a material_id (M7c/M7d).
        self._materials_host = []
        self._d_materials = None
        self._cloth_material = self.add_material(
            (0.2, 0.45, 0.85), roughness=0.6, metallic=0.0,
            base_color_back=(0.85, 0.6, 0.2))
        self._sphere_material = self.add_material(
            (0.75, 0.2, 0.2), roughness=0.1, metallic=0.0)
        self._ground_material = self.add_material(
            (0.6, 0.6, 0.6), roughness=0.9, metallic=0.0)
        p["cloth_material"] = self._cloth_material
        p["sphere_material"] = self._sphere_material
        p["ground_material"] = self._ground_material
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
        # No radius guard: this is a per-frame live setter (cloth.py pushes the
        # interactive radius every frame), and the analytic intersector uses the
        # radius only as r*r, so a stray negative value renders as |r| rather
        # than corrupting anything -- not worth crashing a running session over.
        p = self._h_params[0]
        p["sphere_center"] = _vec3(center)
        p["sphere_radius"] = float(radius)
        if albedo is not None:
            self.update_material(int(p["sphere_material"]), base_color=albedo)

    def set_ground(self, y, albedo=None):
        p = self._h_params[0]
        p["ground_y"] = float(y)
        if albedo is not None:
            self.update_material(int(p["ground_material"]), base_color=albedo)

    def set_light(self, direction, color=(1.0, 1.0, 1.0)):
        p = self._h_params[0]
        p["light_dir"] = _norm3(_vec3(direction))
        p["light_color"] = _vec3(color)

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
        self.update_material(int(self._h_params[0]["cloth_material"]),
                             base_color=front,
                             base_color_back=back if back is not None else front)

    def set_cloth_material(self, roughness, metallic=0.0):
        """Cloth Disney material. ``roughness`` is used directly as GGX alpha."""
        self.update_material(int(self._h_params[0]["cloth_material"]),
                             roughness=roughness, metallic=metallic)

    def set_sphere_material(self, roughness, metallic=0.0):
        self.update_material(int(self._h_params[0]["sphere_material"]),
                             roughness=roughness, metallic=metallic)

    def set_ground_material(self, roughness, metallic=0.0):
        self.update_material(int(self._h_params[0]["ground_material"]),
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
    # Geometry / acceleration structure
    # ------------------------------------------------------------------ #
    def set_geometry(self, vertices, indices, normals=None):
        """Register triangle geometry and build the GAS.

        ``vertices`` is a ``wp.array(dtype=wp.vec3)``; ``indices`` a
        ``wp.array(dtype=wp.int32)`` of shape ``(num_tris, 3)`` (or a flat
        length-``3*num_tris`` array). ``normals``, if given, is a
        ``wp.array(dtype=wp.vec3)`` of per-vertex smooth normals (one per vertex)
        used for shading; when omitted the geometric (flat) normal is used. All
        stay owned by the caller (the physics keeps writing ``vertices`` and
        ``normals`` in place); we only keep their pointers.
        """
        self._vertices = vertices          # kept for the per-frame prev snapshot
        self._vtx_ptr = _device_ptr(vertices)
        self._num_vertices = int(len(vertices))
        if self._num_vertices < 3:
            raise ValueError(
                f"set_geometry needs at least 3 vertices, got "
                f"{self._num_vertices}")
        self._idx_ptr = _device_ptr(indices)
        shape = getattr(indices, "shape", None)
        if shape is not None and len(shape) == 2:
            if shape[1] != 3:
                raise ValueError(
                    "set_geometry expects a (num_tris, 3) index array, got "
                    f"shape {tuple(shape)}")
            self._num_triangles = int(shape[0])
        else:
            n_idx = int(len(indices))
            if n_idx % 3 != 0:
                raise ValueError(
                    "set_geometry expects a flat index count that is a "
                    f"multiple of 3, got {n_idx}")
            self._num_triangles = n_idx // 3
        if self._num_triangles < 1:
            raise ValueError("set_geometry needs at least 1 triangle")

        # Smooth shading normals (optional). The pointer is wired once and read
        # by the closesthit program every frame; the caller updates the contents
        # in place. 0 => device falls back to the geometric normal. A count
        # mismatch would make the closesthit read past the buffer (the device
        # indexes cloth_normals by vertex id), so reject it here.
        self._normals = normals
        if normals is not None and int(len(normals)) != self._num_vertices:
            raise ValueError(
                f"normals length ({int(len(normals))}) must match vertex count "
                f"({self._num_vertices})")
        self._h_params[0]["cloth_normals"] = (
            int(_device_ptr(normals)) if normals is not None else 0)

        # Previous-frame vertex snapshot for motion vectors. The closesthit
        # program reads params.prev_vertices[tri.xyz]; the index buffer itself
        # doubles as uint3* (int32 triplets are bit-identical to uint3), so
        # tri_indices just aliases the caller's index buffer.
        self._prev_vertices = wp.zeros(self._num_vertices, dtype=wp.vec3,
                                       device=self._device)
        wp.copy(self._prev_vertices, self._vertices)   # frame 0: prev == current
        self._h_params[0]["prev_vertices"] = int(self._prev_vertices.ptr)
        self._h_params[0]["tri_indices"] = int(self._idx_ptr)

        self._build_gas(update=False)

    def _triangle_input(self):
        optix = self._optix
        tri = optix.BuildInputTriangleArray()
        tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
        tri.numVertices = self._num_vertices
        tri.vertexBuffers = [self._vtx_ptr]
        tri.vertexStrideInBytes = 12       # wp.vec3 is tightly packed (3x f32)
        tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
        tri.numIndexTriplets = self._num_triangles
        tri.indexBuffer = self._idx_ptr
        tri.indexStrideInBytes = 12
        tri.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
        tri.numSbtRecords = 1
        return tri

    def _build_flags(self):
        optix = self._optix
        return int(optix.BUILD_FLAG_ALLOW_UPDATE
                   | optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
                   | optix.BUILD_FLAG_PREFER_FAST_TRACE)

    def _build_gas(self, update):
        optix = self._optix
        cp = self._cp
        tri = self._triangle_input()
        opts = optix.AccelBuildOptions(
            buildFlags=self._build_flags(),
            operation=(optix.BUILD_OPERATION_UPDATE if update
                       else optix.BUILD_OPERATION_BUILD),
        )
        if not update:
            sizes = self._ctx.accelComputeMemoryUsage([opts], [tri])
            self._d_temp_size = max(sizes.tempSizeInBytes,
                                    sizes.tempUpdateSizeInBytes)
            self._gas_output_size = sizes.outputSizeInBytes
            self._d_temp = cp.cuda.alloc(self._d_temp_size)
            self._d_gas = cp.cuda.alloc(self._gas_output_size)
        self._gas_handle = self._ctx.accelBuild(
            self._stream_ptr, [opts], [tri],
            self._d_temp.ptr, self._d_temp_size,
            self._d_gas.ptr, self._gas_output_size, [])
        self._h_params[0]["handle"] = int(self._gas_handle)

    def refit(self):
        """In-place GAS update after the physics moved the vertices. Cheap
        relative to a full rebuild; topology/counts must be unchanged."""
        if self._d_gas is None:
            raise RuntimeError("refit() called before set_geometry()")
        self._build_gas(update=True)

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
        if self._gas_handle == 0:
            raise RuntimeError("render() called before set_geometry()")
        spp = max(1, int(spp))
        if reset:
            self._subframe = 0

        p = self._h_params[0]
        p["exposure"] = self.exposure
        p["handle"] = int(self._gas_handle)
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
        """Freeze this frame's camera, sphere and cloth vertices as the
        ``previous`` state the next frame's motion vectors reproject against.

        Runs after the launch has consumed the *current* prev_* fields. The
        vertex copy is issued on the render stream (the Warp default stream,
        shared with the physics), so it is ordered strictly after this frame's
        trace and strictly before next frame's physics overwrites the vertices
        in place -- no extra sync, no race on the shared buffer.
        """
        p = self._h_params[0]
        p["prev_cam_eye"] = tuple(p["cam_eye"])
        p["prev_cam_u"] = tuple(p["cam_u"])
        p["prev_cam_v"] = tuple(p["cam_v"])
        p["prev_cam_w"] = tuple(p["cam_w"])
        p["sphere_center_prev"] = tuple(p["sphere_center"])
        if self._prev_vertices is not None and self._vertices is not None:
            wp.copy(self._prev_vertices, self._vertices)
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
            "_env_data", "_env_cdf", "_prev_vertices",
            "_d_params", "_d_rg", "_d_ms", "_d_ch",
            "_d_state", "_d_scratch", "_d_intensity", "_d_avg_color",
            "_d_gas", "_d_temp",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self._gas_handle = 0

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
