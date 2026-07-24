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
    (single-frame HDR in M1; temporal + upscale come in M3), then tone-mapped
    (ACES) by a Warp kernel straight into an OpenGL PBO for display.

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
# field. align=True reproduces C struct padding: the three 8-byte members come
# first (offsets 0/8/16), then the 4-byte scalars, then the tightly packed
# float3s (each an ('f4', (3,)) subarray == float3). itemsize rounds to 208.
# --------------------------------------------------------------------------- #
_PARAMS_NAMES = [
    "accum", "output", "handle",
    "width", "height", "subframe", "exposure",
    "cam_eye", "cam_u", "cam_v", "cam_w",
    "light_dir", "light_color", "sky_top", "sky_bottom",
    "sphere_center", "sphere_albedo", "sphere_radius",
    "ground_albedo", "ground_y",
    "cloth_albedo_front", "cloth_albedo_back",
]
_PARAMS_FORMATS = [
    "u8", "u8", "u8",
    "u4", "u4", "u4", "f4",
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)), "f4",
    ("f4", (3,)), "f4",
    ("f4", (3,)), ("f4", (3,)),
]
# Build with align=True, then pin itemsize up to the 8-byte struct alignment so
# it equals the CUDA sizeof(Params) exactly (numpy stops at the last field, 204;
# C rounds the struct to a multiple of alignof == 8, i.e. 208).
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
                 log_level=0):
        self.width = int(width)
        self.height = int(height)
        self.exposure = float(exposure)
        self._device = device
        self._subframe = 0

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
        self._vtx_ptr = 0
        self._num_vertices = 0
        self._idx_ptr = 0
        self._num_triangles = 0

        # GL present state (created lazily on the first present, when a GL
        # context is current).
        self._gl_ready = False
        self._tex = None
        self._pbo = None
        self._pbo_reg = None

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
            numPayloadValues=4,     # t + geometric normal (xyz)
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

        ch_desc = optix.ProgramGroupDesc()
        ch_desc.hitgroupModuleCH = module
        ch_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
        ch_group, _ = self._ctx.programGroupCreate([ch_desc])

        self._rg_group = rg_group[0]
        self._ms_group = ms_group[0]
        self._ch_group = ch_group[0]
        program_groups = [self._rg_group, self._ms_group, self._ch_group]

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

        def header_record(group):
            itemsize = _round_up(
                np.dtype({"names": ["header"], "formats": [header_fmt],
                          "align": True}).itemsize, align)
            dt = np.dtype({"names": ["header"], "formats": [header_fmt],
                           "itemsize": itemsize, "align": True})
            h = np.array([0], dtype=dt)
            optix.sbtRecordPackHeader(group, h)
            d = cp.cuda.alloc(dt.itemsize)
            d.copy_from_host(ctypes.c_void_p(h.ctypes.data), dt.itemsize)
            return d, dt.itemsize

        self._d_rg, rg_stride = header_record(self._rg_group)
        self._d_ms, ms_stride = header_record(self._ms_group)
        self._d_ch, ch_stride = header_record(self._ch_group)
        self._sbt = optix.ShaderBindingTable(
            raygenRecord=self._d_rg.ptr,
            missRecordBase=self._d_ms.ptr,
            missRecordStrideInBytes=ms_stride,
            missRecordCount=1,
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
        self.d_denoised = wp.zeros(n, dtype=wp.vec4, device=self._device)
        # Offscreen LDR target (present() writes into a mapped PBO instead).
        self.d_ldr = wp.zeros(n * 4, dtype=wp.uint8, device=self._device)

    def _image2d(self, wp_arr):
        oi = self._optix.Image2D()
        oi.data = int(wp_arr.ptr)
        oi.width = self.width
        oi.height = self.height
        oi.rowStrideInBytes = self.width * 16   # FLOAT4
        oi.pixelStrideInBytes = 16
        oi.format = self._optix.PIXEL_FORMAT_FLOAT4
        return oi

    def _create_denoiser(self):
        optix = self._optix
        cp = self._cp
        dn_options = optix.DenoiserOptions()
        dn_options.guideAlbedo = 0
        dn_options.guideNormal = 0
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
        self._dn_guide = optix.DenoiserGuideLayer()
        self._dn_params = optix.DenoiserParams()
        self._dn_params.denoiseAlpha = 0
        self._dn_params.hdrIntensity = self._d_intensity
        self._dn_params.hdrAverageColor = 0
        self._dn_params.blendFactor = 0.0

    def _init_params(self):
        self._h_params = np.zeros(1, PARAMS_DTYPE)
        self._d_params = self._cp.cuda.alloc(PARAMS_DTYPE.itemsize)
        p = self._h_params[0]
        p["accum"] = int(self.d_accum.ptr)
        p["output"] = int(self.d_output.ptr)
        p["width"] = self.width
        p["height"] = self.height
        p["exposure"] = self.exposure
        # Sensible scene defaults; callers override via the setters below.
        p["cam_eye"] = (0.0, 1.0, 5.0)
        p["cam_u"] = (1.0, 0.0, 0.0)
        p["cam_v"] = (0.0, 1.0, 0.0)
        p["cam_w"] = (0.0, 0.0, -1.0)
        p["light_dir"] = _norm3((0.4, 1.0, 0.3))
        p["light_color"] = (1.0, 1.0, 1.0)
        p["sky_top"] = (0.35, 0.55, 0.9)
        p["sky_bottom"] = (0.9, 0.9, 0.95)
        p["sphere_center"] = (0.0, 1.5, 0.0)
        p["sphere_albedo"] = (0.75, 0.2, 0.2)
        p["sphere_radius"] = 0.5
        p["ground_albedo"] = (0.6, 0.6, 0.6)
        p["ground_y"] = 0.0
        p["cloth_albedo_front"] = (0.2, 0.45, 0.85)
        p["cloth_albedo_back"] = (0.85, 0.6, 0.2)

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
        w = _sub3(target, eye)          # not normalized: |W| is the focal length
        wlen = _len3(w)
        u = _norm3(_cross3(w, up))
        v = _norm3(_cross3(u, w))
        vlen = wlen * math.tan(0.5 * math.radians(fov_y_deg))
        ulen = vlen * aspect
        self.set_camera(eye, _mul3(u, ulen), _mul3(v, vlen), w)

    def set_sphere(self, center, radius, albedo=None):
        p = self._h_params[0]
        p["sphere_center"] = _vec3(center)
        p["sphere_radius"] = float(radius)
        if albedo is not None:
            p["sphere_albedo"] = _vec3(albedo)

    def set_ground(self, y, albedo=None):
        p = self._h_params[0]
        p["ground_y"] = float(y)
        if albedo is not None:
            p["ground_albedo"] = _vec3(albedo)

    def set_light(self, direction, color=(1.0, 1.0, 1.0)):
        p = self._h_params[0]
        p["light_dir"] = _norm3(_vec3(direction))
        p["light_color"] = _vec3(color)

    def set_sky(self, top, bottom):
        p = self._h_params[0]
        p["sky_top"] = _vec3(top)
        p["sky_bottom"] = _vec3(bottom)

    def set_cloth_albedo(self, front, back=None):
        p = self._h_params[0]
        p["cloth_albedo_front"] = _vec3(front)
        p["cloth_albedo_back"] = _vec3(back if back is not None else front)

    # ------------------------------------------------------------------ #
    # Geometry / acceleration structure
    # ------------------------------------------------------------------ #
    def set_geometry(self, vertices, indices):
        """Register triangle geometry and build the GAS.

        ``vertices`` is a ``wp.array(dtype=wp.vec3)``; ``indices`` a
        ``wp.array(dtype=wp.int32)`` of shape ``(num_tris, 3)`` (or a flat
        length-``3*num_tris`` array). Both stay owned by the caller (the physics
        keeps writing ``vertices`` in place); we only keep their pointers.
        """
        self._vtx_ptr = _device_ptr(vertices)
        self._num_vertices = int(len(vertices))
        self._idx_ptr = _device_ptr(indices)
        shape = getattr(indices, "shape", None)
        if shape is not None and len(shape) == 2:
            self._num_triangles = int(shape[0])
        else:
            self._num_triangles = int(len(indices)) // 3
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
    def render(self, reset=True):
        """Trace + accumulate one sample, then denoise into ``d_denoised``.

        ``reset=True`` restarts accumulation (use it every frame while the scene
        animates); ``reset=False`` keeps accumulating (progressive refinement
        while paused). Call ``present()`` or ``download_ldr()`` afterwards.
        """
        if self._gas_handle == 0:
            raise RuntimeError("render() called before set_geometry()")
        if reset:
            self._subframe = 0

        p = self._h_params[0]
        p["subframe"] = self._subframe
        p["exposure"] = self.exposure
        p["handle"] = int(self._gas_handle)
        self._d_params.copy_from_host(
            ctypes.c_void_p(self._h_params.ctypes.data), PARAMS_DTYPE.itemsize)

        self._optix.launch(
            self._pipeline, self._stream_ptr, self._d_params.ptr,
            PARAMS_DTYPE.itemsize, self._sbt, self.width, self.height, 1)

        self._denoise()
        self._subframe += 1

    def _denoise(self):
        self._denoiser.computeIntensity(
            self._stream_ptr, self._dn_input, self._d_intensity.data.ptr,
            self._d_scratch.data.ptr, self._d_scratch.nbytes)
        self._denoiser.invokeTiled(
            self._stream_ptr, self._dn_params,
            self._d_state.data.ptr, self._d_state.nbytes,
            self._dn_guide, [self._dn_layer],
            self._d_scratch.data.ptr, self._d_scratch.nbytes,
            0, self.width, self.height)

    def _tonemap_into(self, out_u8):
        wp.launch(_tonemap_kernel, dim=self.width * self.height,
                  inputs=[self.d_denoised, out_u8, self.exposure],
                  device=self._device, stream=self._wp_stream)

    def download_ldr(self):
        """Tone-map into ``d_ldr`` and return an (H, W, 4) uint8 numpy image
        (row 0 = bottom, matching the ray-gen convention). For offscreen use."""
        self._tonemap_into(self.d_ldr)
        wp.synchronize_stream(self._wp_stream)
        flat = self.d_ldr.numpy()
        return flat.reshape(self.height, self.width, 4)

    # ------------------------------------------------------------------ #
    # OpenGL present (fixed-function; robust across the target's GL version)
    # ------------------------------------------------------------------ #
    def init_gl(self):
        """Create the present texture + PBO. Must be called with a GL context
        current (i.e. from inside the shaderbang render loop)."""
        from OpenGL.GL import (
            glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
            glGenBuffers, glBindBuffer, glBufferData,
            GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
            GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
            GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
            GL_PIXEL_UNPACK_BUFFER, GL_STREAM_DRAW,
        )
        self._tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._pbo = int(glGenBuffers(1))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self.width * self.height * 4,
                     None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # Register the PBO with Warp so the tone-map kernel writes into it with
        # no CPU copy (same pattern cloth.py uses for its vertex buffers).
        self._pbo_reg = wp.RegisteredGLBuffer(
            self._pbo, self._device,
            wp.RegisteredGLBuffer.WRITE_DISCARD)  # VERIFY-ON-TARGET (ctor args)
        self._gl_ready = True

    def present(self):
        """Tone-map ``d_denoised`` into the PBO and draw it full-screen. Must be
        called with a GL context current."""
        if not self._gl_ready:
            self.init_gl()
        from OpenGL.GL import (
            glBindBuffer, glBindTexture, glTexSubImage2D,
            GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_PIXEL_UNPACK_BUFFER,
        )
        n = self.width * self.height
        mapped = self._pbo_reg.map(dtype=wp.uint8, shape=(n * 4,))
        self._tonemap_into(mapped)
        self._pbo_reg.unmap()   # ensures the CUDA write is complete for GL

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                        GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._draw_fullscreen_quad()

    def _draw_fullscreen_quad(self):
        from OpenGL.GL import (
            glMatrixMode, glLoadIdentity, glPushMatrix, glPopMatrix,
            glOrtho, glViewport, glDisable, glEnable, glBindTexture, glColor3f,
            glBegin, glEnd, glVertex2f, glTexCoord2f, glDepthMask,
            GL_PROJECTION, GL_MODELVIEW, GL_TEXTURE_2D, GL_DEPTH_TEST,
            GL_LIGHTING, GL_QUADS, GL_TRUE,
        )
        glViewport(0, 0, self.width, self.height)
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
