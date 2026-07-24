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
    when constructed with ``upscale=2``), then tone-mapped (ACES) by a Warp kernel
    straight into an OpenGL PBO for display.

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
    "prev_vertices", "tri_indices", "flow", "handle",
    "width", "height", "subframe", "exposure",
    "cam_eye", "cam_u", "cam_v", "cam_w",
    "prev_cam_eye", "prev_cam_u", "prev_cam_v", "prev_cam_w",
    "light_dir", "light_color", "sky_top", "sky_bottom",
    "sphere_center", "sphere_albedo", "sphere_radius",
    "sphere_center_prev",
    "ground_albedo", "ground_y",
    "cloth_albedo_front", "cloth_albedo_back",
]
_PARAMS_FORMATS = [
    "u8", "u8", "u8", "u8",
    "u8", "u8", "u8", "u8",
    "u4", "u4", "u4", "f4",
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)), ("f4", (3,)), ("f4", (3,)),
    ("f4", (3,)), ("f4", (3,)), "f4",
    ("f4", (3,)),
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
                 upscale=1, log_level=0):
        self.width = int(width)
        self.height = int(height)
        # Output extent. The temporal-upscale denoiser (M3b) produces a 2x-larger
        # image than it is fed, so the beauty/guide/flow buffers stay at the
        # render res (self.width/height) while the denoised result, tone-map
        # target, present texture and PBO use the output res (self._out_*).
        self.upscale = int(upscale)
        self._out_width = self.width * self.upscale
        self._out_height = self.height * self.upscale
        self.exposure = float(exposure)
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
        self._prev_vertices = None   # previous-frame vertex snapshot (motion vec)

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
            numPayloadValues=7,     # t + geometric normal (xyz) + prev-pos (xyz)
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
        # prev_vertices / tri_indices are wired by set_geometry (0 until then).
        p["prev_vertices"] = 0
        p["tri_indices"] = 0
        p["width"] = self.width
        p["height"] = self.height
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
        p["sphere_albedo"] = (0.75, 0.2, 0.2)
        p["sphere_radius"] = 0.5
        p["sphere_center_prev"] = tuple(p["sphere_center"])
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
        self._vertices = vertices          # kept for the per-frame prev snapshot
        self._vtx_ptr = _device_ptr(vertices)
        self._num_vertices = int(len(vertices))
        self._idx_ptr = _device_ptr(indices)
        shape = getattr(indices, "shape", None)
        if shape is not None and len(shape) == 2:
            self._num_triangles = int(shape[0])
        else:
            self._num_triangles = int(len(indices)) // 3

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
        if self._temporal:
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     self._out_width, self._out_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._pbo = int(glGenBuffers(1))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     self._out_width * self._out_height * 4,
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
        self._draw_fullscreen_quad()

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
