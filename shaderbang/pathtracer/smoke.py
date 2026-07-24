# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""OptiX / Blackwell toolchain de-risk smoke test (path-tracer milestone M0).

This is the single load-bearing test that must pass on the target RTX GPU
*before* any real rendering code is written. It exercises, in isolation, the
parts of the path-tracer design that cannot be tried on the CUDA-less dev box:

  1. import the stack: warp, cupy, optix (otk-pyoptix), cuda.bindings.nvrtc;
  2. ``optix.init()`` and report the OptiX / driver / GPU (Blackwell = sm_120);
  3. create an OptiX device context on Warp's CUDA context — i.e. the shared
     CUDA *primary* context (``cu_ctx = 0`` binds OptiX to the context that is
     current on this thread, which is the primary context Warp and CuPy also
     use), so OptiX, Warp and CuPy share device memory and streams;
  4. compile a trivial raygen/miss/closest-hit pipeline from embedded CUDA via
     NVRTC (needs the OptiX + CUDA headers — see the env vars below);
  5. build a GAS whose triangle vertices live in a ``wp.array`` and trace one
     ray per pixel against it, confirming a hit with **no CPU copy** of the
     geometry (the crux of the "reuse the physics geometry" constraint);
  6. run the OptiX AI denoiser (HDR) end to end on device buffers.

The CUDA<->GL PBO interop leg is intentionally *not* here: it needs a live GL
context (only the native shaderbang render loop provides one) and reuses the
``wp.RegisteredGLBuffer`` pattern ``examples/cloth.py`` already ships, so it is
validated in milestone M1's real ``Input`` instead of a standalone script.

Prerequisites on the target (see docs/pathtracer.md):
  - an NVIDIA RTX GPU, recent driver (OptiX 9.x needs R570+/R590+), CUDA 12.x;
  - ``otk-pyoptix`` built and importable as ``optix``;
  - ``warp-lang``, ``cupy-cuda12x``, ``cuda-python`` installed;
  - the OptiX SDK headers reachable via ``OPTIX_INCLUDE_DIR`` (or
    ``OPTIX_PATH``/``OPTIX_ROOT``/``OptiX_INSTALL_DIR`` pointing at the SDK
    root, whose ``include/`` is used);
  - the CUDA headers reachable via ``CUDA_INCLUDE_DIR``, ``CUDA_HOME``/
    ``CUDA_PATH``, or ``/usr/local/cuda/include``.

Run:
    python -m shaderbang.pathtracer.smoke

Exit status is 0 iff every check passes.
"""

import os
import sys
import traceback


# --------------------------------------------------------------------------- #
# Embedded OptiX device program (compiled to PTX by NVRTC at runtime).
#
# Self-contained: it only includes <optix.h> and uses component-wise math and
# CUDA's built-in vector types / make_* helpers, so it needs no vec_math.h.
# Params must match the host-side struct packed in _launch() field-for-field.
# --------------------------------------------------------------------------- #
CUDA_SRC = rb"""
#include <optix.h>

struct Params
{
    uchar4*                image;
    unsigned int           width;
    unsigned int           height;
    float3                 cam_eye;
    float3                 cam_u;
    float3                 cam_v;
    float3                 cam_w;
    OptixTraversableHandle handle;
};

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Normalised device coordinates in [-1, 1].
    const float dx = 2.0f * ( (float)idx.x + 0.5f ) / (float)dim.x - 1.0f;
    const float dy = 2.0f * ( (float)idx.y + 0.5f ) / (float)dim.y - 1.0f;

    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;

    float3 dir = make_float3( dx * U.x + dy * V.x + W.x,
                              dx * U.y + dy * V.y + W.y,
                              dx * U.z + dy * V.z + W.z );
    const float inv_len = rsqrtf( dir.x * dir.x + dir.y * dir.y + dir.z * dir.z );
    dir.x *= inv_len; dir.y *= inv_len; dir.z *= inv_len;

    unsigned int p0 = 0u, p1 = 0u, p2 = 0u;
    optixTrace(
            params.handle,
            params.cam_eye,
            dir,
            0.0f,                        // tmin
            1e16f,                       // tmax
            0.0f,                        // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE,
            0,                           // SBT offset
            1,                           // SBT stride
            0,                           // missSBTIndex
            p0, p1, p2 );

    float r = __uint_as_float( p0 );
    float g = __uint_as_float( p1 );
    float b = __uint_as_float( p2 );

    const unsigned char cr = (unsigned char)( 255.99f * fminf( fmaxf( r, 0.0f ), 1.0f ) );
    const unsigned char cg = (unsigned char)( 255.99f * fminf( fmaxf( g, 0.0f ), 1.0f ) );
    const unsigned char cb = (unsigned char)( 255.99f * fminf( fmaxf( b, 0.0f ), 1.0f ) );
    params.image[idx.y * params.width + idx.x] = make_uchar4( cr, cg, cb, 255 );
}

extern "C" __global__ void __miss__ms()
{
    // Constant dark-blue background (no SBT data needed).
    setPayload( make_float3( 0.05f, 0.05f, 0.15f ) );
}

extern "C" __global__ void __closesthit__ch()
{
    // Built-in triangle intersection provides barycentrics; encode them as
    // colour so a hit is unmistakable from the background.
    const float2 bc = optixGetTriangleBarycentrics();
    setPayload( make_float3( bc.x, bc.y, 1.0f ) );
}
"""


class Check:
    """Ordered pass/fail reporter for the smoke-test steps."""

    def __init__(self):
        self.results = []  # (name, ok, detail)

    def ok(self, name, detail=""):
        self.results.append((name, True, detail))
        print(f"  [ OK ] {name}" + (f" -- {detail}" if detail else ""))

    def fail(self, name, detail=""):
        self.results.append((name, False, detail))
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))

    def passed(self):
        return all(ok for _, ok, _ in self.results)

    def summary(self):
        n = len(self.results)
        n_ok = sum(1 for _, ok, _ in self.results if ok)
        print()
        print(f"  {n_ok}/{n} checks passed")
        for name, ok, detail in self.results:
            if not ok:
                print(f"    - FAILED: {name}" + (f" ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
# Toolchain include-path discovery (env-driven; NVRTC needs the OptiX headers).
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
    for inc in ("/usr/local/cuda/include",):
        if os.path.isdir(inc):
            return inc
    return None


# --------------------------------------------------------------------------- #
# Small ctypes-struct packing helpers (mirrors otk-pyoptix examples).
# --------------------------------------------------------------------------- #
def _round_up(val, mult_of):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def _aligned_itemsize(formats, alignment):
    import numpy as np
    names = [f"x{i}" for i in range(len(formats))]
    dt = np.dtype({"names": names, "formats": formats, "align": True})
    return _round_up(dt.itemsize, alignment)


def _to_device(np_array, cp):
    import ctypes
    byte_size = np_array.size * np_array.dtype.itemsize
    h_ptr = ctypes.c_void_p(np_array.ctypes.data)
    d_mem = cp.cuda.memory.alloc(byte_size)
    # Synchronous H2D so the buffer is ready regardless of which stream the
    # subsequent launch/denoiser call runs on (no cross-stream race).
    d_mem.copy_from_host(h_ptr, byte_size)
    return d_mem


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


# --------------------------------------------------------------------------- #
# NVRTC compilation (embedded CUDA -> PTX).
# --------------------------------------------------------------------------- #
def _compile_ptx(nvrtc, optix_inc, cuda_inc):
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
        b"-rdc",
        b"true",
        f"-I{optix_inc}".encode(),
        f"-I{cuda_inc}".encode(),
    ]
    prog = check(nvrtc.nvrtcCreateProgram(CUDA_SRC, b"pathtracer_smoke.cu", 0, [], []))
    check(nvrtc.nvrtcCompileProgram(prog, len(options), options), prog)
    ptx_size = check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptx_size
    check(nvrtc.nvrtcGetPTX(prog, ptx))
    return ptx


# --------------------------------------------------------------------------- #
# The smoke test.
# --------------------------------------------------------------------------- #
def run():
    print("=" * 70)
    print("shaderbang path-tracer M0 smoke test (OptiX / Blackwell de-risk)")
    print("=" * 70)
    chk = Check()

    # 1. Imports -----------------------------------------------------------
    print("\n[1] Imports")
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        chk.fail("import numpy", str(e))
        chk.summary()
        return 1
    chk.ok("import numpy", np.__version__)

    try:
        import warp as wp
        chk.ok("import warp", wp.config.version)
    except Exception as e:  # noqa: BLE001
        chk.fail("import warp", str(e))
        chk.summary()
        return 1

    try:
        import cupy as cp
        chk.ok("import cupy", cp.__version__)
    except Exception as e:  # noqa: BLE001
        chk.fail("import cupy", str(e))
        chk.summary()
        return 1

    try:
        import optix
        chk.ok("import optix (otk-pyoptix)")
    except Exception as e:  # noqa: BLE001
        chk.fail("import optix (otk-pyoptix)", str(e))
        chk.summary()
        return 1

    try:
        from cuda.bindings import nvrtc
        chk.ok("import cuda.bindings.nvrtc")
    except Exception as e:  # noqa: BLE001
        try:
            from cuda import nvrtc  # older cuda-python layout
            chk.ok("import cuda.nvrtc (legacy)")
        except Exception as e2:  # noqa: BLE001
            chk.fail("import nvrtc", f"{e} / {e2}")
            chk.summary()
            return 1

    # 2. Warp device + OptiX init -----------------------------------------
    print("\n[2] Warp device + OptiX init")
    try:
        wp.init()
        wp.set_device("cuda:0")
        # Establish/select the CUDA primary context that OptiX will share.
        cp.cuda.Device(0).use()
        cp.cuda.runtime.free(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        cc = cp.cuda.Device(0).compute_capability
        chk.ok("select cuda:0", f"{name}, sm_{cc}")
        if cc.startswith("12"):
            chk.ok("Blackwell (sm_120) detected", f"sm_{cc}")
        else:
            chk.ok("compute capability", f"sm_{cc} (target is Blackwell sm_120)")
    except Exception as e:  # noqa: BLE001
        chk.fail("Warp/CUDA device selection", str(e))
        chk.summary()
        return 1

    try:
        optix.init()
        ver = optix.version()
        chk.ok("optix.init()", f"OptiX {ver[0]}.{ver[1]}.{ver[2] if len(ver) > 2 else 0}")
    except Exception as e:  # noqa: BLE001
        chk.fail("optix.init()", str(e))
        chk.summary()
        return 1

    # 3. Device context on the shared primary context ----------------------
    print("\n[3] OptiX device context on Warp's CUDA context")

    class _Logger:
        def __call__(self, level, tag, msg):
            print(f"    [optix][{level}][{tag}] {msg}")

    try:
        ctx_options = optix.DeviceContextOptions(
            logCallbackFunction=_Logger(),
            logCallbackLevel=4,
        )
        try:
            ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
        except Exception:  # noqa: BLE001
            pass
        # cu_ctx = 0 -> use the CUDA context current on this thread, which is
        # the primary context Warp initialised above. This is the mechanism by
        # which OptiX shares memory/streams with Warp (no second context).
        ctx = optix.deviceContextCreate(0, ctx_options)
        chk.ok("deviceContextCreate on primary context (shared with Warp)")
    except Exception as e:  # noqa: BLE001
        chk.fail("deviceContextCreate", str(e))
        chk.summary()
        return 1

    # 4. NVRTC compile -----------------------------------------------------
    print("\n[4] NVRTC compile of the OptiX device program")
    optix_inc = _optix_include_dir()
    cuda_inc = _cuda_include_dir()
    if not optix_inc:
        chk.fail("locate OptiX headers",
                 "set OPTIX_INCLUDE_DIR (or OPTIX_PATH/OPTIX_ROOT to the SDK root)")
        chk.summary()
        return 1
    if not cuda_inc:
        chk.fail("locate CUDA headers",
                 "set CUDA_INCLUDE_DIR (or CUDA_HOME/CUDA_PATH)")
        chk.summary()
        return 1
    chk.ok("OptiX headers", optix_inc)
    chk.ok("CUDA headers", cuda_inc)
    try:
        ptx = _compile_ptx(nvrtc, optix_inc, cuda_inc)
        chk.ok("NVRTC -> PTX", f"{len(ptx)} bytes")
    except Exception as e:  # noqa: BLE001
        chk.fail("NVRTC compile", str(e))
        chk.summary()
        return 1

    # 5. Module / program groups / pipeline / SBT --------------------------
    print("\n[5] Module, program groups, pipeline, SBT")
    try:
        pipeline_options = optix.PipelineCompileOptions(
            usesMotionBlur=False,
            traversableGraphFlags=int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
            numPayloadValues=3,
            numAttributeValues=3,
            exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
            pipelineLaunchParamsVariableName="params",
            usesPrimitiveTypeFlags=optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE,
        )
        module_options = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        module, _log = ctx.moduleCreate(module_options, pipeline_options, ptx)
        chk.ok("moduleCreate")

        rg_desc = optix.ProgramGroupDesc()
        rg_desc.raygenModule = module
        rg_desc.raygenEntryFunctionName = "__raygen__rg"
        rg_group, _ = ctx.programGroupCreate([rg_desc])

        ms_desc = optix.ProgramGroupDesc()
        ms_desc.missModule = module
        ms_desc.missEntryFunctionName = "__miss__ms"
        ms_group, _ = ctx.programGroupCreate([ms_desc])

        ch_desc = optix.ProgramGroupDesc()
        ch_desc.hitgroupModuleCH = module
        ch_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
        ch_group, _ = ctx.programGroupCreate([ch_desc])

        program_groups = [rg_group[0], ms_group[0], ch_group[0]]
        chk.ok("programGroupCreate (raygen/miss/hitgroup)")

        max_trace_depth = 1
        link_options = optix.PipelineLinkOptions()
        link_options.maxTraceDepth = max_trace_depth
        pipeline = ctx.pipelineCreate(
            pipeline_options, link_options, program_groups, "")

        stack_sizes = optix.StackSizes()
        for pg in program_groups:
            try:
                optix.util.accumulateStackSizes(pg, stack_sizes, pipeline)
            except TypeError:
                optix.util.accumulateStackSizes(pg, stack_sizes)
        dc_trav, dc_state, cc_size = optix.util.computeStackSizes(
            stack_sizes, max_trace_depth, 0, 0)
        pipeline.setStackSize(dc_trav, dc_state, cc_size, 1)
        chk.ok("pipelineCreate + setStackSize")
    except Exception as e:  # noqa: BLE001
        chk.fail("pipeline setup", str(e))
        traceback.print_exc()
        chk.summary()
        return 1

    # SBT: header-only records (no per-record data needed).
    try:
        header_fmt = f"{optix.SBT_RECORD_HEADER_SIZE}B"

        def _header_record(group):
            itemsize = _aligned_itemsize([header_fmt], optix.SBT_RECORD_ALIGNMENT)
            dt = np.dtype({"names": ["header"], "formats": [header_fmt],
                           "itemsize": itemsize, "align": True})
            h = np.array([0], dtype=dt)
            optix.sbtRecordPackHeader(group, h)
            return _to_device(h, cp), dt.itemsize

        d_rg, rg_stride = _header_record(rg_group[0])
        d_ms, ms_stride = _header_record(ms_group[0])
        d_ch, ch_stride = _header_record(ch_group[0])
        # Keep device SBT buffers alive for the launch.
        _keep = (d_rg, d_ms, d_ch)
        sbt = optix.ShaderBindingTable(
            raygenRecord=d_rg.ptr,
            missRecordBase=d_ms.ptr,
            missRecordStrideInBytes=ms_stride,
            missRecordCount=1,
            hitgroupRecordBase=d_ch.ptr,
            hitgroupRecordStrideInBytes=ch_stride,
            hitgroupRecordCount=1,
        )
        chk.ok("shader binding table")
    except Exception as e:  # noqa: BLE001
        chk.fail("SBT setup", str(e))
        traceback.print_exc()
        chk.summary()
        return 1

    # 6. GAS from a wp.array + trace (the zero-copy crux) ------------------
    print("\n[6] GAS from wp.array geometry + trace")
    gas_handle = None
    try:
        # A single triangle facing +Z, stored in Warp device memory. Passing
        # its raw device pointer to OptiX proves geometry is shared with the
        # physics with no CPU round-trip.
        tri = np.array([[-0.5, -0.5, 0.0],
                        [0.5, -0.5, 0.0],
                        [0.0, 0.5, 0.0]], dtype=np.float32)
        wp_vertices = wp.array(tri, dtype=wp.vec3)  # kept alive below
        vtx_ptr = _device_ptr(wp_vertices)
        chk.ok("wp.array vertices", f"ptr=0x{vtx_ptr:x}, n={len(wp_vertices)}")

        accel_options = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
            operation=optix.BUILD_OPERATION_BUILD,
        )
        tri_input = optix.BuildInputTriangleArray()
        tri_input.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
        tri_input.numVertices = len(wp_vertices)
        tri_input.vertexBuffers = [vtx_ptr]
        tri_input.flags = [optix.GEOMETRY_FLAG_NONE]
        tri_input.numSbtRecords = 1

        sizes = ctx.accelComputeMemoryUsage([accel_options], [tri_input])
        d_temp = cp.cuda.alloc(sizes.tempSizeInBytes)
        d_gas = cp.cuda.alloc(sizes.outputSizeInBytes)  # kept alive below
        gas_handle = ctx.accelBuild(
            0, [accel_options], [tri_input],
            d_temp.ptr, sizes.tempSizeInBytes,
            d_gas.ptr, sizes.outputSizeInBytes, [])
        chk.ok("accelBuild (GAS) from wp.array pointer")
    except Exception as e:  # noqa: BLE001
        chk.fail("GAS build", str(e))
        traceback.print_exc()
        chk.summary()
        return 1

    try:
        width, height = 256, 256
        d_image = cp.zeros((height, width, 4), dtype=cp.uint8)

        # Camera looking down -Z at the triangle (matches otk triangle example).
        params = [
            ("u8", "image", int(d_image.data.ptr)),
            ("u4", "width", width),
            ("u4", "height", height),
            ("f4", "cam_eye_x", 0.0), ("f4", "cam_eye_y", 0.0), ("f4", "cam_eye_z", 2.0),
            ("f4", "cam_u_x", 1.10457), ("f4", "cam_u_y", 0.0), ("f4", "cam_u_z", 0.0),
            ("f4", "cam_v_x", 0.0), ("f4", "cam_v_y", 0.828427), ("f4", "cam_v_z", 0.0),
            ("f4", "cam_w_x", 0.0), ("f4", "cam_w_y", 0.0), ("f4", "cam_w_z", -2.0),
            ("u8", "handle", int(gas_handle)),
        ]
        formats = [p[0] for p in params]
        names = [p[1] for p in params]
        values = [p[2] for p in params]
        itemsize = _aligned_itemsize(formats, 8)
        params_dtype = np.dtype({"names": names, "formats": formats,
                                 "itemsize": itemsize, "align": True})
        h_params = np.array([tuple(values)], dtype=params_dtype)
        d_params = _to_device(h_params, cp)

        stream = cp.cuda.Stream()
        optix.launch(pipeline, stream.ptr, d_params.ptr, h_params.dtype.itemsize,
                     sbt, width, height, 1)
        stream.synchronize()

        img = cp.asnumpy(d_image)
        center = img[height // 2, width // 2]
        corner = img[2, 2]
        # Center ray hits the triangle -> blue channel forced to 255 by the CH.
        hit_center = int(center[2]) > 200
        # Corner ray misses -> dark-blue background (~[13, 13, 38]).
        miss_corner = int(corner[2]) < 120 and int(corner[0]) < 60
        detail = f"center={tuple(int(c) for c in center)}, corner={tuple(int(c) for c in corner)}"
        if hit_center and miss_corner:
            chk.ok("optixTrace hit/miss against Warp geometry", detail)
        else:
            chk.fail("optixTrace hit/miss against Warp geometry", detail)
    except Exception as e:  # noqa: BLE001
        chk.fail("launch/trace", str(e))
        traceback.print_exc()

    # 7. OptiX denoiser (HDR) end-to-end ----------------------------------
    print("\n[7] OptiX AI denoiser (HDR)")
    try:
        dn_w, dn_h = 256, 256

        def _optix_image(cp_buf, w, h):
            oi = optix.Image2D()
            oi.data = int(cp_buf.data.ptr)
            oi.width = w
            oi.height = h
            oi.rowStrideInBytes = w * 4 * 4  # FLOAT4
            oi.pixelStrideInBytes = 4 * 4
            oi.format = optix.PIXEL_FORMAT_FLOAT4
            return oi

        # Noisy HDR input: a smooth gradient plus salt noise, so we can check
        # the denoiser actually ran (output finite, noise reduced).
        base = np.zeros((dn_h, dn_w, 4), dtype=np.float32)
        yy = np.linspace(0.0, 1.0, dn_h, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, dn_w, dtype=np.float32)[None, :]
        base[..., 0] = xx
        base[..., 1] = yy
        base[..., 2] = 0.5
        base[..., 3] = 1.0
        noisy = base.copy()
        rng = np.random.default_rng(0)
        spikes = rng.random((dn_h, dn_w)) > 0.85
        noisy[spikes, 0:3] += 3.0

        d_noisy = cp.asarray(noisy)
        d_out = cp.zeros((dn_h, dn_w, 4), dtype=cp.float32)

        dn_options = optix.DenoiserOptions()
        dn_options.guideAlbedo = 0
        dn_options.guideNormal = 0
        denoiser = ctx.denoiserCreate(optix.DENOISER_MODEL_KIND_HDR, dn_options)

        mem = denoiser.computeMemoryResources(dn_w, dn_h)
        d_state = cp.empty((mem.stateSizeInBytes,), dtype=cp.uint8)
        d_scratch = cp.empty((mem.withoutOverlapScratchSizeInBytes,), dtype=cp.uint8)
        d_intensity = cp.empty((1,), dtype=cp.float32)
        denoiser.setup(0, dn_w, dn_h,
                       d_state.data.ptr, d_state.nbytes,
                       d_scratch.data.ptr, d_scratch.nbytes)
        chk.ok("denoiserCreate + setup",
               f"state={mem.stateSizeInBytes}B scratch={d_scratch.nbytes}B")

        layer = optix.DenoiserLayer()
        layer.input = _optix_image(d_noisy, dn_w, dn_h)
        layer.output = _optix_image(d_out, dn_w, dn_h)
        guide_layer = optix.DenoiserGuideLayer()

        # Field assignments mirror the otk-pyoptix denoiser example verbatim:
        # hdrIntensity takes the CuPy buffer (the binding reads its pointer),
        # the other fields are plain scalars.
        params = optix.DenoiserParams()
        params.denoiseAlpha = 0
        params.hdrIntensity = d_intensity
        params.hdrAverageColor = 0
        params.blendFactor = 0.0

        denoiser.computeIntensity(0, layer.input, d_intensity.data.ptr,
                                  d_scratch.data.ptr, d_scratch.nbytes)
        denoiser.invokeTiled(0, params,
                             d_state.data.ptr, d_state.nbytes,
                             guide_layer, [layer],
                             d_scratch.data.ptr, d_scratch.nbytes,
                             0, dn_w, dn_h)
        cp.cuda.Stream.null.synchronize()

        out = cp.asnumpy(d_out)
        finite = bool(np.all(np.isfinite(out)))
        # A denoiser must reduce the spike variance vs. the noisy input.
        noisy_var = float(np.var(noisy[..., 0:3]))
        out_var = float(np.var(out[..., 0:3]))
        if finite and out_var < noisy_var:
            chk.ok("denoiserInvoke",
                   f"var {noisy_var:.4f} -> {out_var:.4f}")
        else:
            chk.fail("denoiserInvoke",
                     f"finite={finite}, var {noisy_var:.4f} -> {out_var:.4f}")
    except Exception as e:  # noqa: BLE001
        chk.fail("denoiser", str(e))
        traceback.print_exc()

    chk.summary()
    return 0 if chk.passed() else 1


def main():
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
