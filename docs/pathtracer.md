# Path tracing for shaderbang

A reusable, real-time, denoised path tracer for shaderbang, first consumed by
[`examples/cloth.py`](../examples/cloth.py). It renders the cloth (and the rest
of the scene) with a Monte-Carlo path tracer running on the GPU's RT cores via
NVIDIA OptiX, denoised live with the OptiX AI Denoiser, and presented through
OpenGL — the substrate shaderbang already uses for its EGL/DRM/KMS scanout.

> **Status.** Design + phased implementation. The renderer is **RTX-on-target
> only**: it requires an NVIDIA GPU (developed against an RTX 5090 / Blackwell,
> CUDA 12.8+, driver R590+), CUDA, and OptiX 9.x. It cannot run on the CUDA-less
> dev box (macOS); only the Warp physics stays CPU-testable. Each milestone is
> committed separately and verified on-target.

## Goals and constraints

1. **Reuse the simulation geometry with no CPU round-trips.** The cloth's vertex
   positions, normals and indices live in GPU memory owned by the physics
   (NVIDIA Warp). The renderer consumes those buffers directly — no host copies
   on the render hot path.
2. **Live and denoised.** The target is real-time animated rendering with a
   temporal denoiser, not merely clean stills when paused.
3. **Reusable.** Packaged as `shaderbang/pathtracer/`, an `Input` subclass that
   fits shaderbang's `init` / `pre_render` / `render` / `post_render` model. The
   module takes CUDA **device arrays** for geometry — agnostic to whether the
   consumer happens to back them with OpenGL — and owns exactly one GL buffer
   internally (the present target).
4. **Borrow proven algorithms.** The shading math (Disney BSDF, MIS, NEE,
   environment importance sampling) is ported from
   [GLSL-PathTracer](https://github.com/knightcrawler25/GLSL-PathTracer) (MIT)
   into OptiX device code. The ACES tonemap GLSL is kept verbatim on the present
   quad.

## Architecture

```
Warp (physics / CUDA)                OptiX (rendering / CUDA)               OpenGL (present)
─────────────────────                ────────────────────────               ────────────────
cloth.simulate()                                                            
  writes pos/normals (wp.array) ─┐                                          
  refit collision LBVH           │  (device pointers, zero copy)            
    (collision ONLY)             └──►  GAS build / refit (per frame)        
                                       raygen megakernel:                    
                                         optixTrace on RT cores              
                                         iterative bounce loop               
                                         Disney BSDF / MIS / NEE             
                                         → HDR radiance + AOVs               
                                           (albedo, normal, flow, …)         
                                       OptiX AI Denoiser                     
                                         TEMPORAL + 2× upscale               
                                         1080p → 4K                          
                                         writes into ──(mapped GL PBO)──►  glTexSubImage2D
                                                                          → RGBA32F texture
                                                                          → fullscreen ACES quad
```

Three subsystems share **one CUDA context and stream**:

- **Warp** runs the cloth simulation, owns the geometry as plain `wp.array`s,
  and keeps the collision LBVH (`wp.Mesh`) — now used for **collision only**
  (swept CCD + PDT self-collision), which OptiX cannot serve.
- **OptiX** builds a geometry acceleration structure (**GAS**) from the *same*
  device pointers each frame, traces primary + secondary rays on the RT cores,
  writes HDR radiance and denoiser guide buffers (AOVs), then runs the AI
  Denoiser.
- **OpenGL** is the **presentation layer only**. There is no CUDA→display path
  in shaderbang, so the final image must cross into a GL texture to be scanned
  out. The denoiser writes into a CUDA-mapped GL pixel buffer object (PBO); a
  `glTexSubImage2D` uploads it into a texture drawn by a fullscreen ACES quad.

End state: **two GPU acceleration structures over one geometry source** — the
Warp collision LBVH (refit per physics substep) and the OptiX render GAS (refit
per frame). Both derive from the same `wp.array`s; neither involves a host copy.

### Why this shape

The four design decisions that produced it:

1. **Engine + denoiser: OptiX.** OptiX is NVIDIA's actively-developed ray-tracing
   + AI-denoising SDK (9.0 Feb 2025 added Blackwell support; 9.1 Dec 2025). Its
   AI Denoiser is the reference real-time/temporal denoiser and is the only one
   in this class reachable from a Python + CUDA + raw-OpenGL Linux stack. The
   alternatives were rejected on **accessibility, not quality**:
   - **NVIDIA NRD** (ReBLUR/ReLAX/SIGMA) — DX11/12/Vulkan + HLSL compute; no
     CUDA/GL/Python path.
   - **DLSS Ray Reconstruction** — Streamline/NGX, a DX12/Vulkan framework with
     no CUDA entry point and no Python bindings.
   - **Vulkan Ray Tracing** — reaches the same RT cores but breaks the
     CUDA/Warp interop that anchors the stack, and has no Python story.
   - **Intel OIDN** — native CUDA + Python, but still not temporal as of
     2026-07 (OIDN 3 temporal unreleased). Kept only as a **non-temporal
     fallback** behind the same interface.

2. **Binding: `otk-pyoptix` (first-party), not `python-optix`.** The
   "OptiX is unmaintained" concern was a conflation: the *SDK* is maintained;
   only the third-party `python-optix` binding (mortacious, last commit 2023,
   pinned to OptiX 7.6) is stale. NVIDIA ships
   [`otk-pyoptix`](https://github.com/NVIDIA/otk-pyoptix) (part of the OptiX
   Toolkit), current through 2026, supporting OptiX 9.1 on CUDA 12.x/Blackwell.
   Fallback is a self-owned thin `ctypes` wrapper over the denoiser host API —
   dependency-free, but it must reproduce the version-specific
   `OptixFunctionTable` ABI struct (which is exactly what `otk-pyoptix`
   absorbs), so it is the *fallback*, not the primary.

3. **Traversal: OptiX GAS (RT cores) from v1.** Software `wp.mesh_query_ray`
   traversal on the Warp LBVH leaves the RT cores idle. Since we already depend
   on OptiX for denoising, we let it do traversal too, on the RT cores. A GAS is
   a *second* acceleration structure, but it is built from the same GL/CUDA
   vertex buffers (no CPU copy), costs tens of MB, and refits in sub-millisecond
   time. This relaxes the original "no second BVH" constraint in exchange for
   the performance needed to be *live*; the "no CPU transfer" constraint is
   preserved.

4. **Resolution: 1080p + OptiX 2× temporal upscale.** v1 renders and denoises at
   1080p and upscales to 4K with the denoiser's temporal-upscale model, rather
   than tracing native 4K. Native 4K is a later option (M5).

## GL/CUDA/OptiX interop

GL↔CUDA interop is a **CUDA driver** feature, not an OptiX or Warp feature.
OptiX consumes plain `CUdeviceptr`s (`OptixImage2D.data`,
`OptixBuildInputTriangleArray.vertexBuffers`). A GL buffer becomes such a
pointer, zero-copy, via `cudaGraphicsGLRegisterBuffer` (register once at init) +
`cudaGraphicsMapResources` / `cudaGraphicsResourceGetMappedPointer` (map per
frame) — exactly what `wp.RegisteredGLBuffer` wraps. So the denoiser writes
directly into the GL PBO's memory.

Only **one** GL-registered buffer is needed: the present PBO. The intermediate
AOVs (radiance, albedo, normal, flow, …) are plain on-device buffers that never
touch GL. The cloth geometry is plain `wp.array`s (pure-PT: nothing rasterizes
them, so they need no GL backing).

Interop rules: the mapped pointer is valid only between map and unmap — remap
each frame, don't cache it; keep map → OptiX/CUDA work → unmap on one stream so
the unmap doesn't hand the buffer back to GL before the GPU finishes; GL must not
touch the buffer while it is mapped. Map/unmap provide implicit GL↔CUDA
synchronization (no external semaphores, unlike the Vulkan path).

*Optimization (M6, done):* register the GL **texture** directly
(`cudaGraphicsGLRegisterImage` + a CUDA surface object) so the tone-map writes
straight into the texture, eliminating the PBO→texture copy. Warp interop is
buffer-only, so this drops to the cuda-python driver API; opt in with
`interop="texture"`/`"auto"` (see M6 below).

## Integration gotchas

- **Shared CUDA context.** Create the OptiX device context on *Warp's* current
  `CUcontext` (`optixDeviceContextCreate`) so the GAS, denoiser and physics
  share one context and stream. Otherwise device pointers do not alias across
  contexts.
- **Denoiser outside CUDA-graph capture.** The denoiser allocates state/scratch
  and computes the HDR intensity on first invoke — it must not be captured into
  a CUDA graph. Warm it up once in `init` (like the cloth warms up `refit`).
- **Stream handle.** `wp.get_stream().cuda_stream` is a raw integer; APIs that
  want a stream object (e.g. CuPy) need `cp.cuda.ExternalStream(int(...))`.
- **Denoiser image formats.** The noisy input must not be `FLOAT4`; use
  `HALF4`/`FLOAT3`. Albedo/normal guides are typically 3-channel.
- **Motion vectors.** The cloth's existing `prevPos` is overwritten at the start
  of every physics substep, so it holds the second-to-last substep, not the last
  displayed frame — it is **unusable** for motion vectors. A dedicated
  `mvPrevPos` is snapshotted (`wp.copy(mvPrevPos, pos)`) in `pre_render`
  **before** `simulate`, **every rendered frame** (including when paused), and
  initialized to `pos`. Screen-space flow is computed in the raygen from the
  cached previous/current view-projection matrices.
- **Two-sided cloth.** The cloth is a thin sheet; shading normals are flipped on
  back-face hits (via the hit's front/back-face flag) or half of it renders
  black.
- **CPU import.** `shaderbang/__init__.py` loads `_shaderbang.so` at import, and
  that C extension does not build on macOS. The load is guarded so the pure-Warp
  physics remains importable/testable on the CUDA-less dev box; `warp-lang` is a
  declared dependency.

## Milestones

Each milestone is a separate commit, verified on the RTX 5090.

### M0 — De-risk the OptiX/Blackwell toolchain
The single load-bearing test, before any real rendering code, shipped as a
standalone script (`python -m shaderbang.pathtracer.smoke`). On the 5090:
import `otk-pyoptix`; `optix.init()` + device context on Warp's `CUcontext`
(the shared CUDA primary context) against the R590 driver; compile a trivial
raygen pipeline via NVRTC; build a GAS from a `wp.array` pointer and trace one
ray, confirming a hit with no CPU copy; run the OptiX denoiser (HDR) on device
buffers. Plus the CPU-side enablers: guard the `_shaderbang.so` load so the
package imports off-target, add `warp-lang` (extras) to `pyproject.toml`.

The CUDA↔GL PBO interop leg (`glMapBuffer`-free `wp.RegisteredGLBuffer` →
`glTexSubImage2D`) is validated in M1 instead: it needs a live GL context, which
only the native shaderbang render loop provides, and it reuses the exact
`wp.RegisteredGLBuffer` pattern `cloth.py` already ships — so it is low-risk and
best exercised in the real `Input`, not a standalone script that would have to
stand up its own EGL context.

### M1 — First live frame
`shaderbang/pathtracer/` module: OptiX pipeline (raygen / miss / closest-hit
`.cu` compiled by NVRTC), indexed GAS from `wp.array` geometry, 1 spp,
single-bounce Lambert shading + analytic sphere + ground, HDR accumulation,
single-frame (non-temporal) denoise, PBO → `glTexSubImage2D` → textured quad.
Integrated into `cloth.py`, replacing the fixed-function GL cloth/sphere draw;
geometry moves to plain `wp.array`s.

Lands in two commits for reviewability:

- **M1a — reusable renderer.** `programs.cu` (device programs) + `renderer.py`
  (the `PathTracer` class: pipeline, indexed GAS build + in-place refit,
  accumulation, HDR denoise, present), plus `offscreen.py`, an on-target
  self-test (`python -m shaderbang.pathtracer.offscreen`) that renders a curved
  triangle sheet over the sphere/ground to a PNG with **no GL context** — so the
  renderer is validated in isolation before touching cloth.py's live loop.
- **M1b — cloth integration.** Wire the `PathTracer` into `cloth.py`, move the
  cloth geometry off GL buffers onto plain `wp.array`s, and present the traced
  frame in place of the fixed-function draws. As implemented: `Cloth.init`
  allocates `pos` / `normals` / `triIds` as plain device arrays (the same arrays
  still back the `wp.Mesh` LBVH used for self-collision), and a new
  `PathTracerView` Input — constructed *last*, so it renders after the physics
  step and the camera/sphere updates — refits the GAS from `cloth.pos`, pushes
  the live camera basis (from `gluPerspective`/`gluLookAt`'s 40° vertical FOV)
  and sphere transform, traces, denoises, and presents. `Cloth`/`Ground`/`Sphere`
  `render()` become no-ops; the cloth mesh, sphere and ground are now the traced
  geometry (cloth GAS) and the two analytic colliders. Accumulation restarts
  whenever the scene moves (simulation running, or camera/sphere changed) and
  otherwise refines progressively while paused and still; the GAS is refit only
  on frames where the cloth actually deformed. Because a restart discards the
  cross-frame HDR accumulation, animating frames trace a small per-frame **spp
  burst** (`render(reset=True, spp=PathTracerView.SPP)`, default 4) into the
  accumulator before the single denoise, giving the temporal denoiser a
  cleaner-than-1-spp input so fast cloth motion does not ghost; still frames
  drop back to 1 spp/frame and refine progressively on the retained
  accumulation. The denoiser, history and motion-vector snapshot advance once
  per frame regardless of the burst. The anchor gizmos are the one
  remaining fixed-function draw, layered on top of the traced frame
  (`Cloth.draw_anchors`, called by `PathTracerView` after `present()`). Renders
  at the native framebuffer resolution for M1 — the 1080p→4K upscale is M3.

Refinement vs. the original sketch: the ACES tone-map runs as a **post-denoise
Warp kernel** that writes RGBA8 straight into the PBO (rather than a GLSL quad),
so the present path is one texture upload + one fixed-function quad — robust
across whatever GL version the native context exposes, and keeping the GL surface
minimal in the pure-PT design.

### M2 — On-target benchmark
Instrumentation to measure rays/s and frame time on the 5090 at 1080p and 4K.
This sets the real spp/bounce budget — the numbers that were only hypotheses at
design time. Implemented as `shaderbang/pathtracer/benchmark.py` (`python -m
shaderbang.pathtracer.benchmark`): headless (no GL context, so it runs over SSH),
it traces a deforming sheet sized to match cloth.py's default 320k-triangle mesh
against the analytic sphere/ground and times each pipeline phase — GAS refit,
trace, HDR denoise, ACES tone-map — reporting per-frame ms, fps and primary
rays/s at 1080p and 4K. `trace` is derived as `render − denoise` (the fused
launch+denoise `render()` call minus a denoise-only loop); timing is wall-clock
around an explicit stream sync, one sync per `iters`-long phase loop so the
figure is the sustained pipelined cost. The native-4K numbers stand in for the
upper bound until M3 adds the 1080p→4K temporal upscale path.

### M3 — Temporal quality
Split into **M3a** (guide AOVs), **M3b-1** (patch the binding to expose the
temporal/upscale surface) and **M3b-2** (the temporal + 2× upscale render path).

**M3a — albedo/normal guide AOVs (done).** The raygen program now writes two
extra per-pixel buffers alongside the beauty: `d_albedo` (the hit's surface
albedo; the background pixel gets the sky colour) and `d_normal` (the front-
facing shading normal in *view space*, with the forward axis negated so +z points
back toward the camera). Both are `FLOAT4`, same extent as the beauty, and are
overwritten every subframe rather than accumulated — they are deterministic per
pixel and the sub-pixel jitter only perturbs them at silhouette edges, which is
fine for a denoiser guide. Host side: `DenoiserOptions.guideAlbedo`/`guideNormal`
are enabled and the two buffers are attached as `DenoiserGuideLayer.albedo` /
`.normal`. The guides let the HDR denoiser keep albedo and geometric edges the
beauty alone smears at 1 spp — this is the achievable subset of the original
"albedo/normal guide AOVs + irradiance demodulation" item (guide-based edge
preservation subsumes manual demodulation while all albedos stay flat; explicit
demodulation is revisited in M4 once textures/BSDF give it something to do).

**M3b-1 — patch the binding (done).** The published `otk-pyoptix` binding
(`src/main.cpp`, master) cannot express the temporal/upscale path, and worse, it
has a latent bug: `DENOISER_MODEL_KIND_TEMPORAL` and `_TEMPORAL_AOV` are both
bound to the *AOV* enum value, so selecting "temporal" silently creates a plain
AOV denoiser. On top of that it exposes no upscale model kinds, no `FLOAT2`/`HALF2`
or `INTERNAL_GUIDE_LAYER` pixel formats, no internal-guide-layer double buffers on
`DenoiserGuideLayer`, no `temporalModeUsePreviousLayers` on `DenoiserParams`, and
no `internalGuideLayerPixelSizeInBytes` on `DenoiserSizes`.

Per the "fork + rebuild" decision, `shaderbang/pathtracer/patches/` carries a
small, version-gated patch (`otk-pyoptix-optix9-denoiser.patch`) that adds exactly
this surface to `src/main.cpp` and fixes the TEMPORAL/_TEMPORAL_AOV mapping —
everything it adds already exists in the OptiX 9 SDK headers. The patch is gated
with `#if OPTIX_VERSION >= 70500` / `IF_OPTIX77(...)` so the source still builds
against older SDKs. `patches/README.md` documents provenance (the exact upstream
commit + blob it was generated against), the build/apply steps, and a `hasattr`
check to verify the new symbols are present. The extension is built from source on
the target box; the stock binding still works for the single-frame path.

**M3b-2 — temporal denoiser + motion vectors + 2× upscale.** With the patched
binding in place: `mvPrevPos`-style history snapshot + a screen-space **flow** AOV
(OptiX convention `flow = current − previous`, FLOAT2 at input/1× resolution);
`previousOutput` history feedback carried through the double-buffered internal
guide layers; firefly clamp before accumulation; and the OptiX
**TEMPORAL_UPSCALE2X** model rendering at 1080p and presenting 4K.

Device side (`programs.cu`): the closest-hit program interpolates the *previous*
world position of the hit via `optixGetTriangleBarycentrics()` over
`params.prev_vertices[params.tri_indices[prim]]` and returns it as payload; the
raygen program projects both current and previous positions to pixel coordinates
(camera projection inverted via Cramer's rule) and writes `curr_pix − prev_pix`
into the flow buffer, with per-`which` handling for cloth / sphere / ground /
background.

Host side (`renderer.py`): `PathTracer(..., upscale=2)` selects
`TEMPORAL_UPSCALE2X`, allocates a 2× output double-buffer and two
`INTERNAL_GUIDE_LAYER` buffers (swapped each frame), tags the flow guide `FLOAT2`,
uses `computeAverageColor` → `hdrAverageColor` (not `computeIntensity`, which the
upscale family does not use), sets `temporalModeUsePreviousLayers = 0` on the
first frame of a sequence and `1` afterward, zeroes the previous internal guide on
frame 0, and invokes the denoiser non-tiled. Per-frame `_snapshot_prev()` copies
camera basis, sphere centre, and cloth vertices into their `prev_*` slots after the
render so the next frame's flow is correct.

### M4 — Full shading
Disney BSDF, multiple importance sampling, next-event estimation, and
environment-map importance sampling ported from GLSL-PathTracer into OptiX device
code; smooth shading normals from the cloth's per-vertex normals; two-sided cloth
via the hit's front/back-face flag. Split into **M4a** (BSDF + multi-bounce GI),
**M4b** (next-event estimation: shadowed delta sun) and **M4c** (env-map
importance sampling + MIS).

**M4a — Disney BSDF + multi-bounce GI + smooth normals + two-sided cloth (done).**
The single-hit Lambert of M1 is replaced by an iterative multi-bounce path tracer
in `__raygen__rg`: every bounce runs a unified `sceneIntersect` (cloth GAS +
analytic sphere/ground), evaluates the ported Disney principled BSDF, importance-
samples the next direction (`disneySample`), and applies Russian roulette past
`rr_depth`. All `optixTrace` calls still issue from the raygen program, so the
pipeline's `maxTraceDepth` stays 1. Shading uses the cloth's per-vertex smooth
normals (barycentric-interpolated in the closesthit program, wired via
`set_geometry(..., normals=)`), and cloth is two-sided (front/back albedo chosen
from the geometric face). Per-object roughness/metallic are exposed through
`set_cloth_material` / `set_sphere_material` / `set_ground_material`, and the bounce
budget through `set_path_depth`; `roughness` is used directly as the GGX α (Disney
linear roughness). A stand-in *unshadowed* directional sun keeps the frame lit
until NEE lands in M4b (below), and the analytic sky is the environment on a miss. The
albedo/normal/flow guide AOVs + motion vectors are preserved (the normal guide now
uses the smooth shading normal); a whole-path NaN sink + firefly clamp protect the
accumulator, and all normalizes/guide writes are zero/NaN-safe. The host `Params`
struct is guarded against host/device ABI drift by a compile-time
`static_assert(sizeof(Params) == PARAMS_EXPECTED_SIZE)` fed from `PARAMS_DTYPE`.

**M4b — next-event estimation: shadowed delta sun (done).** The M4a unshadowed
stand-in is replaced by a proper `directLight` that shadow-tests the directional
sun each bounce, so the frame gains contact shadows and correct occlusion. The sun
is a **delta light**, so it is next-event-estimated *only* — a BSDF-sampled ray has
zero probability of hitting an infinitesimal light, so there is nothing to combine
and no double counting (this mirrors GLSL-PathTracer's `SampleDistantLight`, which
forces `misWeight = 1`). Shadow rays reuse a second miss program (`__miss__shadow`,
miss SBT index 1) traced with closesthit disabled + terminate-on-first-hit, plus
cheap analytic sphere/ground occluder tests, all inside `sceneOcclude`; the trace
still issues from raygen so `maxTraceDepth` stays 1 and `Params` is unchanged (no
ABI churn). The sky remains BSDF-sampling only and is gathered on a miss at full
weight. **MIS is intentionally deferred to M4c:** it first becomes necessary when
the environment is *importance-sampled* (an env direction can be reached by both
NEE and BSDF sampling), so the power heuristic + carried BSDF pdf land there rather
than as dead scaffolding here.

**M4c — environment-map importance sampling + MIS (done).** An optional HDR
lat-long environment (`renderer.set_environment(image, intensity, rotation)`)
becomes the light on a miss; the analytic sky gradient (`set_sky`) stays the
default and `clear_environment()` restores it. This is where multiple importance
sampling arrives: an env direction can be reached by both env-NEE and BSDF
sampling, so the two are combined with the β=2 power heuristic
`PowerHeuristic(a,b) = a²/(a²+b²)`. The env-NEE half lives in `directLight`
(weighted `PowerHeuristic(envPdf, bsdfPdf)`); the BSDF-sampling half is gathered
in the raygen miss handler (weighted `PowerHeuristic(bsdfPdf, envPdf)`), carrying
the scalar BSDF pdf of the ray being traced across bounces — weight 1 on the
primary ray (depth 0), which has no NEE competitor. The delta sun stays NEE-only
and is never gathered on a miss, so sun and env never double-count.

Design choices (deviations from the GLSL-PathTracer reference, all deliberate):

- **`sin(θ)` row weighting + pbrt-exact pdf.** The host builds a single flat,
  row-major, unnormalized running-sum CDF over all `W·H` texels, weighted by
  Rec.709 luminance × `sin(θ_center)` (the reference omits `sin(θ)`, oversampling
  the poles). A texel is thus selected with probability `L·sin(θ_center)/Σ`. The
  device solid-angle pdf keeps the lat-long Jacobian's `1/sin(θ_actual)` term:
  `pdf_ω = L·sin(θ_center)·W·H / (Σ · 2π² · sin(θ_actual))`. The two `sin(θ)`s do
  **not** cancel — `θ_center` is the texel row center baked into the CDF, while
  `θ_actual` is the sampled/queried direction's polar angle — so both are kept.
  This is exactly pbrt's `InfiniteAreaLight` pdf and is unbiased; dropping
  `1/sin(θ_actual)` would bias the estimate by `sin(θ_center)/sin(θ_actual)`,
  O(1) near the poles. The identical pdf-of-direction is evaluated on both MIS
  sides so the strategies share one measure. Hierarchical binary search over the
  flat CDF (last column = marginal over rows, running sum within a row =
  conditional over columns) picks the texel.
- **Plain device buffers + manual bilinear filtering — no CUDA texture objects.**
  Consistent with the M0-validated plain-pointer approach (texture objects were
  never de-risked on target). The caller supplies the env image as a plain array
  (numpy / `wp.array` / any `__cuda_array_interface__` provider), so there are no
  new dependencies. Radiance is bilinearly filtered (wrap `u`, clamp `v`); the
  pdf uses the discrete per-texel luminance to stay consistent with the CDF.
- **Continuous, jittered sampler.** After the CDF picks a texel, the sample is
  jittered uniformly within that texel's uv cell (the reference point-samples the
  texel corner). `sampleEnv` draws exactly three uniforms whenever the env is
  enabled (one for the CDF value, two for the jitter), so the RNG stream stays
  deterministic per launch regardless of branch.

### M5 — Hardening + fallback

**M5a — Robustness pass + native-4K (done).** Every public setter now validates
its inputs up front and raises a clear `ValueError` instead of silently
producing a corrupt launch or an out-of-bounds device read: the constructor
checks `width`/`height >= 1`, `upscale ∈ {1, 2}` and `exposure > 0`;
`set_camera_lookat` rejects a coincident eye/target, an up vector parallel to the
view direction, and a non-positive aspect or out-of-range FOV; and `set_geometry`
rejects fewer than three vertices, a non-`(N, 3)` / non-multiple-of-3 index
buffer, and a smooth-normal count that does not match the vertex count. The
live per-frame setters (`set_sphere`) are deliberately left permissive so an
extreme interactive value can never crash a running session. A `close()` method
(also a context manager)
tears the instance down deterministically — `destroy()` on the pipeline, program
groups, module, denoiser and device context, then the GL texture/PBO, then the
device buffers — and is idempotent and safe after a partially-failed
construction. **Native 4K** needs no special path: it is simply `upscale=1` at a
4K extent (`PathTracer(3840, 2160, upscale=1)`), which selects the single-frame
HDR denoiser at full resolution instead of the 1080p→4K temporal upscaler.

**M5b — OIDN fallback (done).** A second denoiser backend, selected with
`PathTracer(..., denoiser="oidn")`, runs Intel Open Image Denoise instead of the
OptiX AI denoiser. It shares the whole rest of the pipeline — the OptiX trace and
the same beauty + albedo + normal AOVs still feed it, and the result still flows
through the device ACES tone-map — so from the caller's side it is the same
`render()` / `present()` loop; only the denoise step swaps. OIDN is non-temporal,
so it is valid only at `upscale=1` (the constructor rejects `denoiser="oidn"`
with `upscale=2`).

It is a deliberate **still-frame / portability fallback**, not a replacement for
the live path: it is the one place that takes a CPU roundtrip (download the FLOAT4
beauty/albedo/normal AOVs to host, run OIDN's `"RT"` filter with the guides in HDR
mode, upload the denoised RGB back for the tone-map). That does not breach the
"no CPU-GPU transfer on the hot path" rule — the hot path is the OptiX temporal
backend; OIDN is opt-in for portability (e.g. where the patched temporal binding
isn't available) or for high-quality progressive stills. The `oidn` package is a
**soft dependency**: it is imported lazily and only when this backend is chosen,
and a clear `RuntimeError` (with an install hint) is raised if it is missing. The
binding targeted is the ctypes `oidn` wrapper whose functions mirror the C API
(`NewDevice` / `NewFilter` / `SetSharedFilterImage` / `ExecuteFilter`) with numpy
arrays as shared buffers; the exact binding surface and the HDR-flag setter name
(which differs between OIDN 1.x and 2.x, so both spellings are probed) are marked
`VERIFY-ON-TARGET`.

### M6 — Direct-to-texture present (done)

The optimization deferred under *GL/CUDA/OptiX interop* above: present the
tone-mapped frame **straight into the GL texture** via a CUDA surface object,
dropping the per-frame PBO→`glTexSubImage2D` copy. Selected with a new
`PathTracer(..., interop=...)` argument:

- `"pbo"` (default) — the original path, unchanged: tone-map (Warp kernel) into a
  `wp.RegisteredGLBuffer` PBO, then `glTexSubImage2D` the PBO into the texture.
  The only present path with no raw-CUDA dependency, so it stays the portable
  default and the fallback.
- `"texture"` — register the GL texture with `cuGraphicsGLRegisterImage`
  (`…SURFACE_LDST`) and, each frame, map it on the render stream, bind a CUDA
  **surface object** over the mapped array, and run a `surf2Dwrite` tone-map
  kernel straight into it, then unmap. No PBO, no upload copy. Raises if the
  interop cannot be set up.
- `"auto"` — prefer `"texture"`, but fall back to `"pbo"` (creating the PBO
  instead) if the raw-CUDA/texture interop is unavailable at `init_gl` time
  (missing `cuda-python`, a kernel-compile failure, or a registration error).
  `cloth.py` uses this, so the example gets the fast path where it works and the
  proven PBO path everywhere else.

Because Warp's GL interop is buffer-only (it wraps `cudaGraphicsGLRegisterBuffer`,
which cannot register a texture, and it has no surface-object write), this leg
drops below Warp to the **cuda-python driver API** for registration
(`cuGraphicsGLRegisterImage`), per-frame map/unmap
(`cuGraphicsMapResources` / `cuGraphicsSubResourceGetMappedArray` /
`cuGraphicsUnmapResources`), and the surface object
(`cuSurfObjectCreate` / `cuSurfObjectDestroy`). The tone-map kernel is the one
piece Warp cannot express, so it is compiled at runtime with CuPy's `RawModule`
and kept **pixel-identical** to the Warp `_tonemap_kernel` (same ACES curve, same
1/2.2 gamma, same round-to-nearest u8); surface writes are byte-addressed on x
(`x*4` for `uchar4`), and row 0 stays the bottom row, so the fullscreen quad's
texcoords are unchanged across both paths.

Lifetime details: the mapped array handle is stable for a fixed-size texture, so
the surface object is built once and reused, and rebuilt only if the driver hands
back a different array across maps (a sync precedes the rebuild so no in-flight
kernel references the surface being destroyed). Map and unmap and the kernel all
issue on the shared render stream, so the write orders after the denoise and
before GL reclaims the texture; the unmap runs in a `finally` so a launch error
never leaves the resource mapped. `close()` drains the stream, destroys the
surface and unregisters the texture *before* the texture is deleted. The gain is
one device→device copy per frame (largest at native 4K, ~34 MB/frame); it is
**not** a host transfer either way, so the "no CPU-GPU transfer on the hot path"
rule held before and after. The whole leg is `VERIFY-ON-TARGET`: it cannot run on
the CUDA-less dev box (the `interop` validation, `_cuda_check` unpacking and
surface-rebuild bookkeeping are unit-tested off-target; the driver calls, the
`RawModule` surf2Dwrite compile and the map/write/unmap loop are not).

### M7 — Generic, consumer-agnostic scene (done)

Through M6 the *pipeline* was reusable but the *scene* was hardcoded for cloth:
one deformable mesh in global `Params` pointers, a single analytic sphere +
ground baked into the raygen, a 3-way material switch, and one directional sun.
M7 turns the scene into **data** — device buffers of materials, lights and
analytic primitives, plus an **instanced multi-mesh graph** — driven by a small
generic `add_*` API. `cloth.py` becomes just one consumer; a second minimal
example, [`examples/pathtracer_demo.py`](../examples/pathtracer_demo.py), proves
reuse. The present / tone-map / denoiser / AOV stack is scene-agnostic and was
untouched. Each sub-milestone changed the ABI for **its slice only** — the numpy
`PARAMS_DTYPE`, the device `struct Params`, and the `-DPARAMS_EXPECTED_SIZE`
`static_assert` move in lockstep — and left the tracer rendering.

The scene model referenced from `Params`:

- **Materials** (`M7a`) — a device `Material materials[]` + `num_materials`,
  indexed by `material_id`; `makeMaterial` reads the table (front/back albedo,
  roughness, metallic) instead of the old 3-way switch. `add_material()` appends
  a slot and returns its id.
- **Lights** (`M7b`) — a device `Light lights[]` + `num_lights`; `directLight`
  loops over them. Both kinds are delta lights next-event-estimated with
  `misWeight = 1` (no `/pdf`): **directional** (a direction, the old sun path) and
  **point** (a position, radiance attenuated `1/dist²`, shadow ray to the point).
  `add_light("directional"|"point", …)`. The env map is retained.
- **Analytic primitives** (`M7c`) — device `Sphere spheres[]` / `Plane planes[]`
  + counts, looped over in `sceneIntersect` / `sceneOcclude`; each carries its own
  `material_id` (a sphere also stores its previous-frame centre for motion
  vectors). `add_sphere` / `add_plane` + per-frame `update_sphere`. Unlike the
  material/light tables (uploaded eagerly on edit), these mutate every frame, so
  `add_/update_` only mark them dirty and `render()` uploads once per launch —
  the tiny transfer stays off the meaningful hot path. The trailing `uint`
  `material_id` is packed bit-for-bit into its `float` slot (a `np.uint32` view),
  matching the device struct's trailing `unsigned int`.
- **Meshes** (`M7d`) — moved off `Params` globals into a **single-level IAS**:
  each mesh is one per-mesh **GAS** + one `OptixInstance` + one per-instance SBT
  **hitgroup record** (`HitGroupData { uint3* indices; float3* normals;
  float3* prev_vertices; uint material_id; uint flags; float prev_xform[12]; }`,
  guarded by its own `static_assert(sizeof == 80)` via
  `-DHITGROUP_DATA_EXPECTED_SIZE`). `__closesthit__` reaches its mesh data through
  `optixGetSbtDataPointer()` instead of globals; `Params.handle` is now the IAS.
  Instance `i` gets `instanceId = i` and `sbtOffset = i`, so with the raygen's
  `optixTrace(SBToffset=0, SBTstride=1)` over single-build-input GASes the hit
  selects hitgroup record `i`. Pipeline `traversableGraphFlags` becomes
  `ALLOW_SINGLE_LEVEL_INSTANCING`.

**Generic motion vectors** (M7d): the flow AOV that feeds the temporal denoiser is
reconstructed per hit from the mesh kind. A **deformable** mesh
(`prev_vertices != 0`) interpolates the previous world position with the hit's
barycentrics over `prev_vertices` (the original cloth path). A **rigid** mesh
(`prev_vertices == 0`) reprojects the *object-space* hit point
(`objRayOrigin + objRayDir * tMax`) through a per-instance **previous
object→world transform** stored in the hitgroup record. (`optixGetObjectRayDirection`
is unnormalized, so the same `tMax` parameterizes the world and object rays; the
`safeNormalize` zero-guard survives because `transformNormal(0) = 0`.) Two dirty
flags drive the per-frame update: `_instances_dirty` (a rigid transform or GAS
handle changed → re-pack + upload the `OptixInstance` array and refit the IAS) and
`_sbt_dirty` (a `prev_xform`/material advanced in `_snapshot_prev` → re-pack the
hitgroup records). `add_mesh(deformable=…)` builds the GAS + instance + record;
deformable meshes update in place via `refit()`, rigid ones move via
`set_instance_transform()`. All geometry arrays (vertices, indices, normals,
prev-vertices) are **retained** by the mesh so the raw device pointers baked into
the SBT record stay valid for as long as the mesh — closesthit dereferences them
every frame, not just at GAS-build time.

**Migration + second consumer** (`M7e`): `cloth.py` moved to the generic API
(`add_material` / `add_mesh(deformable=True)` / `add_sphere` / `add_plane` /
`add_light`, per-frame `update_sphere` + `refit`), and the temporary
cloth-specific setters/shims (`set_geometry`, `set_sphere`, `set_ground`,
`set_light`, `set_cloth_*`, `set_sphere_material`, `set_ground_material`) and the
default scene slots they seeded were **deleted** — a fresh `PathTracer` now starts
empty and the consumer builds the whole scene. `examples/pathtracer_demo.py` is a
minimal second consumer with no cloth/Warp-sim dependency: two spinning rigid cube
instances + an analytic sphere on a ground plane, under a directional key + warm
point light, from a slowly orbiting camera — exercising `add_mesh(deformable=False)`,
`set_instance_transform` (rigid motion vectors), the analytic prims, the material
and light tables, and per-frame camera motion through the same `Input` callbacks.

## References

- [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) — SDK,
  programming guide, denoiser host API.
- [`otk-pyoptix`](https://github.com/NVIDIA/otk-pyoptix) — first-party Python
  bindings (OptiX Toolkit).
- [NVIDIA Warp](https://github.com/NVIDIA/warp) — physics, `wp.Mesh`/LBVH,
  `wp.RegisteredGLBuffer` interop.
- [GLSL-PathTracer](https://github.com/knightcrawler25/GLSL-PathTracer) (MIT) —
  source of the ported Disney BSDF / MIS / NEE / env-sampling and the ACES
  tonemap.
