# otk-pyoptix denoiser patch (OptiX 9 temporal + 2x upscale)

The path tracer's live-denoise path (milestone M3b) needs the OptiX AI denoiser's
**temporal** and **2x upscaling** surface. The published
[`NVIDIA/otk-pyoptix`](https://github.com/NVIDIA/otk-pyoptix) pybind11 binding
does not expose it (and has a latent bug), so we build the extension from source
with a small patch. Everything the patch adds already exists in the OptiX 9 SDK
headers; the binding just never wired it up.

## What the patch changes

`otk-pyoptix-optix9-denoiser.patch` edits a single file, `src/main.cpp`:

- **`PixelFormat`** — adds `FLOAT2`, `HALF2` (motion-vector / flow AOV) and
  `INTERNAL_GUIDE_LAYER` (the temporal denoiser's internal state image), plus the
  7.7+ `FLOAT1`/`HALF1`.
- **`DenoiserModelKind`** — adds `UPSCALE2X` and `TEMPORAL_UPSCALE2X`, and **fixes
  a bug**: the published binding maps `DENOISER_MODEL_KIND_TEMPORAL` and
  `_TEMPORAL_AOV` to the *wrong* enum value (`_AOV`), which would silently select
  the non-temporal model. The patch points them at the correct values.
- **`DenoiserGuideLayer`** — adds `previousOutputInternalGuideLayer`,
  `outputInternalGuideLayer` (the double-buffered internal guide images the
  temporal model requires) and the 7.7+ `flowTrustworthiness`.
- **`DenoiserParams`** — adds `temporalModeUsePreviousLayers` (0 on the first
  frame of a sequence, 1 afterwards).
- **`DenoiserSizes`** — adds `internalGuideLayerPixelSizeInBytes` (needed to size
  the internal guide buffers).
- **`DenoiserLayer`** — adds the 7.7+ AOV `type` hint (optional).

Every addition is version-gated (`#if OPTIX_VERSION >= 70500` or the existing
`IF_OPTIX77(...)` macro), so the patched source still compiles against older
OptiX SDKs; the new symbols simply appear only where the SDK defines them.
Nothing in the existing denoiser binding is removed, and `denoiseAlpha` is already
correctly gated for 8.0+ upstream, so the build does not break against OptiX 9.

## Provenance

Generated against `NVIDIA/otk-pyoptix` **master** commit
`3144f224c0fd18733925faf3d8fb82c7376b8dcf` (`src/main.cpp` blob
`0fa8a7c6eb01586926076525c72907f135ba2fc9`). If upstream has moved on and the
patch does not apply cleanly, the edits are small and localized to the
denoiser-enum and denoiser-struct binding blocks — re-apply them by hand from the
diff, or regenerate against the current `src/main.cpp`.

## Build

Requires the OptiX 9 SDK headers, a CUDA toolkit, and CMake. On the target box:

```sh
git clone https://github.com/NVIDIA/otk-pyoptix.git
cd otk-pyoptix
git checkout 3144f224c0fd18733925faf3d8fb82c7376b8dcf   # or master, if the patch still applies
git apply /path/to/shaderbang/shaderbang/pathtracer/patches/otk-pyoptix-optix9-denoiser.patch

# Point the build at the OptiX 9 SDK (the dir that contains include/optix.h):
export PYOPTIX_CMAKE_ARGS="-DOptiX_INSTALL_DIR=/opt/optix"   # adjust path

# Build + install into the active environment (matches shaderbang's .venv):
pip install .            # or:  python -m pip install --no-build-isolation .
```

Verify the new surface is present:

```sh
python -c "import optix; \
print(hasattr(optix.DenoiserModelKind, 'DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X')); \
print(hasattr(optix.DenoiserParams(), 'temporalModeUsePreviousLayers')); \
print(hasattr(optix.DenoiserGuideLayer(), 'outputInternalGuideLayer')); \
print(hasattr(optix.DenoiserSizes(), 'internalGuideLayerPixelSizeInBytes'))"
# expect: True / True / True / True
```

Once installed, `shaderbang.pathtracer.renderer.PathTracer(..., upscale=2)` selects
the `TEMPORAL_UPSCALE2X` model automatically. With the stock binding (`upscale=1`)
the renderer falls back to the single-frame HDR denoiser and does not require this
patch.
