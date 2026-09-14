# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Plasma globe simulation and rendering support package for ``examples/plasma_globe.py``.

Modules
-------
dbm       dielectric-breakdown growth: harmonic-split potential, conductor charges, per-tree
          Gumbel-max growth, persistent filament nodes, touch charges
gas       Boussinesq stable-fluids gas convection on a masked 3D grid (Warp)
spectra   noble-gas emission colours (NIST lines x CIE 1931 -> linear sRGB)
apsf      Narasimhan-Nayar / Kim-Lin atmospheric point spread function -> Gaussian pyramid fit
glctx     headless desktop-GL context via the EGL device platform (benchmarks and tests)
glutil    GL program / image / buffer / timer helpers shared by the app and the benchmarks
interop   Warp <-> OpenGL data path (batched map/unmap, 3D texture upload)
harness   headless validation metrics (fractal dimension, spacing histograms, plume speed, ...)
"""
