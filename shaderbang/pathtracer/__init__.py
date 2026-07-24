# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""Real-time OptiX path tracer for shaderbang.

This package renders the same geometry the Warp physics writes (plain
``wp.array``s, no CPU copy) with a full NVIDIA OptiX path-tracing pipeline and
the OptiX AI denoiser, presenting the result through a single OpenGL PBO. See
``docs/pathtracer.md`` for the design.

It is RTX-on-target only: it requires an NVIDIA GPU with CUDA 12.x, the OptiX
SDK, and the ``otk-pyoptix`` bindings. On a CUDA-less / non-Linux box importing
this package is fine, but nothing here will run.

The de-risk smoke test lives in :mod:`shaderbang.pathtracer.smoke` and is run
with ``python -m shaderbang.pathtracer.smoke``.
"""
