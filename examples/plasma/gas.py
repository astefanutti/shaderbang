# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Gas convection solver
=====================

Boussinesq "stable fluids" solver for the neon fill of the plasma globe, on a masked
``wp.array3d`` grid, one sub-step per rendered frame (dt = 1/60 s), entirely inside a CUDA graph.
The hot filament channels are deposited as Gaussian line sources, the heated gas rises, hits the
glass, cools through the only Robin sink (the glass; the electrode is a Dirichlet T0 sink) and
recirculates.

Physical model (plan 4.7)
-------------------------
Per frame, on the ping-pong pair ``state_in -> state_out``::

    1. heat        Q(x) = sum_k q'_k G_line(x; sigma_q = 1.5 dx)         (segment list, atomics)
    2. advection   semi-Lagrangian RK2 back-trace of u and T, samples clamped to the fluid annulus
    3. diffusion   implicit Euler for u (nu) and T (alpha): fused red-black Gauss-Seidel, 4 sweeps;
                   the Robin glass sink and the Dirichlet electrode enter the same solve
    4. buoyancy    u_y += dt g_sign g (1 - T0 / T)                       (exact ideal gas)
    5. projection  div = grad.u - mean_fluid(grad.u); the all-Neumann Poisson problem, warm-started
                   from the previous frame, gets 6 fine red-black GS sweeps and then twice {one
                   geometric multigrid V-cycle (64 -> 32 -> 16 -> 8, 2 / 2 / 16 coarse sweeps),
                   6 fine sweeps}; p is re-centred every fine sweep by a device reduction; u -= grad q

Velocity is stored MAC-staggered inside a ``vec3`` cell array (``u[i, j, k]`` holds the x-face at
i - 1/2, the y-face at j - 1/2 and the z-face at k - 1/2), so divergence, gradient and Laplacian
are one consistent compact stencil and the projection is exact up to the solver residual. A face
carries a velocity only when both adjacent cells are fluid (no-slip everywhere else). ``p`` holds
the pressure pre-multiplied by dt / rho0 (units m^2/s). The sealed rigid vessel makes the
Boussinesq constraint div u = 0 exact (the volume integral of the thermal expansion is zero).

Multigrid: the coarse levels use the fine mask's own connectivity, i.e. the coupling between two
coarse cells is the fraction of open (fluid-fluid) fine faces across their common face and a coarse
cell is fluid when any child is, so the coarse operators are the finite-volume restriction of the
fine one; residuals are averaged over the fluid children, corrections are prolonged by a
mask-weighted trilinear interpolation. One cycle reduces the residual ~5x (40 plain GS sweeps: 8 %),
so the convergence no longer depends on the temporal warm start: an impulsive start from rest or a
gravity flip keeps max |div u| dx / max |u| < 1e-3 on every frame (``--test divergence``).

Determinism: every atomic accumulation (heat / emission stamps, the divergence and pressure sums)
is done in int64 fixed point, so any thread order gives the same bits and a replay is bit-identical
(``--test determinism``). Scales: heat and emission 2^-20 W/m^3 per count, divergence sum 2^-36
1/s, pressure sums 2^-40 m^2/s. ``time_scale = 0`` (pause) is an exact no-op: advection copies the
state through and the diffusion, buoyancy, pressure and projection kernels return early.

Provenance-tagged constants (plan 4.1)
--------------------------------------
======================  =====================  ==========================================
R1 electrode bulb       0.015 m                MEASURED (PPPL-4485, 1.5-2 cm)
R2i inner glass         0.075 m                DECISION (PPPL Table 1, R1 / R2 = 0.2)
gas                     Ne + 2 % Xe, 740 Torr  MEASURED (PPPL Table 1)
rho0, cp                0.798 kg/m^3, 1030 J/kgK   DERIVED (ideal gas, monatomic)
k, nu, alpha            0.0491 W/mK, 3.97e-5, 5.97e-5 m^2/s   DERIVED (Pr = 0.67)
beta                    1 / T0 exactly         DERIVED; the code uses g (1 - T0 / T)
h_ext                   7.5 W/m^2K             CHOSEN (free convection on the outer glass)
sigma_q                 1.5 dx                 CHOSEN (the 1 mm channel is conduction-smeared:
                                               Pe(a) = 0.84, so the coarse Gaussian is physical)
ice cap                 T_amb - 20 K, 20 deg   CHOSEN (PPPL Sec. 10b ice-cube experiment)
grid                    64^3 over 0.16 m       CHOSEN (dx 2.5 mm); 96^3 is a resolution change
dt                      1 / 60 s, 1 sub-step   CHOSEN; alpha dt / dx^2 = 0.16 (64^3), 0.36 (96^3)
                                               against the explicit limit 1/6 -> implicit
SEG_MAX                 6144                   CHOSEN, compile-time
======================  =====================  ==========================================

Contract for other modules
--------------------------
``GasState.grid`` is a ``GasGrid`` struct that can be passed to any kernel; ``sample_u(grid, x)``
and ``sample_T(grid, x)`` are world-space trilinear samplers (the growth engine advects nodes
with the former and gates the reduced field with the latter). ``GasSources`` is the device
segment list {p0, p1, q', e} the filament code fills inside the frame graph (q' in W/m heats the
gas, e is the per-metre emission weight of the ambient-glow splat, count in a device int array so
an empty list costs one early-out). ``GasControl`` is the small device parameter array
{time scale, g_sign, ice, h_ext, q scale, T_amb, ice dT}; nothing per-frame is baked into a
capture. ``GasSolver.capture(state_in, state_out, control, dt)`` returns the graph of one step
followed by the copy of ``state_out`` back into ``state_in`` so the same graph can be replayed
every frame; ``GasSolver.pack_volume(state, vol)`` resamples the grid into the renderer's coarse
96^3 RGBA16F volume {T - T0 [K], emission, |u| [m/s], mask} in (z, y, x) memory order.

The class shape (``step(state_in, state_out, control, dt)``) follows ``newton.solvers.SolverBase``
from the Newton physics engine (Apache-2.0, https://github.com/newton-physics/newton).

Self-tests (plan 9/M3)::

    python examples/plasma/gas.py --test all --res 64
    python examples/plasma/gas.py --test plume --qprime 1.3 --invert --ice
    python examples/plasma/gas.py --test bench --res 96 --segments 6144
"""

import math
import time

import numpy as np
import warp as wp


# Gas and vessel (plan 4.1)
RHO0 = 0.798          # kg/m^3
CP = 1030.0           # J/(kg K)
K_GAS = 0.0491        # W/(m K)
NU = 3.97e-5          # m^2/s
ALPHA = 5.97e-5       # m^2/s
T0 = 300.0            # K
G = 9.81              # m/s^2
R1 = 0.011            # m, electrode bulb (matches plasma.params.R1)
R2I = 0.075           # m, inner glass
H_EXT = 7.5           # W/(m^2 K), Robin sink on the glass
ICE_DT = 20.0         # K, ambient depression under the ice cap
ICE_CAP_COS = math.cos(math.radians(20.0))

# Discretisation
EXTENT = 0.16         # m, cube side
RES = 64
DT = 1.0 / 60.0
SEG_MAX = 6144
SIGMA_Q_CELLS = 1.5   # heat stamp sigma in cells
EMIS_SIGMA = 4.0e-3   # m, ambient emission splat sigma
STAMP_SIGMAS = 3.0    # heat stamp truncation radius in sigmas
EMIS_SIGMAS = 2.5     # emission splat truncation radius in sigmas
EMIS_UNIT = 1.0e3     # the volume stores emission density / 1e3 (kW/m^3 for e = q'; float16 range)
DIFF_SWEEPS = 4
PRESSURE_SWEEPS = 12  # fine red-black GS sweeps per V-cycle (half before, half after)
MG_LEVELS = 3         # coarse levels below the fine grid (64 -> 32 -> 16 -> 8)
MG_SWEEPS = 2         # pre- and post-smoothing sweeps on each coarse level
MG_BOTTOM_SWEEPS = 16 # sweeps on the coarsest level
MG_CYCLES = 2         # V-cycles per frame
GS_ONLY_SWEEPS = 40   # the plan's pure Gauss-Seidel baseline (mg_levels = 0)
VOL_RES = 96
VOL_EXTENT = 2.0 * R2I
BLOCK_DIM = 256       # block-level reductions need n^3 / 2 to be a multiple of this

# Fixed-point scales of the int64 accumulators (integer sums are order independent)
FIXED_SRC = 1048576.0            # 2^20 counts per W/m^3 (heat and emission stamps)
FIXED_SRC_INV = 1.0 / FIXED_SRC
FIXED_DIV = 68719476736.0        # 2^36 counts per 1/s (divergence sum)
FIXED_DIV_INV = 1.0 / FIXED_DIV
FIXED_P = 1099511627776.0        # 2^40 counts per m^2/s (pressure sums)
FIXED_P_INV = 1.0 / FIXED_P

# Cell word: type bits | fluid-neighbour mask << NBF_SHIFT | thermal-neighbour mask << NBT_SHIFT
FLUID = 1
GLASS = 2
ELECTRODE = 4
NBF_SHIFT = 3
NBT_SHIFT = 9
MX = 1
PX = 2
MY = 4
PY = 8
MZ = 16
PZ = 32

# Control array slots
CTRL_TIME_SCALE = 0   # 1 = real time, 0 = frozen (an exact no-op step, all fields bit-identical)
CTRL_G_SIGN = 1       # +1 upright, -1 globe upside down (gravity stays world-vertical)
CTRL_ICE = 2          # 1 = ice cap on the +y pole
CTRL_H_EXT = 3        # W/(m^2 K)
CTRL_Q_SCALE = 4      # multiplier on every q' (heat only; the emission weight e is not scaled)
CTRL_T_AMB = 5        # K, ambient outside the glass
CTRL_ICE_DT = 6       # K
CTRL_T_ELECTRODE = 7  # K above T0: the electrode is a heated sphere (its sheath power, globe.fill_params)
CTRL_SIZE = 8

# Reduction slots: divergence sum, then two rotating {red, black} pairs of pressure sums
ACC_DIV = 0
ACC_P = 1
ACC_SIZE = 5


@wp.struct
class GasGrid:
    n: int
    dx: float
    inv_dx: float
    origin: wp.vec3
    r1: float
    r2: float
    u: wp.array3d(dtype=wp.vec3)
    T: wp.array3d(dtype=float)
    cell: wp.array3d(dtype=int)


@wp.func
def cell_center(origin: wp.vec3, dx: float, i: int, j: int, k: int) -> wp.vec3:
    return origin + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx


@wp.func
def clamp_annulus(r1: float, r2: float, x: wp.vec3) -> wp.vec3:
    r = wp.length(x)
    if r > r2:
        return x * (r2 / r)
    if r < r1:
        if r < 1.0e-9:
            return wp.vec3(0.0, r1, 0.0)
        return x * (r1 / r)
    return x


# The sampler internals take plain arrays and scalars on purpose: passing the GasGrid struct
# (three array descriptors) by value down a four-deep call chain, four times per thread,
# overflows the per-thread CUDA stack (measured: illegal memory access in k_advect).

@wp.func
def lattice(n: int, origin: wp.vec3, inv_dx: float, x: wp.vec3, off: wp.vec3):
    """Base index and fractional position of x on the lattice shifted by off (in cells)."""
    p = (x - origin) * inv_dx - off
    i = wp.clamp(int(wp.floor(p[0])), 0, n - 2)
    j = wp.clamp(int(wp.floor(p[1])), 0, n - 2)
    k = wp.clamp(int(wp.floor(p[2])), 0, n - 2)
    f = wp.vec3(wp.clamp(p[0] - float(i), 0.0, 1.0),
                wp.clamp(p[1] - float(j), 0.0, 1.0),
                wp.clamp(p[2] - float(k), 0.0, 1.0))
    return i, j, k, f


@wp.func
def trilinear(c000: float, c100: float, c010: float, c110: float,
              c001: float, c101: float, c011: float, c111: float, f: wp.vec3) -> float:
    c00 = c000 + (c100 - c000) * f[0]
    c10 = c010 + (c110 - c010) * f[0]
    c01 = c001 + (c101 - c001) * f[0]
    c11 = c011 + (c111 - c011) * f[0]
    c0 = c00 + (c10 - c00) * f[1]
    c1 = c01 + (c11 - c01) * f[1]
    return c0 + (c1 - c0) * f[2]


@wp.func
def face_value(u: wp.array3d(dtype=wp.vec3), n: int, origin: wp.vec3, inv_dx: float,
               x: wp.vec3, axis: int, off: wp.vec3) -> float:
    i, j, k, f = lattice(n, origin, inv_dx, x, off)
    return trilinear(u[i, j, k][axis], u[i + 1, j, k][axis],
                     u[i, j + 1, k][axis], u[i + 1, j + 1, k][axis],
                     u[i, j, k + 1][axis], u[i + 1, j, k + 1][axis],
                     u[i, j + 1, k + 1][axis], u[i + 1, j + 1, k + 1][axis], f)


@wp.func
def velocity(u: wp.array3d(dtype=wp.vec3), n: int, origin: wp.vec3, inv_dx: float, x: wp.vec3) -> wp.vec3:
    return wp.vec3(face_value(u, n, origin, inv_dx, x, 0, wp.vec3(0.0, 0.5, 0.5)),
                   face_value(u, n, origin, inv_dx, x, 1, wp.vec3(0.5, 0.0, 0.5)),
                   face_value(u, n, origin, inv_dx, x, 2, wp.vec3(0.5, 0.5, 0.0)))


@wp.func
def fixed_scalar(a: wp.array3d(dtype=wp.int64), n: int, origin: wp.vec3, inv_dx: float, x: wp.vec3) -> float:
    """Trilinear sample of a cell-centred fixed-point field (FIXED_SRC counts per unit)."""
    i, j, k, f = lattice(n, origin, inv_dx, x, wp.vec3(0.5, 0.5, 0.5))
    return trilinear(wp.float32(a[i, j, k]), wp.float32(a[i + 1, j, k]),
                     wp.float32(a[i, j + 1, k]), wp.float32(a[i + 1, j + 1, k]),
                     wp.float32(a[i, j, k + 1]), wp.float32(a[i + 1, j, k + 1]),
                     wp.float32(a[i, j + 1, k + 1]), wp.float32(a[i + 1, j + 1, k + 1]), f) * FIXED_SRC_INV


@wp.func
def blend_corner(T: wp.array3d(dtype=float), cell: wp.array3d(dtype=int), i: int, j: int, k: int,
                 w: float, ts: float, wt: float, wf: float):
    c = cell[i, j, k]
    if c & (FLUID | ELECTRODE):
        ts += w * T[i, j, k]
        wt += w
    if c & FLUID:
        wf += w
    return ts, wt, wf


@wp.func
def temperature(T: wp.array3d(dtype=float), cell: wp.array3d(dtype=int), n: int, origin: wp.vec3,
                inv_dx: float, x: wp.vec3):
    """Fluid-weighted trilinear temperature at x (K) and the trilinear fluid mask in [0, 1].

    Glass cells are excluded from the weights (Neumann wall), electrode cells count with their
    Dirichlet T0, so a sample next to the glass never blends in a fake cold value.
    """
    i, j, k, f = lattice(n, origin, inv_dx, x, wp.vec3(0.5, 0.5, 0.5))
    ts = float(0.0)
    wt = float(0.0)
    wf = float(0.0)
    ts, wt, wf = blend_corner(T, cell, i, j, k, (1.0 - f[0]) * (1.0 - f[1]) * (1.0 - f[2]), ts, wt, wf)
    ts, wt, wf = blend_corner(T, cell, i + 1, j, k, f[0] * (1.0 - f[1]) * (1.0 - f[2]), ts, wt, wf)
    ts, wt, wf = blend_corner(T, cell, i, j + 1, k, (1.0 - f[0]) * f[1] * (1.0 - f[2]), ts, wt, wf)
    ts, wt, wf = blend_corner(T, cell, i + 1, j + 1, k, f[0] * f[1] * (1.0 - f[2]), ts, wt, wf)
    ts, wt, wf = blend_corner(T, cell, i, j, k + 1, (1.0 - f[0]) * (1.0 - f[1]) * f[2], ts, wt, wf)
    ts, wt, wf = blend_corner(T, cell, i + 1, j, k + 1, f[0] * (1.0 - f[1]) * f[2], ts, wt, wf)
    ts, wt, wf = blend_corner(T, cell, i, j + 1, k + 1, (1.0 - f[0]) * f[1] * f[2], ts, wt, wf)
    ts, wt, wf = blend_corner(T, cell, i + 1, j + 1, k + 1, f[0] * f[1] * f[2], ts, wt, wf)
    t = T0
    if wt > 0.0:
        t = ts / wt
    return t, wf


@wp.func
def departure(u: wp.array3d(dtype=wp.vec3), n: int, origin: wp.vec3, inv_dx: float,
              r1: float, r2: float, x: wp.vec3, dt: float) -> wp.vec3:
    """RK2 (midpoint) semi-Lagrangian departure point, kept inside the fluid annulus."""
    v1 = velocity(u, n, origin, inv_dx, x)
    xm = clamp_annulus(r1, r2, x - 0.5 * dt * v1)
    v2 = velocity(u, n, origin, inv_dx, xm)
    return clamp_annulus(r1, r2, x - dt * v2)


@wp.func
def sample_u(g: GasGrid, x: wp.vec3) -> wp.vec3:
    """World-space gas velocity at x (m/s), trilinear per MAC component; zero inside walls."""
    return velocity(g.u, g.n, g.origin, g.inv_dx, x)


@wp.func
def sample_T(g: GasGrid, x: wp.vec3) -> float:
    """World-space gas temperature at x (K), fluid-weighted trilinear (T0 inside the electrode)."""
    t, mask = temperature(g.T, g.cell, g.n, g.origin, g.inv_dx, x)
    return t


@wp.func
def ambient(x: wp.vec3, ctrl: wp.array(dtype=float)) -> float:
    t = ctrl[CTRL_T_AMB]
    if ctrl[CTRL_ICE] > 0.5 and x[1] > ICE_CAP_COS * wp.length(x):
        t -= ctrl[CTRL_ICE_DT]
    return t


@wp.kernel
def k_deposit(g: GasGrid, src: wp.array3d(dtype=wp.int64), emis: wp.array3d(dtype=wp.int64),
              seg_p0: wp.array(dtype=wp.vec3), seg_p1: wp.array(dtype=wp.vec3),
              seg_q: wp.array(dtype=float), seg_e: wp.array(dtype=float), seg_count: wp.array(dtype=int),
              ctrl: wp.array(dtype=float), sigma_q: float, sigma_e: float, r_box: float, slabs: int):
    """Gaussian line sources -> volumetric heat (W/m^3) and emission density fields, in int64
    fixed point (FIXED_SRC counts per unit) so the atomic order does not change the result.

    Thread (s, slab) stamps every slabs-th x-slab of segment s's bounding box; the axial profile
    is the exact erf integral along the segment, the radial profile a 2D Gaussian normalised over
    its truncation disk so the deposited power integrates to q' L (emission to e L).
    """
    s, slab = wp.tid()
    if s >= seg_count[0]:
        return
    q = seg_q[s] * ctrl[CTRL_Q_SCALE]
    e = seg_e[s]
    if q == 0.0 and e == 0.0:
        return
    p0 = seg_p0[s]
    d = seg_p1[s] - p0
    length = wp.length(d)
    if length < 1.0e-9:
        return
    axis = d / length
    lo = wp.min(p0, seg_p1[s]) - wp.vec3(r_box, r_box, r_box)
    hi = wp.max(p0, seg_p1[s]) + wp.vec3(r_box, r_box, r_box)
    i0 = wp.clamp(int(wp.ceil((lo[0] - g.origin[0]) * g.inv_dx - 0.5)), 0, g.n - 1)
    j0 = wp.clamp(int(wp.ceil((lo[1] - g.origin[1]) * g.inv_dx - 0.5)), 0, g.n - 1)
    k0 = wp.clamp(int(wp.ceil((lo[2] - g.origin[2]) * g.inv_dx - 0.5)), 0, g.n - 1)
    i1 = wp.clamp(int(wp.floor((hi[0] - g.origin[0]) * g.inv_dx - 0.5)), 0, g.n - 1)
    j1 = wp.clamp(int(wp.floor((hi[1] - g.origin[1]) * g.inv_dx - 0.5)), 0, g.n - 1)
    k1 = wp.clamp(int(wp.floor((hi[2] - g.origin[2]) * g.inv_dx - 0.5)), 0, g.n - 1)
    rq = STAMP_SIGMAS * sigma_q
    re = EMIS_SIGMAS * sigma_e
    inv_sq = 1.0 / (wp.sqrt(2.0) * sigma_q)
    inv_se = 1.0 / (wp.sqrt(2.0) * sigma_e)
    norm_q = q * FIXED_SRC / (2.0 * wp.pi * sigma_q * sigma_q * (1.0 - wp.exp(-0.5 * STAMP_SIGMAS * STAMP_SIGMAS)))
    norm_e = e * FIXED_SRC / (2.0 * wp.pi * sigma_e * sigma_e * (1.0 - wp.exp(-0.5 * EMIS_SIGMAS * EMIS_SIGMAS)))
    r_box2 = r_box * r_box
    for i in range(i0 + slab, i1 + 1, slabs):
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                if (g.cell[i, j, k] & FLUID) == 0:
                    continue
                r = cell_center(g.origin, g.dx, i, j, k) - p0
                t = wp.dot(r, axis)
                perp2 = wp.max(wp.dot(r, r) - t * t, 0.0)
                if perp2 > r_box2:
                    continue
                perp = wp.sqrt(perp2)
                if perp <= rq and q != 0.0:
                    axial = 0.5 * (wp.erf((length - t) * inv_sq) + wp.erf(t * inv_sq))
                    wp.atomic_add(src, i, j, k, wp.int64(norm_q * axial * wp.exp(-0.5 * perp2 / (sigma_q * sigma_q))))
                if perp <= re and e != 0.0:
                    axial = 0.5 * (wp.erf((length - t) * inv_se) + wp.erf(t * inv_se))
                    wp.atomic_add(emis, i, j, k, wp.int64(norm_e * axial * wp.exp(-0.5 * perp2 / (sigma_e * sigma_e))))


@wp.kernel
def k_advect(gin: GasGrid, gout: GasGrid, src: wp.array3d(dtype=wp.int64),
             u_rhs: wp.array3d(dtype=wp.vec3), T_rhs: wp.array3d(dtype=float),
             ctrl: wp.array(dtype=float), dt0: float):
    """Semi-Lagrangian RK2 transport of T and of the three MAC faces owned by each cell, plus the
    heat source; writes the diffusion right-hand sides (also the Gauss-Seidel initial guess).
    A frozen step (time scale 0) copies the state through untouched."""
    i, j, k = wp.tid()
    c = gin.cell[i, j, k]
    t = gin.T[i, j, k]
    u = wp.vec3(0.0, 0.0, 0.0)
    scale = ctrl[CTRL_TIME_SCALE]
    if scale == 0.0:
        u = gin.u[i, j, k]
    elif c & FLUID:
        u_in = gin.u
        n = gin.n
        origin = gin.origin
        inv_dx = gin.inv_dx
        r1 = gin.r1
        r2 = gin.r2
        dt = dt0 * scale
        x = cell_center(origin, gin.dx, i, j, k)
        t, mask = temperature(gin.T, gin.cell, n, origin, inv_dx,
                              departure(u_in, n, origin, inv_dx, r1, r2, x, dt))
        t += dt * wp.float32(src[i, j, k]) * (FIXED_SRC_INV / (RHO0 * CP))
        nbf = (c >> NBF_SHIFT) & 63
        h = 0.5 * gin.dx
        if nbf & MX:
            xd = departure(u_in, n, origin, inv_dx, r1, r2, x - wp.vec3(h, 0.0, 0.0), dt)
            u[0] = face_value(u_in, n, origin, inv_dx, xd, 0, wp.vec3(0.0, 0.5, 0.5))
        if nbf & MY:
            xd = departure(u_in, n, origin, inv_dx, r1, r2, x - wp.vec3(0.0, h, 0.0), dt)
            u[1] = face_value(u_in, n, origin, inv_dx, xd, 1, wp.vec3(0.5, 0.0, 0.5))
        if nbf & MZ:
            xd = departure(u_in, n, origin, inv_dx, r1, r2, x - wp.vec3(0.0, 0.0, h), dt)
            u[2] = face_value(u_in, n, origin, inv_dx, xd, 2, wp.vec3(0.5, 0.5, 0.0))
    gout.T[i, j, k] = t
    gout.u[i, j, k] = u
    T_rhs[i, j, k] = t
    u_rhs[i, j, k] = u


@wp.kernel
def k_electrode_temperature(g: GasGrid, ctrl: wp.array(dtype=float)):
    """The electrode cells hold the Dirichlet value the diffusion sees: T0 + the surface excess the
    electrode's dissipated sheath power sustains (a heated sphere: its plume and the thinner gas
    around it are what the corona and the roots' boundary layer feel)."""
    i, j, k = wp.tid()
    if (g.cell[i, j, k] & ELECTRODE) != 0:
        g.T[i, j, k] = T0 + ctrl[CTRL_T_ELECTRODE]


@wp.kernel
def k_diffuse(g: GasGrid, u_rhs: wp.array3d(dtype=wp.vec3), T_rhs: wp.array3d(dtype=float),
              wall: wp.array3d(dtype=float), ctrl: wp.array(dtype=float), dt0: float, color: int):
    """One red-black Gauss-Seidel half-sweep of the implicit diffusion of u (nu) and T (alpha).

    T: glass neighbours are Neumann (dropped), electrode neighbours Dirichlet T0 (they hold T0),
    the Robin glass sink -k dT/dn = h_ext (T - T_amb) is folded in through the cell's glass area.
    u: a neighbouring wall face holds zero (no-slip Dirichlet), so the plain 6-sum applies.
    """
    i, j, kk = wp.tid()
    if ctrl[CTRL_TIME_SCALE] == 0.0:
        return
    k = 2 * kk + ((i + j + color) & 1)
    c = g.cell[i, j, k]
    if (c & FLUID) == 0:
        return
    dt = dt0 * ctrl[CTRL_TIME_SCALE]
    a_t = ALPHA * dt * g.inv_dx * g.inv_dx
    a_u = NU * dt * g.inv_dx * g.inv_dx
    nbt = (c >> NBT_SHIFT) & 63
    ts = float(0.0)
    cnt = float(0.0)
    if nbt & MX:
        ts += g.T[i - 1, j, k]
        cnt += 1.0
    if nbt & PX:
        ts += g.T[i + 1, j, k]
        cnt += 1.0
    if nbt & MY:
        ts += g.T[i, j - 1, k]
        cnt += 1.0
    if nbt & PY:
        ts += g.T[i, j + 1, k]
        cnt += 1.0
    if nbt & MZ:
        ts += g.T[i, j, k - 1]
        cnt += 1.0
    if nbt & PZ:
        ts += g.T[i, j, k + 1]
        cnt += 1.0
    robin = dt * ctrl[CTRL_H_EXT] * wall[i, j, k] / (RHO0 * CP * g.dx * g.dx * g.dx)
    t_amb = ambient(cell_center(g.origin, g.dx, i, j, k), ctrl)
    g.T[i, j, k] = (T_rhs[i, j, k] + a_t * ts + robin * t_amb) / (1.0 + a_t * cnt + robin)
    nbf = (c >> NBF_SHIFT) & 63
    us = (g.u[i - 1, j, k] + g.u[i + 1, j, k] + g.u[i, j - 1, k] + g.u[i, j + 1, k]
          + g.u[i, j, k - 1] + g.u[i, j, k + 1])
    u = (u_rhs[i, j, k] + a_u * us) / (1.0 + 6.0 * a_u)
    if (nbf & MX) == 0:
        u[0] = 0.0
    if (nbf & MY) == 0:
        u[1] = 0.0
    if (nbf & MZ) == 0:
        u[2] = 0.0
    g.u[i, j, k] = u


@wp.kernel
def k_buoyancy(g: GasGrid, ctrl: wp.array(dtype=float), dt0: float):
    """u_y += dt g_sign g (1 - T0 / T) on interior y-faces (T averaged over the two cells)."""
    i, j, k = wp.tid()
    if ctrl[CTRL_TIME_SCALE] == 0.0:
        return
    c = g.cell[i, j, k]
    if (c & FLUID) and (((c >> NBF_SHIFT) & MY) != 0):
        dt = dt0 * ctrl[CTRL_TIME_SCALE]
        t = 0.5 * (g.T[i, j, k] + g.T[i, j - 1, k])
        u = g.u[i, j, k]
        u[1] += dt * ctrl[CTRL_G_SIGN] * G * (1.0 - T0 / t)
        g.u[i, j, k] = u


@wp.kernel
def k_divergence(g: GasGrid, div: wp.array3d(dtype=float), acc: wp.array(dtype=wp.int64)):
    """Raw divergence of u in fluid cells (1/s) and its fluid sum in int64 fixed point (block
    reduction, one atomic per block); the sweeps subtract the fluid mean so the all-Neumann
    problem is compatible."""
    i, j, k = wp.tid()
    d = float(0.0)
    if g.cell[i, j, k] & FLUID:
        d = (g.u[i + 1, j, k][0] - g.u[i, j, k][0]
             + g.u[i, j + 1, k][1] - g.u[i, j, k][1]
             + g.u[i, j, k + 1][2] - g.u[i, j, k][2]) * g.inv_dx
        div[i, j, k] = d
    total = wp.tile_sum(wp.tile(wp.int64(d * FIXED_DIV)))
    wp.tile_atomic_add(acc, total, offset=(ACC_DIV,))


@wp.func
def div_mean(acc: wp.array(dtype=wp.int64), inv_n_fluid: float) -> float:
    return wp.float32(acc[ACC_DIV]) * FIXED_DIV_INV * inv_n_fluid


@wp.kernel
def k_pressure(g: GasGrid, p: wp.array3d(dtype=float), div: wp.array3d(dtype=float),
               acc: wp.array(dtype=wp.int64), inv_n_fluid: float, ctrl: wp.array(dtype=float),
               color: int, slot: int):
    """One red-black Gauss-Seidel half-sweep of lap q = div - mean(div), Neumann on walls.

    The red half-sweep subtracts the mean of the previous complete iterate (its red + black sums
    from slots [1 - slot]); because the all-Neumann operator is shift-invariant the following
    black half-sweep inherits the shift, so the pair re-centres the whole iterate at no extra
    launch. Each half-sweep accumulates its own sum into slots [slot] (int64 fixed point); the
    black half-sweep zeroes the pair the next iteration will fill.
    """
    i, j, kk = wp.tid()
    if ctrl[CTRL_TIME_SCALE] == 0.0:
        return
    k = 2 * kk + ((i + j + color) & 1)
    c = g.cell[i, j, k]
    v = float(0.0)
    if c & FLUID:
        nbf = (c >> NBF_SHIFT) & 63
        ps = float(0.0)
        cnt = float(0.0)
        if nbf & MX:
            ps += p[i - 1, j, k]
            cnt += 1.0
        if nbf & PX:
            ps += p[i + 1, j, k]
            cnt += 1.0
        if nbf & MY:
            ps += p[i, j - 1, k]
            cnt += 1.0
        if nbf & PY:
            ps += p[i, j + 1, k]
            cnt += 1.0
        if nbf & MZ:
            ps += p[i, j, k - 1]
            cnt += 1.0
        if nbf & PZ:
            ps += p[i, j, k + 1]
            cnt += 1.0
        rhs = (div[i, j, k] - div_mean(acc, inv_n_fluid)) * g.dx * g.dx
        v = (ps - rhs) / wp.max(cnt, 1.0)
        if color == 0:
            v -= wp.float32(acc[ACC_P + 2 * (1 - slot)] + acc[ACC_P + 2 * (1 - slot) + 1]) * FIXED_P_INV * inv_n_fluid
        p[i, j, k] = v
    total = wp.tile_sum(wp.tile(wp.int64(v * FIXED_P)))
    wp.tile_atomic_add(acc, total, offset=(ACC_P + 2 * slot + color,))
    if color == 1 and i == 0 and j == 0 and kk == 0:
        acc[ACC_P + 2 * (1 - slot)] = wp.int64(0)
        acc[ACC_P + 2 * (1 - slot) + 1] = wp.int64(0)


@wp.kernel
def k_mg_restrict_fine(g: GasGrid, p: wp.array3d(dtype=float), div: wp.array3d(dtype=float),
                       acc: wp.array(dtype=wp.int64), inv_n_fluid: float, ctrl: wp.array(dtype=float),
                       rc: wp.array3d(dtype=float)):
    """Residual (div - mean) - lap p of the fine problem (1/s) as a volume average over the 8
    children of every coarse cell (solid children count as zero, so a partially solid coarse cell
    carries its fluid fraction; this keeps the coarse equation the sum of the fine flux balances)."""
    ci, cj, ck = wp.tid()
    if ctrl[CTRL_TIME_SCALE] == 0.0:
        return
    mean = div_mean(acc, inv_n_fluid)
    inv_h2 = g.inv_dx * g.inv_dx
    total = float(0.0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = 2 * ci + di
                j = 2 * cj + dj
                k = 2 * ck + dk
                c = g.cell[i, j, k]
                if c & FLUID:
                    nbf = (c >> NBF_SHIFT) & 63
                    q = p[i, j, k]
                    lap = float(0.0)
                    if nbf & MX:
                        lap += p[i - 1, j, k] - q
                    if nbf & PX:
                        lap += p[i + 1, j, k] - q
                    if nbf & MY:
                        lap += p[i, j - 1, k] - q
                    if nbf & PY:
                        lap += p[i, j + 1, k] - q
                    if nbf & MZ:
                        lap += p[i, j, k - 1] - q
                    if nbf & PZ:
                        lap += p[i, j, k + 1] - q
                    total += (div[i, j, k] - mean) - lap * inv_h2
    rc[ci, cj, ck] = 0.125 * total


@wp.func
def mg_neighbours(wlo: wp.array3d(dtype=wp.vec3), whi: wp.array3d(dtype=wp.vec3),
                  e: wp.array3d(dtype=float), i: int, j: int, k: int):
    """Conductance-weighted neighbour sum and total conductance of coarse cell (i, j, k); a zero
    weight marks a closed face and is never dereferenced (boundary cells stay in range)."""
    lo = wlo[i, j, k]
    hi = whi[i, j, k]
    s = float(0.0)
    ws = float(0.0)
    if lo[0] > 0.0:
        s += lo[0] * e[i - 1, j, k]
        ws += lo[0]
    if hi[0] > 0.0:
        s += hi[0] * e[i + 1, j, k]
        ws += hi[0]
    if lo[1] > 0.0:
        s += lo[1] * e[i, j - 1, k]
        ws += lo[1]
    if hi[1] > 0.0:
        s += hi[1] * e[i, j + 1, k]
        ws += hi[1]
    if lo[2] > 0.0:
        s += lo[2] * e[i, j, k - 1]
        ws += lo[2]
    if hi[2] > 0.0:
        s += hi[2] * e[i, j, k + 1]
        ws += hi[2]
    return s, ws


@wp.kernel
def k_mg_smooth(mask: wp.array3d(dtype=int), wlo: wp.array3d(dtype=wp.vec3), whi: wp.array3d(dtype=wp.vec3),
                e: wp.array3d(dtype=float), r: wp.array3d(dtype=float), h2: float,
                ctrl: wp.array(dtype=float), color: int):
    """One red-black Gauss-Seidel half-sweep of the coarse correction equation lap_H e = r."""
    i, j, kk = wp.tid()
    if ctrl[CTRL_TIME_SCALE] == 0.0:
        return
    k = 2 * kk + ((i + j + color) & 1)
    if (mask[i, j, k] & FLUID) == 0:
        return
    s, ws = mg_neighbours(wlo, whi, e, i, j, k)
    if ws > 0.0:
        e[i, j, k] = (s - h2 * r[i, j, k]) / ws


@wp.kernel
def k_mg_restrict(mask: wp.array3d(dtype=int), wlo: wp.array3d(dtype=wp.vec3), whi: wp.array3d(dtype=wp.vec3),
                  e: wp.array3d(dtype=float), r: wp.array3d(dtype=float), inv_h2: float,
                  ctrl: wp.array(dtype=float), rc: wp.array3d(dtype=float)):
    """Residual r - lap_H e of a coarse level, volume-averaged over the 8 children of every cell
    of the next level (solid children count as zero)."""
    ci, cj, ck = wp.tid()
    if ctrl[CTRL_TIME_SCALE] == 0.0:
        return
    total = float(0.0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = 2 * ci + di
                j = 2 * cj + dj
                k = 2 * ck + dk
                if mask[i, j, k] & FLUID:
                    s, ws = mg_neighbours(wlo, whi, e, i, j, k)
                    total += r[i, j, k] - (s - ws * e[i, j, k]) * inv_h2
    rc[ci, cj, ck] = 0.125 * total


@wp.func
def prolong_corner(mask: wp.array3d(dtype=int), e: wp.array3d(dtype=float), i: int, j: int, k: int,
                   w: float, s: float, ws: float):
    if mask[i, j, k] & FLUID:
        s += w * e[i, j, k]
        ws += w
    return s, ws


@wp.kernel
def k_mg_prolong(mask_f: wp.array3d(dtype=int), e_f: wp.array3d(dtype=float), dx_f: float,
                 mask_c: wp.array3d(dtype=int), e_c: wp.array3d(dtype=float), n_c: int, inv_dx_c: float,
                 origin: wp.vec3, ctrl: wp.array(dtype=float)):
    """Add the mask-weighted trilinear interpolation of the coarse correction to the finer level
    (the fine level's unknown is p itself)."""
    i, j, k = wp.tid()
    if ctrl[CTRL_TIME_SCALE] == 0.0:
        return
    if (mask_f[i, j, k] & FLUID) == 0:
        return
    x = cell_center(origin, dx_f, i, j, k)
    ci, cj, ck, f = lattice(n_c, origin, inv_dx_c, x, wp.vec3(0.5, 0.5, 0.5))
    s = float(0.0)
    ws = float(0.0)
    s, ws = prolong_corner(mask_c, e_c, ci, cj, ck, (1.0 - f[0]) * (1.0 - f[1]) * (1.0 - f[2]), s, ws)
    s, ws = prolong_corner(mask_c, e_c, ci + 1, cj, ck, f[0] * (1.0 - f[1]) * (1.0 - f[2]), s, ws)
    s, ws = prolong_corner(mask_c, e_c, ci, cj + 1, ck, (1.0 - f[0]) * f[1] * (1.0 - f[2]), s, ws)
    s, ws = prolong_corner(mask_c, e_c, ci + 1, cj + 1, ck, f[0] * f[1] * (1.0 - f[2]), s, ws)
    s, ws = prolong_corner(mask_c, e_c, ci, cj, ck + 1, (1.0 - f[0]) * (1.0 - f[1]) * f[2], s, ws)
    s, ws = prolong_corner(mask_c, e_c, ci + 1, cj, ck + 1, f[0] * (1.0 - f[1]) * f[2], s, ws)
    s, ws = prolong_corner(mask_c, e_c, ci, cj + 1, ck + 1, (1.0 - f[0]) * f[1] * f[2], s, ws)
    s, ws = prolong_corner(mask_c, e_c, ci + 1, cj + 1, ck + 1, f[0] * f[1] * f[2], s, ws)
    if ws > 0.0:
        e_f[i, j, k] = e_f[i, j, k] + s / ws


@wp.kernel
def k_project(g: GasGrid, p: wp.array3d(dtype=float), ctrl: wp.array(dtype=float)):
    """u -= grad q on interior faces; every other face is a no-slip wall and is zeroed."""
    i, j, k = wp.tid()
    if ctrl[CTRL_TIME_SCALE] == 0.0:
        return
    c = g.cell[i, j, k]
    u = wp.vec3(0.0, 0.0, 0.0)
    if c & FLUID:
        nbf = (c >> NBF_SHIFT) & 63
        q = p[i, j, k]
        u = g.u[i, j, k]
        if nbf & MX:
            u[0] -= (q - p[i - 1, j, k]) * g.inv_dx
        else:
            u[0] = 0.0
        if nbf & MY:
            u[1] -= (q - p[i, j - 1, k]) * g.inv_dx
        else:
            u[1] = 0.0
        if nbf & MZ:
            u[2] -= (q - p[i, j, k - 1]) * g.inv_dx
        else:
            u[2] = 0.0
    g.u[i, j, k] = u


@wp.kernel
def k_pack_volume(g: GasGrid, emis: wp.array3d(dtype=wp.int64), vol: wp.array3d(dtype=wp.vec4h),
                  vol_corner: wp.vec3, dxv: float):
    """Resample {T - T0 [K], emission density / EMIS_UNIT, |u| [m/s], fluid mask} into the render
    volume, stored in (z, y, x) memory order (x fastest) like publish.k_pack_volume and the
    glTexSubImage3D upload: vol[k, j, i] is the cell at x = corner.x + (i + 1/2) dxv. Threads run
    along x so the vec4h stores coalesce; the gas-grid reads are then strided (measured 0.065 ms
    at 96^3 against 0.023 ms for a same-order copy)."""
    k, j, i = wp.tid()
    x = vol_corner + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dxv
    t, mask = temperature(g.T, g.cell, g.n, g.origin, g.inv_dx, x)
    e = fixed_scalar(emis, g.n, g.origin, g.inv_dx, x) / EMIS_UNIT
    speed = wp.length(velocity(g.u, g.n, g.origin, g.inv_dx, x))
    vol[k, j, i] = wp.vec4h(wp.float16(t - T0), wp.float16(e), wp.float16(speed), wp.float16(mask))


class GasControl:
    """Device parameter array read by every kernel (nothing per-frame is baked into the graph).

    ``set(...)`` updates the pinned host mirror and issues the copy on the current stream. Outside
    a capture it first waits (through an event) until the previous copy has consumed the mirror,
    so two consecutive set() calls in an unsynchronised replay loop both reach the device in
    stream order; a value becomes visible to the frame the GPU executes after the copy, not to
    frames that were already enqueued. Inside a capture the copy is a memcpy node that reads the
    live mirror at every replay; the integrator may instead ``wp.copy`` a slice of its own pinned
    params array into ``array`` inside its frame graph.
    """

    NAMES = {"time_scale": CTRL_TIME_SCALE, "g_sign": CTRL_G_SIGN, "ice": CTRL_ICE,
             "h_ext": CTRL_H_EXT, "q_scale": CTRL_Q_SCALE, "t_amb": CTRL_T_AMB, "ice_dt": CTRL_ICE_DT}

    def __init__(self, device):
        self.device = wp.get_device(device)
        self.host = wp.zeros(CTRL_SIZE, dtype=float, device="cpu", pinned=True)
        self.array = wp.zeros(CTRL_SIZE, dtype=float, device=self.device)
        self.event = wp.Event(self.device) if self.device.is_cuda else None
        self.pending = False
        self.set(time_scale=1.0, g_sign=1.0, ice=0.0, h_ext=H_EXT, q_scale=1.0, t_amb=T0, ice_dt=ICE_DT)

    def set(self, **values):
        capturing = self.device.is_cuda and wp.get_stream(self.device).is_capturing
        if self.pending and not capturing:
            wp.synchronize_event(self.event)
            self.pending = False
        h = self.host.numpy()
        for name, value in values.items():
            h[self.NAMES[name]] = value
        wp.copy(self.array, self.host)
        if self.event is not None and not capturing:
            wp.record_event(self.event)
            self.pending = True


class GasSources:
    """Device list of heat segments {p0, p1, q' (W/m), e} with a device count (<= capacity).

    ``e`` is the per-metre weight of the ambient-emission splat (renderer units; it is not scaled
    by the heat knob CTRL_Q_SCALE). ``set`` defaults it to q', which makes the volume's emission
    channel a heat density in kW/m^3.
    """

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.p0 = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.p1 = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.q = wp.zeros(capacity, dtype=float, device=device)
        self.e = wp.zeros(capacity, dtype=float, device=device)
        self.count = wp.zeros(1, dtype=int, device=device)

    def set(self, p0, p1, q, e=None):
        """Host upload (tests and calibration; not for use inside a capture)."""
        n = len(q)
        if n > self.capacity:
            raise ValueError(f"{n} segments exceed the capacity {self.capacity}")
        device = self.q.device
        if n:
            e = q if e is None else e
            wp.copy(self.p0, wp.array(np.asarray(p0, np.float32), dtype=wp.vec3, device=device), count=n)
            wp.copy(self.p1, wp.array(np.asarray(p1, np.float32), dtype=wp.vec3, device=device), count=n)
            wp.copy(self.q, wp.array(np.asarray(q, np.float32), dtype=float, device=device), count=n)
            wp.copy(self.e, wp.array(np.asarray(e, np.float32), dtype=float, device=device), count=n)
        self.count.fill_(n)

    def clear(self):
        self.count.zero_()


def line_segments(start, end, qprime, count=24):
    """Split a straight line source into count segments of linear power qprime (W/m)."""
    start = np.asarray(start, np.float64)
    end = np.asarray(end, np.float64)
    s = np.linspace(0.0, 1.0, count + 1)[:, None]
    points = start + s * (end - start)
    return points[:-1], points[1:], np.full(count, qprime)


def synthetic_line_source(qprime=1.0, x=0.03, y0=-0.03, y1=0.03, z=0.0, count=24):
    """The M3 test source: a vertical 6 cm line at x = 3 cm, comfortably inside the annulus."""
    return line_segments((x, y0, z), (x, y1, z), qprime, count)


class GasState:
    """One side of the ping-pong pair: MAC velocity u (m/s), temperature T (K) and the scaled
    pressure p = dt / rho0 * pressure (m^2/s); ``grid`` is the kernel-side view."""

    def __init__(self, solver):
        self.solver = solver
        n = solver.n
        self.u = wp.zeros((n, n, n), dtype=wp.vec3, device=solver.device)
        self.T = wp.full((n, n, n), T0, dtype=float, device=solver.device)
        self.p = wp.zeros((n, n, n), dtype=float, device=solver.device)
        self.grid = solver.make_grid(self.u, self.T)

    def swap(self, other):
        """Exchange the storage of two states (uncaptured stepping loops)."""
        self.u, other.u = other.u, self.u
        self.T, other.T = other.T, self.T
        self.p, other.p = other.p, self.p
        self.grid, other.grid = other.grid, self.grid

    def assign(self, other):
        """Copy the fields of other into self; a valid graph node sequence."""
        wp.copy(self.u, other.u)
        wp.copy(self.T, other.T)
        wp.copy(self.p, other.p)

    def reset(self):
        """Rest state; for a bit-identical replay use ``GasSolver.reset`` (it also clears the
        reduction slots the warm-started pressure sweeps read)."""
        self.u.zero_()
        self.T.fill_(T0)
        self.p.zero_()


class MGLevel:
    """One coarse multigrid level: n^3 cells of size dx, fluid mask, face conductances (fraction of
    open fine faces across each coarse face, minus and plus side), correction e and residual r."""

    def __init__(self, n, dx, mask, wlo, whi, device):
        self.n = n
        self.dx = dx
        self.h2 = dx * dx
        self.inv_dx = 1.0 / dx
        with wp.ScopedDevice(device):
            self.mask = wp.array(mask, dtype=int)
            self.wlo = wp.array(wlo, dtype=wp.vec3)
            self.whi = wp.array(whi, dtype=wp.vec3)
            self.e = wp.zeros((n, n, n), dtype=float)
            self.r = wp.zeros((n, n, n), dtype=float)


class GasSolver:
    """Boussinesq gas solver on an n^3 grid over a cube of side extent centred on the globe.

    The pressure solve runs ``pressure_sweeps // 2`` fine red-black sweeps, then ``mg_cycles`` times
    {one V-cycle over ``mg_levels`` coarse levels (``mg_sweeps`` pre/post smoothing,
    ``mg_bottom_sweeps`` on the coarsest), ``pressure_sweeps - pressure_sweeps // 2`` fine sweeps};
    ``mg_levels = 0`` is the plan's pure Gauss-Seidel baseline (use ``pressure_sweeps = GS_ONLY_SWEEPS``).
    """

    def __init__(self, res=RES, extent=EXTENT, device="cuda:0", r1=R1, r2=R2I, seg_max=SEG_MAX,
                 diff_sweeps=DIFF_SWEEPS, pressure_sweeps=PRESSURE_SWEEPS, mg_levels=MG_LEVELS,
                 mg_sweeps=MG_SWEEPS, mg_bottom_sweeps=MG_BOTTOM_SWEEPS, mg_cycles=MG_CYCLES):
        wp.init()
        if res % 8 != 0 or res < 16:
            raise ValueError("res must be a multiple of 8 (block-level reductions), got %d" % res)
        self.device = wp.get_device(device)
        self.n = res
        self.extent = extent
        self.dx = extent / res
        self.origin = -0.5 * extent
        self.r1 = r1
        self.r2 = r2
        self.seg_max = seg_max
        self.diff_sweeps = diff_sweeps
        self.pressure_sweeps = pressure_sweeps
        self.mg_levels = mg_levels
        self.mg_sweeps = mg_sweeps
        self.mg_bottom_sweeps = mg_bottom_sweeps
        self.mg_cycles = mg_cycles if mg_levels > 0 else 1
        self.sigma_q = SIGMA_Q_CELLS * self.dx
        self.sigma_e = EMIS_SIGMA
        self.r_box = max(STAMP_SIGMAS * self.sigma_q, EMIS_SIGMAS * self.sigma_e)
        self.slabs = int(math.ceil(2.0 * self.r_box / self.dx)) + 2
        self.allocated = False

    def make_grid(self, u, T):
        g = GasGrid()
        g.n = self.n
        g.dx = self.dx
        g.inv_dx = 1.0 / self.dx
        g.origin = wp.vec3(self.origin, self.origin, self.origin)
        g.r1 = self.r1
        g.r2 = self.r2
        g.u = u
        g.T = T
        g.cell = self.cell
        return g

    def allocate(self):
        """Build the cell classification, the multigrid hierarchy and every fixed-capacity array
        (nothing allocates later)."""
        n, dx = self.n, self.dx
        c = self.origin + (np.arange(n) + 0.5) * dx
        x, y, z = np.meshgrid(c, c, c, indexing="ij")
        r = np.sqrt(x * x + y * y + z * z)
        fluid = (r >= self.r1) & (r <= self.r2)
        electrode = r < self.r1
        glass = r > self.r2
        if fluid[0].any() or fluid[-1].any() or fluid[:, 0].any() or fluid[:, -1].any() \
                or fluid[:, :, 0].any() or fluid[:, :, -1].any():
            raise ValueError("the fluid annulus touches the grid boundary; enlarge extent")

        def neighbour(mask, axis, sign):
            out = np.zeros_like(mask)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            src[axis] = slice(1, None) if sign > 0 else slice(None, -1)
            dst[axis] = slice(None, -1) if sign > 0 else slice(1, None)
            out[tuple(dst)] = mask[tuple(src)]
            return out

        dirs = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]
        nbf = np.zeros((n, n, n), np.int32)
        nbt = np.zeros((n, n, n), np.int32)
        n_glass_faces = np.zeros((n, n, n), np.int32)
        n_electrode_faces = np.zeros((n, n, n), np.int32)
        for bit, (axis, sign) in enumerate(dirs):
            nbf |= neighbour(fluid, axis, sign).astype(np.int32) << bit
            nbt |= neighbour(fluid | electrode, axis, sign).astype(np.int32) << bit
            n_glass_faces += neighbour(glass, axis, sign)
            n_electrode_faces += neighbour(electrode, axis, sign)
        cell = (fluid * FLUID | glass * GLASS | electrode * ELECTRODE).astype(np.int32)
        cell |= (nbf << NBF_SHIFT) | (nbt << NBT_SHIFT)
        n_glass_faces = np.where(fluid, n_glass_faces, 0)
        # Glass area per boundary cell, scaled so the total conductance is exactly h_ext 4 pi R2^2
        # (a voxelised sphere over-counts its area by ~1.5 through the staircase).
        area_scale = 4.0 * math.pi * self.r2 ** 2 / (n_glass_faces.sum() * dx * dx)
        wall = (n_glass_faces * dx * dx * area_scale).astype(np.float32)

        self.fluid = fluid
        self.boundary = n_glass_faces > 0
        self.electrode_faces = np.where(fluid, n_electrode_faces, 0)
        self.centers = np.stack([x, y, z], axis=-1).astype(np.float32)
        self.n_fluid = int(fluid.sum())
        self.area_scale = area_scale
        with wp.ScopedDevice(self.device):
            self.cell = wp.array(cell, dtype=int)
            self.wall = wp.array(wall, dtype=float)
            self.src = wp.zeros((n, n, n), dtype=wp.int64)
            self.emis = wp.zeros((n, n, n), dtype=wp.int64)
            self.div = wp.zeros((n, n, n), dtype=float)
            self.u_rhs = wp.zeros((n, n, n), dtype=wp.vec3)
            self.T_rhs = wp.zeros((n, n, n), dtype=float)
            self.acc = wp.zeros(ACC_SIZE, dtype=wp.int64)
            self.div_sum = self.acc[ACC_DIV:ACC_DIV + 1]
            self.sources = GasSources(self.seg_max, self.device)
        self.levels = self.build_levels(fluid)
        self.allocated = True
        return self

    def build_levels(self, fluid):
        """Coarse levels from the fine mask: a coarse cell is fluid when any child is; the
        conductance of a coarse face is the fraction of open (fluid-fluid) fine faces across it,
        so every coarse operator is the finite-volume restriction of the fine stencil."""
        n = self.n
        open_faces = []
        for axis in range(3):
            o = np.zeros((n, n, n), np.float64)
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            o[tuple(hi)] = fluid[tuple(lo)] & fluid[tuple(hi)]
            open_faces.append(o)
        levels = []
        nl, f = n, 1
        for _ in range(self.mg_levels):
            if nl % 2 or nl // 2 < 4:
                break
            nl //= 2
            f *= 2
            mask = fluid.reshape(nl, f, nl, f, nl, f).any(axis=(1, 3, 5))
            wx = open_faces[0][::f].reshape(nl, nl, f, nl, f).sum(axis=(2, 4))
            wy = open_faces[1][:, ::f].reshape(nl, f, nl, nl, f).sum(axis=(1, 4))
            wz = open_faces[2][:, :, ::f].reshape(nl, f, nl, f, nl).sum(axis=(1, 3))
            wlo = np.stack([wx, wy, wz], axis=-1) / (f * f)
            whi = np.zeros_like(wlo)
            whi[:-1, :, :, 0] = wlo[1:, :, :, 0]
            whi[:, :-1, :, 1] = wlo[:, 1:, :, 1]
            whi[:, :, :-1, 2] = wlo[:, :, 1:, 2]
            levels.append(MGLevel(nl, self.dx * f, (mask * FLUID).astype(np.int32), wlo.astype(np.float32),
                                  whi.astype(np.float32), self.device))
        return levels

    def state(self):
        if not self.allocated:
            raise RuntimeError("call allocate() first")
        return GasState(self)

    def control(self):
        if not self.allocated:
            raise RuntimeError("call allocate() first")
        return GasControl(self.device)

    def reset(self, *states):
        """Rest state for the given states plus the solver's reduction slots (capturable); a run
        from here replays bit-identically."""
        for s in states:
            s.reset()
        self.acc.zero_()

    def step(self, state_in, state_out, control, dt=DT):
        """One Boussinesq sub-step state_in -> state_out (graph-capturable, allocation-free)."""
        with wp.ScopedDevice(self.device):
            for stage in self.stages(state_in, state_out, control, dt).values():
                stage()

    def stages(self, state_in, state_out, control, dt=DT):
        """The ordered launch groups of one step, as closures (step() runs them; profile() times
        each one in its own graph)."""
        n = self.n
        gin, gout = state_in.grid, state_out.grid
        ctrl = control.array
        seg = self.sources

        def deposit():
            self.src.zero_()
            self.emis.zero_()
            wp.launch(k_deposit, dim=(seg.capacity, self.slabs),
                      inputs=[gin, self.src, self.emis, seg.p0, seg.p1, seg.q, seg.e, seg.count, ctrl,
                              self.sigma_q, self.sigma_e, self.r_box, self.slabs])

        def advect():
            wp.launch(k_advect, dim=(n, n, n),
                      inputs=[gin, gout, self.src, self.u_rhs, self.T_rhs, ctrl, dt])

        def diffuse():
            for _ in range(self.diff_sweeps):
                for color in (0, 1):
                    wp.launch(k_diffuse, dim=(n, n, n // 2),
                              inputs=[gout, self.u_rhs, self.T_rhs, self.wall, ctrl, dt, color])

        def buoyancy():
            wp.launch(k_buoyancy, dim=(n, n, n), inputs=[gout, ctrl, dt])
            self.div_sum.zero_()
            wp.launch(k_divergence, dim=(n, n, n), inputs=[gout, self.div, self.acc], block_dim=BLOCK_DIM)

        def pressure():
            wp.copy(state_out.p, state_in.p)
            sweep = 0
            pre = self.pressure_sweeps // 2
            for cycle in range(self.mg_cycles):
                for _ in range(pre if cycle == 0 else 0):
                    sweep = self.fine_sweep(gout, state_out.p, ctrl, sweep)
                self.v_cycle(gout, state_out.p, ctrl)
                for _ in range(self.pressure_sweeps - pre):
                    sweep = self.fine_sweep(gout, state_out.p, ctrl, sweep)

        def project():
            wp.launch(k_project, dim=(n, n, n), inputs=[gout, state_out.p, ctrl])

        return {"deposit": deposit, "advect": advect, "diffuse": diffuse, "buoyancy+div": buoyancy,
                "pressure": pressure, "project": project}

    def fine_sweep(self, g, p, ctrl, sweep):
        """One complete (red + black) fine Gauss-Seidel sweep; returns the next sweep index."""
        n = self.n
        for color in (0, 1):
            wp.launch(k_pressure, dim=(n, n, n // 2),
                      inputs=[g, p, self.div, self.acc, 1.0 / self.n_fluid, ctrl, color, sweep % 2],
                      block_dim=BLOCK_DIM)
        return sweep + 1

    def coarse_sweeps(self, level, ctrl, count):
        for _ in range(count):
            for color in (0, 1):
                wp.launch(k_mg_smooth, dim=(level.n, level.n, level.n // 2),
                          inputs=[level.mask, level.wlo, level.whi, level.e, level.r, level.h2, ctrl, color])

    def v_cycle(self, g, p, ctrl):
        """One multigrid V-cycle correction of p (no-op when there are no coarse levels)."""
        levels = self.levels
        origin = wp.vec3(self.origin, self.origin, self.origin)
        for l, level in enumerate(levels):
            level.e.zero_()
            if l == 0:
                wp.launch(k_mg_restrict_fine, dim=(level.n,) * 3,
                          inputs=[g, p, self.div, self.acc, 1.0 / self.n_fluid, ctrl, level.r])
            else:
                prev = levels[l - 1]
                wp.launch(k_mg_restrict, dim=(level.n,) * 3,
                          inputs=[prev.mask, prev.wlo, prev.whi, prev.e, prev.r, 1.0 / prev.h2, ctrl, level.r])
            self.coarse_sweeps(level, ctrl, self.mg_bottom_sweeps if l == len(levels) - 1 else self.mg_sweeps)
        for l in range(len(levels) - 1, -1, -1):
            level = levels[l]
            if l == 0:
                wp.launch(k_mg_prolong, dim=(self.n,) * 3,
                          inputs=[self.cell, p, self.dx, level.mask, level.e, level.n, level.inv_dx, origin, ctrl])
            else:
                prev = levels[l - 1]
                wp.launch(k_mg_prolong, dim=(prev.n,) * 3,
                          inputs=[prev.mask, prev.e, prev.dx, level.mask, level.e, level.n, level.inv_dx, origin, ctrl])
                self.coarse_sweeps(prev, ctrl, self.mg_sweeps)

    def capture(self, state_in, state_out, control, dt=DT, copy_back=True):
        """Graph of one step; with copy_back the result lands back in state_in so the graph can
        be replayed every frame and state_in is always the current state."""
        with wp.ScopedCapture(device=self.device) as capture:
            self.step(state_in, state_out, control, dt)
            if copy_back:
                state_in.assign(state_out)
        return capture.graph

    def pack_volume(self, state, vol, extent=VOL_EXTENT):
        """Resample the gas fields into vol, an array3d of vec4h over the cube of side extent
        centred on the globe, in (z, y, x) memory order (vol[k, j, i], x fastest)."""
        res = vol.shape
        vol_corner = wp.vec3(-0.5 * extent, -0.5 * extent, -0.5 * extent)
        with wp.ScopedDevice(self.device):
            wp.launch(k_pack_volume, dim=res, inputs=[state.grid, self.emis, vol, vol_corner, extent / res[0]])

    # Host-side diagnostics (synchronising; tests and calibration only)

    def divergence_stats(self, state):
        """max |div u| (1/s), max |u| (m/s) and the dimensionless max |div u| dx / max |u|."""
        acc = wp.zeros(ACC_SIZE, dtype=wp.int64, device=self.device)
        div = wp.zeros((self.n,) * 3, dtype=float, device=self.device)
        wp.launch(k_divergence, dim=(self.n,) * 3, inputs=[state.grid, div, acc], block_dim=BLOCK_DIM,
                  device=self.device)
        d = np.abs(div.numpy()[self.fluid]).max()
        speed = self.max_speed(state)
        return d, speed, d * self.dx / max(speed, 1e-12)

    def max_speed(self, state):
        u = state.u.numpy()
        return float(np.sqrt((u * u).sum(-1)).max())

    def fields(self, state):
        """(T, u) as numpy arrays after a device sync."""
        wp.synchronize_device(self.device)
        return state.T.numpy(), state.u.numpy()

    def source_field(self):
        """The last deposited heat field in W/m^3 (numpy)."""
        return self.src.numpy().astype(np.float64) * FIXED_SRC_INV

    def emission_field(self):
        """The last deposited emission density (e per m^3, numpy)."""
        return self.emis.numpy().astype(np.float64) * FIXED_SRC_INV

    def pressure_nodes(self):
        """Symbolic list of the pressure stage's graph nodes (for reports)."""
        pre = self.pressure_sweeps // 2
        nodes = ["copy p"] + ["gs"] * 2 * (pre + self.mg_cycles * (self.pressure_sweeps - pre))
        for _ in range(self.mg_cycles):
            for l, level in enumerate(self.levels):
                nodes += ["memset", "restrict"] + ["smooth"] * 2 * (
                    self.mg_bottom_sweeps if l == len(self.levels) - 1 else self.mg_sweeps)
            for l in range(len(self.levels) - 1, -1, -1):
                nodes += ["prolong"] + (["smooth"] * 2 * self.mg_sweeps if l > 0 else [])
        return nodes


# ----------------------------------------------------------------------------------------------
# Self-tests (plan 9 / M3, validation matrix 10)
# ----------------------------------------------------------------------------------------------

class Runner:
    """Steps a solver frame by frame through the captured graph (or eagerly on the CPU)."""

    def __init__(self, solver, control):
        self.solver = solver
        self.control = control
        self.s0 = solver.state()
        self.s1 = solver.state()
        self.graph = solver.capture(self.s0, self.s1, control) if solver.device.is_cuda else None
        self.frame = 0

    def run(self, frames):
        if self.graph is not None:
            with wp.ScopedDevice(self.solver.device):
                for _ in range(frames):
                    wp.capture_launch(self.graph)
        else:
            for _ in range(frames):
                self.solver.step(self.s0, self.s1, self.control)
                self.s0.assign(self.s1)
        self.frame += frames

    def reset(self):
        self.solver.reset(self.s0, self.s1)
        self.frame = 0

    def snapshot(self):
        T, u = self.solver.fields(self.s0)
        return T, u

    def arrays(self):
        wp.synchronize_device(self.solver.device)
        return self.s0.T.numpy(), self.s0.u.numpy(), self.s0.p.numpy()


def plume_metrics(solver, T, u):
    fluid = solver.fluid
    dT = np.where(fluid, T - T0, 0.0)
    hot = np.clip(dT, 0.0, None)
    total = max(hot.sum(), 1e-12)
    centroid = (hot[..., None] * solver.centers).sum((0, 1, 2)) / total
    lead = dT > 0.25 * dT.max()
    uy = np.where(fluid, u[..., 1], 0.0)
    return dict(dT_max=float(dT.max()), centroid=centroid,
                front=float(solver.centers[..., 1][lead].max()), back=float(solver.centers[..., 1][lead].min()),
                uy_hot=float((hot * uy).sum() / total), uy_max=float(uy.max()), uy_min=float(uy.min()),
                speed_max=float(np.sqrt((u * u).sum(-1))[fluid].max()))


def band(value, lo=0.01, hi=0.07):
    return "PASS" if lo <= value <= hi else "FAIL"


def make_solver(args, **kw):
    """GasSolver for the tests: --res, --device and the pressure-solver knobs from the CLI."""
    return GasSolver(res=args.res, device=args.device, pressure_sweeps=args.sweeps, mg_levels=args.mg_levels,
                     mg_cycles=args.mg_cycles, **kw).allocate()


def test_plume(args, qprime, g_sign=1.0, ice=0.0, x=0.03, y0=-0.03, y1=0.03, label=None):
    """Synthetic 6 cm vertical line at q' W/m from rest.

    Reports the leading-edge speed (cells above 25 % of the peak dT) while the front travels, the
    temperature-weighted centroid speed over the same window (the brief's preferred metric; the T
    maximum sits on the fixed source and never moves), the steady temperature-weighted mean
    vertical velocity <u_y>_hot, the steady peak u_y and the time to 90 %.
    """
    solver = make_solver(args)
    control = solver.control()
    control.set(g_sign=g_sign, ice=ice)
    solver.sources.set(*synthetic_line_source(qprime, x=x, y0=y0, y1=y1))
    runner = Runner(solver, control)
    label = label or f"plume q'={qprime} W/m g_sign={g_sign:+.0f} ice={int(ice)}"
    print(f"[{label}] res {solver.n}^3 dx {solver.dx * 1e3:.2f} mm, {args.frames} frames")
    interval = 15
    history = []
    for f in range(0, args.frames, interval):
        runner.run(interval)
        T, u = runner.snapshot()
        m = plume_metrics(solver, T, u)
        m["t"] = runner.frame * DT
        m["div"] = solver.divergence_stats(runner.s0)[2]
        history.append(m)
        if args.verbose or f % (interval * 20) == 0:
            print(f"  t={m['t']:6.2f}s dTmax={m['dT_max']:6.2f}K lead y={m['front'] * 100:6.2f}cm "
                  f"centroid y={m['centroid'][1] * 100:6.2f} x={m['centroid'][0] * 100:5.2f}cm "
                  f"<uy>hot={m['uy_hot'] * 100:5.2f} uy max={m['uy_max'] * 100:5.2f} min={m['uy_min'] * 100:6.2f} cm/s "
                  f"div*dx/|u|={m['div']:.2e}")
    if not np.all(np.isfinite([m["dT_max"] for m in history])):
        print("  NaN detected")
        return None
    sign = 1.0 if g_sign >= 0 else -1.0
    t = np.array([m["t"] for m in history])
    lead = np.array([m["front"] if sign > 0 else m["back"] for m in history])
    cy = np.array([m["centroid"][1] for m in history])
    # Travel window: from the first sample until the leading edge is within one cell of its
    # farthest excursion (the wall or where the plume cools below the threshold).
    farthest = lead.max() if sign > 0 else lead.min()
    arrived = np.where(sign * (farthest - lead) <= solver.dx)[0][0]
    window = slice(0, max(arrived + 1, 3))
    speeds = dict(front=sign * np.polyfit(t[window], lead[window], 1)[0],
                  centroid=sign * np.polyfit(t[window], cy[window], 1)[0],
                  window=(float(t[window][0]), float(t[window][-1])),
                  lead_travel=sign * (lead[arrived] - lead[0]))
    hot = sign * np.array([m["uy_hot"] for m in history])
    core = np.array([m["uy_max"] if sign > 0 else -m["uy_min"] for m in history])
    steady = slice(-max(1, len(t) // 5), None)
    speeds["hot_steady"] = float(hot[steady].mean())
    speeds["core_steady"] = float(core[steady].mean())
    speeds["reach_t"] = float(t[np.argmax(hot >= 0.9 * speeds["hot_steady"])])
    speeds["div_max"] = max(m["div"] for m in history[1:])
    speeds["dT_max_final"] = history[-1]["dT_max"]
    speeds["centroid_final"] = history[-1]["centroid"]
    speeds["uy_min_final"] = history[-1]["uy_min"]
    speeds["history"] = history
    print(f"  leading edge {speeds['front'] * 100:.2f} cm/s [{band(speeds['front'])}] and centroid "
          f"{speeds['centroid'] * 100:.2f} cm/s [{band(speeds['centroid'])}] over "
          f"t = {speeds['window'][0]:.2f}-{speeds['window'][1]:.2f} s ({speeds['lead_travel'] * 100:.1f} cm of travel); "
          f"steady <u_y>_hot {speeds['hot_steady'] * 100:.2f} cm/s (90 % at t = {speeds['reach_t']:.2f} s), "
          f"steady peak u_y {speeds['core_steady'] * 100:.2f} cm/s [{band(speeds['core_steady'])}]; "
          f"max div*dx/|u| after frame {interval}: {speeds['div_max']:.2e}")
    return speeds


def test_robin(args):
    """1 W total on the 6 cm line -> glass boundary layer dT vs P / (h_ext 4 pi R2^2)."""
    solver = make_solver(args)
    control = solver.control()
    power = 1.0
    solver.sources.set(*synthetic_line_source(power / 0.06))
    runner = Runner(solver, control)
    frames = max(args.frames, 1800)
    print(f"[robin] {power} W, {frames} frames ({frames * DT:.0f} s), res {solver.n}^3")
    expected = power / (H_EXT * 4.0 * math.pi * solver.r2 ** 2)
    cap = RHO0 * CP * solver.n_fluid * solver.dx ** 3
    last = None
    for f in range(0, frames, 300):
        runner.run(300)
        T, u = runner.snapshot()
        wall_dT = float((T - T0)[solver.boundary].mean())
        mean_dT = float((T - T0)[solver.fluid].mean())
        sink = H_EXT * (solver.wall.numpy() * (T - T0)).sum()
        electrode = K_GAS * (solver.electrode_faces * (T - T0)).sum() * solver.dx
        print(f"  t={runner.frame * DT:6.1f}s glass-layer dT={wall_dT:.3f} K, fluid mean dT={mean_dT:.3f} K, "
              f"max dT={float((T - T0)[solver.fluid].max()):.2f} K, glass sink={sink:.3f} W, "
              f"electrode sink={electrode:.3f} W, stored energy={cap * mean_dT:.2f} J")
        last = dict(wall_dT=wall_dT, mean_dT=mean_dT, sink=sink, electrode=electrode)
    # Deposited power check: one frame of the source field integrated over the fluid.
    deposited = float(solver.source_field().sum() * solver.dx ** 3)
    emitted = float(solver.emission_field().sum() * solver.dx ** 3)
    print(f"  deposited {deposited:.4f} W of {power} W nominal (emission integral {emitted:.4f}); expected glass "
          f"dT = P / (h A) = {expected:.3f} K (A = 4 pi R2i^2 = {4 * math.pi * solver.r2 ** 2:.4f} m^2; the plan's "
          f"~1 K assumed R = 0.10 m); measured {last['wall_dT']:.3f} K; budget: glass {last['sink']:.3f} + "
          f"electrode {last['electrode']:.3f} = {last['sink'] + last['electrode']:.3f} W of {deposited:.3f} W deposited")
    return dict(expected=expected, measured=last["wall_dT"], deposited=deposited, sink=last["sink"],
                electrode=last["electrode"])


def divergence_run(solver, control, runner, frames, label, verbose_frames=()):
    """Step frame by frame; return the per-frame post-projection ratio max|div| dx / max|u|."""
    ratios = []
    for f in range(frames):
        runner.run(1)
        wp.synchronize_device(solver.device)
        d, speed, ratio = solver.divergence_stats(runner.s0)
        ratios.append(ratio)
        if f + 1 in verbose_frames:
            print(f"  {label} +{f + 1:3d} (frame {runner.frame:5d}): max|div|={d:.3e} 1/s max|u|={speed * 100:.2f} cm/s "
                  f"ratio={ratio:.2e}")
    return np.array(ratios)


def test_divergence(args, solver=None):
    """Post-projection max |div u| dx / max |u|: impulsive start from rest, developed flow, then
    the g_sign flip protocol (300 frames, flip, 60 frames, flip back after 300 more, 60 frames)
    and an ice toggle."""
    solver = solver or make_solver(args)
    control = solver.control()
    solver.sources.set(*synthetic_line_source(1.3))
    runner = Runner(solver, control)
    print(f"[divergence] q'=1.3 W/m, res {solver.n}^3, {len(solver.levels)} MG levels "
          f"{[lv.n for lv in solver.levels]}, {solver.pressure_sweeps} fine sweeps x {solver.mg_cycles} cycle(s), "
          f"{len(solver.pressure_nodes())} pressure nodes")
    start = divergence_run(solver, control, runner, 300, "start", verbose_frames=(1, 2, 3, 5, 10, 15, 30, 60, 150, 300))
    control.set(g_sign=-1.0)
    flip = divergence_run(solver, control, runner, 300, "flip", verbose_frames=(1, 2, 5, 10, 30, 60))
    control.set(g_sign=1.0)
    back = divergence_run(solver, control, runner, 300, "flip back", verbose_frames=(1, 2, 5, 10, 30, 60))
    control.set(ice=1.0)
    ice = divergence_run(solver, control, runner, 60, "ice on", verbose_frames=(1, 2, 5, 30, 60))
    control.set(ice=0.0)
    T, u = runner.snapshot()
    m = plume_metrics(solver, T, u)
    results = dict(start_worst=float(start.max()), start_first_ok=int(np.argmax(start < 1e-3)) + 1,
                   start_over=int((start >= 1e-3).sum()), developed_worst=float(start[30:].max()),
                   flip_worst=float(flip[:60].max()), flip_over=int((flip[:60] >= 1e-3).sum()),
                   back_worst=float(back[:60].max()), back_over=int((back[:60] >= 1e-3).sum()),
                   ice_worst=float(ice.max()), steady=float(np.median(start[150:])))
    print(f"  start: worst {results['start_worst']:.2e}, first frame < 1e-3: {results['start_first_ok']}, "
          f"frames >= 1e-3: {results['start_over']} of 300; developed (frame > 30) worst {results['developed_worst']:.2e}, "
          f"steady median {results['steady']:.2e}")
    print(f"  g_sign flip: worst {results['flip_worst']:.2e} over 60 frames ({results['flip_over']} frames >= 1e-3); "
          f"flip back: worst {results['back_worst']:.2e} ({results['back_over']} frames >= 1e-3); "
          f"ice toggle: worst {results['ice_worst']:.2e}")
    print(f"  after 960 frames: dT max {m['dT_max']:.2f} K, <u_y>_hot {m['uy_hot'] * 100:.2f} cm/s (upright again), "
          f"finite={bool(np.isfinite(T).all() and np.isfinite(u).all())}")
    worst = max(results["start_worst"], results["flip_worst"], results["back_worst"], results["ice_worst"])
    print(f"  worst ratio overall {worst:.2e} (threshold 1e-3): {'PASS' if worst < 1e-3 else 'FAIL'}")
    results["worst"] = worst
    return results


def test_stability(args):
    """Implicit diffusion only (no buoyancy, no source): a hot Gaussian blob must decay
    monotonically and spread with <r^2> growing at 6 alpha (3D); the default sweep count is
    compared with a better-converged 8-sweep solve."""
    results = {}
    for sweeps in (DIFF_SWEEPS, 8):
        solver = make_solver(args, diff_sweeps=sweeps)
        control = solver.control()
        control.set(g_sign=0.0)
        runner = Runner(solver, control)
        c = np.array([0.045, 0.0, 0.0], np.float32)
        sig0 = 2.0 * solver.dx
        r2 = ((solver.centers - c) ** 2).sum(-1)
        T = T0 + 50.0 * np.exp(-0.5 * r2 / sig0 ** 2)
        T = np.where(solver.fluid, T, T0).astype(np.float32)
        runner.s0.T.assign(T)
        a = ALPHA * DT / solver.dx ** 2
        maxes, moments, times = [], [], []
        for f in range(0, 120, 6):
            T, u = runner.snapshot()
            dT = np.where(solver.fluid, T - T0, 0.0)
            maxes.append(dT.max())
            moments.append((dT * r2).sum() / dT.sum())
            times.append(runner.frame * DT)
            runner.run(6)
        maxes = np.array(maxes)
        slope = np.polyfit(times, moments, 1)[0]
        T, u = runner.snapshot()
        finite = bool(np.isfinite(T).all() and np.isfinite(u).all())
        monotone = bool(np.all(np.diff(maxes) <= 1e-4))
        print(f"[stability] res {solver.n}^3, {sweeps} GS sweeps: alpha dt/dx^2 = {a:.3f} (explicit limit 1/6), "
              f"nu dt/dx^2 = {NU * DT / solver.dx ** 2:.3f}; max dT {maxes[0]:.2f} -> {maxes[-1]:.2f} K "
              f"monotone={monotone}, min dT {float((T - T0)[solver.fluid].min()):.1e} K, finite={finite}, "
              f"d<r^2>/dt = {slope:.3e} vs 6 alpha = {6 * ALPHA:.3e} m^2/s (ratio {slope / (6 * ALPHA):.3f})")
        results[sweeps] = dict(monotone=monotone, finite=finite, ratio=slope / (6 * ALPHA))
    return results


def test_nan(args):
    """10k frames with the 1.3 W/m plume; report finiteness and extremes."""
    solver = make_solver(args)
    control = solver.control()
    solver.sources.set(*synthetic_line_source(1.3))
    runner = Runner(solver, control)
    frames = max(args.frames, 10000)
    print(f"[nan] {frames} frames at res {solver.n}^3, q'=1.3 W/m")
    ok = True
    t0 = time.perf_counter()
    for f in range(0, frames, 1000):
        runner.run(1000)
        T, u = runner.snapshot()
        finite = bool(np.isfinite(T).all() and np.isfinite(u).all())
        ok &= finite
        print(f"  frame {runner.frame}: finite={finite} max dT={float((T - T0)[solver.fluid].max()):.2f} K "
              f"max|u|={solver.max_speed(runner.s0) * 100:.2f} cm/s")
    print(f"  {frames} frames NaN-free: {ok} ({time.perf_counter() - t0:.1f} s wall)")
    return ok


def test_determinism(args):
    """Two fresh 600-frame runs and a reset + replay of the same graph must agree bit for bit
    (T, u, p), including with the ice cap and inverted gravity toggled mid-run."""
    frames = max(args.frames, 600)

    def make():
        solver = make_solver(args)
        control = solver.control()
        solver.sources.set(*synthetic_line_source(1.3))
        return solver, control, Runner(solver, control)

    def run(runner, control):
        runner.run(frames // 3)
        control.set(ice=1.0)
        runner.run(frames // 3)
        control.set(g_sign=-1.0)
        runner.run(frames - 2 * (frames // 3))
        control.set(ice=0.0, g_sign=1.0)
        return runner.arrays()

    solver_a, control_a, runner_a = make()
    ref = run(runner_a, control_a)
    runner_a.reset()
    replay = run(runner_a, control_a)
    solver_b, control_b, runner_b = make()
    fresh = run(runner_b, control_b)
    names = ("T", "u", "p")
    same_replay = [bool(np.array_equal(a, b)) for a, b in zip(ref, replay)]
    same_fresh = [bool(np.array_equal(a, b)) for a, b in zip(ref, fresh)]
    diffs = [float(np.abs(a.astype(np.float64) - b).max()) for a, b in zip(ref, fresh)]
    print(f"[determinism] res {solver_a.n}^3, {frames} frames (ice on at {frames // 3}, inverted at {2 * (frames // 3)})")
    print(f"  reset + replay of the same graph bit-identical: {dict(zip(names, same_replay))}")
    print(f"  fresh solver bit-identical: {dict(zip(names, same_fresh))} (max abs diff {dict(zip(names, diffs))})")
    print(f"  max dT {float((ref[0] - T0)[solver_a.fluid].max()):.2f} K, max|u| {float(np.sqrt((ref[1] ** 2).sum(-1)).max()) * 100:.2f} cm/s")
    ok = all(same_replay) and all(same_fresh)
    print(f"  determinism: {'PASS' if ok else 'FAIL'}")
    return ok


def test_pause(args):
    """time_scale = 0 must be an exact no-op over a long pause (T, u, p bit-identical after 1000
    frozen frames on a developed flow), the flow must resume, and two control changes issued
    without a synchronisation in between must both reach the device in order."""
    solver = make_solver(args)
    control = solver.control()
    solver.sources.set(*synthetic_line_source(1.3))
    runner = Runner(solver, control)
    runner.run(300)
    before = runner.arrays()
    m0 = plume_metrics(solver, before[0], before[1])
    control.set(time_scale=0.0)
    runner.run(1000)
    after = runner.arrays()
    same = [bool(np.array_equal(a, b)) for a, b in zip(before, after)]
    diffs = [float(np.abs(a.astype(np.float64) - b).max()) for a, b in zip(before, after)]
    print(f"[pause] res {solver.n}^3: 1000 frozen frames after 300 live ones -> bit-identical "
          f"{dict(zip(('T', 'u', 'p'), same))} (max abs diff {dict(zip(('T', 'u', 'p'), diffs))})")
    control.set(time_scale=1.0)
    runner.run(60)
    T, u = runner.snapshot()
    m1 = plume_metrics(solver, T, u)
    print(f"  resumed: dT max {m0['dT_max']:.2f} -> {m1['dT_max']:.2f} K, <u_y>_hot {m0['uy_hot'] * 100:.2f} -> "
          f"{m1['uy_hot'] * 100:.2f} cm/s, finite={bool(np.isfinite(T).all())}")
    # Control ordering: flip, 300 unsynchronised replays, flip back; the device must have seen -1.
    control.set(g_sign=-1.0)
    runner.run(300)
    control.set(g_sign=1.0)
    T, u = runner.snapshot()
    m2 = plume_metrics(solver, T, u)
    inverted = m2["uy_hot"] < 0.0
    print(f"  control ordering: after set(g_sign=-1), 300 replays, set(g_sign=+1): <u_y>_hot = {m2['uy_hot'] * 100:.2f} cm/s "
          f"({'inverted flow seen' if inverted else 'toggle LOST'})")
    ok = all(same) and inverted
    print(f"  pause: {'PASS' if ok else 'FAIL'}")
    return ok


def test_bench(args):
    """Captured-graph cost per frame (wp.ScopedTimer(synchronize=True), median of 3 trials) and
    the render-volume layout check (hot centroid at x = +3 cm along the last array axis)."""
    solver = make_solver(args)
    control = solver.control()
    if args.segments:
        solver.sources.set(*random_segments(args.segments))
    else:
        solver.sources.set(*synthetic_line_source(1.0))
    runner = Runner(solver, control)
    vol = wp.zeros((VOL_RES,) * 3, dtype=wp.vec4h, device=solver.device)
    with wp.ScopedCapture(device=solver.device) as capture:
        solver.pack_volume(runner.s0, vol)
    pack_graph = capture.graph
    print(f"[bench] res {solver.n}^3, {len(solver.pressure_nodes())} pressure nodes, MG levels {[lv.n for lv in solver.levels]}")
    if not args.segments:
        # Layout check on a young plume (0.5 s: the hot gas still sits on the line at x = 3 cm, z = 0).
        runner.run(30)
        with wp.ScopedDevice(solver.device):
            wp.capture_launch(pack_graph)
        hot = np.clip(vol.numpy().astype(np.float32)[..., 0], 0.0, None)
        coords = (np.arange(VOL_RES) + 0.5) * VOL_EXTENT / VOL_RES - 0.5 * VOL_EXTENT
        centroid = [float((hot.sum(axis=tuple(a for a in range(3) if a != ax)) * coords).sum() / hot.sum()) for ax in range(3)]
        layout_ok = abs(centroid[2] - 0.03) < 0.005 and abs(centroid[0]) < 0.005 and abs(centroid[1]) < 0.01
        print(f"  volume layout at t = 0.5 s: hot centroid in array-index order (axis0, axis1, axis2) = "
              f"({centroid[0] * 100:.2f}, {centroid[1] * 100:.2f}, {centroid[2] * 100:.2f}) cm, expected (z, y, x) = "
              f"(0, ~0.3, 3.0): {'PASS' if layout_ok else 'FAIL'}")
        assert layout_ok, "render volume is not in (z, y, x) order"
    runner.run(60)
    results = {}
    for name, graph in (("step", runner.graph), ("pack", pack_graph)):
        times = []
        for trial in range(3):
            with wp.ScopedDevice(solver.device):
                with wp.ScopedTimer(name, synchronize=True, print=False) as timer:
                    for _ in range(200):
                        wp.capture_launch(graph)
            times.append(timer.elapsed / 200.0)
        results[name] = float(np.median(times))
        print(f"  {name} segments {args.segments}: {' '.join(f'{x:.3f}' for x in times)} ms -> median {results[name]:.3f} ms/frame")
    T, u = runner.snapshot()
    v = vol.numpy().astype(np.float32)
    print(f"  volume {VOL_RES}^3: dT [{v[..., 0].min():.2f}, {v[..., 0].max():.2f}] K, "
          f"emission max {v[..., 1].max():.3f} (e / m^3 / {EMIS_UNIT:g}), |u| max {v[..., 2].max() * 100:.2f} cm/s, "
          f"mask mean {v[..., 3].mean():.3f}")
    return results


def profile(args):
    """Per-stage device time: every launch group of one step captured in its own graph and
    replayed (wp.TIMING_KERNEL records nothing on this Warp build, so stages are timed instead)."""
    solver = make_solver(args)
    control = solver.control()
    if args.segments:
        solver.sources.set(*random_segments(args.segments))
    else:
        solver.sources.set(*synthetic_line_source(1.0))
    runner = Runner(solver, control)
    runner.run(60)
    vol = wp.zeros((VOL_RES,) * 3, dtype=wp.vec4h, device=solver.device)
    stages = solver.stages(runner.s0, runner.s1, control)
    stages["copy back"] = lambda: runner.s0.assign(runner.s1)
    stages["pack volume"] = lambda: solver.pack_volume(runner.s0, vol)
    stages["whole step"] = lambda: wp.capture_launch(runner.graph)
    print(f"[profile] res {solver.n}^3, {args.segments or 24} segments, 200 replays x 3 trials, median ms; "
          f"pressure = {len(solver.pressure_nodes())} nodes ({solver.pressure_sweeps} fine sweeps, "
          f"MG {[lv.n for lv in solver.levels]} with {solver.mg_sweeps}/{solver.mg_bottom_sweeps} sweeps)")
    total = 0.0
    for name, stage in stages.items():
        with wp.ScopedDevice(solver.device):
            if name == "whole step":
                graph = runner.graph
            else:
                with wp.ScopedCapture(device=solver.device) as capture:
                    stage()
                graph = capture.graph
            times = []
            for trial in range(3):
                with wp.ScopedTimer(name, synchronize=True, print=False) as timer:
                    for _ in range(200):
                        wp.capture_launch(graph)
                times.append(timer.elapsed / 200.0)
        ms = float(np.median(times))
        if name not in ("whole step", "pack volume"):
            total += ms
        print(f"  {name:14s} {ms:.3f} ms")
    print(f"  {'sum of stages':14s} {total:.3f} ms (stage graphs replay on a frozen state)")


def random_segments(count, seed=0, spacing=1.5e-3, power=0.75):
    """count random filament-like segments of length spacing inside the annulus sharing power W."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p0 = d * rng.uniform(R1 + 0.005, R2I - 0.005, size=(count, 1))
    t = rng.normal(size=(count, 3))
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return p0, p0 + spacing * t, np.full(count, power / (count * spacing))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gas convection solver self-tests (plan M3)")
    parser.add_argument("--test", default="all",
                        choices=["all", "plume", "robin", "divergence", "invert", "ice", "stability", "nan",
                                 "determinism", "pause", "bench"])
    parser.add_argument("--res", type=int, default=RES)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--profile", action="store_true", help="per-stage device time (each launch group in its own graph)")
    parser.add_argument("--qprime", type=float, default=None, help="W/m for the plume test (default: 0.6 and 1.3)")
    parser.add_argument("--invert", action="store_true", help="g_sign = -1 for the plume test")
    parser.add_argument("--ice", action="store_true", help="ice cap on for the plume test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segments", type=int, default=0, help="random filament segments for the benchmark")
    parser.add_argument("--sweeps", type=int, default=PRESSURE_SWEEPS, help="fine pressure sweeps per V-cycle")
    parser.add_argument("--mg-levels", type=int, default=MG_LEVELS, help="coarse multigrid levels (0 = plain GS)")
    parser.add_argument("--mg-cycles", type=int, default=MG_CYCLES, help="V-cycles per frame")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.profile:
        profile(args)
        return
    tests = args.test
    if tests in ("plume", "all"):
        for q in ([args.qprime] if args.qprime else [0.6, 1.3]):
            test_plume(args, q, g_sign=-1.0 if args.invert else 1.0, ice=1.0 if args.ice else 0.0)
    if tests in ("invert", "all"):
        r = test_plume(args, args.qprime or 1.3, g_sign=-1.0, label="inverted g_sign=-1")
        if r:
            print(f"  inverted: centroid moved to y={r['centroid_final'][1] * 100:.2f} cm "
                  f"({'toward -y = up in the world frame' if r['centroid_final'][1] < 0 else 'NOT inverted'})")
    if tests in ("ice", "all"):
        base = test_plume(args, args.qprime or 1.3, x=0.02, y0=0.0, y1=0.05, label="under the cap, ice off")
        iced = test_plume(args, args.qprime or 1.3, x=0.02, y0=0.0, y1=0.05, ice=1.0, label="under the cap, ice on")
        if base and iced:
            cb, ci = base["centroid_final"], iced["centroid_final"]
            print(f"  ice deflection: hot-gas centroid (x, y) {cb[0] * 100:.2f}, {cb[1] * 100:.2f} cm -> "
                  f"{ci[0] * 100:.2f}, {ci[1] * 100:.2f} cm; peak dT {base['dT_max_final']:.2f} -> {iced['dT_max_final']:.2f} K; "
                  f"steady <u_y>_hot {base['hot_steady'] * 100:.2f} -> {iced['hot_steady'] * 100:.2f} cm/s; "
                  f"strongest downdraft {base['uy_min_final'] * 100:.2f} -> {iced['uy_min_final'] * 100:.2f} cm/s")
    if tests in ("robin", "all"):
        test_robin(args)
    if tests in ("divergence", "all"):
        test_divergence(args)
    if tests in ("stability", "all"):
        test_stability(args)
    if tests in ("nan", "all"):
        test_nan(args)
    if tests in ("determinism", "all"):
        test_determinism(args)
    if tests in ("pause", "all"):
        test_pause(args)
    if tests in ("bench", "all"):
        test_bench(args)


if __name__ == "__main__":
    main()
