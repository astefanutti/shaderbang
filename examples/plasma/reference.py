# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Reference Laplace solver for the growth-formulation validation (dossier followups[0] protocol).

7-point finite differences on an N^3 grid of cell centres over [-R2, R2]^3, Dirichlet phi = 0
on r <= R1 and on the cells holding aggregate nodes (the containing cell always, plus the cells
whose centre lies within ``node_radius``), phi = 1 (optionally a touch bump
f = 1 + A exp(-(theta/sigma)^2) around a finger direction) on r >= R2, solved by conjugate
gradient on the GPU (Warp, float64) to a relative residual of 1e-8. Validated against the
analytic annulus solution c1 + c2/r.

Two ways to compare a point-charge model with it: sample the grid by trilinear interpolation at
off-lattice candidates (``sample``), or reproduce the dossier's lattice protocol
(``snap_to_cells`` the nodes, ``frontier_cells`` = the free cells 6-adjacent to node cells,
exact grid values, no interpolation). ``dense_charges`` / ``dense_potential`` are the float64
dense conductor system of the dossier (M_jk = 1/r_jk, M_jj = 1/a) with an optional Kelvin image
series: ``images`` 0 none, 1 the electrode image (what the engine uses), 2 the alternating
electrode + glass image series (the node charges then vanish on both spheres).
"""

import math

import numpy as np
import warp as wp

FIXED_ZERO = 1
FIXED_ONE = 2
FIXED_NODE = 3   # aggregate cell (Dirichlet 0, distinguished for the lattice frontier)


@wp.kernel
def k_setup(P: wp.array(dtype=wp.float64), kind: wp.array3d(dtype=wp.int32),
            phi: wp.array3d(dtype=wp.float64), finger: wp.vec3d, amp: wp.float64,
            sigma: wp.float64, N: int):
    i, j, k = wp.tid()
    n = wp.float64(N)
    r2 = P[1]
    dx = wp.float64(2.0) * r2 / n
    x = wp.vec3d(-r2 + (wp.float64(i) + wp.float64(0.5)) * dx,
                 -r2 + (wp.float64(j) + wp.float64(0.5)) * dx,
                 -r2 + (wp.float64(k) + wp.float64(0.5)) * dx)
    r = wp.length(x)
    kind[i, j, k] = 0
    phi[i, j, k] = wp.float64(0.0)
    if r <= P[0]:
        kind[i, j, k] = FIXED_ZERO
    elif r >= r2:
        kind[i, j, k] = FIXED_ONE
        f = wp.float64(1.0)
        if amp > wp.float64(0.0):
            th = wp.acos(wp.clamp(wp.dot(x, finger) / r, wp.float64(-1.0), wp.float64(1.0)))
            f = f + amp * wp.exp(-(th / sigma) * (th / sigma))
        phi[i, j, k] = f


@wp.kernel
def k_rasterise(nodes: wp.array(dtype=wp.vec3d), radius: wp.float64, r2: wp.float64,
                kind: wp.array3d(dtype=wp.int32), phi: wp.array3d(dtype=wp.float64), N: int):
    """Marks the cell containing each node (always) and every cell whose centre lies within
    `radius` of it as aggregate cells."""
    m = wp.tid()
    x = nodes[m]
    dx = wp.float64(2.0) * r2 / wp.float64(N)
    span = int(wp.ceil(radius / dx))
    ci = int(wp.floor((x[0] + r2) / dx))
    cj = int(wp.floor((x[1] + r2) / dx))
    ck = int(wp.floor((x[2] + r2) / dx))
    for di in range(-span, span + 1):
        for dj in range(-span, span + 1):
            for dk in range(-span, span + 1):
                i = ci + di
                j = cj + dj
                k = ck + dk
                if i >= 0 and j >= 0 and k >= 0 and i < N and j < N and k < N:
                    c = wp.vec3d(-r2 + (wp.float64(i) + wp.float64(0.5)) * dx,
                                 -r2 + (wp.float64(j) + wp.float64(0.5)) * dx,
                                 -r2 + (wp.float64(k) + wp.float64(0.5)) * dx)
                    if (di == 0 and dj == 0 and dk == 0) or wp.length(c - x) <= radius:
                        kind[i, j, k] = FIXED_NODE
                        phi[i, j, k] = wp.float64(0.0)


@wp.func
def value(phi: wp.array3d(dtype=wp.float64), kind: wp.array3d(dtype=wp.int32),
          i: int, j: int, k: int, N: int, fixed_only: int):
    """Neighbour value: fixed cells contribute their Dirichlet value (or 0 in the operator),
    out-of-grid neighbours count as the outer shell (phi = 1)."""
    if i < 0 or j < 0 or k < 0 or i >= N or j >= N or k >= N:
        if fixed_only != 0:
            return wp.float64(1.0)
        return wp.float64(0.0)
    if kind[i, j, k] != 0:
        if fixed_only != 0:
            return phi[i, j, k]
        return wp.float64(0.0)
    if fixed_only != 0:
        return wp.float64(0.0)
    return phi[i, j, k]


@wp.kernel
def k_apply(v: wp.array3d(dtype=wp.float64), kind: wp.array3d(dtype=wp.int32),
            out: wp.array3d(dtype=wp.float64), N: int, mode: int):
    """mode 0: out = A v (6 v - sum of free neighbours); mode 1: out = b (sum of fixed
    neighbour values)."""
    i, j, k = wp.tid()
    if kind[i, j, k] != 0:
        out[i, j, k] = wp.float64(0.0)
        return
    fixed_only = mode
    s = value(v, kind, i - 1, j, k, N, fixed_only) + value(v, kind, i + 1, j, k, N, fixed_only) \
        + value(v, kind, i, j - 1, k, N, fixed_only) + value(v, kind, i, j + 1, k, N, fixed_only) \
        + value(v, kind, i, j, k - 1, N, fixed_only) + value(v, kind, i, j, k + 1, N, fixed_only)
    if mode == 0:
        out[i, j, k] = wp.float64(6.0) * v[i, j, k] - s
    else:
        out[i, j, k] = s


@wp.kernel
def k_axpy(y: wp.array3d(dtype=wp.float64), x: wp.array3d(dtype=wp.float64),
           alpha: wp.array(dtype=wp.float64), sign: wp.float64):
    i, j, k = wp.tid()
    y[i, j, k] = y[i, j, k] + sign * alpha[0] * x[i, j, k]


@wp.kernel
def k_xpay(p: wp.array3d(dtype=wp.float64), r: wp.array3d(dtype=wp.float64),
           beta: wp.array(dtype=wp.float64)):
    i, j, k = wp.tid()
    p[i, j, k] = r[i, j, k] + beta[0] * p[i, j, k]


@wp.kernel
def k_dot(x: wp.array3d(dtype=wp.float64), y: wp.array3d(dtype=wp.float64),
          out: wp.array(dtype=wp.float64)):
    i, j, k = wp.tid()
    wp.atomic_add(out, 0, x[i, j, k] * y[i, j, k])


@wp.kernel
def k_scalar(cg: wp.array(dtype=wp.float64), mode: int):
    """mode 0: alpha = rr / pAp; mode 1: beta = rr_new / rr, rr = rr_new."""
    if mode == 0:
        cg[2] = cg[0] / cg[1]
    else:
        cg[3] = cg[4] / cg[0]
        cg[0] = cg[4]


@wp.kernel
def k_combine(phi: wp.array3d(dtype=wp.float64), kind: wp.array3d(dtype=wp.int32),
              x: wp.array3d(dtype=wp.float64)):
    i, j, k = wp.tid()
    if kind[i, j, k] == 0:
        phi[i, j, k] = x[i, j, k]


class Reference:
    """Grid Laplace reference in the annulus R1 < r < R2 with the aggregate as phi = 0 cells."""

    def __init__(self, N, r1, r2, device="cuda:0"):
        self.N, self.r1, self.r2, self.device = N, r1, r2, device
        self.dx = 2.0 * r2 / N
        with wp.ScopedDevice(device):
            self.P = wp.array([r1, r2], dtype=wp.float64)
            shape = (N, N, N)
            self.kind = wp.zeros(shape, dtype=wp.int32)
            self.phi = wp.zeros(shape, dtype=wp.float64)
            self.x = wp.zeros(shape, dtype=wp.float64)
            self.r = wp.zeros(shape, dtype=wp.float64)
            self.p = wp.zeros(shape, dtype=wp.float64)
            self.Ap = wp.zeros(shape, dtype=wp.float64)
            self.cg = wp.zeros(8, dtype=wp.float64)
            self.dot_out = wp.zeros(1, dtype=wp.float64)
        self.iterations = 0
        self.residual = 0.0

    def setup(self, nodes=None, node_radius=0.0, finger=None, amp=0.0, sigma=0.25):
        N = self.N
        with wp.ScopedDevice(self.device):
            f = wp.vec3d(0.0, 0.0, 1.0) if finger is None else wp.vec3d(*[float(c) for c in finger])
            wp.launch(k_setup, dim=(N, N, N), inputs=[self.P, self.kind, self.phi, f,
                                                     wp.float64(amp), wp.float64(sigma), N])
            if nodes is not None and len(nodes):
                arr = wp.array(np.asarray(nodes, dtype=np.float64), dtype=wp.vec3d)
                wp.launch(k_rasterise, dim=len(nodes), inputs=[arr, wp.float64(node_radius),
                                                               wp.float64(self.r2), self.kind,
                                                               self.phi, N])

    def _dot(self, a, b):
        self.dot_out.zero_()
        wp.launch(k_dot, dim=a.shape, inputs=[a, b, self.dot_out])
        return self.dot_out

    def solve(self, tol=1.0e-8, max_iter=5000, warm=False):
        """CG on the free cells; returns the relative residual reached."""
        N = self.N
        dim = (N, N, N)
        with wp.ScopedDevice(self.device):
            if not warm:
                self.x.zero_()
            b = self.r
            wp.launch(k_apply, dim=dim, inputs=[self.phi, self.kind, b, N, 1])
            bb = float(self._dot(b, b).numpy()[0])
            wp.launch(k_apply, dim=dim, inputs=[self.x, self.kind, self.Ap, N, 0])
            ones = wp.array([1.0], dtype=wp.float64)
            wp.launch(k_axpy, dim=dim, inputs=[self.r, self.Ap, ones, wp.float64(-1.0)])
            wp.copy(self.p, self.r)
            rr = float(self._dot(self.r, self.r).numpy()[0])
            self.cg.zero_()
            wp.copy(self.cg, wp.array([rr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float64,
                                      device="cpu"))
            it = 0
            while it < max_iter and rr > tol * tol * bb:
                wp.launch(k_apply, dim=dim, inputs=[self.p, self.kind, self.Ap, N, 0])
                pap = self._dot(self.p, self.Ap)
                wp.copy(self.cg, pap, dest_offset=1, count=1)
                wp.launch(k_scalar, dim=1, inputs=[self.cg, 0])
                alpha = wp.array(ptr=self.cg.ptr + 2 * 8, dtype=wp.float64, shape=(1,),
                                 device=self.device, copy=False)
                wp.launch(k_axpy, dim=dim, inputs=[self.x, self.p, alpha, wp.float64(1.0)])
                wp.launch(k_axpy, dim=dim, inputs=[self.r, self.Ap, alpha, wp.float64(-1.0)])
                rr_new = self._dot(self.r, self.r)
                wp.copy(self.cg, rr_new, dest_offset=4, count=1)
                wp.launch(k_scalar, dim=1, inputs=[self.cg, 1])
                beta = wp.array(ptr=self.cg.ptr + 3 * 8, dtype=wp.float64, shape=(1,),
                                device=self.device, copy=False)
                wp.launch(k_xpay, dim=dim, inputs=[self.p, self.r, beta])
                it += 1
                if it % 25 == 0:
                    rr = float(self.cg.numpy()[0])
            rr = float(self.cg.numpy()[0])
            wp.launch(k_combine, dim=dim, inputs=[self.phi, self.kind, self.x])
            wp.synchronize()
        self.iterations = it
        self.residual = math.sqrt(rr / bb) if bb > 0 else 0.0
        return self.residual

    def grid(self):
        return self.phi.numpy()

    def sample(self, points):
        """Trilinear interpolation of phi at world points (clamped to the grid)."""
        phi = self.grid()
        N = self.N
        pts = np.asarray(points, dtype=np.float64)
        g = (pts + self.r2) / self.dx - 0.5
        g = np.clip(g, 0.0, N - 1.000001)
        i0 = np.floor(g).astype(np.int64)
        f = g - i0
        i1 = np.minimum(i0 + 1, N - 1)
        out = np.zeros(len(pts))
        for dx_ in (0, 1):
            wx = f[:, 0] if dx_ else 1.0 - f[:, 0]
            ix = i1[:, 0] if dx_ else i0[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1.0 - f[:, 1]
                iy = i1[:, 1] if dy else i0[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1.0 - f[:, 2]
                    iz = i1[:, 2] if dz else i0[:, 2]
                    out += wx * wy * wz * phi[ix, iy, iz]
        return out

    def cell_centres(self, points):
        """Indices (M, 3) of the cells containing `points`."""
        pts = np.asarray(points, dtype=np.float64)
        return np.clip(np.floor((pts + self.r2) / self.dx).astype(np.int64), 0, self.N - 1)

    def snap_to_cells(self, points):
        """Unique cell centres of the cells containing `points` (the dossier's lattice aggregate)."""
        cells = np.unique(self.cell_centres(points), axis=0)
        return (cells + 0.5) * self.dx - self.r2

    def frontier_cells(self):
        """Centres and grid potentials of the free cells 6-adjacent to an aggregate cell (the
        lattice frontier of the dossier protocol; exact grid values, no interpolation)."""
        kind = self.kind.numpy()
        node = kind == FIXED_NODE
        near = np.zeros_like(node)
        for axis in range(3):
            for shift in (-1, 1):
                rolled = np.roll(node, shift, axis=axis)
                edge = [slice(None)] * 3
                edge[axis] = 0 if shift == 1 else -1
                rolled[tuple(edge)] = False
                near |= rolled

        cells = np.argwhere((kind == 0) & near)
        return (cells + 0.5) * self.dx - self.r2, self.grid()[cells[:, 0], cells[:, 1], cells[:, 2]]

    def analytic_error(self):
        """(max, mean) relative error against c1 + c2/r over 1.6 R1 < r < 0.92 R2."""
        N = self.N
        c = (np.arange(N) + 0.5) * self.dx - self.r2
        X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
        r = np.sqrt(X * X + Y * Y + Z * Z)
        c1 = self.r2 / (self.r2 - self.r1)
        c2 = -self.r1 * self.r2 / (self.r2 - self.r1)
        exact = c1 + c2 / np.maximum(r, 1e-12)
        mask = (r > 1.6 * self.r1) & (r < 0.92 * self.r2)
        err = np.abs(self.grid()[mask] - exact[mask]) / exact[mask]
        return float(err.max()), float(err.mean())


# --- dense conductor system (float64, host) -------------------------------------------------------

def image_charges(pos, r1, r2, images, n_reflections=6):
    """Positions (n, m, 3) and charge factors (n, m) of each node charge and its Kelvin images:
    images 0 -> the charge alone; 1 -> plus its image in the electrode (-R1/l at R1^2/l);
    2 -> plus the two alternating reflection chains between the electrode and the glass."""
    pos = np.asarray(pos, dtype=np.float64)
    chains = [[(pos, np.ones(len(pos)))]]
    starts = [] if images == 0 else ([r1] if images == 1 else [r1, r2])
    for first in starts:
        p, f = pos, np.ones(len(pos))
        terms = []
        for n in range(1 if images == 1 else n_reflections):
            R = first if n % 2 == 0 else (r2 if first == r1 else r1)
            l = np.linalg.norm(p, axis=1)
            f = -f * R / l
            p = p * (R * R / (l * l))[:, None]
            terms.append((p, f))
        chains.append(terms)
    terms = [t for chain in chains for t in chain]
    return np.stack([t[0] for t in terms], 1), np.stack([t[1] for t in terms], 1)


def dense_potential(points, pos, q, r1, r2, images, n_reflections=6):
    """u0 plus the potential of the node charges `q` (with their images) at `points`."""
    pts = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(pts, axis=1)
    phi = (1.0 / r1 - 1.0 / r) / (1.0 / r1 - 1.0 / r2)
    P, F = image_charges(pos, r1, r2, images, n_reflections)
    q = np.asarray(q, dtype=np.float64)
    for start in range(0, len(pts), 128):
        xs = pts[start:start + 128]
        d = np.linalg.norm(xs[:, None, None, :] - P[None, :, :, :], axis=3)
        phi[start:start + 128] += np.einsum("j,jm,ijm->i", q, F, 1.0 / np.maximum(d, 1e-12))
    return phi


def dense_charges(pos, r1, r2, a, images, n_reflections=6):
    """Solves the dense conductor system M q = -u0(x_j) (M_jk = the image-series kernel,
    M_jj = 1/a plus the node's own images seen at the node) in float64; returns q."""
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    P, F = image_charges(pos, r1, r2, images, n_reflections)
    M = np.empty((n, n))
    for start in range(0, n, 128):
        rows = np.arange(start, min(start + 128, n))
        d = np.linalg.norm(pos[rows, None, None, :] - P[None, :, :, :], axis=3)
        d[np.arange(len(rows)), rows, 0] = a
        M[rows] = np.einsum("km,jkm->jk", F, 1.0 / np.maximum(d, 1e-12))
    r = np.linalg.norm(pos, axis=1)
    b = -(1.0 / r1 - 1.0 / r) / (1.0 / r1 - 1.0 / r2)
    return np.linalg.solve(M, b)
