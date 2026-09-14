# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
APSF glow profile, pyramid weights and blue noise
=================================================

Camera / eye glare kernel for the plasma-globe renderer (plan 5.5), from the Narasimhan & Nayar
atmospheric point spread function (CVPR 2003) in the form used by Kim & Lin (TVCG 2007, eqs.
5-12)::

    I(T, mu) = sum_m (g_m(T) + g_{m+1}(T)) L_m(mu)        g_m(T) = exp(-beta_m T - alpha_m ln T)
    alpha_m = m + 1     beta_m = ((2m + 1) / m) (1 - q^(m-1))     g_0 = 0

with Legendre polynomials by the bottom-up recurrence L_i(x) = ((2i-1) x L_{i-1} - (i-1) L_{i-2}) / i.
``q`` is the Henyey-Greenstein forward-scattering parameter, ``T`` the optical thickness and
``mu = cos(theta)`` the cosine of the angle between the exit ray and the outward normal of the
sphere of radius R around the source. The series only converges for T > 1 (g_m ~ T^-(m+1)),
``apsf()`` asserts it; at T = 1.1 the sum is converged to < 0.1 % at m_max = 64 (Kim & Lin used
200; both are checked by the CLI).

Kernel geometry (Kim & Lin fig. 13, eqs. 10-12): a pinhole eye, image plane at kappa = 0.025 m,
source at D = 2000 m, glow radius R (their user knob); a kernel sample at image-plane offset nu
sees the ray that exits the sphere at axial depth omega, and
theta = pi - atan(kappa / nu) - asin((D - omega) / R); the kernel is 0 where nu / kappa > R / D
and the smallest non-zero value (the value on that silhouette, ~7 % of the peak at q 0.9, T 1.1)
is subtracted so the disc edge vanishes (pedestal subtraction). Their published table I (n 64,
M 1.0 with kappa 0.025, D 2000, R 100) puts the whole disc inside one pixel, so M is not taken
from the paper: ``apsf_profile(kernel_px=...)`` chooses the pixel pitch M / n = kappa R /
(D kernel_px) so that the silhouette falls at ``kernel_px`` pixels - the glow radius in pixels is
the knob and R only enters the (tiny) near-field correction to mu.

Gaussian fit and pyramid weights
--------------------------------
The 1-D radial profile is fitted with a sum of 4 zero-mean Gaussians (Levenberg-Marquardt in
log-parameters from ``FIT_STARTS`` initial spreads, best kept; no scipy). Validated box at 64 px
(fit RMSE < ``FIT_TARGET`` = 2 % of the peak): q 0.9 or 0.95 with T 1.05-1.3, q 0.7-0.8 with
T <= 1.1; flatter domes (T >= 1.5) collapse two sigmas onto the 0.25 px bound and miss the target
(3-7 %) - they would need a top-hat term. The fit does not vanish at the silhouette: at
r = kernel_px it is +4 % of the peak (max residual, reported next to the RMSE).

The Gaussians are mapped to the weights of the renderer's Jimenez/Kawase pyramid:
``GLOW_LEVELS`` = 6 downsampled levels 1..6 below the image itself (level 0), 13-tap 6x6
downsample, 3x3 tent upsample of bilinear fetches one coarse texel apart, progressive up-chain
U_l = w_l D_l + up(U_{l+1}). The effective kernel of level l is the response up^l(down^l(delta));
its second-moment sigma is ``PYRAMID_SIGMA[l]`` = 0, 2.12, 4.74, 9.72, 19.56, 39.17, 78.37 px
(~1.06 * 2^l for l >= 1; measured by ``pyramid_kernels()`` / ``pyramid_sigmas()`` over delta
phases chosen so that every level up to 2^3 sees each of its phases equally often - the sigma is
identical to 3 decimals for every phase and the kernel shape varies by < 0.05 % RMSE; the Karis
1/(1+luma) weighting of the first downsample is non-linear and ignored). ``pyramid_weights()``
solves the non-negative least squares problem  sum_l w_l G(sigma_l) ~ sum_k a_k G(sigma_k)  on
the radial profile (uniform in r, unit-energy 2-D Gaussians for l >= 1, a one-pixel delta for
l = 0) and returns energy fractions (sum 1) for levels 0..6; ``pyramid_weights_default()`` is the
renderer entry point (profile -> fit -> weights for the downsampled levels 1..6, ~10 ms).
``validate_pyramid()`` synthesises the real pyramid response with those weights and measures the
profile error against the 2-D APSF kernel. Measured: a 64 px kernel is reproduced to 1.7 % of the
peak (RMS over r <= 64; weights 0.36 / 0.61 on levels 4 / 5, nothing on 6), a 128 px kernel to
1.6 % (0.36 / 0.61 on levels 5 / 6). The widest fitted Gaussian (~0.435 kernel_px) must stay
below ``SIGMA_MARGIN`` = 1.1 x the widest level sigma, i.e. kernel_px <= ~195 px with 6
downsampled levels (2.3 % at 208 px, 6.4 % at 256 px; ``pyramid_weights()`` warns). Like the fit,
the pyramid does not vanish at the silhouette (+6 % of the peak at r = kernel_px) and its soft
Gaussian tails carry ~16 % of the glow energy beyond r = kernel_px: the rendered glow is
~1.3-1.5x wider than Kim & Lin's compact kernel - the price of the pyramid.

Renderer contract: the weights assume ``glow_up.comp``'s tent radius = 1.0 coarse texel and
that every level, the coarsest included, is multiplied by its own weight (U_6 = w_6 D_6, no tent
on it). Deviations measured with the numpy model: a plain bilinear (no tent) last step 1 -> 0
changes the level sigmas to 1.58, 4.53, 9.62, 19.51, 39.15, 78.36 px (< 1 % for l >= 3, where all
the weight is); a same-resolution 3x3 tent applied to the coarsest level widens it from 78.4 to
90.5 px; a tent radius of 0.5 / 1.5 / 2.0 texels scales every level sigma by ~0.82 / 1.25 / 1.53
(``pyramid_kernels(radius=...)`` re-measures them). Change the glow width through ``kernel_px``
(re-fit with ``pyramid_weights_default()``), not through the tent radius.

Blue noise
----------
A 64x64 blue-noise dither texture for the 8-bit present pass (plan 5.7) is generated by
Ulichney's void-and-cluster method (``void_and_cluster()``, Gaussian sigma 1.5 on a torus, ranks
mapped to 16 pixels per grey level); ``blue_noise()`` loads ``data/blue_noise_64.png`` or
regenerates it in memory when the file is missing.

Run ``python examples/plasma/apsf.py`` for the parameters, the fit and pyramid errors and the
plot ``/tmp/plasma_apsf.png`` (~7 s: the 7-level synthesis works on 1024^2 images);
``--blue-noise PNG`` writes the texture (the shipped one is ``data/blue_noise_64.png``, seed 0).
"""

import argparse
import math
import warnings
from pathlib import Path

import numpy as np


Q = 0.9              # Henyey-Greenstein parameter (Kim & Lin table I, fig. 15)
T = 1.1              # optical thickness (must be > 1)
M_MAX = 64           # Legendre terms
KAPPA = 0.025        # pinhole to image plane [m]
D = 2000.0           # source distance [m]
R = 100.0            # glow sphere radius [m], Kim & Lin's width knob
GLOW_LEVELS = 6      # downsampled pyramid levels 1..6 (renderer GLOW_LEVELS); level 0 is the image
LEVELS = GLOW_LEVELS + 1                                        # pyramid levels modelled, 0..6
PYRAMID_SIGMA = (0.0, 2.12, 4.74, 9.72, 19.56, 39.17, 78.37)   # px, measured second moment per level
SIGMA_MARGIN = 1.1   # widest fitted sigma / widest level sigma beyond which the mapping exceeds 2 %
FIT_TARGET = 0.02    # RMSE target of the fit and of the pyramid, fraction of the peak
FIT_STARTS = ((16.0, 1.5), (8.0, 1.2))   # initial sigma spreads (support / lo .. support / hi)
BLUE_NOISE_PATH = Path(__file__).resolve().parent / "data" / "blue_noise_64.png"

# 13-tap Jimenez downsample as a 6x6 kernel over source texels 2j-2 .. 2j+3: bilinear fetches
# (each a 2x2 box) at (+-1, +-1) weight 1/8, (0, 0) 1/8, (+-2, 0) / (0, +-2) 1/16, (+-2, +-2) 1/32.
DOWN_TAPS = np.zeros((6, 6))
for (_ox, _oy), _w in {(1, 1): .125, (1, -1): .125, (-1, 1): .125, (-1, -1): .125, (0, 0): .125,
                       (2, 0): .0625, (-2, 0): .0625, (0, 2): .0625, (0, -2): .0625,
                       (2, 2): .03125, (2, -2): .03125, (-2, 2): .03125, (-2, -2): .03125}.items():
    DOWN_TAPS[_ox + 2:_ox + 4, _oy + 2:_oy + 4] += _w / 4.0
UP_TAPS = ((-1, 0.25), (0, 0.5), (1, 0.25))   # tent per axis, offsets in coarse texels x radius


def apsf(mu, T=T, q=Q, m_max=M_MAX):
    """Narasimhan-Nayar APSF I(T, mu) (unnormalised) for an array of mu = cos(theta)."""
    assert T > 1.0, "the APSF Legendre series diverges for T <= 1"
    mu = np.asarray(mu, dtype=np.float64)
    m = np.arange(0, m_max + 2, dtype=np.float64)
    alpha = m + 1.0
    beta = np.zeros_like(m)
    beta[1:] = (2.0 * m[1:] + 1.0) / m[1:] * (1.0 - q ** (m[1:] - 1.0))
    g = np.exp(-beta * T - alpha * math.log(T))
    g[0] = 0.0
    l_prev = np.ones_like(mu)
    l_cur = mu.copy()
    out = (g[0] + g[1]) * l_prev + (g[1] + g[2]) * l_cur
    for i in range(2, m_max + 1):
        l_next = ((2 * i - 1) * mu * l_cur - (i - 1) * l_prev) / i
        out += (g[i] + g[i + 1]) * l_next
        l_prev, l_cur = l_cur, l_next
    return out


def kim_lin_mu(r_px, kernel_px, R=R, D=D, kappa=KAPPA):
    """(mu, inside) for kernel samples at radius ``r_px`` pixels, Kim & Lin eqs. 10-12 with the
    pixel pitch M / n = kappa R / (D kernel_px) (silhouette nu / kappa = R / D at ``kernel_px``)."""
    nu = np.asarray(r_px, dtype=np.float64) * (kappa * R / (D * kernel_px))
    inside = nu / kappa <= R / D * (1.0 + 1e-12)
    root = np.sqrt(np.maximum(-nu * nu * D * D + kappa * kappa * R * R + nu * nu * R * R, 0.0))
    omega = (kappa * kappa * D - kappa * root) / (kappa * kappa + nu * nu)
    theta = np.pi - np.arctan2(kappa, nu) - np.arcsin(np.clip((D - omega) / R, -1.0, 1.0))
    return np.cos(theta), inside


def apsf_profile(q=Q, T=T, m_max=M_MAX, kernel_px=64, R=R, D=D, kappa=KAPPA, step=1.0):
    """1-D radial glow profile (r_px, I) in pixels: peak 1 at r = 0, pedestal subtracted, 0 for
    r >= kernel_px."""
    r_px = np.arange(0.0, kernel_px + step * 0.5, step)
    return r_px, _kernel_values(r_px, q, T, m_max, kernel_px, R, D, kappa)


def apsf_kernel(kernel_px=64, q=Q, T=T, m_max=M_MAX, R=R, D=D, kappa=KAPPA, pedestal=True):
    """(2 kernel_px + 1)^2 image of the same kernel (peak 1), for the silhouette check and the
    pyramid validation."""
    y, x = np.mgrid[-kernel_px:kernel_px + 1, -kernel_px:kernel_px + 1]
    return _kernel_values(np.hypot(x, y), q, T, m_max, kernel_px, R, D, kappa, pedestal)


def _kernel_values(r_px, q, T, m_max, kernel_px, R, D, kappa, pedestal=True):
    mu, inside = kim_lin_mu(r_px, kernel_px, R, D, kappa)
    values = np.where(inside, apsf(mu, T, q, m_max), 0.0)
    if pedestal:
        values = np.where(inside, values - values[inside].min(), 0.0)
    return np.maximum(values, 0.0) / values.max()


def gaussian_sum(r, params):
    """sum_k a_k exp(-r^2 / (2 sigma_k^2)) for params (k, 2) = [amplitude, sigma]."""
    r = np.asarray(r, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    return np.exp(-0.5 * (r[..., None] / params[:, 1]) ** 2) @ params[:, 0]


def nnls(A, b, tolerance=1e-12):
    """Non-negative least squares min |A x - b|, x >= 0 (Lawson & Hanson active set)."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[1]
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    scale = max(1.0, float(np.abs(A.T @ b).max()))
    while (~passive).any() and w[~passive].max() > tolerance * scale:
        passive[int(np.argmax(np.where(passive, -np.inf, w)))] = True
        while True:
            z = np.zeros(n)
            z[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if z[passive].min() > 0.0:
                x = z
                break
            blocking = passive & (z <= 0.0)
            alpha = np.min(x[blocking] / (x[blocking] - z[blocking]))
            x = x + alpha * (z - x)
            passive &= x > tolerance
            x[~passive] = 0.0
        w = A.T @ (b - A @ x)
    return x


def fit_gaussians(r, I, k=4, iterations=300, starts=FIT_STARTS):
    """Fit ``I(r)`` with ``k`` zero-mean Gaussians by Levenberg-Marquardt in log(amplitude),
    log(sigma), from every initial spread in ``starts`` (sigmas geometric between support / lo
    and support / hi, amplitudes by NNLS), keeping the lowest cost. Returns params (k, 2) =
    [amplitude, sigma] sorted by sigma."""
    r = np.asarray(r, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    best = None
    for lo, hi in starts:
        params, cost = _fit_once(r, I, k, iterations, lo, hi)
        if best is None or cost < best[1]:
            best = (params, cost)
    return best[0]


def _fit_once(r, I, k, iterations, lo, hi):
    peak = I.max()
    support = r[I > 0.01 * peak].max()
    sigma = np.geomspace(support / lo, support / hi, k)
    basis = np.exp(-0.5 * (r[:, None] / sigma[None, :]) ** 2)
    amplitude = np.maximum(nnls(basis, I), 1e-4 * peak)
    p = np.log(np.concatenate([amplitude, sigma]))
    # keep the parameters in the range a sane fit can use (no runaway sigma on spiky profiles)
    lower = np.log(np.concatenate([np.full(k, 1e-6 * peak), np.full(k, 0.25)]))
    upper = np.log(np.concatenate([np.full(k, 10.0 * peak), np.full(k, 4.0 * max(support, 1.0))]))

    def evaluate(p):
        a, s = np.exp(p[:k]), np.exp(p[k:])
        basis = np.exp(-0.5 * (r[:, None] / s[None, :]) ** 2)
        return basis @ a - I, basis, a, s

    residual, basis, a, s = evaluate(p)
    cost = residual @ residual
    damping = 1e-2
    for _ in range(iterations):
        jacobian = np.concatenate([basis * a, basis * a * (r[:, None] / s[None, :]) ** 2], axis=1)
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        step = np.linalg.solve(hessian + damping * np.diag(np.diag(hessian)) + 1e-18 * np.eye(2 * k), -gradient)
        candidate = np.clip(p + step, lower, upper)
        residual_new, basis_new, a_new, s_new = evaluate(candidate)
        cost_new = residual_new @ residual_new
        if cost_new < cost:
            converged = cost - cost_new < 1e-14 * cost
            p, residual, basis, a, s, cost = candidate, residual_new, basis_new, a_new, s_new, cost_new
            damping = max(damping / 3.0, 1e-12)
            if converged:
                break
        else:
            damping *= 4.0
    order = np.argsort(s)
    return np.stack([a[order], s[order]], axis=1), cost


def fit_error(r, I, params):
    """(uniform RMSE, area-weighted RMSE, max |residual|) of the Gaussian fit, all relative to
    the peak."""
    residual = gaussian_sum(r, params) - I
    uniform = math.sqrt(np.mean(residual ** 2))
    area = math.sqrt(np.sum(r * residual ** 2) / max(np.sum(r), 1e-12))
    return uniform / I.max(), area / I.max(), float(np.abs(residual).max()) / I.max()


def downsample(image):
    """Jimenez 13-tap 2x downsample (``DOWN_TAPS``), zero padded."""
    n = image.shape[0]
    padded = np.zeros((n + 5, n + 5))
    padded[2:n + 2, 2:n + 2] = image
    out = np.zeros((n // 2, n // 2))
    for a in range(6):
        for b in range(6):
            out += DOWN_TAPS[a, b] * padded[a:a + n:2, b:b + n:2]
    return out


def upsample(image, radius=1.0):
    """2x upsample as ``glow_up.comp``: every fine pixel (coarse coordinate c -/+ 0.25 for the
    even / odd pixel of a pair) is the 3x3 tent ``UP_TAPS`` of bilinear fetches ``radius``
    coarse texels apart, zero outside the image. Separable: per axis and pixel parity the tent
    of bilinear pairs is folded into one compact FIR over the coarse samples."""
    def axis_pass(c, axis):
        c = np.moveaxis(c, axis, 0)
        n = c.shape[0]
        out = np.zeros((2 * n,) + c.shape[1:])
        for parity, delta in ((0, -0.25), (1, 0.25)):
            fir = {}
            for k, t in UP_TAPS:
                position = delta + k * radius
                m = math.floor(position)
                f = position - m
                fir[m] = fir.get(m, 0.0) + t * (1.0 - f)
                fir[m + 1] = fir.get(m + 1, 0.0) + t * f
            pad = max(abs(m) for m in fir)
            padded = np.zeros((n + 2 * pad,) + c.shape[1:])
            padded[pad:pad + n] = c
            for m, w in fir.items():
                if w != 0.0:
                    out[parity::2] += w * padded[pad + m:pad + m + n]
        return np.moveaxis(out, 0, axis)

    return axis_pass(axis_pass(image, 0), 1)


def pyramid_kernels(levels=LEVELS, size=None, phases=8, radius=1.0):
    """Phase-averaged effective kernel of every pyramid level (unit sum, centred in a size^2
    image, default 16 x the coarsest block so nothing is truncated): the response to a delta
    pushed l levels down and l levels back up (tent ``radius``), averaged over ``phases``^2 delta
    positions (j m mod block, m odd) so that every level l <= log2(phases) sees each of its 4^l
    phases equally often."""
    block = 2 ** (levels - 1)
    size = size or max(512, 16 * block)
    multiplier = max(block // phases, 1) | 1
    positions = [(j * multiplier) % block for j in range(min(phases, block))]
    kernels = [np.zeros((size, size)) for _ in range(levels)]
    for px in positions:
        for py in positions:
            image = np.zeros((size, size))
            image[size // 2 + px, size // 2 + py] = 1.0
            kernels[0] += np.roll(image, (-px, -py), axis=(0, 1))
            current = image
            for level in range(1, levels):
                current = downsample(current)
                response = current
                for _ in range(level):
                    response = upsample(response, radius)
                kernels[level] += np.roll(response, (-px, -py), axis=(0, 1))
    return [k / len(positions) ** 2 for k in kernels]


def pyramid_sigmas(kernels):
    """Second-moment sigma (px) of each kernel: sigma^2 = sum K r^2 / (2 sum K)."""
    size = kernels[0].shape[0]
    y, x = np.mgrid[0:size, 0:size]
    r2 = (y - size // 2) ** 2 + (x - size // 2) ** 2
    return np.array([math.sqrt((k * r2).sum() / (2.0 * k.sum())) for k in kernels])


def _radial_basis(sigmas, r):
    """Unit-energy 2-D Gaussian profiles (a one-pixel delta for sigma 0) sampled at radii ``r``,
    normalised on the pixel grid."""
    basis = np.empty((r.size, len(sigmas)))
    for level, sigma in enumerate(sigmas):
        if sigma == 0.0:
            basis[:, level] = (r == 0.0)
        else:
            extent = int(math.ceil(4.0 * sigma))
            y, x = np.mgrid[-extent:extent + 1, -extent:extent + 1]
            basis[:, level] = np.exp(-0.5 * (r / sigma) ** 2) / np.exp(-0.5 * (x * x + y * y) / sigma ** 2).sum()
    return basis


def pyramid_weights(params, levels=LEVELS, sigmas=PYRAMID_SIGMA, radius=None, normalised=True):
    """Weights w_l of a ``levels``-level pyramid (levels 0 .. levels-1) whose level kernels are
    unit-energy Gaussians of ``sigmas`` (a delta for sigma 0), non-negative least squares on the
    radial profile (uniform in r, 0..``radius`` px, default 3 sigma of the widest fitted
    Gaussian) against sum_k a_k exp(-r^2 / 2 sigma_k^2). ``normalised``: energy fractions
    (sum 1); otherwise the raw weights reproduce the peak-1 profile. Warns when the widest fitted
    Gaussian exceeds ``SIGMA_MARGIN`` x the widest level sigma (the mapping then misses 2 %)."""
    params = np.asarray(params, dtype=np.float64)
    if len(sigmas) < levels:
        raise ValueError(f"{levels} levels need {levels} level sigmas, got {len(sigmas)}")
    sigmas = np.asarray(sigmas[:levels], dtype=np.float64)
    widest = params[:, 1].max()
    if widest > SIGMA_MARGIN * sigmas.max():
        warnings.warn(f"widest fitted Gaussian {widest:.1f} px exceeds {SIGMA_MARGIN} x the widest "
                      f"level sigma {sigmas.max():.1f} px: {levels} levels cannot reproduce this kernel "
                      "(reduce kernel_px or add a level)", stacklevel=2)
    if radius is None:
        radius = int(math.ceil(3.0 * widest))
    r = np.arange(0.0, radius + 1.0)
    w = nnls(_radial_basis(sigmas, r), gaussian_sum(r, params))
    return w / w.sum() if normalised else w


def pyramid_weights_default(levels=GLOW_LEVELS, kernel_px=64, q=Q, T=T, m_max=M_MAX, sigmas=PYRAMID_SIGMA):
    """Renderer entry point: the energy fractions (sum 1) of the ``levels`` downsampled pyramid
    levels 1 .. levels for a glow of radius ``kernel_px`` internal pixels - APSF profile ->
    4-Gaussian fit -> ``pyramid_weights`` over levels 0 .. levels, the level-0 delta weight
    dropped (it is 0 for kernel_px >= 16 px; the image itself is not part of the glow). Index i
    of the result is the weight of the level downsampled i + 1 times; the coarsest level must be
    multiplied by its own weight like the others. Warns when the fit misses ``FIT_TARGET``."""
    r, profile = apsf_profile(q, T, m_max, kernel_px)
    params = fit_gaussians(r, profile)
    rmse = fit_error(r, profile, params)[0]
    if rmse > FIT_TARGET:
        warnings.warn(f"4-Gaussian fit of the APSF (q {q}, T {T}, {kernel_px} px) has RMSE "
                      f"{100 * rmse:.2f} % of the peak (target {100 * FIT_TARGET:.0f} %)", stacklevel=2)
    w = pyramid_weights(params, levels + 1, sigmas)[1:]
    total = w.sum()
    return w / total if total > 0.0 else w


def pyramid_weights_direct(kernel, kernels, radius=None):
    """Energy fractions fitted the same way but against the radial profiles of synthesised level
    kernels (``pyramid_kernels``), for comparison with the Gaussian mapping."""
    size = kernels[0].shape[0]
    r, target = radial_profile(_embed(kernel, size))
    if radius is None:
        radius = int(1.5 * (kernel.shape[0] // 2))
    basis = np.stack([radial_profile(k)[1] for k in kernels], axis=1)
    w = nnls(basis[:radius + 1], target[:radius + 1])
    return w / w.sum()


def pyramid_response(weights, kernels):
    """sum_l w_l K_l."""
    return sum(w * k for w, k in zip(weights, kernels))


def validate_pyramid(weights, kernel, kernels):
    """Synthesise the pyramid response with ``weights`` (any scale), fit one gain to the peak-1
    2-D ``kernel`` on the radial profile over its support, and report
    {'rmse': profile RMSE uniform in r / peak, 'rmse_area': area-weighted, 'max': max |residual|
    / peak, 'edge': residual at r = kernel_px / peak, 'gain', 'leak': energy fraction of the
    response outside the kernel radius} and (r, target, response)."""
    size = kernels[0].shape[0]
    kernel_px = kernel.shape[0] // 2
    r, target = radial_profile(_embed(kernel, size))
    response_2d = pyramid_response(weights, kernels)
    _, response = radial_profile(response_2d)
    n = kernel_px + 1
    gain = float(response[:n] @ target[:n] / (response[:n] @ response[:n]))
    response = gain * response
    residual = response[:n] - target[:n]
    y, x = np.mgrid[0:size, 0:size]
    outside = np.hypot(x - size // 2, y - size // 2) > kernel_px
    return {
        "rmse": math.sqrt(np.mean(residual ** 2)) / target[0],
        "rmse_area": math.sqrt(np.sum(r[:n] * residual ** 2) / np.sum(r[:n])) / target[0],
        "max": float(np.abs(residual).max()) / target[0],
        "edge": float(residual[-1]) / target[0],
        "gain": gain,
        "leak": float(response_2d[outside].sum() / response_2d.sum()),
    }, (r, target, response)


def radial_profile(image):
    """Mean of ``image`` in integer-radius bins around its centre: (r, mean)."""
    size = image.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    bins = np.rint(np.hypot(x - size // 2, y - size // 2)).astype(np.int64).ravel()
    total = np.bincount(bins, weights=image.ravel())
    count = np.bincount(bins)
    r = np.arange(size // 2)
    return r, total[:size // 2] / count[:size // 2]


def _embed(kernel, size):
    half = kernel.shape[0] // 2
    out = np.zeros((size, size))
    out[size // 2 - half:size // 2 + half + 1, size // 2 - half:size // 2 + half + 1] = kernel
    return out


def void_and_cluster(size=64, sigma=1.5, seed=0):
    """Ulichney void-and-cluster blue-noise ranks as an 8-bit (size, size) array; every grey
    level holds size^2 / 256 pixels."""
    rng = np.random.default_rng(seed)
    n = size * size
    d = np.minimum(np.arange(size), size - np.arange(size)).astype(np.float64)
    kernel = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2.0 * sigma * sigma))   # centred at [0, 0]

    def add(energy, index, sign):
        energy += sign * np.roll(kernel, (index // size, index % size), axis=(0, 1))

    def tightest(energy, pattern):
        return int(np.argmax(np.where(pattern, energy, -np.inf)))

    def largest_void(energy, pattern):
        return int(np.argmin(np.where(pattern, np.inf, energy)))

    ones = n // 10
    pattern = np.zeros(n, dtype=bool)
    pattern[rng.choice(n, ones, replace=False)] = True
    energy = np.zeros((size, size))
    for index in np.flatnonzero(pattern):
        add(energy, int(index), 1.0)
    while True:
        cluster = tightest(energy.ravel(), pattern)
        pattern[cluster] = False
        add(energy, cluster, -1.0)
        void = largest_void(energy.ravel(), pattern)
        pattern[void] = True
        add(energy, void, 1.0)
        if void == cluster:
            break
    rank = np.zeros(n, dtype=np.int64)
    work, e = pattern.copy(), energy.copy()
    for k in range(ones - 1, -1, -1):
        cluster = tightest(e.ravel(), work)
        work[cluster] = False
        add(e, cluster, -1.0)
        rank[cluster] = k
    work, e = pattern.copy(), energy.copy()
    for k in range(ones, n):
        void = largest_void(e.ravel(), work)
        work[void] = True
        add(e, void, 1.0)
        rank[void] = k
    return (rank * 256 // n).astype(np.uint8).reshape(size, size)


def blue_noise(path=BLUE_NOISE_PATH, size=64, seed=0):
    """The shipped blue-noise texture as a uint8 (size, size) array, generated when missing."""
    path = Path(path)
    if path.exists():
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return void_and_cluster(size, seed=seed)


def spectrum_ratio(image, low=0.125, high=0.25):
    """Ratio of the power below radial frequency ``low`` (cycles/px, DC excluded) to the power
    at or above ``high``; blue noise -> small, white noise -> the band-area ratio."""
    v = image.astype(np.float64)
    v -= v.mean()
    power = np.abs(np.fft.fft2(v)) ** 2
    f = np.fft.fftfreq(image.shape[0])
    rho = np.hypot(f[:, None], f[None, :])
    return float(power[(rho > 0.0) & (rho < low)].sum() / power[rho >= high].sum())


def blue_noise_metric(image, seed=0, trials=16):
    """{'blue': ratio, 'white': mean ratio of shuffled copies, 'gain': white / blue}."""
    rng = np.random.default_rng(seed)
    blue = spectrum_ratio(image)
    white = float(np.mean([spectrum_ratio(rng.permutation(image.ravel()).reshape(image.shape))
                           for _ in range(trials)]))
    return {"blue": blue, "white": white, "gain": white / blue}


SURFACE, INK, INK_SOFT, GRID = (252, 252, 251), (11, 11, 11), (82, 81, 78), (230, 229, 225)
SERIES = ((42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0))


def line_panel(draw, font, box, x, series, title, y_label, y_range=None):
    """One line chart: axes with 4 grid lines, 2 px lines, legend. ``series`` = [(label, y)]."""
    left, top, right, bottom = box
    x = np.asarray(x, dtype=np.float64)
    ys = [np.asarray(y, dtype=np.float64) for _, y in series]
    lo, hi = y_range or (min(float(y.min()) for y in ys), max(float(y.max()) for y in ys))
    if hi - lo < 1e-12:
        hi = lo + 1.0
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    px, py = left + 56, top + 22
    width, height = right - 12 - px, bottom - 30 - py

    def to_x(v):
        return px + (v - x[0]) / (x[-1] - x[0]) * width

    def to_y(v):
        return py + (hi - v) / (hi - lo) * height

    draw.text((left + 8, top + 2), f"{title}   [y: {y_label}]", fill=INK, font=font)
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        yy = to_y(v)
        draw.line([(px, yy), (px + width, yy)], fill=GRID, width=1)
        draw.text((left + 4, yy - 7), f"{v:.3g}", fill=INK_SOFT, font=font)
    for k in range(5):
        v = x[0] + (x[-1] - x[0]) * k / 4
        xx = to_x(v)
        draw.line([(xx, py + height), (xx, py + height + 4)], fill=INK_SOFT, width=1)
        draw.text((xx - 8, py + height + 6), f"{v:.3g}", fill=INK_SOFT, font=font)
    draw.line([(px, py + height), (px + width, py + height)], fill=INK_SOFT, width=1)
    draw.text((px + width - 60, py + height + 16), "r [px]", fill=INK_SOFT, font=font)
    for i, (label, y) in enumerate(zip([s[0] for s in series], ys)):
        colour = SERIES[i % len(SERIES)]
        points = [(to_x(a), to_y(b)) for a, b in zip(x, y)]
        draw.line(points, fill=colour, width=2)
        ly = py + 4 + 16 * i
        draw.line([(px + width - 150, ly + 7), (px + width - 130, ly + 7)], fill=colour, width=2)
        draw.text((px + width - 124, ly), label, fill=INK, font=font)


def plot(path, r, profile, params, validation, kernel_px, levels=LEVELS):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()
    width, height = 900, 640
    image = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(image)
    fit = gaussian_sum(r, params)
    _, _, pyramid = validation
    pyramid = pyramid[:r.size]
    line_panel(draw, font, (0, 0, width, height // 2), r,
               [("APSF (Kim-Lin)", profile), ("4-Gaussian fit", fit), (f"{levels}-level pyramid", pyramid)],
               f"APSF radial profile, kernel {kernel_px} px (q {Q}, T {T}, m {M_MAX})", "I / I(0)")
    line_panel(draw, font, (0, height // 2, width, height), r,
               [("fit - APSF", 100.0 * (fit - profile)), ("pyramid - APSF", 100.0 * (pyramid - profile))],
               "residuals", "% of peak")
    image.save(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="APSF glow profile -> Gaussian fit -> pyramid weights")
    parser.add_argument("--kernel-px", type=int, default=64, help="glow radius in pixels (silhouette)")
    parser.add_argument("--q", type=float, default=Q)
    parser.add_argument("--T", type=float, default=T)
    parser.add_argument("--m", type=int, default=M_MAX)
    parser.add_argument("--R", type=float, default=R)
    parser.add_argument("--levels", type=int, default=LEVELS, help="pyramid levels including level 0 (the image)")
    parser.add_argument("--phases", type=int, default=8, help="delta phases per axis for the pyramid synthesis")
    parser.add_argument("--radius", type=float, default=1.0, help="upsample tent radius in coarse texels")
    parser.add_argument("--plot", metavar="PNG", default="/tmp/plasma_apsf.png")
    parser.add_argument("--blue-noise", metavar="PNG", help="generate the blue-noise texture to PNG")
    parser.add_argument("--blue-noise-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    r, profile = apsf_profile(args.q, args.T, args.m, args.kernel_px, args.R)
    r_ref, profile_ref = apsf_profile(args.q, args.T, 4 * args.m, args.kernel_px, args.R)
    mu, inside = kim_lin_mu(r, args.kernel_px, args.R)
    raw = apsf(mu[inside], args.T, args.q, args.m)
    print(f"APSF q {args.q} T {args.T} m {args.m} kappa {KAPPA} D {D} R {args.R}: kernel radius {args.kernel_px} px, "
          f"pixel pitch {KAPPA * args.R / (D * args.kernel_px):.3e} m, theta at the edge {math.degrees(math.acos(mu[inside][-1])):.2f} deg")
    print(f"  pedestal (edge / peak before subtraction) {raw.min() / raw.max():.4f}; "
          f"truncation m {args.m} vs {4 * args.m}: max |dI| {np.abs(profile - profile_ref).max():.2e} of peak")
    kernel = apsf_kernel(args.kernel_px, args.q, args.T, args.m, args.R)
    kernel_raw = apsf_kernel(args.kernel_px, args.q, args.T, args.m, args.R, pedestal=False)
    y, x = np.mgrid[-args.kernel_px:args.kernel_px + 1, -args.kernel_px:args.kernel_px + 1]
    rim = (np.hypot(x, y) <= args.kernel_px) & (np.hypot(x, y) > args.kernel_px - 1.0)
    print(f"  outermost pixel ring (r in ({args.kernel_px - 1}, {args.kernel_px}]) max value {kernel[rim].max():.4f} with the pedestal "
          f"subtracted (continuous ramp to 0) vs a {kernel_raw[rim].max():.4f} step without")
    half = 0.5 * profile[0]
    print(f"  half-maximum radius {r[np.argmax(profile < half)]:.0f} px, I(r/2) {profile[len(profile) // 2]:.3f}")

    params = fit_gaussians(r, profile, 4)
    uniform, area, largest = fit_error(r, profile, params)
    edge = gaussian_sum(r[-1], params) - profile[-1]
    print(f"4-Gaussian fit (amplitude, sigma px): " + ", ".join(f"({a:.4f}, {s:.2f})" for a, s in params))
    print(f"  RMSE {100 * uniform:.3f} % of peak (uniform), {100 * area:.3f} % (area-weighted); max |residual| "
          f"{100 * largest:.2f} %, residual at r = {args.kernel_px} px {100 * edge:+.2f} %; target < {100 * FIT_TARGET:.0f} %"
          + ("  PASS" if uniform < FIT_TARGET else "  FAIL"))

    kernels = pyramid_kernels(args.levels, None, args.phases, args.radius)
    sigmas = pyramid_sigmas(kernels)
    print(f"pyramid level sigmas (measured, {args.phases}^2 phases, {kernels[0].shape[0]}^2 px, tent radius {args.radius}): "
          + ", ".join(f"{s:.2f}" for s in sigmas)
          + "  (constants " + ", ".join(f"{s:.2f}" for s in PYRAMID_SIGMA[:args.levels]) + ")")
    for label, weights in (("Gaussian mapping", pyramid_weights(params, args.levels, sigmas)),
                           ("direct NNLS on synthesised kernels", pyramid_weights_direct(kernel, kernels))):
        report, validation = validate_pyramid(weights, kernel, kernels)
        print(f"pyramid weights ({label}, energy fractions, levels 0..{args.levels - 1}): "
              + ", ".join(f"{w:.4f}" for w in weights))
        print(f"  synthesised pyramid vs APSF profile (r <= {args.kernel_px} px): RMSE {100 * report['rmse']:.2f} % of peak "
              f"(uniform in r), {100 * report['rmse_area']:.2f} % (area-weighted); max |residual| {100 * report['max']:.2f} %, "
              f"at r = {args.kernel_px} px {100 * report['edge']:+.2f} %; gain {report['gain']:.3f}; "
              f"{100 * report['leak']:.1f} % of the energy beyond the kernel radius"
              + ("  PASS" if report["rmse"] < FIT_TARGET else "  FAIL"))
        if label == "Gaussian mapping":
            plot_validation = validation
            single = pyramid_kernels(args.levels, kernels[0].shape[0], 1, args.radius)
            print(f"  single-phase (delta on the block corner): RMSE {100 * validate_pyramid(weights, kernel, single)[0]['rmse']:.2f} %")
    renderer = pyramid_weights_default(args.levels - 1, args.kernel_px, args.q, args.T, args.m)
    print(f"renderer weights pyramid_weights_default({args.levels - 1}, {args.kernel_px}) (downsampled levels 1..{args.levels - 1}, "
          "PYRAMID_SIGMA constants, sum 1): " + ", ".join(f"{w:.4f}" for w in renderer))
    if args.plot:
        print(f"plot: {plot(args.plot, r, profile, params, plot_validation, args.kernel_px, args.levels)}")

    if args.blue_noise:
        from PIL import Image

        noise = void_and_cluster(args.blue_noise_size, seed=args.seed)
        Image.fromarray(noise, mode="L").save(args.blue_noise)
        metric = blue_noise_metric(noise, args.seed)
        counts = np.bincount(noise.ravel(), minlength=256)
        print(f"blue noise {args.blue_noise_size}x{args.blue_noise_size} -> {args.blue_noise}: "
              f"low/high spectral power {metric['blue']:.2e} vs white {metric['white']:.2e} "
              f"({metric['gain']:.1f}x less low-frequency energy); histogram {counts.min()}-{counts.max()} px per level")


if __name__ == "__main__":
    main()
