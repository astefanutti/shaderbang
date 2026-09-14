# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Headless validation metrics for the plasma-globe growth lab (plan section 10).

Pure numpy (no scipy): fractal dimension by box counting and by mass-radius, inter-tree
minimum-distance histograms, Spearman rank correlation, a two-sample Kolmogorov-Smirnov
statistic, tip / shell-contact / angle-to-finger statistics, a NaN scan over an object's
arrays (``find_nonfinite_members``) and GPU timing helpers around ``wp.ScopedTimer``.
"""

import math
import statistics

import numpy as np
import warp as wp


# --- fractal dimension ------------------------------------------------------------------------

def box_counting_dimension(pos, h, n_scales=6):
    """Slope of log N(eps) vs log(1/eps) for box sizes from 2h up to extent/4.

    Returns (D, scales, counts)."""
    pos = np.asarray(pos, dtype=np.float64)
    lo = pos.min(0)
    extent = float((pos.max(0) - lo).max())
    eps_min, eps_max = 2.0 * h, extent / 4.0
    if eps_max <= eps_min:
        eps_max = 2.0 * eps_min
    scales = np.geomspace(eps_min, eps_max, n_scales)
    counts = []
    for eps in scales:
        cells = np.floor((pos - lo) / eps).astype(np.int64)
        counts.append(len(np.unique(cells, axis=0)))
    counts = np.asarray(counts, dtype=np.float64)
    slope = -np.polyfit(np.log(scales), np.log(counts), 1)[0]
    return float(slope), scales, counts


def mass_radius_dimension(pos, center, r_min, r_max, n_radii=12):
    """Slope of log n(R) vs log R for the number of nodes within R of `center`, R in [r_min, r_max].

    Returns (D, radii, counts)."""
    pos = np.asarray(pos, dtype=np.float64)
    d = np.linalg.norm(pos - np.asarray(center, dtype=np.float64), axis=1)
    radii = np.geomspace(r_min, r_max, n_radii)
    counts = np.array([(d <= R).sum() for R in radii], dtype=np.float64)
    keep = counts > 0
    slope = np.polyfit(np.log(radii[keep]), np.log(counts[keep]), 1)[0]
    return float(slope), radii, counts


# --- spacing -----------------------------------------------------------------------------------

def min_distance_between_trees(pos, tree):
    """For every node, the distance to the nearest node of a different tree (nan if none)."""
    pos = np.asarray(pos, dtype=np.float64)
    tree = np.asarray(tree)
    out = np.full(len(pos), np.nan)
    for t in np.unique(tree):
        mine = tree == t
        others = pos[~mine]
        if len(others) == 0:
            continue
        for start in range(0, mine.sum(), 1024):
            xs = pos[mine][start:start + 1024]
            d = np.linalg.norm(xs[:, None, :] - others[None, :, :], axis=2)
            out[np.nonzero(mine)[0][start:start + 1024]] = d.min(1)
    return out


def min_distance_between_nodes(pos, exclude_parent=None):
    """Nearest-neighbour distance per node (optionally ignoring the parent edge)."""
    pos = np.asarray(pos, dtype=np.float64)
    out = np.empty(len(pos))
    for start in range(0, len(pos), 1024):
        xs = pos[start:start + 1024]
        d = np.linalg.norm(xs[:, None, :] - pos[None, :, :], axis=2)
        idx = np.arange(start, start + len(xs))
        d[np.arange(len(xs)), idx] = np.inf
        if exclude_parent is not None:
            par = np.asarray(exclude_parent)[idx]
            ok = par >= 0
            d[np.nonzero(ok)[0], par[ok]] = np.inf
        out[start:start + 1024] = d.min(1)
    return out


# --- statistics ---------------------------------------------------------------------------------

def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic D and its asymptotic p-value."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    d = float(np.abs(cdf_a - cdf_b).max())
    n = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2.0 * sum((-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --- morphology ---------------------------------------------------------------------------------

def distinct_tips(pos, r_min, cos_threshold=0.35):
    """Greedy angular clustering of the nodes beyond r_min; clusters closer than the cosine
    threshold merge. Returns the number of clusters."""
    pos = np.asarray(pos, dtype=np.float64)
    r = np.linalg.norm(pos, axis=1)
    far = pos[r > r_min]
    if len(far) == 0:
        return 0
    dirs = far / np.linalg.norm(far, axis=1)[:, None]
    centers = []
    for d in dirs[np.argsort(-r[r > r_min])]:
        if all(np.dot(d, c) < cos_threshold for c in centers):
            centers.append(d)
    return len(centers)


def shell_contacts(pos, r2, a, tol=1.0e-5):
    r = np.linalg.norm(np.asarray(pos, dtype=np.float64), axis=1)
    return int((r >= r2 - a - tol).sum())


def angles_to_direction(pos, direction):
    pos = np.asarray(pos, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    u = pos / np.linalg.norm(pos, axis=1)[:, None]
    return np.arccos(np.clip(u @ d, -1.0, 1.0))


def halo_profile(potential, r, angles_deg, axis=(0.0, 0.0, 1.0), n_azimuth=24):
    """Min-max normalised mean potential on rings of polar angle `angles_deg` around `axis`
    at radius r (the screening halo probe of dossier followups[0])."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    phis = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
    means = []
    for deg in angles_deg:
        th = math.radians(deg)
        pts = r * (math.cos(th) * axis[None, :]
                   + math.sin(th) * (np.cos(phis)[:, None] * e1 + np.sin(phis)[:, None] * e2))
        means.append(float(np.mean(potential(pts))))
    means = np.asarray(means)
    lo, hi = means.min(), means.max()
    return (means - lo) / (hi - lo) if hi > lo else means * 0.0, means


# --- health ------------------------------------------------------------------------------------

def tree_topology_violations(nodes, h, alive=1, decaying=4, factor=1.5):
    """Counts the live non-root nodes that break the node-SoA topology contract: parent not
    alive, parent of another tree, parent decaying while the node is not, parent farther than
    `factor` h. `nodes` is ``Dbm.nodes()``; returns a dict of counts (all zero when healthy)."""
    ids, parent = np.asarray(nodes["ids"]), np.asarray(nodes["parent"])
    pos, tree, flags = np.asarray(nodes["pos"], dtype=np.float64), np.asarray(nodes["tree"]), np.asarray(nodes["flags"])
    index = np.full(max(int(ids.max()) + 1 if len(ids) else 0, int(parent.max()) + 1 if len(parent) else 0, 1), -1)
    index[ids] = np.arange(len(ids))
    child = parent >= 0
    pi = np.where(child, index[np.where(child, parent, 0)], -1)
    dead = child & (pi < 0)
    ok = child & (pi >= 0)
    pj = np.where(ok, pi, 0)
    out = dict(children=int(child.sum()), parent_dead=int(dead.sum()),
               parent_other_tree=int((ok & (tree[pj] != tree)).sum()),
               parent_decaying=int((ok & ((flags[pj] & decaying) != 0) & ((flags & decaying) == 0)).sum()),
               parent_far=int((ok & (np.linalg.norm(pos - pos[pj], axis=1) > factor * h)).sum()))
    out["total"] = out["parent_dead"] + out["parent_other_tree"] + out["parent_decaying"] + out["parent_far"]
    return out


def find_nonfinite_members(obj, prefix=""):
    """Names of the numpy / Warp array attributes of `obj` that contain NaN or Inf.

    Pattern after Newton's `newton.utils.find_nonfinite_members` (Apache-2.0,
    github.com/newton-physics/newton), re-implemented here without the dependency."""
    bad = []
    for name in sorted(vars(obj)):
        value = getattr(obj, name)
        if isinstance(value, wp.array):
            if value.size == 0 or value.dtype in (wp.int32, wp.int64, wp.uint32, wp.int8, wp.uint8):
                continue
            data = value.numpy()
        elif isinstance(value, np.ndarray):
            data = value
        else:
            continue
        if data.dtype.kind in "fc" and not np.isfinite(data).all():
            bad.append(prefix + name)
    return bad


# --- timing ------------------------------------------------------------------------------------

def gpu_time_ms(fn, reps=20, repeats=3, warmup=2):
    """Median (over `repeats`) of the mean wall time per call, synchronised via ScopedTimer."""
    for _ in range(warmup):
        fn()
    wp.synchronize()
    samples = []
    for _ in range(repeats):
        with wp.ScopedTimer("bench", print=False, synchronize=True) as timer:
            for _ in range(reps):
                fn()
        samples.append(timer.elapsed / reps)
    return statistics.median(samples)


def verdict(ok):
    return "PASS" if ok else "FAIL"
