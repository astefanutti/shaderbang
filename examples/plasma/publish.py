# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Publish stage: turn the persistent filament tree into the buffers the GLSL renderer reads.

Everything runs inside the frame graph on fixed-capacity device arrays (plan 5.1 P0a):

* ``segments`` -- one 112-byte record per live node (the corner-cut span of the link parent ->
  node, see ``k_segments``), as 7 vec4: ``[p0.xyz, radius] [p1.xyz, pack(tree, class, alpha)]
  [rgb_power.xyz, x_ion] [prev_p0.xyz, tree] [prev_p1.xyz, flags] [n_start.xyz, 0] [n_end.xyz, 0]``.
  Node ids are stable across frames, so ``prev_p0/prev_p1`` are the same span built from the
  previous positions: exact per-segment motion vectors for the temporal upscale. ``n_start`` /
  ``n_end`` are the mitre planes at the joints (bisectors of consecutive span directions): the
  tracer clips each capsule's body to them, so consecutive spans tile space with neither the
  wedge gap nor the double count a plain capsule union has at every bend.
* ``cell_start / cell_count / cell_items`` -- uniform-grid CSR over the inner-sphere cube (96^3 cells of
  1.56 mm, dilation radius 1 cell) traversed by Amanatides-Woo DDA in the shader; built by
  count -> exclusive scan -> scatter (no sort).
* ``lights`` -- per-tree coarse polylines of the main channel (<= 8 long segments carrying the
  summed power of their member segments) for the analytic Karis line-light shading.
* ``ambient`` -- a wide Gaussian splat of the segment power into a 96^3 grid for the coarse
  volume march (glow / heat shimmer); packed with the gas fields into a float16 RGBA volume.

Channel classes follow Kim & Lin (main / secondary / side = 1 : 0.25 : 0.06): the main channel
is the foot -> root chain; off-main nodes are classed by their descendant count (a GPU proxy
for "longest path in each subtree", cf. the ancestor counting of triggered-discharge).

DECISION (deviation from the plan's single interleaved Frame SSBO): the render buffers are
separate GL buffers mapped with ONE batched ``cuGraphicsMapResources`` per frame (measured
0.19 ms for 8 resources), which keeps every buffer a plain typed ``wp.array``.
"""

import math

import numpy as np
import warp as wp

from plasma.params import (
    PlasmaParams, R1, R2I, NODE_SPACING,
    NODE_ALIVE, NODE_DECAYING, NODE_FOOT, NODE_ROOT, NODE_MAINCUT,
    TREE_ATTACHED, TREE_GROW, TREE_RETRACT,
)

N_MAX = 8192                 # node capacity (must match plasma.dbm)
F_MAX = 32
SEG_MAX = N_MAX              # one segment per node
SEG_STRIDE = 7               # vec4 per segment record (see k_segments)
ROOT_R = 0.011                # electrode radius (plasma.params.R1)
ROOT_FLARE = 2.5              # extra brightness at the root (x3.5 at the bulb surface)
ROOT_FLARE_LEN = 2.5e-3       # m: e-folding length of the root flare
GRID_N = 96
GRID_CELLS = GRID_N * GRID_N * GRID_N
GRID_EXTENT = R2I            # grid covers [-R2I, R2I]^3
CELL_SIZE = 2.0 * GRID_EXTENT / GRID_N
DILATION = 2                 # cells: the sheath (4 core radii, ~2 mm) must stay inside the footprint a ray
                             # visits, else the halo is clipped at cell boundaries (blocky); the shader
                             # fades the halo to zero at DILATION cells (VIS_RADIUS)
LONG_SEG_CELLS = 2           # a segment spanning more cells than this on an axis is rasterised by
                             # sampling along it (an advected link can reach centimetres: its AABB
                             # would cost thousands of atomics per segment)
ITEMS_MAX = SEG_MAX * 125    # dilation 2 -> at most 125 cells per ~1-cell segment
LIGHTS_PER_TREE = 8
BRUSH_FEET = 4               # must match plasma.dbm
LIGHT_MAX = F_MAX * LIGHTS_PER_TREE
AMBIENT_RADIUS = 8.0e-3      # m, wide splat for the volume glow (the diffuse discharge around the channels)
CLASS_MAIN = 1.0
CLASS_SECONDARY = 0.30       # the foot brush at the glass shares the foot's current; the transient
CLASS_SIDE = 0.02            # streamer branches of a strike are faint (Kim & Lin: 0.25 / 0.06 for lightning)
BRUSH_R = R2I - 0.012        # off-main nodes beyond this radius are brush (see dbm.FOOT_BRUSH)
SECONDARY_MIN_DESCENDANTS = 8
NODE_MAIN = 128              # transient flag bit: node is on a main channel this frame
NODE_BRUSH = 256             # transient flag bit: node is on a brush-foot chain (the channel's current is
                             # split between the trunk's foot and its brush feet)

# node flag helpers ---------------------------------------------------------------------------------

@wp.func
def pack_tree_class_alpha(tree: int, cls: float, alpha: float) -> float:
    """tree (0..31) in the integer part, class code (0/1/2) in tenths, alpha in thousandths."""
    code = float(0.0)
    if cls == CLASS_MAIN:
        code = 1.0
    elif cls == CLASS_SECONDARY:
        code = 2.0
    return float(tree) + 0.1 * code + 0.001 * wp.clamp(alpha, 0.0, 0.99)


@wp.kernel
def k_mark_main(tree_state: wp.array(dtype=wp.int32),
                tree_foot: wp.array(dtype=wp.int32),
                tree_tip: wp.array(dtype=wp.int32),
                node_parent: wp.array(dtype=wp.int32),
                node_flags: wp.array(dtype=wp.int32),
                node_next: wp.array(dtype=wp.int32),
                tree_foot2: wp.array(dtype=wp.int32),
                tree_nfeet: wp.array(dtype=wp.int32)):
    """Flag the main channel of every tree (foot -> root when attached, else tip -> root) and
    the chains of its secondary (brush) feet, count the feet, and record each main node's
    successor towards its foot (``node_next``, -1 elsewhere)."""
    k = wp.tid()
    nfeet = int(1)
    for j in range(BRUSH_FEET):
        if tree_foot2[k * BRUSH_FEET + j] >= 0:
            nfeet += 1
    tree_nfeet[k] = nfeet
    if tree_state[k] == 0:
        return
    for j in range(BRUSH_FEET + 1):
        n = tree_foot[k]
        if j == 0:
            if n < 0:
                n = tree_tip[k]
        else:
            n = tree_foot2[k * BRUSH_FEET + j - 1]
        prev = int(-1)
        steps = int(0)
        while n >= 0 and steps < 4096:
            f = node_flags[n]
            if j > 0 and (f & NODE_MAIN) != 0:
                break                      # joined the main channel
            node_flags[n] = f | NODE_MAIN | wp.where(j > 0, NODE_BRUSH, 0)
            node_next[n] = prev
            prev = n
            n = node_parent[n]
            steps += 1


@wp.func
def chain_smooth(x: wp.array(dtype=wp.vec3), parent: wp.array(dtype=wp.int32), nxt: wp.array(dtype=wp.int32),
                 flags: wp.array(dtype=wp.int32), i: int) -> wp.vec3:
    """Render position of node i: a 5-tap binomial average along its main chain
    (x[gp] + 4 x[p] + 6 x[i] + 4 x[c] + x[cc]) / 16, ends clamped, for main-channel nodes; the
    node itself elsewhere. The node spacing (1.5 mm) is a discretisation choice, so the channel
    is drawn as the low-passed (quadratic B-spline like) path, not the lattice walk."""
    if (flags[i] & NODE_MAIN) == 0:
        return x[i]
    p = parent[i]
    if p < 0:
        p = i
    g = parent[p]
    if g < 0:
        g = p
    c = nxt[i]
    if c < 0:
        c = i
    cc = nxt[c]
    if cc < 0:
        cc = c
    return (x[g] + 4.0 * x[p] + 6.0 * x[i] + 4.0 * x[c] + x[cc]) / 16.0


@wp.kernel
def k_count_descendants(node_flags: wp.array(dtype=wp.int32),
                        node_parent: wp.array(dtype=wp.int32),
                        node_desc: wp.array(dtype=wp.int32)):
    """Each off-main live node credits every off-main ancestor up to the main channel."""
    i = wp.tid()
    f = node_flags[i]
    if (f & NODE_ALIVE) == 0 or (f & NODE_MAIN) != 0:
        return
    n = node_parent[i]
    steps = int(0)
    while n >= 0 and steps < 4096:
        if (node_flags[n] & NODE_MAIN) != 0:
            break
        wp.atomic_add(node_desc, n, 1)
        n = node_parent[n]
        steps += 1


@wp.kernel
def k_segments(node_pos: wp.array(dtype=wp.vec3),
               node_prev_pos: wp.array(dtype=wp.vec3),
               node_parent: wp.array(dtype=wp.int32),
               node_tree: wp.array(dtype=wp.int32),
               node_flags: wp.array(dtype=wp.int32),
               node_next: wp.array(dtype=wp.int32),
               node_alpha: wp.array(dtype=wp.float32),
               node_desc: wp.array(dtype=wp.int32),
               node_xion: wp.array(dtype=wp.float32),
               tree_current: wp.array(dtype=wp.float32),
               tree_radius: wp.array(dtype=wp.float32),
               tree_nfeet: wp.array(dtype=wp.int32),
               color_neutral: wp.array(dtype=wp.vec3),
               color_ion: wp.array(dtype=wp.vec3),
               segments: wp.array(dtype=wp.vec4),
               seg_valid: wp.array(dtype=wp.int32),
               node_class: wp.array(dtype=wp.float32)):
    """One segment record per live node with a parent; invalid slots get radius 0.

    The rendered polyline is the corner-cut (one Chaikin pass) of the node chain: node i emits
    the span from the midpoint of its link to the midpoint of its parent's link (the root itself
    when the parent is the root). The 1.5 mm growth zigzag halves in amplitude, the channel keeps
    its path, and each leaf ends half a link short of its node (inside the foot disc at the
    glass). Previous positions are cut the same way so the motion vectors stay consistent."""
    i = wp.tid()
    f = node_flags[i]
    p = node_parent[i]
    valid = (f & NODE_ALIVE) != 0 and p >= 0
    seg_valid[i] = wp.where(valid, 1, 0)
    base = SEG_STRIDE * i
    if not valid:
        segments[base] = wp.vec4(0.0)
        node_class[i] = 0.0
        return
    t = node_tree[i]
    cls = CLASS_SIDE
    if (f & NODE_MAIN) != 0:
        cls = CLASS_MAIN
    elif node_desc[i] >= SECONDARY_MIN_DESCENDANTS or ((f & NODE_DECAYING) == 0 and wp.length(node_pos[i]) > BRUSH_R):
        cls = CLASS_SECONDARY
    node_class[i] = cls
    alpha = float(1.0)
    if (f & NODE_DECAYING) != 0:
        alpha = node_alpha[i]
    cur = tree_current[t]
    radius = tree_radius[t]
    if (f & NODE_BRUSH) != 0:
        # a brush branch carries its share of the channel's current: 1/(feet) of the power and
        # the matching thinner core (r ~ I^0.4)
        share = 1.0 / float(tree_nfeet[t])
        cur = cur * share
        radius = radius * wp.pow(share, 0.4)
    x_ion = node_xion[i]
    col = color_neutral[t] * (1.0 - x_ion) + color_ion[t] * x_ion
    # radiance grows faster than the current (a hotter channel): P ~ I^1.5; the footage's touched
    # channel (~1 mA) saturates all three camera channels while a 40 uA one is a thin violet line
    power = cls * cur * alpha * wp.sqrt(wp.max(cur, 1.0e-9) / 5.0e-5)
    # root flare: the first millimetres of a channel at the electrode are brighter (the footage's
    # pink-white flares where the filaments leave the bulb), e-folding ROOT_FLARE_LEN
    power = power * (1.0 + ROOT_FLARE * wp.exp(-(wp.length(node_pos[i]) - ROOT_R) / ROOT_FLARE_LEN))
    g = node_parent[p]
    xi = chain_smooth(node_pos, node_parent, node_next, node_flags, i)
    xp = chain_smooth(node_pos, node_parent, node_next, node_flags, p)
    p1 = 0.5 * (xi + xp)
    qp = chain_smooth(node_prev_pos, node_parent, node_next, node_flags, p)
    q1 = 0.5 * (chain_smooth(node_prev_pos, node_parent, node_next, node_flags, i) + qp)
    p0 = xp
    q0 = qp
    xg = xp
    if g >= 0:
        xg = chain_smooth(node_pos, node_parent, node_next, node_flags, g)
        p0 = 0.5 * (xp + xg)
        q0 = 0.5 * (qp + chain_smooth(node_prev_pos, node_parent, node_next, node_flags, g))
    # mitre planes: bisectors between this span and its neighbours (flat cut where there is none)
    u = p1 - p0
    u = u / wp.max(wp.length(u), 1.0e-9)
    u_prev = u
    if g >= 0:
        gg = node_parent[g]
        p0_prev = xg
        if gg >= 0:
            p0_prev = 0.5 * (xg + chain_smooth(node_pos, node_parent, node_next, node_flags, gg))
        v = p0 - p0_prev
        if wp.length(v) > 1.0e-6:
            u_prev = v / wp.length(v)
    u_next = u
    c = node_next[i]
    if c >= 0:
        v2 = 0.5 * (chain_smooth(node_pos, node_parent, node_next, node_flags, c) + xi) - p1
        if wp.length(v2) > 1.0e-6:
            u_next = v2 / wp.length(v2)
    n_a = u_prev + u
    if wp.length(n_a) > 1.0e-6:
        n_a = n_a / wp.length(n_a)
    else:
        n_a = u
    n_b = u + u_next
    if wp.length(n_b) > 1.0e-6:
        n_b = n_b / wp.length(n_b)
    else:
        n_b = u
    segments[base + 0] = wp.vec4(p0[0], p0[1], p0[2], radius)
    segments[base + 1] = wp.vec4(p1[0], p1[1], p1[2], pack_tree_class_alpha(t, cls, alpha))
    segments[base + 2] = wp.vec4(col[0] * power, col[1] * power, col[2] * power, x_ion)
    segments[base + 3] = wp.vec4(q0[0], q0[1], q0[2], float(t))
    segments[base + 4] = wp.vec4(q1[0], q1[1], q1[2], float(f))
    segments[base + 5] = wp.vec4(n_a[0], n_a[1], n_a[2], 0.0)
    segments[base + 6] = wp.vec4(n_b[0], n_b[1], n_b[2], 0.0)


@wp.func
def cell_coord(p: wp.vec3) -> wp.vec3i:
    c = (p + wp.vec3(GRID_EXTENT)) / CELL_SIZE
    return wp.vec3i(int(wp.floor(c[0])), int(wp.floor(c[1])), int(wp.floor(c[2])))


@wp.func
def cell_index(c: wp.vec3i) -> int:
    return (c[2] * GRID_N + c[1]) * GRID_N + c[0]


@wp.func
def seg_extent(a: wp.vec3, b: wp.vec3):
    """Cell AABB of a segment and its largest axis span in cells."""
    lo = cell_coord(wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2])))
    hi = cell_coord(wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2])))
    ext = wp.max(wp.max(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2])
    return lo, hi, ext


@wp.func
def in_cube(c: wp.vec3i, p: wp.vec3i) -> bool:
    return wp.abs(c[0] - p[0]) <= DILATION and wp.abs(c[1] - p[1]) <= DILATION and wp.abs(c[2] - p[2]) <= DILATION


@wp.kernel
def k_csr_count(segments: wp.array(dtype=wp.vec4), seg_valid: wp.array(dtype=wp.int32),
                cell_count: wp.array(dtype=wp.int32)):
    """Dilated cells touched by a segment: the AABB for ordinary (~1 cell) segments, else samples
    at <= 1 cell spacing along it, each dilated, skipping the cells the previous sample covered."""
    s = wp.tid()
    if seg_valid[s] == 0:
        return
    a = segments[SEG_STRIDE * s]
    b = segments[SEG_STRIDE * s + 1]
    pa = wp.vec3(a[0], a[1], a[2])
    pb = wp.vec3(b[0], b[1], b[2])
    lo, hi, ext = seg_extent(pa, pb)
    if ext <= LONG_SEG_CELLS:
        for z in range(wp.max(lo[2] - DILATION, 0), wp.min(hi[2] + DILATION, GRID_N - 1) + 1):
            for y in range(wp.max(lo[1] - DILATION, 0), wp.min(hi[1] + DILATION, GRID_N - 1) + 1):
                for x in range(wp.max(lo[0] - DILATION, 0), wp.min(hi[0] + DILATION, GRID_N - 1) + 1):
                    wp.atomic_add(cell_count, cell_index(wp.vec3i(x, y, z)), 1)
        return
    n = ext + 1
    prev = wp.vec3i(-1000, -1000, -1000)
    for k in range(n + 1):
        ck = cell_coord(pa + (pb - pa) * (float(k) / float(n)))
        if ck[0] == prev[0] and ck[1] == prev[1] and ck[2] == prev[2]:
            continue
        for z in range(wp.max(ck[2] - DILATION, 0), wp.min(ck[2] + DILATION, GRID_N - 1) + 1):
            for y in range(wp.max(ck[1] - DILATION, 0), wp.min(ck[1] + DILATION, GRID_N - 1) + 1):
                for x in range(wp.max(ck[0] - DILATION, 0), wp.min(ck[0] + DILATION, GRID_N - 1) + 1):
                    c = wp.vec3i(x, y, z)
                    if not in_cube(c, prev):
                        wp.atomic_add(cell_count, cell_index(c), 1)
        prev = ck


@wp.kernel
def k_csr_scatter(segments: wp.array(dtype=wp.vec4), seg_valid: wp.array(dtype=wp.int32),
                  cell_start: wp.array(dtype=wp.int32), cell_fill: wp.array(dtype=wp.int32),
                  cell_items: wp.array(dtype=wp.int32)):
    """Same enumeration as k_csr_count (the two must agree cell for cell)."""
    s = wp.tid()
    if seg_valid[s] == 0:
        return
    a = segments[SEG_STRIDE * s]
    b = segments[SEG_STRIDE * s + 1]
    pa = wp.vec3(a[0], a[1], a[2])
    pb = wp.vec3(b[0], b[1], b[2])
    lo, hi, ext = seg_extent(pa, pb)
    if ext <= LONG_SEG_CELLS:
        for z in range(wp.max(lo[2] - DILATION, 0), wp.min(hi[2] + DILATION, GRID_N - 1) + 1):
            for y in range(wp.max(lo[1] - DILATION, 0), wp.min(hi[1] + DILATION, GRID_N - 1) + 1):
                for x in range(wp.max(lo[0] - DILATION, 0), wp.min(hi[0] + DILATION, GRID_N - 1) + 1):
                    c = cell_index(wp.vec3i(x, y, z))
                    slot = cell_start[c] + wp.atomic_add(cell_fill, c, 1)
                    if slot < ITEMS_MAX:
                        cell_items[slot] = s
        return
    n = ext + 1
    prev = wp.vec3i(-1000, -1000, -1000)
    for k in range(n + 1):
        ck = cell_coord(pa + (pb - pa) * (float(k) / float(n)))
        if ck[0] == prev[0] and ck[1] == prev[1] and ck[2] == prev[2]:
            continue
        for z in range(wp.max(ck[2] - DILATION, 0), wp.min(ck[2] + DILATION, GRID_N - 1) + 1):
            for y in range(wp.max(ck[1] - DILATION, 0), wp.min(ck[1] + DILATION, GRID_N - 1) + 1):
                for x in range(wp.max(ck[0] - DILATION, 0), wp.min(ck[0] + DILATION, GRID_N - 1) + 1):
                    cc = wp.vec3i(x, y, z)
                    if not in_cube(cc, prev):
                        c = cell_index(cc)
                        slot = cell_start[c] + wp.atomic_add(cell_fill, c, 1)
                        if slot < ITEMS_MAX:
                            cell_items[slot] = s
        prev = ck


@wp.kernel
def k_lights(tree_state: wp.array(dtype=wp.int32),
             tree_foot: wp.array(dtype=wp.int32),
             tree_tip: wp.array(dtype=wp.int32),
             tree_length: wp.array(dtype=wp.float32),
             node_pos: wp.array(dtype=wp.vec3),
             node_parent: wp.array(dtype=wp.int32),
             node_s_arc: wp.array(dtype=wp.float32),
             segments: wp.array(dtype=wp.vec4),
             lights: wp.array(dtype=wp.vec4)):
    """Resample each tree's main channel into <= LIGHTS_PER_TREE spans, summing member power.

    lights[(k*LIGHTS_PER_TREE + j)*3 + 0..2] = [p0, valid] [p1, tree] [rgb_power, 0]."""
    k = wp.tid()
    base = k * LIGHTS_PER_TREE * 3
    for j in range(LIGHTS_PER_TREE):
        lights[base + 3 * j] = wp.vec4(0.0)
        lights[base + 3 * j + 1] = wp.vec4(0.0)
        lights[base + 3 * j + 2] = wp.vec4(0.0)
    if tree_state[k] == 0:
        return
    n = tree_foot[k]
    if n < 0:
        n = tree_tip[k]
    if n < 0:
        return
    length = node_s_arc[n]
    if length <= 0.0:
        return
    span = length / float(LIGHTS_PER_TREE)
    j = int(LIGHTS_PER_TREE - 1)
    end_pos = node_pos[n]
    power = wp.vec3(0.0)
    steps = int(0)
    while n >= 0 and steps < 4096:
        seg = segments[SEG_STRIDE * n + 2]
        power += wp.vec3(seg[0], seg[1], seg[2])
        pnode = node_parent[n]
        s = node_s_arc[n]
        # close the current span when we cross its lower boundary
        if pnode < 0 or node_s_arc[pnode] <= float(j) * span:
            start_pos = node_pos[n]
            if pnode >= 0:
                start_pos = node_pos[pnode]
            lights[base + 3 * j] = wp.vec4(start_pos[0], start_pos[1], start_pos[2], 1.0)
            lights[base + 3 * j + 1] = wp.vec4(end_pos[0], end_pos[1], end_pos[2], float(k))
            lights[base + 3 * j + 2] = wp.vec4(power[0], power[1], power[2], 0.0)
            end_pos = start_pos
            power = wp.vec3(0.0)
            j -= 1
            if j < 0:
                break
        n = pnode
        steps += 1


@wp.kernel
def k_ambient_splat(segments: wp.array(dtype=wp.vec4), seg_valid: wp.array(dtype=wp.int32),
                    ambient: wp.array3d(dtype=wp.float32)):
    """Wide Gaussian splat of the segment power (luminance) around its midpoint."""
    s = wp.tid()
    if seg_valid[s] == 0:
        return
    a = segments[SEG_STRIDE * s]
    b = segments[SEG_STRIDE * s + 1]
    e = segments[SEG_STRIDE * s + 2]
    lum = 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]
    if lum <= 0.0:
        return
    mid = 0.5 * (wp.vec3(a[0], a[1], a[2]) + wp.vec3(b[0], b[1], b[2]))
    c = cell_coord(mid)
    r = int(wp.ceil(2.0 * AMBIENT_RADIUS / CELL_SIZE))
    inv2s2 = 1.0 / (2.0 * AMBIENT_RADIUS * AMBIENT_RADIUS)
    for z in range(wp.max(c[2] - r, 0), wp.min(c[2] + r, GRID_N - 1) + 1):
        for y in range(wp.max(c[1] - r, 0), wp.min(c[1] + r, GRID_N - 1) + 1):
            for x in range(wp.max(c[0] - r, 0), wp.min(c[0] + r, GRID_N - 1) + 1):
                q = (wp.vec3(float(x), float(y), float(z)) + wp.vec3(0.5)) * CELL_SIZE - wp.vec3(GRID_EXTENT)
                d2 = wp.length_sq(q - mid)
                wp.atomic_add(ambient, x, y, z, lum * wp.exp(-d2 * inv2s2))


@wp.kernel
def k_pack_volume(ambient: wp.array3d(dtype=wp.float32),
                  temperature: wp.array3d(dtype=wp.float32),
                  speed: wp.array3d(dtype=wp.float32),
                  volume: wp.array3d(dtype=wp.vec4h)):
    """RGBA16F volume: {T - T0, ambient emission, |u|, mask (1 inside the gas)}."""
    x, y, z = wp.tid()
    q = (wp.vec3(float(x), float(y), float(z)) + wp.vec3(0.5)) * CELL_SIZE - wp.vec3(GRID_EXTENT)
    r = wp.length(q)
    mask = wp.where(r < R2I and r > R1, 1.0, 0.0)
    volume[z, y, x] = wp.vec4h(wp.float16(temperature[x, y, z]), wp.float16(ambient[x, y, z]),
                               wp.float16(speed[x, y, z]), wp.float16(mask))


class Publisher:
    """Fixed-capacity render buffers + the per-frame publish launches (graph-capturable)."""

    def __init__(self, device="cuda:0"):
        self.device = device
        z = lambda dtype, shape: wp.zeros(shape, dtype=dtype, device=device)
        self.segments = z(wp.vec4, SEG_MAX * SEG_STRIDE)
        self.seg_valid = z(wp.int32, SEG_MAX)
        self.node_desc = z(wp.int32, N_MAX)
        self.node_next = wp.full(N_MAX, -1, dtype=wp.int32, device=device)
        self.no_foot2 = wp.full(F_MAX * BRUSH_FEET, -1, dtype=wp.int32, device=device)
        self.tree_nfeet = wp.ones(F_MAX, dtype=wp.int32, device=device)
        self.node_class = z(wp.float32, N_MAX)
        self.cell_count = z(wp.int32, GRID_CELLS)
        self.cell_start = z(wp.int32, GRID_CELLS)   # exclusive prefix sum; end of cell c = start[c] + count[c]
        self.cell_fill = z(wp.int32, GRID_CELLS)
        self.cell_items = z(wp.int32, ITEMS_MAX)
        self.lights = z(wp.vec4, LIGHT_MAX * 3)
        self.ambient = z(wp.float32, (GRID_N, GRID_N, GRID_N))
        self.zero_grid = z(wp.float32, (GRID_N, GRID_N, GRID_N))
        self.volume = z(wp.vec4h, (GRID_N, GRID_N, GRID_N))
        # warm up the scan's temporary storage before any graph capture
        wp.utils.array_scan(self.cell_count, self.cell_start, inclusive=False)

    def launch(self, nodes, trees, temperature=None, speed=None, pack_volume=True):
        """nodes / trees: objects exposing the arrays named below (plasma.dbm's SoA)."""
        d = self.device
        self.node_next.fill_(-1)
        foot2 = getattr(trees, "foot2", None)
        wp.launch(k_mark_main, dim=F_MAX,
                  inputs=[trees.state, trees.foot, trees.tip, nodes.parent, nodes.flags, self.node_next,
                          foot2 if foot2 is not None else self.no_foot2, self.tree_nfeet], device=d)
        self.node_desc.zero_()
        wp.launch(k_count_descendants, dim=N_MAX, inputs=[nodes.flags, nodes.parent, self.node_desc], device=d)
        wp.launch(k_segments, dim=N_MAX,
                  inputs=[nodes.pos, nodes.prev_pos, nodes.parent, nodes.tree, nodes.flags, self.node_next,
                          nodes.alpha, self.node_desc, nodes.x_ion, trees.current, trees.radius, self.tree_nfeet,
                          trees.color_neutral, trees.color_ion,
                          self.segments, self.seg_valid, self.node_class], device=d)
        self.cell_count.zero_()
        self.cell_fill.zero_()
        wp.launch(k_csr_count, dim=SEG_MAX, inputs=[self.segments, self.seg_valid, self.cell_count], device=d)
        wp.utils.array_scan(self.cell_count, self.cell_start, inclusive=False)
        wp.launch(k_csr_scatter, dim=SEG_MAX,
                  inputs=[self.segments, self.seg_valid, self.cell_start, self.cell_fill, self.cell_items], device=d)
        wp.launch(k_lights, dim=F_MAX,
                  inputs=[trees.state, trees.foot, trees.tip, trees.length, nodes.pos, nodes.parent,
                          nodes.s_arc, self.segments, self.lights], device=d)
        if pack_volume:   # the coupled app lets the gas solver pack the volume instead
            wp.copy(self.ambient, self.zero_grid)
            wp.launch(k_ambient_splat, dim=SEG_MAX, inputs=[self.segments, self.seg_valid, self.ambient], device=d)
            wp.launch(k_pack_volume, dim=(GRID_N, GRID_N, GRID_N),
                      inputs=[self.ambient, temperature if temperature is not None else self.zero_grid,
                              speed if speed is not None else self.zero_grid, self.volume], device=d)
        # clear the transient main-channel bit for the next frame
        wp.launch(_k_clear_main, dim=N_MAX, inputs=[nodes.flags], device=d)


@wp.kernel
def _k_clear_main(node_flags: wp.array(dtype=wp.int32)):
    i = wp.tid()
    node_flags[i] = node_flags[i] & ~(NODE_MAIN | NODE_BRUSH)


# ---- self-test on a synthetic star ------------------------------------------------------------------

class _SoA:
    pass


def synthetic_star(n_trees=13, n_nodes=150, seed=1, device="cuda:0"):
    """13 tortuous chains of 150 nodes from the electrode to the glass, with a side branch each."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((N_MAX, 3), np.float32); parent = -np.ones(N_MAX, np.int32)
    tree = np.zeros(N_MAX, np.int32); flags = np.zeros(N_MAX, np.int32); s_arc = np.zeros(N_MAX, np.float32)
    t_state = np.zeros(F_MAX, np.int32); t_foot = -np.ones(F_MAX, np.int32); t_tip = -np.ones(F_MAX, np.int32)
    t_len = np.zeros(F_MAX, np.float32)
    nid = 0
    for k in range(n_trees):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        p = R1 * d
        prev = -1
        for i in range(n_nodes):
            step = d + 0.35 * rng.normal(size=3); step /= np.linalg.norm(step)
            d = 0.7 * d + 0.3 * step; d /= np.linalg.norm(d)
            p = p + NODE_SPACING * d
            if np.linalg.norm(p) >= R2I - CHARGE_R:
                break
            pos[nid] = p; parent[nid] = prev; tree[nid] = k; flags[nid] = NODE_ALIVE
            s_arc[nid] = (i + 1) * NODE_SPACING
            prev = nid; nid += 1
        flags[prev] |= NODE_FOOT; t_foot[k] = prev; t_tip[k] = prev; t_len[k] = s_arc[prev]; t_state[k] = TREE_ATTACHED
        # a side branch off the middle of the chain
        mid = prev - n_nodes // 2 if prev - n_nodes // 2 > 0 else prev
        bp = pos[mid].copy(); bd = np.cross(pos[mid], [0, 1, 0]); bd /= np.linalg.norm(bd) + 1e-9
        bprev = mid
        for j in range(20):
            bp = bp + NODE_SPACING * bd
            pos[nid] = bp; parent[nid] = bprev; tree[nid] = k; flags[nid] = NODE_ALIVE
            s_arc[nid] = s_arc[mid] + (j + 1) * NODE_SPACING
            bprev = nid; nid += 1
    nodes = _SoA()
    nodes.pos = wp.array(pos, dtype=wp.vec3, device=device)
    nodes.prev_pos = wp.array(pos - np.float32(1e-4), dtype=wp.vec3, device=device)
    nodes.parent = wp.array(parent, dtype=wp.int32, device=device)
    nodes.tree = wp.array(tree, dtype=wp.int32, device=device)
    nodes.flags = wp.array(flags, dtype=wp.int32, device=device)
    nodes.alpha = wp.ones(N_MAX, dtype=wp.float32, device=device)
    nodes.s_arc = wp.array(s_arc, dtype=wp.float32, device=device)
    nodes.x_ion = wp.full(N_MAX, 0.7, dtype=wp.float32, device=device)
    trees = _SoA()
    trees.state = wp.array(t_state, dtype=wp.int32, device=device)
    trees.foot = wp.array(t_foot, dtype=wp.int32, device=device)
    trees.tip = wp.array(t_tip, dtype=wp.int32, device=device)
    trees.length = wp.array(t_len, dtype=wp.float32, device=device)
    trees.current = wp.full(F_MAX, 5e-5, dtype=wp.float32, device=device)
    trees.radius = wp.full(F_MAX, 5e-4, dtype=wp.float32, device=device)
    trees.color_neutral = wp.full(F_MAX, wp.vec3(1.0, 0.45, 0.2), dtype=wp.vec3, device=device)
    trees.color_ion = wp.full(F_MAX, wp.vec3(0.4, 0.5, 1.0), dtype=wp.vec3, device=device)
    return nodes, trees, nid


CHARGE_R = NODE_SPACING / 4.0

if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
    wp.init()
    dev = args.device
    nodes, trees, n_live = synthetic_star(device=dev)
    pub = Publisher(dev)
    pub.launch(nodes, trees); wp.synchronize()
    with wp.ScopedCapture(device=dev) as cap:
        pub.launch(nodes, trees)
    g = cap.graph
    wp.capture_launch(g); wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        wp.capture_launch(g)
    wp.synchronize()
    ms = (time.perf_counter() - t0) / 100 * 1e3
    seg = pub.segments.numpy().reshape(SEG_MAX, SEG_STRIDE, 4); valid = pub.seg_valid.numpy()
    cls = pub.node_class.numpy()
    cs = pub.cell_start.numpy(); cc = pub.cell_count.numpy()
    occ = int((cc > 0).sum()); items = int(cs[-1] + cc[-1])
    lights = pub.lights.numpy().reshape(LIGHT_MAX, 3, 4)
    nl = int((lights[:, 0, 3] > 0).sum())
    amb = pub.ambient.numpy()
    print(f"live nodes {n_live}, segments {int(valid.sum())}, classes main/sec/side = "
          f"{int((cls == CLASS_MAIN).sum())}/{int((cls == CLASS_SECONDARY).sum())}/{int((cls == CLASS_SIDE).sum())}")
    print(f"CSR: occupied cells {occ}/{GRID_CELLS} ({100*occ/GRID_CELLS:.2f} %), items {items} (cap {ITEMS_MAX}), "
          f"max per cell {int(cc.max())}, mean per occupied {items/max(occ,1):.1f}")
    print(f"lights: {nl} spans (expected {13*LIGHTS_PER_TREE}); light power sum {lights[:, 2, :3].sum():.3e} vs segment power sum on main channels {seg[cls == CLASS_MAIN, 2, :3].sum():.3e}")
    print(f"ambient volume: max {amb.max():.3e}, nonzero cells {(amb > 0).sum()}")
    print(f"publish graph: {ms:.3f} ms/frame (13x150 star, {int(valid.sum())} segments)")
    ok = items < ITEMS_MAX and nl == 13 * LIGHTS_PER_TREE and (cls == CLASS_MAIN).sum() > 0
    print("publish self-test:", "PASS" if ok else "FAIL")
