# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Dielectric-breakdown growth engine for the plasma globe (design plan sections 4.2-4.4, 4.8).

Everything lives in persistent, fixed-capacity Warp arrays so that one frame of growth
(``Dbm.step``) captures into a single CUDA graph and replays bit-identically for a fixed seed.

Potential (dimensionless: electrode + aggregate = 0, glass = 1), harmonic split::

    u0(x)  = (1/R1 - 1/|x|) / (1/R1 - 1/R2)                      exact n = 0 annulus solution
    phi(x) = u0(x)
           + sum_f q_f [ 1/|x - d_f| - (R1/|d_f|) / |x - d_f'| ]   finger + Kelvin image in the electrode
           + sum_j q_j g(x - x_j)                                   ONE shared charge list (all live nodes)
    g(r)   = 1/sqrt(|r|^2 + a^2)                                    inverse multiquadric: a charge smeared over
                                                                    the conductor radius (1/r beyond a few a)
    M q    = -(u0(x_j) + finger terms(x_j)),  M_jk = g(x_j - x_k),  M_jj = 1/a

The inverse multiquadric (CHOSEN 2026-09-14, replacing bare 1/r) keeps M strictly positive
definite for *any* node configuration: the coupled app advects nodes with the gas and packs
them closer than a, where the 1/r system lost definiteness and the warm-started PCG diverged.

``q`` is solved by a matrix-free Jacobi-preconditioned conjugate gradient (``n_cg`` warm-started
iterations per frame); nodes appended inside a frame carry the Born estimate
``q = -phi(x) / M_jj`` (the diagonal of the same system, ``1/a`` minus the node's own image) until
the next solve.

Growth (per tree pool, per unrolled step ``s`` of ``S_MAX``; a committed node spawns K = 8
candidates in a forward cone about its growth direction, ``cone_dir``: the straight continuation
must always be on offer, see the note there)::

    E_c   = V phi_c / h                                  tip field at candidate c
    gate  = E_c (T_c/T0) >= E_thr                        reduced-field breakdown gate
    Phi_c = (phi_c - min_pool) / (max_pool - min_pool)   Kim et al. 2007 Eq.13, previous step's extrema
    w_c   = gate [ Phi_c (T_c/T0)^gamma s_c ]^eta
    key_c = ln w_c + Gumbel(seed, frame S_MAX + s, c)
    winner = atomic_max over the pool of (quantised key << 32 | c)     p(c) is exactly proportional to w_c

Candidates are off-lattice, K per new node at distance h in a randomly rotated octahedral frame
(the eight octant directions), rejected inside R1 + a, projected onto R2 - a (foot) and rejected
within h/2 of any node. An "electrode pool" of E_POOL candidates on r = R1 + h (Fibonacci sphere,
random rotation per frame) is keyed against the full potential with E_thr = E_bd0; at most one
new tree per frame, only when the circuit admits it.

Parameter provenance (plan 4.1):

    R1 = 0.015 m, R2 = 0.075 m         MEASURED (PPPL-4485) / DECISION
    h = 1.5 mm, a = h/4                 CHOSEN / DERIVED (a/h swept in M2)
    E_bd0 = 190 kV/m, E_prop0 = E_bd0/4 CHOSEN (strike ~ 2.5 kV at V/h scaling)
    V = 5 kV, T0 = 300 K                MEASURED
    eta = 3 (plan start 1.5), gamma = 2 MEASURED in the M2 lab: eta 1.5-2 gives D_box 1.8-1.9 and
                                        no glass contact within 2000 nodes; eta 3 gives D_box
                                        1.65-1.75 and contact at 600-1250 nodes; eta 4 D_box ~1.57
    h_f = 0.045 R2, q_f = 0.05          MEASURED / CHOSEN (calibrated in the touch lab)

Tree lifecycle (plan 4.4): FREE -> GROW (claimed by the electrode pool) -> ATTACHED (a node
reaches r = R2 - a: the foot) -> RETRACT (no gated candidate for ``starve_steps`` steps after
``starve_age``, or I_k < I_sus for 3 frames) -> FREE. With ``enable_reroute`` an ATTACHED tree
re-routes *in place* (hybrid model): on (a) main-channel stretch ``t_stretch > 1.5``
(``t_stretch`` is written by the advection kernel of the coupled app: current length of the
root-foot channel over its length at attachment; it stays 1 here, so growth alone never trips
it) or (c) a Poisson timer floor of mean 2 s (armed when the slot is claimed, re-armed at every
attachment), a fork arc length ``fork_s`` in [0.35, 0.75] L is drawn, the pool's candidates
distal of it are killed and K fresh ones are seeded around the fork node (the retained charged
node of largest s_arc <= fork_s); the tree keeps its state, current, heat and foot while this
*leader* grows (``t_regrow`` 1) through the field of the still-charged old channel, which
repels it. When the leader reaches the glass the foot moves to it and the old distal channel
(nodes born before the trigger with s_arc past the fork) enters the decaying pool (uncharged,
alpha -> 0, freed after ``decay_time``; ``t_regrow`` 2 -> 0). A leader without a gated site for
``REGROW_PATIENCE x starve_steps`` steps is re-seeded from a new fork; if no leader has reached
the glass ``REGROW_TIMEOUT`` after the trigger, the stretched channel extinguishes anyway: its
distal part fades, the tree drops to GROW (no foot, no current) and goes on with its leader or
retracts. The frame after any attachment every charged node off the main channel fades too
(only the conducting channel persists). Candidates of decaying nodes die at the next
re-base, so no node is ever born on a decaying parent (topology invariant: every live non-root
node has a live parent of its own tree, decaying only if the node is). The coupled app advects
the nodes with the gas between frames; a link stretched past ``SUBDIV_LINK`` h is split at its
midpoint at the start of the next frame (the channel is a material line), so links stay ~h and
the re-route walk sees the true channel length.

Determinism rules used throughout: no float atomic adds (block reductions by ``wp.tile_sum`` +
a fixed-order finish kernel), integer/min/max atomics only, node ids assigned by rank inside
the serial per-tree commit kernel, candidate slots assigned from a per-tree ring counter. Random
streams are ``wp.rand_init(step_seed(seed, step, salt), offset)`` with one salt per consumer
(Gumbel keys, electrode rotation, spawn rotation, re-route timers) so no two streams overlap.

Runs on ``cuda`` and (slowly) on ``cpu``: tiled kernels loop with stride ``wp.block_dim()``,
which is 1 on the CPU backend.
"""

import math

import numpy as np
import warp as wp

# --- capacities (compile-time defaults; a Dbm instance fixes them at construction) --------------

N_MAX = 8192        # nodes
F_MAX = 32          # trees (pools)
C_F = 2048          # candidates per tree pool
C_MAX = F_MAX * C_F # 65536 tree candidates
K = 8               # candidates per new node
S_MAX = 16          # unrolled growth steps per frame
E_POOL = 256        # electrode-seeding candidates
FINGER_MAX = 10
HOT_MAX = 8         # thermal-channel segments of the default temperature sampler
TILE = 128          # lanes per tiled block (re-base, matvec: many blocks)
SPAWN_TILE = 256    # lanes per commit/spawn block (few blocks: more lanes hide the node-loop latency)
ROWS = 4            # rows (candidates / matrix rows) per tiled block
DOT_TILE = 1024     # lanes of the single-block reductions

# --- physics defaults (plan 4.1) ----------------------------------------------------------------

R1 = 0.011                  # matches plasma.params.R1 (the coupled app configures r1 explicitly)
R2I = 0.075
H = 0.0015
A_OVER_H = 0.25
V_DEFAULT = 5000.0
T0 = 300.0
E_BD0 = 190.0e3
E_PROP0 = 0.25 * E_BD0
ETA = 3.0                   # plan start 1.5; measured in the M2 lab (see the module docstring)
GAMMA = 1.0                 # exponent of T/T0 in the weight: 1 = the reduced field E/N (physical); the
                            # growth exponent eta then applies to (E T/T0), so a hot channel is
                            # preferred by eta * ln(T/T0) in the key (the thermal re-strike memory)
H_F = 0.045 * R2I
Q_F = 0.05
DT = 1.0 / 60.0
RETRACT_SPEED = 0.02        # m / frame
STARVE_STEPS = 3
STARVE_AGE = 0.5            # s
DECAY_TIME = 0.1            # s
REGROW_TIMEOUT = 0.25       # s: grace period of a re-route; if no leader has reached the glass by then
                            # the stretched old channel extinguishes anyway (pruned, tree -> GROW)
REGROW_PATIENCE = 2         # x starve_steps without a gated candidate re-seeds the leader elsewhere
SUBDIV_MAX_NODES = 400      # no link subdivision beyond this many nodes per tree (folded channels)
BRUSH_FEET = 4              # secondary feet per tree: a touched channel branches near the glass to the
                            # other fingers of the hand (brush leaders, see k_tree_frame)
BRUSH_FORK = 0.02           # m before the main foot where a brush leader forks off the channel
FOOT_BRUSH = 0.012          # m: branches that leave the main channel within this distance of the glass
                            # survive the post-attachment prune (the surface discharge spreads the foot
                            # into a brush of 2-4 short branches, as in every photo of a globe)
RESTRIKE_P = 0.3            # share of timer events that end the filament (retract) instead of re-routing
REROUTE_MEAN = 0.25         # s: mean of the Poisson re-route timer (the footage: a filament keeps its path
                            # for ~0.1-0.3 s, then jumps to a nearby one; plan 4.4 said 2 s)
FORK_LO, FORK_HI = 0.10, 0.60   # fork arc fraction range: most of the channel regrows (a whole-path jump)
                            # it: a fresh strike then lands where the electrode is least screened, so
                            # roots keep spreading instead of collecting where the gas carries them
STRETCH_TRIGGER = 1.5       # re-route when the main channel has stretched to this multiple of its arc length
                            # at attachment (CHOSEN; the channel voltage grows with its length and the
                            # supply cannot follow it far: 2 kV of 5 kV at 6 cm; 1.5 re-struck every ~1.5 s
                            # at the drift the footage shows)
STRETCH_FORK = 1.2          # the stretch trigger forks at the most distal main-channel node whose
                            # root-side channel is still within this factor of its arc length at birth
SUBDIV_LINK = 2.0           # x h: an advected link longer than this is split at its midpoint each frame
                            # (the channel is a material line; a polyline chord must stay ~h)
I_DEFAULT = 0.05e-3         # A, constant per-tree current until the circuit (M4) drives it
BORN_GAIN = 1.0             # multiplier on the in-frame Born estimate q = -a phi (1 = plan)

# --- node flags ----------------------------------------------------------------------------------

ALIVE = 1
CHARGED = 2
DECAYING = 4
FOOT = 8
ROOT = 16

# --- tree states ---------------------------------------------------------------------------------

FREE = 0
SEED = 1        # the electrode pool stage (a claimed tree slot enters GROW directly)
GROW = 2
ATTACHED = 3
RETRACT = 4
REROUTE = 5
FROZEN = 6      # lab only: keeps the nodes charged, no growth (sequential-pool protocol)

# --- float params (device array P) ---------------------------------------------------------------

P_R1, P_R2, P_H, P_A, P_V, P_T0, P_E_BD0, P_E_PROP0, P_ETA, P_GAMMA, P_HF, P_DT, \
    P_RETRACT_SPEED, P_STARVE_AGE, P_I_SUS, P_DECAY_TIME, P_BORN_GAIN, P_E_CH, P_COUNT = range(19)

# --- int params (device array IP) ----------------------------------------------------------------

IP_SEED, IP_GLOBAL_NORM, IP_USE_SIGMA, IP_GROW_AFTER_ATTACH, IP_PAUSE, IP_HOT_COUNT, \
    IP_ENABLE_REROUTE, IP_STARVE_STEPS, IP_IMAGES, IP_COUNT = range(10)

# --- random-stream salts (one per consumer of step_seed) ------------------------------------------

SALT_GUMBEL, SALT_ELECTRODE, SALT_SPAWN, SALT_TIMER_STEP, SALT_TIMER_FRAME, SALT_FORK = range(6)

# --- device counters (array cnt) -----------------------------------------------------------------

CNT_FRAME, CNT_FREE_TOP, CNT_NODE_HW, CNT_COUNT = range(4)

# --- CG scalars (array cg) -----------------------------------------------------------------------

CG_RZ, CG_PAP, CG_ALPHA, CG_BETA, CG_RZ_NEW, CG_RR, CG_BB, CG_COUNT = range(8)

KEY_BIAS = 96.0
KEY_SCALE = 16777216.0      # 2^24 -> quantised key resolution 6e-8
BIG = 1.0e30


@wp.func
def u0(x: wp.vec3, r1: float, r2: float):
    """Exact concentric-sphere Dirichlet solution: 0 on r = R1, 1 on r = R2."""
    return (1.0 / r1 - 1.0 / wp.length(x)) / (1.0 / r1 - 1.0 / r2)


@wp.func
def pair_potential(x: wp.vec3, xj: wp.vec3, r1: float, a: float, images: int):
    """Potential at x of a unit node charge at x_j: the inverse multiquadric 1/sqrt(r^2 + a^2)
    (a charge smeared over the conductor radius a: 1/r beyond a few a, 1/a at the node, and a
    strictly positive definite kernel for *any* node configuration, so the conductor system stays
    SPD when advection packs nodes closer than a), minus the Kelvin image of x_j in the electrode
    sphere when images != 0 (keeps phi(R1) = 0 for the node charges; the fingers use the same
    construction)."""
    d = x - xj
    g = 1.0 / wp.sqrt(wp.dot(d, d) + a * a)
    if images != 0:
        lj = wp.length(xj)
        di = x - xj * (r1 * r1 / (lj * lj))
        g -= (r1 / lj) / wp.sqrt(wp.dot(di, di) + a * a)
    return g


@wp.func
def channel_k(P: wp.array(dtype=float)) -> float:
    """Dimensionless channel potential per metre of arc length: a resistive filament drops E_ch
    (V/m) along its length, so a node at arc length s sits at phi = E_ch s / V above the
    electrode instead of at 0 (a channel held at the electrode potential screens the electrode
    and the glass far too strongly; V_ch ~ 2 kV of a 5 kV drive over a 6 cm channel)."""
    return P[P_E_CH] / wp.max(P[P_V], 1.0)


@wp.func
def self_potential(xj: wp.vec3, r1: float, a: float, images: int):
    """Diagonal of the conductor system: pair_potential at r = 0, i.e. 1/a minus the node's own
    image seen at the node."""
    d = 1.0 / a
    if images != 0:
        lj = wp.length(xj)
        ri = (lj * lj - r1 * r1) / lj
        d -= (r1 / lj) / wp.sqrt(ri * ri + a * a)
    return d


@wp.func
def finger_potential(x: wp.vec3, r1: float, r2: float, hf: float,
                     finger_dir: wp.array(dtype=wp.vec3), finger_q: wp.array(dtype=float),
                     n_fingers: int):
    """External positive point charges at (R2 + h_f) n_f with their Kelvin images in the electrode."""
    acc = float(0.0)
    for f in range(n_fingers):
        d = (r2 + hf) * finger_dir[f]
        dl = wp.length(d)
        image = d * (r1 * r1 / (dl * dl))
        acc += finger_q[f] * (1.0 / wp.length(x - d) - (r1 / dl) / wp.length(x - image))
    return acc


@wp.func
def boundary_potential(x: wp.vec3, P: wp.array(dtype=float),
                       finger_dir: wp.array(dtype=wp.vec3), finger_q: wp.array(dtype=float),
                       n_fingers: int):
    return u0(x, P[P_R1], P[P_R2]) + finger_potential(x, P[P_R1], P[P_R2], P[P_HF],
                                                      finger_dir, finger_q, n_fingers)


@wp.func
def temperature_at(x: wp.vec3, t0: float,
                   hot_p0: wp.array(dtype=wp.vec3), hot_p1: wp.array(dtype=wp.vec3),
                   hot_r: wp.array(dtype=float), hot_T: wp.array(dtype=float), n_hot: int):
    """Default temperature sampler: T0 plus hot channels (capsules).

    The coupled app replaces this with a trilinear sample of the gas grid; the signature is the
    hook (position in, temperature out)."""
    t = t0
    for i in range(n_hot):
        p0 = hot_p0[i]
        seg = hot_p1[i] - p0
        l2 = wp.dot(seg, seg)
        u = float(0.0)
        if l2 > 0.0:
            u = wp.clamp(wp.dot(x - p0, seg) / l2, 0.0, 1.0)
        if wp.length(x - (p0 + u * seg)) <= hot_r[i]:
            t = wp.max(t, hot_T[i])
    return t


@wp.func
def random_quat(state: wp.uint32):
    """Uniform random rotation (Shoemake)."""
    u1 = wp.randf(state)
    u2 = wp.randf(state)
    u3 = wp.randf(state)
    a = wp.sqrt(1.0 - u1)
    b = wp.sqrt(u1)
    return wp.quat(a * wp.sin(6.283185307 * u2), a * wp.cos(6.283185307 * u2),
                   b * wp.sin(6.283185307 * u3), b * wp.cos(6.283185307 * u3))


@wp.func
def octant_dir(k: int):
    """The eight octant directions (+-1, +-1, +-1) / sqrt(3) of an octahedral frame."""
    s = 0.57735027
    return wp.vec3(float((k & 1) * 2 - 1) * s, float(((k >> 1) & 1) * 2 - 1) * s,
                   float(((k >> 2) & 1) * 2 - 1) * s)


@wp.func
def cone_dir(k: int, z: wp.vec3, phase: float):
    """K = 8 candidate directions in the frame of the growth direction z: straight ahead, three
    at 35 deg and four at 80 deg (azimuths offset by a random phase). A streamer head propagates
    along the field at its own tip, so the candidate set must always contain the straight
    continuation; a randomly rotated octahedral frame (mean deflection ~55 deg per step) gave
    channels 1.8x longer than their chord regardless of eta. Lateral candidates keep branching
    and the field-driven curvature."""
    if k == 0:
        return z
    up = wp.vec3(0.0, 1.0, 0.0)
    if wp.abs(z[1]) > 0.9:
        up = wp.vec3(1.0, 0.0, 0.0)
    u = wp.normalize(wp.cross(z, up))
    v = wp.cross(z, u)
    polar = float(0.610865)          # 35 deg
    az = phase + float(k - 1) * 2.0943951
    if k >= 4:
        polar = 1.3962634            # 80 deg
        az = phase + 0.7853982 + float(k - 4) * 1.5707963
    s = wp.sin(polar)
    return z * wp.cos(polar) + (u * wp.cos(az) + v * wp.sin(az)) * s


@wp.func
def fibonacci_dir(i: int, n: int):
    z = 1.0 - 2.0 * (float(i) + 0.5) / float(n)
    r = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    t = 2.39996323 * float(i)
    return wp.vec3(r * wp.cos(t), r * wp.sin(t), z)


@wp.func
def step_seed(seed: int, step: int, salt: int):
    """Per-(step, consumer) RNG seed; the mixing wraps in uint32 (no signed overflow)."""
    mixed = wp.uint32(seed) + wp.uint32(1640531527) * wp.uint32(step + 1) \
        + wp.uint32(668265261) * wp.uint32(salt + 1)
    return int(mixed)


@wp.func
def reroute_timer(seed: int, step: int, t: int, salt: int):
    """Poisson re-route floor (plan 4.4 trigger c): exponential, mean REROUTE_MEAN."""
    st = wp.rand_init(step_seed(seed, step, salt), t)
    return -REROUTE_MEAN * wp.log(wp.max(wp.randf(st), 1.0e-6))


@wp.func
def pool_of(c: int, c_f: int, c_max: int, f_max: int):
    if c >= c_max:
        return f_max
    return c // c_f


@wp.func
def growing(state: int, grow_after_attach: int, regrow: int):
    """A pool grows while its tree is GROW, or ATTACHED with a re-route leader (regrow == 1)."""
    if state == GROW:
        return True
    if state == ATTACHED and (grow_after_attach != 0 or regrow == 1):
        return True
    return False


# --- per-frame kernels ---------------------------------------------------------------------------

@wp.kernel
def k_subdivide_mark(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
                     pos: wp.array(dtype=wp.vec3), parent: wp.array(dtype=int),
                     flags: wp.array(dtype=int), tree: wp.array(dtype=int), t_nodes: wp.array(dtype=int),
                     mark: wp.array(dtype=int)):
    """Marks live links (child i -> parent) stretched past SUBDIV_LINK h by the app's advection."""
    i = wp.tid()
    mark[i] = 0
    if i >= cnt[CNT_NODE_HW] or IP[IP_PAUSE] != 0:
        return
    f = flags[i]
    if (f & ALIVE) == 0 or (f & ROOT) != 0 or (f & DECAYING) != 0:
        return
    p = parent[i]
    if p < 0:
        return
    fp = flags[p]
    if (fp & ALIVE) == 0 or (fp & DECAYING) != 0 or tree[p] != tree[i]:
        return
    if t_nodes[tree[i]] >= SUBDIV_MAX_NODES:
        return
    if wp.length(pos[i] - pos[p]) > SUBDIV_LINK * P[P_H]:
        mark[i] = 1


@wp.kernel
def k_subdivide_insert(cnt: wp.array(dtype=int), scan: wp.array(dtype=int), free_ids: wp.array(dtype=int),
                       pos: wp.array(dtype=wp.vec3), prev_pos: wp.array(dtype=wp.vec3),
                       parent: wp.array(dtype=int), tree: wp.array(dtype=int), birth: wp.array(dtype=int),
                       s_arc: wp.array(dtype=float), q: wp.array(dtype=float), flags: wp.array(dtype=int),
                       stamp: wp.array(dtype=int), mark: wp.array(dtype=int), t_nodes: wp.array(dtype=int)):
    """Splits every marked link at its midpoint with a node popped from the free list (rank order,
    from the top like the growth commits). The new node inherits the child's tree, birth, decay
    state and stamp (so re-route pruning and retraction treat it like the link it came from) and
    interpolates position, previous position, arc length and charge (CG warm start)."""
    i = wp.tid()
    if mark[i] == 0:
        return
    idx = cnt[CNT_FREE_TOP] - 1 - (scan[i] - 1)
    if idx < 0:
        return                      # free list exhausted: the link stays a chord this frame
    m = free_ids[idx]
    p = parent[i]
    pos[m] = 0.5 * (pos[i] + pos[p])
    prev_pos[m] = 0.5 * (prev_pos[i] + prev_pos[p])
    parent[m] = p
    parent[i] = m
    tree[m] = tree[i]
    birth[m] = birth[i]
    s_arc[m] = 0.5 * (s_arc[i] + s_arc[p])
    q[m] = 0.5 * (q[i] + q[p])
    stamp[m] = stamp[i]
    flags[m] = flags[i] & (ALIVE | CHARGED | DECAYING)
    wp.atomic_add(t_nodes, tree[i], 1)
    wp.atomic_max(cnt, CNT_NODE_HW, m + 1)


@wp.kernel
def k_subdivide_finish(cnt: wp.array(dtype=int), scan: wp.array(dtype=int), n_max: int):
    cnt[CNT_FREE_TOP] = cnt[CNT_FREE_TOP] - wp.min(scan[n_max - 1], cnt[CNT_FREE_TOP])


@wp.kernel
def k_frame_begin(pos: wp.array(dtype=wp.vec3), prev_pos: wp.array(dtype=wp.vec3),
                  cnt: wp.array(dtype=int)):
    """prev_pos <- pos is the first kernel of every frame (paused included)."""
    i = wp.tid()
    prev_pos[i] = pos[i]
    if i == 0:
        cnt[CNT_FRAME] = cnt[CNT_FRAME] + 1


@wp.kernel
def k_tree_frame(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
                 pos: wp.array(dtype=wp.vec3), parent: wp.array(dtype=int), s_arc: wp.array(dtype=float),
                 t_foot: wp.array(dtype=int),
                 t_state: wp.array(dtype=int), t_birth: wp.array(dtype=int),
                 t_I: wp.array(dtype=float), t_starve_I: wp.array(dtype=int),
                 t_L: wp.array(dtype=float), t_chord: wp.array(dtype=float),
                 t_ring_age: wp.array(dtype=float), t_timer: wp.array(dtype=float),
                 t_retract_len: wp.array(dtype=float), t_fork_s: wp.array(dtype=float),
                 t_stretch: wp.array(dtype=float), t_fork_key: wp.array(dtype=wp.int64),
                 t_regrow: wp.array(dtype=int), t_regrow_frame: wp.array(dtype=int),
                 t_leader_frame: wp.array(dtype=int),
                 t_prune_side: wp.array(dtype=int), main_mark: wp.array(dtype=int),
                 t_reroute_req: wp.array(dtype=int),
                 t_foot2: wp.array(dtype=int), t_brush: wp.array(dtype=int),
                 t_brush_req: wp.array(dtype=int), t_foot2_drop: wp.array(dtype=int),
                 t_hold: wp.array(dtype=int),
                 t_free_snapshot: wp.array(dtype=int), t_new: wp.array(dtype=int),
                 stage_min: wp.array(dtype=float), stage_max: wp.array(dtype=float),
                 slot: wp.array(dtype=wp.int64)):
    """Per-tree frame bookkeeping (F_MAX + 1 threads, the last one is the electrode pool).

    ``t_hold`` (app: the filament is under a finger) slows the Poisson re-route timer 4x and disables re-strikes: a touched
    filament is pinned by the finger and does not wander; stretch and requests still apply.

    Brush leaders (``t_brush_req`` from the circuit: an unserved finger next to this touched
    channel): a leader forks BRUSH_FORK before the main foot; when it reaches the glass it becomes
    a secondary foot (``t_foot2``) of the same channel instead of replacing it, so one filament
    splits to the fingers of a hand. Secondary feet are dropped on request (finger gone,
    ``t_foot2_drop``), on any re-route and on retraction; their chains are marked with the main
    channel so the prune keeps them.

    ``t_reroute_req`` is the app's trigger (b): a re-route requested from outside the engine (a
    finger landed near this tree's foot); consumed here like the timer trigger.

    ``t_regrow``: 0 idle, 1 leader out, 2 leader attached (prune next frame), 3 leader starved
    (re-seed here), 4 grace period over (the stretched channel extinguishes: k_node_frame prunes
    it, k_free_finish drops the tree to GROW). ``t_regrow_frame`` is the trigger frame (grace
    clock), ``t_leader_frame`` the frame the current leader was seeded (election, candidate kill,
    pruning key).

    The frame after a tree attaches (first strike or re-route leader) its main channel
    (foot -> root) is stamped in ``main_mark`` and k_node_frame sends every other charged node
    to the decaying pool: only the conducting channel persists as a hot, charged filament; the
    streamer branches that guided the strike cool and fade (plan 4.4, hybrid model).

    Re-routes overlap the old channel (plan 4.4, hybrid): the trigger arms a re-route leader
    (``t_regrow`` 1) that grows from the proximal part of the channel while the tree stays
    ATTACHED (current, heat, foot); the old distal channel is pruned into the decaying pool only
    once the leader reaches the glass (``t_regrow`` 2, k_node_frame), and a leader that starves or
    times out is abandoned (the old channel stays, the stub remains a side branch)."""
    t = wp.tid()
    f1 = t_state.shape[0] + 1
    stage_min[t] = BIG
    stage_max[t] = -BIG
    stage_min[f1 + t] = BIG
    stage_max[f1 + t] = -BIG
    slot[t] = wp.int64(0)
    slot[f1 + t] = wp.int64(0)
    t_new[t] = -1
    if t >= t_state.shape[0]:
        return
    t_fork_key[t] = wp.int64(0)
    state = t_state[t]
    t_free_snapshot[t] = 0
    if state == FREE:
        t_free_snapshot[t] = 1
        return
    if IP[IP_PAUSE] != 0:
        return
    dt = P[P_DT]
    if state == ATTACHED:
        t_ring_age[t] = t_ring_age[t] + dt
        frame = cnt[CNT_FRAME]
        # secondary feet dropped by the circuit (finger gone): their chains fade at this prune
        for j in range(BRUSH_FEET):
            fs = t * BRUSH_FEET + j
            if t_foot2_drop[fs] != 0:
                t_foot2_drop[fs] = 0
                if t_foot2[fs] >= 0:
                    t_foot2[fs] = -1
                    t_prune_side[t] = 1
        rg = t_regrow[t]
        if (rg == 1 or rg == 3) and float(frame - t_regrow_frame[t]) * dt > REGROW_TIMEOUT:
            if t_brush[t] != 0:
                # a brush leader that never made it: prune it, the channel stays as it is
                t_brush[t] = 0
                t_regrow[t] = 0
                t_prune_side[t] = 1
                rg = 0
            else:
                t_regrow[t] = 4          # grace over: the stretched channel extinguishes now
                return
        if t_prune_side[t] != 0:
            for j in range(BRUSH_FEET + 1):
                n = t_foot[t]
                if j > 0:
                    n = t_foot2[t * BRUSH_FEET + j - 1]
                steps = int(0)
                while n >= 0 and steps < 4096:
                    main_mark[n] = frame
                    n = parent[n]
                    steps += 1
        # starvation latch (inert until the circuit writes I_k): I_k < I_sus for 3 frames -> RETRACT
        if t_I[t] < P[P_I_SUS]:
            t_starve_I[t] = t_starve_I[t] + 1
            if t_starve_I[t] >= 3:
                t_state[t] = RETRACT
                t_retract_len[t] = 0.0
                t_regrow[t] = 0
                t_brush[t] = 0
                for j in range(BRUSH_FEET):
                    t_foot2[t * BRUSH_FEET + j] = -1
                return
        else:
            t_starve_I[t] = 0
        # re-route triggers (a) main channel stretched by advection (t_stretch is the advection
        # kernel's; growth never changes it) and (c) the Poisson timer floor. Inert until enabled;
        # never re-armed while a leader is out; a starved leader (3) is re-seeded from a new fork.
        req = t_reroute_req[t]
        t_reroute_req[t] = 0
        breq = t_brush_req[t]
        t_brush_req[t] = 0
        if IP[IP_ENABLE_REROUTE] != 0 and (rg == 0 or rg == 3):
            if rg == 0:
                # a touched filament (t_hold) still re-routes, 4x more slowly, and never retracts:
                # the footage's touched channel keeps changing shape but never disappears
                t_timer[t] = t_timer[t] - dt * wp.where(t_hold[t] == 0, 1.0, 0.25)
            stretched = t_stretch[t] > STRETCH_TRIGGER
            brush = (rg == 3 and t_brush[t] != 0) or (rg == 0 and breq != 0)
            if brush:
                free = int(0)
                for j in range(BRUSH_FEET):
                    if t_foot2[t * BRUSH_FEET + j] < 0:
                        free = 1
                if free != 0 and t_L[t] > BRUSH_FORK + 2.0 * P[P_H]:
                    t_fork_s[t] = t_L[t] - BRUSH_FORK
                    if rg == 0:
                        t_regrow_frame[t] = frame
                    t_regrow[t] = 1
                    t_leader_frame[t] = frame
                    t_brush[t] = 1
                else:
                    t_brush[t] = 0
                    if rg == 3:
                        t_regrow[t] = 0
            elif rg == 3 or stretched or req != 0 or t_timer[t] <= 0.0:
                st = wp.rand_init(step_seed(IP[IP_SEED], frame, SALT_TIMER_FRAME), t)
                if rg == 0 and not stretched and req == 0 and t_hold[t] == 0 and wp.randf(st) < RESTRIKE_P:
                    # the filament dies; the count law strikes a new one elsewhere
                    t_state[t] = RETRACT
                    t_retract_len[t] = 0.0
                    t_regrow[t] = 0
                    t_brush[t] = 0
                    for j in range(BRUSH_FEET):
                        t_foot2[t * BRUSH_FEET + j] = -1
                    return
                fork_s = (FORK_LO + (FORK_HI - FORK_LO) * wp.randf(st)) * t_L[t]
                if stretched:
                    # the discharge abandons the elongated part of its hot channel and keeps the
                    # proximal part that still runs where it was struck: fork at the most distal
                    # main-channel node u with geometric length(root -> u) <= STRETCH_FORK s_arc[u]
                    # (two foot -> root walks: total length, then the first node that qualifies);
                    # none -> re-strike from the root
                    fork_s = float(0.0)
                    total = float(0.0)
                    n = t_foot[t]
                    steps = int(0)
                    while n >= 0 and steps < 4096:
                        pn = parent[n]
                        if pn < 0:
                            break
                        total += wp.length(pos[n] - pos[pn])
                        n = pn
                        steps += 1
                    n2 = t_foot[t]
                    g = float(0.0)
                    steps2 = int(0)
                    found = int(0)
                    while n2 >= 0 and steps2 < 4096 and found == 0:
                        pn2 = parent[n2]
                        if pn2 < 0:
                            break
                        g += wp.length(pos[n2] - pos[pn2])
                        if total - g <= STRETCH_FORK * s_arc[pn2]:
                            fork_s = s_arc[pn2]
                            found = 1
                        n2 = pn2
                        steps2 += 1
                t_fork_s[t] = fork_s
                t_timer[t] = -REROUTE_MEAN * wp.log(wp.max(wp.randf(st), 1.0e-6))
                t_stretch[t] = 1.0
                if rg == 0:
                    t_regrow_frame[t] = frame
                t_regrow[t] = 1
                t_leader_frame[t] = frame
    if state == RETRACT:
        t_retract_len[t] = t_retract_len[t] + P[P_RETRACT_SPEED]


@wp.kernel
def k_node_frame(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
                 flags: wp.array(dtype=int), tree: wp.array(dtype=int), s_arc: wp.array(dtype=float),
                 stamp: wp.array(dtype=int), q: wp.array(dtype=float),
                 birth: wp.array(dtype=int), pos: wp.array(dtype=wp.vec3), parent: wp.array(dtype=int),
                 t_state: wp.array(dtype=int), t_L: wp.array(dtype=float),
                 t_retract_len: wp.array(dtype=float), t_fork_s: wp.array(dtype=float),
                 t_fork_key: wp.array(dtype=wp.int64),
                 t_regrow: wp.array(dtype=int), t_leader_frame: wp.array(dtype=int),
                 t_prune_side: wp.array(dtype=int), main_mark: wp.array(dtype=int),
                 t_nodes: wp.array(dtype=int), mark: wp.array(dtype=int)):
    """Retraction, re-route forks, post-attachment / grace pruning and decaying-pool expiry;
    marks the nodes to free this frame.

    On the frame a re-route is armed, the tree's charged node of largest s_arc <= fork_s is
    elected fork node (atomic max on a (s_arc, id) key: exact and order-independent) and
    k_free_finish seeds the leader's candidates around it. The frame after any attachment
    (t_prune_side, main channel stamped by k_tree_frame) every charged node off the main channel
    enters the decaying pool: the streamer branches of a fresh strike, and after a re-route the
    old distal channel and the leader's own branches."""
    i = wp.tid()
    mark[i] = 0
    if i >= cnt[CNT_NODE_HW]:
        return
    f = flags[i]
    if (f & ALIVE) == 0 or IP[IP_PAUSE] != 0:
        return
    t = tree[i]
    state = t_state[t]
    frame = cnt[CNT_FRAME]
    free = False
    if state == RETRACT:
        if s_arc[i] > t_L[t] - t_retract_len[t]:
            free = True
    elif state == ATTACHED:
        rg = t_regrow[t]
        if rg == 1:
            if t_leader_frame[t] == frame and (f & CHARGED) != 0 and s_arc[i] <= t_fork_s[t]:
                key = (wp.int64(s_arc[i] * 1.0e6) + wp.int64(1)) << wp.int64(32) | wp.int64(i)
                wp.atomic_max(t_fork_key, t, key)
        elif rg == 4:
            # the stretched channel extinguishes: everything distal of the fork that is not the
            # current leader (born at or after its seeding frame) fades
            if (f & DECAYING) == 0 and s_arc[i] > t_fork_s[t] and birth[i] < t_leader_frame[t]:
                flags[i] = (f | DECAYING) & ~(CHARGED | FOOT)
                q[i] = 0.0
                stamp[i] = frame
        if t_prune_side[t] != 0 and (f & CHARGED) != 0 and (f & ROOT) == 0 and main_mark[i] != frame:
            # spare the foot brush: a branch whose ancestors reach the main channel without leaving
            # the last FOOT_BRUSH of the gap (the surface discharge spreading at the glass)
            spare = int(0)
            brush_r = P[P_R2] - FOOT_BRUSH
            if wp.length(pos[i]) > brush_r:
                n = parent[i]
                steps = int(0)
                while n >= 0 and steps < 64:
                    if main_mark[n] == frame:
                        spare = 1
                        break
                    if wp.length(pos[n]) <= brush_r:
                        break
                    n = parent[n]
                    steps += 1
            if spare == 0:
                flags[i] = (f | DECAYING) & ~(CHARGED | FOOT)
                q[i] = 0.0
                stamp[i] = frame
    if (f & DECAYING) != 0:
        if float(frame - stamp[i]) * P[P_DT] >= P[P_DECAY_TIME]:
            free = True
    if free:
        flags[i] = 0
        q[i] = 0.0
        mark[i] = 1
        wp.atomic_sub(t_nodes, t, 1)


@wp.kernel
def k_free_push(mark: wp.array(dtype=int), scan: wp.array(dtype=int), cnt: wp.array(dtype=int),
                free_ids: wp.array(dtype=int)):
    i = wp.tid()
    if mark[i] != 0:
        free_ids[cnt[CNT_FREE_TOP] + scan[i] - 1] = i


@wp.kernel
def k_free_finish(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
                  scan: wp.array(dtype=int), pos: wp.array(dtype=wp.vec3),
                  cand_pos: wp.array(dtype=wp.vec3), cand_phi: wp.array(dtype=float),
                  cand_parent: wp.array(dtype=int), cand_alive: wp.array(dtype=int),
                  cand_stamp: wp.array(dtype=int),
                  t_state: wp.array(dtype=int), t_nodes: wp.array(dtype=int),
                  t_L: wp.array(dtype=float), t_fork_s: wp.array(dtype=float),
                  t_fork_key: wp.array(dtype=wp.int64), t_tip: wp.array(dtype=int),
                  t_tail: wp.array(dtype=int), t_new: wp.array(dtype=int),
                  t_regrow: wp.array(dtype=int), t_leader_frame: wp.array(dtype=int),
                  t_prune_side: wp.array(dtype=int), t_foot: wp.array(dtype=int),
                  t_foot_dir: wp.array(dtype=wp.vec3), t_starve: wp.array(dtype=int),
                  t_foot2: wp.array(dtype=int), t_brush: wp.array(dtype=int),
                  c_f: int, k: int, n_max: int):
    """Closes the free-list push, the RETRACT -> FREE transition and the re-route bookkeeping.

    On the frame a re-route is armed the leader gets K fresh candidates around the fork node
    (positions only: the re-base that follows sets their potential and applies the h/2 rule);
    the pool's older candidates distal of the fork are killed by that re-base, so the leader can
    start anywhere proximal of the fork where the field is strongest. A pruned re-route
    (t_regrow 2, k_node_frame this frame) is closed here."""
    t = wp.tid()
    if t == 0:
        cnt[CNT_FREE_TOP] = cnt[CNT_FREE_TOP] + scan[n_max - 1]
    state = t_state[t]
    if state == RETRACT and t_nodes[t] <= 0:
        t_state[t] = FREE
        t_new[t] = -1
    if state == ATTACHED and t_regrow[t] == 2:
        t_regrow[t] = 0
    if state == ATTACHED and t_regrow[t] == 4:
        # grace over (k_node_frame pruned the old distal channel): the tree loses its foot and
        # current and goes on growing its leader, or retracts if that starves
        t_state[t] = GROW
        t_foot[t] = -1
        t_foot_dir[t] = wp.vec3(0.0, 0.0, 0.0)
        t_L[t] = t_fork_s[t]
        t_regrow[t] = 0
        t_starve[t] = 0
        t_prune_side[t] = 0
        t_brush[t] = 0
        for j in range(BRUSH_FEET):
            t_foot2[t * BRUSH_FEET + j] = -1
    if state == ATTACHED:
        t_prune_side[t] = 0
    if state == ATTACHED and t_regrow[t] == 1 and t_leader_frame[t] == cnt[CNT_FRAME]:
        key = t_fork_key[t]
        if key == wp.int64(0):
            t_regrow[t] = 3             # nothing to fork from below fork_s: draw again next frame
        else:
            fk = int(key & wp.int64(4294967295))
            t_tip[t] = fk
            x0 = pos[fk]
            h = P[P_H]
            a = P[P_A]
            r1 = P[P_R1]
            r2 = P[P_R2]
            st = wp.rand_init(step_seed(IP[IP_SEED], cnt[CNT_FRAME], SALT_FORK), t)
            rot = random_quat(st)
            tail = t_tail[t]
            for kk in range(k):
                x = x0 + h * wp.quat_rotate(rot, octant_dir(kk))
                r = wp.length(x)
                valid = int(1)
                if r < r1 + a:
                    valid = 0
                if r > r2 - a:
                    x = x * ((r2 - a) / r)
                if wp.length(x - x0) < 0.5 * h:
                    valid = 0
                c = t * c_f + (tail + kk) % c_f
                cand_pos[c] = x
                cand_phi[c] = 0.0
                cand_parent[c] = fk
                cand_alive[c] = valid
                cand_stamp[c] = -1
            t_tail[t] = tail + k


@wp.kernel
def k_electrode_pool(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
                     cand_pos: wp.array(dtype=wp.vec3), cand_parent: wp.array(dtype=int),
                     cand_alive: wp.array(dtype=int), cand_stamp: wp.array(dtype=int),
                     c_max: int, e_pool: int):
    """E_POOL seed candidates on r = R1 + h: Fibonacci sphere, one random rotation per frame."""
    i = wp.tid()
    st = wp.rand_init(step_seed(IP[IP_SEED], cnt[CNT_FRAME], SALT_ELECTRODE), 0)
    rot = random_quat(st)
    d = wp.quat_rotate(rot, fibonacci_dir(i, e_pool))
    c = c_max + i
    cand_pos[c] = (P[P_R1] + P[P_H]) * d
    cand_parent[c] = -1
    cand_alive[c] = 1
    cand_stamp[c] = -1


@wp.kernel
def k_cand_phi_full(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
                    pos: wp.array(dtype=wp.vec3), q: wp.array(dtype=float), flags: wp.array(dtype=int),
                    tree: wp.array(dtype=int), s_arc: wp.array(dtype=float), birth: wp.array(dtype=int),
                    finger_dir: wp.array(dtype=wp.vec3), finger_q: wp.array(dtype=float),
                    n_fingers: wp.array(dtype=int), t_state: wp.array(dtype=int),
                    t_regrow: wp.array(dtype=int), t_leader_frame: wp.array(dtype=int),
                    t_fork_s: wp.array(dtype=float),
                    cand_pos: wp.array(dtype=wp.vec3), cand_phi: wp.array(dtype=float),
                    cand_parent: wp.array(dtype=int), cand_alive: wp.array(dtype=int),
                    stage_min: wp.array(dtype=float), stage_max: wp.array(dtype=float),
                    c_f: int, c_max: int, f_max: int, buf: int):
    """Re-bases every live candidate's potential on the solved charges, once per frame.

    ROWS candidates per block, lanes stride the node list; also kills candidates of dead pools,
    orphans (parent node freed by a retract / re-route this frame, decaying after a re-route, or
    its id already reused by another tree) and candidates within h/2 of any node, the parent
    included (electrode seeds included)."""
    blk, lane = wp.tid()
    c0 = blk * ROWS
    alive0 = cand_alive[c0] != 0
    alive1 = cand_alive[c0 + 1] != 0
    alive2 = cand_alive[c0 + 2] != 0
    alive3 = cand_alive[c0 + 3] != 0
    if not (alive0 or alive1 or alive2 or alive3):
        return
    grow_after = IP[IP_GROW_AFTER_ATTACH]
    if c0 < c_max:
        # all ROWS candidates of a block belong to the same pool (c_f is a multiple of ROWS)
        state = t_state[c0 // c_f]
        if not growing(state, grow_after, t_regrow[c0 // c_f]):
            if lane == 0:
                cand_alive[c0] = 0
                cand_alive[c0 + 1] = 0
                cand_alive[c0 + 2] = 0
                cand_alive[c0 + 3] = 0
            return
    x0 = cand_pos[c0]
    x1 = cand_pos[c0 + 1]
    x2 = cand_pos[c0 + 2]
    x3 = cand_pos[c0 + 3]
    a0 = float(0.0)
    a1 = float(0.0)
    a2 = float(0.0)
    a3 = float(0.0)
    d0 = float(BIG)
    d1 = float(BIG)
    d2 = float(BIG)
    d3 = float(BIG)
    n = cnt[CNT_NODE_HW]
    er1 = P[P_R1]
    ea = P[P_A]
    images = IP[IP_IMAGES]
    for base in range(0, n, wp.block_dim()):
        j = base + lane
        if j < n:
            fj = flags[j]
            if (fj & ALIVE) != 0:
                xj = pos[j]
                qj = q[j]
                if (fj & CHARGED) == 0:
                    qj = 0.0
                r0 = wp.max(wp.length(x0 - xj), 1.0e-9)
                r1 = wp.max(wp.length(x1 - xj), 1.0e-9)
                r2 = wp.max(wp.length(x2 - xj), 1.0e-9)
                r3 = wp.max(wp.length(x3 - xj), 1.0e-9)
                a0 += qj * pair_potential(x0, xj, er1, ea, images)
                a1 += qj * pair_potential(x1, xj, er1, ea, images)
                a2 += qj * pair_potential(x2, xj, er1, ea, images)
                a3 += qj * pair_potential(x3, xj, er1, ea, images)
                d0 = wp.min(d0, r0)
                d1 = wp.min(d1, r1)
                d2 = wp.min(d2, r2)
                d3 = wp.min(d3, r3)
    s0 = wp.tile_sum(wp.tile(a0))
    s1 = wp.tile_sum(wp.tile(a1))
    s2 = wp.tile_sum(wp.tile(a2))
    s3 = wp.tile_sum(wp.tile(a3))
    m0 = wp.tile_min(wp.tile(d0))
    m1 = wp.tile_min(wp.tile(d1))
    m2 = wp.tile_min(wp.tile(d2))
    m3 = wp.tile_min(wp.tile(d3))
    if lane == 0:
        pool = pool_of(c0, c_f, c_max, f_max)
        nf = n_fingers[0]
        half_h = 0.5 * P[P_H]
        for r in range(ROWS):
            c = c0 + r
            pp = cand_parent[c]
            if cand_alive[c] != 0 and pp >= 0:
                fp = flags[pp]
                if (fp & ALIVE) == 0 or (fp & DECAYING) != 0 or tree[pp] != pool:
                    cand_alive[c] = 0
                elif pool < f_max and t_regrow[pool] == 1 and t_leader_frame[pool] == cnt[CNT_FRAME] \
                        and birth[pp] < cnt[CNT_FRAME] and s_arc[pp] > t_fork_s[pool]:
                    cand_alive[c] = 0    # re-route armed: the leader starts proximal of the fork
            if cand_alive[c] != 0:
                acc = s0[0]
                dmin = m0[0]
                if r == 1:
                    acc = s1[0]
                    dmin = m1[0]
                elif r == 2:
                    acc = s2[0]
                    dmin = m2[0]
                elif r == 3:
                    acc = s3[0]
                    dmin = m3[0]
                if dmin < half_h:
                    cand_alive[c] = 0
                else:
                    phi = boundary_potential(cand_pos[c], P, finger_dir, finger_q, nf) + acc
                    if pp >= 0:
                        phi -= channel_k(P) * s_arc[pp]      # local: above the parent's channel potential
                    cand_phi[c] = phi
                    wp.atomic_min(stage_min, buf * (f_max + 1) + pool, phi)
                    wp.atomic_max(stage_max, buf * (f_max + 1) + pool, phi)


# --- growth step kernels (two per unrolled step) ------------------------------------------------

COMMIT_MAX = F_MAX + 1      # pools that can commit in one step (trees + the electrode pool)
commit_ids = wp.types.vector(length=COMMIT_MAX, dtype=wp.int32)


@wp.func
def settle_step(cnt: wp.array(dtype=int), t_committed: wp.array(dtype=int), t_tail: wp.array(dtype=int),
                t_state: wp.array(dtype=int), t_free_snapshot: wp.array(dtype=int), k: int, f_max: int):
    """Closes a growth step once every block has read the free list: pops the committed node ids,
    advances the candidate rings (the electrode pool's entry holds 1 + the claimed tree, whose
    ring advances for the root's K candidates), clears the claimed seed slots. One thread, fixed
    order."""
    popped = int(0)
    for u in range(f_max + 1):
        v = t_committed[u]
        if v != 0:
            popped += 1
            t_committed[u] = 0
            if u < f_max:
                t_tail[u] = t_tail[u] + k
            else:
                t_tail[v - 1] = t_tail[v - 1] + k
    cnt[CNT_FREE_TOP] = cnt[CNT_FREE_TOP] - popped
    for u in range(f_max):
        if t_state[u] != FREE:
            t_free_snapshot[u] = 0


@wp.kernel
def k_update_key(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
                 pos: wp.array(dtype=wp.vec3), q: wp.array(dtype=float),
                 t_state: wp.array(dtype=int), t_new: wp.array(dtype=int),
                 t_committed: wp.array(dtype=int), t_tail: wp.array(dtype=int),
                 t_free_snapshot: wp.array(dtype=int),
                 cand_pos: wp.array(dtype=wp.vec3), cand_phi: wp.array(dtype=float),
                 cand_alive: wp.array(dtype=int), cand_stamp: wp.array(dtype=int),
                 cand_s: wp.array(dtype=float), cand_T: wp.array(dtype=float),
                 t_regrow: wp.array(dtype=int),
                 hot_p0: wp.array(dtype=wp.vec3), hot_p1: wp.array(dtype=wp.vec3),
                 hot_r: wp.array(dtype=float), hot_T: wp.array(dtype=float),
                 stage_min: wp.array(dtype=float), stage_max: wp.array(dtype=float),
                 slot: wp.array(dtype=wp.int64),
                 c_f: int, c_max: int, f_max: int, k: int, s_max: int, s: int):
    """Kim Eq.11 incremental update with the previous step's new charges, gate, weight, Gumbel key.

    Extrema and Gumbel slots are double-buffered by step parity: this step reads the extrema
    staged during step s-1 (buffer s % 2) and stages into buffer (s+1) % 2."""
    c = wp.tid()
    f1 = f_max + 1
    if c == 0:
        settle_step(cnt, t_committed, t_tail, t_state, t_free_snapshot, k, f_max)
    if cand_alive[c] == 0 or IP[IP_PAUSE] != 0:
        return
    pool = pool_of(c, c_f, c_max, f_max)
    if pool == f_max:
        if s != 0:
            return
    elif not growing(t_state[pool], IP[IP_GROW_AFTER_ATTACH], t_regrow[pool]):
        return
    frame = cnt[CNT_FRAME]
    step = frame * s_max + s
    cur = (s % 2) * f1
    nxt = ((s + 1) % 2) * f1
    x = cand_pos[c]
    phi = cand_phi[c]
    if cand_stamp[c] != step - 1:
        # candidates spawned in the previous step already saw these charges in their full sum
        half_h = 0.5 * P[P_H]
        for t in range(f_max):
            j = t_new[t]
            if j >= 0:
                xj = pos[j]
                if wp.length(x - xj) < half_h:
                    cand_alive[c] = 0
                    return
                phi += q[j] * pair_potential(x, xj, P[P_R1], P[P_A], IP[IP_IMAGES])
        cand_phi[c] = phi
    wp.atomic_min(stage_min, nxt + pool, phi)
    wp.atomic_max(stage_max, nxt + pool, phi)
    # reduced-field breakdown gate
    t0 = P[P_T0]
    tc = cand_T[c]          # the coupled app samples the gas grid here; 0 -> default capsule sampler
    if tc > 0.0:
        tf = tc / t0
    else:
        tf = temperature_at(x, t0, hot_p0, hot_p1, hot_r, hot_T, IP[IP_HOT_COUNT]) / t0
    e = P[P_V] * phi / P[P_H]
    e_thr = P[P_E_PROP0]
    if pool == f_max:
        e_thr = P[P_E_BD0]
    if e * tf < e_thr:
        return
    # Kim Eq.13 min-max normalisation over the previous step's extrema
    lo = stage_min[cur + pool]
    hi = stage_max[cur + pool]
    if IP[IP_GLOBAL_NORM] != 0 and pool != f_max:
        lo = float(BIG)
        hi = float(-BIG)
        for t in range(f_max):
            if stage_min[cur + t] <= stage_max[cur + t]:
                lo = wp.min(lo, stage_min[cur + t])
                hi = wp.max(hi, stage_max[cur + t])
    if not (lo <= hi):
        return
    big_phi = 1.0
    if hi > lo:
        big_phi = wp.clamp((phi - lo) / (hi - lo), 0.0, 1.0)
    sc = float(1.0)
    if IP[IP_USE_SIGMA] != 0:
        sc = cand_s[c]
    w = big_phi * wp.pow(tf, P[P_GAMMA]) * sc
    if w <= 0.0:
        return
    st = wp.rand_init(step_seed(IP[IP_SEED], step, SALT_GUMBEL), c)
    u = wp.clamp(wp.randf(st), 1.0e-7, 1.0 - 1.0e-7)
    key = P[P_ETA] * wp.log(w) - wp.log(-wp.log(u))
    qk = wp.int64(wp.clamp((key + KEY_BIAS) * KEY_SCALE, 1.0, 2147483000.0))
    wp.atomic_max(slot, cur + pool, (qk << wp.int64(32)) | wp.int64(c))


@wp.kernel
def k_commit_spawn(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
                   admit: wp.array(dtype=int),
                   pos: wp.array(dtype=wp.vec3), prev_pos: wp.array(dtype=wp.vec3),
                   parent: wp.array(dtype=int), tree: wp.array(dtype=int), birth: wp.array(dtype=int),
                   s_arc: wp.array(dtype=float), q: wp.array(dtype=float), flags: wp.array(dtype=int),
                   free_ids: wp.array(dtype=int),
                   finger_dir: wp.array(dtype=wp.vec3), finger_q: wp.array(dtype=float),
                   n_fingers: wp.array(dtype=int),
                   cand_pos: wp.array(dtype=wp.vec3), cand_phi: wp.array(dtype=float),
                   cand_parent: wp.array(dtype=int), cand_alive: wp.array(dtype=int),
                   cand_stamp: wp.array(dtype=int),
                   t_state: wp.array(dtype=int), t_root: wp.array(dtype=int), t_tip: wp.array(dtype=int),
                   t_foot: wp.array(dtype=int), t_root_dir: wp.array(dtype=wp.vec3),
                   t_foot_dir: wp.array(dtype=wp.vec3), t_birth: wp.array(dtype=int),
                   t_nodes: wp.array(dtype=int), t_L: wp.array(dtype=float), t_chord: wp.array(dtype=float),
                   t_ring_age: wp.array(dtype=float), t_starve: wp.array(dtype=int),
                   t_starve_I: wp.array(dtype=int), t_retract_len: wp.array(dtype=float),
                   t_fork_s: wp.array(dtype=float), t_timer: wp.array(dtype=float),
                   t_stretch: wp.array(dtype=float), t_tail: wp.array(dtype=int),
                   t_new: wp.array(dtype=int), t_free_snapshot: wp.array(dtype=int),
                   t_committed: wp.array(dtype=int),
                   t_regrow: wp.array(dtype=int), t_regrow_frame: wp.array(dtype=int),
                   t_leader_frame: wp.array(dtype=int), t_prune_side: wp.array(dtype=int),
                   t_foot2: wp.array(dtype=int), t_brush: wp.array(dtype=int),
                   stage_min: wp.array(dtype=float), stage_max: wp.array(dtype=float),
                   slot: wp.array(dtype=wp.int64),
                   c_f: int, f_max: int, k: int, s_max: int, s: int):
    """Commits every pool's Gumbel-max winner and spawns its K candidates in one launch.

    Block (t, kk): every block re-derives the step's commits from the read-only Gumbel slots
    (rank -> node id from the free list), so the decision is identical in all blocks; lane 0 of
    block (t, 0) is the only writer of the node and tree arrays. New nodes of this step are
    skipped in the node loop and added explicitly (Born charge), so the spawned potentials never
    depend on whether another block's write is visible yet. Thread F_MAX's role (electrode
    seeding) is taken by the blocks of the lowest tree slot that was FREE at frame start.
    A pool that does not commit clears its incremental entry t_new (this step's k_update_key has
    consumed it), so an idle tree never re-adds a stale charge (Kim Eq.11 double counting)."""
    blk, lane = wp.tid()
    t = blk // k
    kk = blk - t * k
    f1 = f_max + 1
    cur = (s % 2) * f1
    nxt = ((s + 1) % 2) * f1
    if blk == 0:
        for u in range(lane, f1, wp.block_dim()):
            slot[nxt + u] = wp.int64(0)
            stage_min[cur + u] = BIG
            stage_max[cur + u] = -BIG
    if IP[IP_PAUSE] != 0:
        return
    frame = cnt[CNT_FRAME]
    step = frame * s_max + s
    h = P[P_H]
    a = P[P_A]
    r1 = P[P_R1]
    r2 = P[P_R2]
    images = IP[IP_IMAGES]
    gain = P[P_BORN_GAIN]
    writer = kk == 0 and lane == 0
    # this step's commits (identical in every block): committing pools in index order
    e_commit = s == 0 and slot[cur + f_max] != wp.int64(0) and admit[0] != 0
    t_e = int(-1)
    if e_commit:
        for u in range(f_max):
            if t_e < 0 and t_free_snapshot[u] != 0:
                t_e = u
        if t_e < 0:
            e_commit = False
    my_commit = slot[cur + t] != wp.int64(0)
    is_claimed = e_commit and t == t_e
    if not (my_commit or is_claimed):
        if writer:
            t_new[t] = -1
            if growing(t_state[t], IP[IP_GROW_AFTER_ATTACH], t_regrow[t]):
                t_starve[t] = t_starve[t] + 1
                if t_state[t] == GROW and t_starve[t] >= IP[IP_STARVE_STEPS] and \
                        float(frame - t_birth[t]) * P[P_DT] > P[P_STARVE_AGE]:
                    t_state[t] = RETRACT
                    t_retract_len[t] = 0.0
                elif t_state[t] == ATTACHED and t_regrow[t] == 1 and \
                        t_starve[t] >= REGROW_PATIENCE * IP[IP_STARVE_STEPS]:
                    t_regrow[t] = 3          # the leader found no gated site: re-seed elsewhere
        return
    free_top = cnt[CNT_FREE_TOP]
    pool = t
    if is_claimed:
        pool = f_max
    ids = commit_ids()
    cs = commit_ids()
    n_commit = int(0)
    my_i = int(-1)
    my_c = int(-1)
    for u in range(f1):
        v = slot[cur + u]
        if v != wp.int64(0) and (u < f_max or e_commit):
            idx = free_top - 1 - n_commit
            if idx >= 0:
                cu = int(v & wp.int64(4294967295))
                iu = free_ids[idx]
                ids[n_commit] = iu
                cs[n_commit] = cu
                if u == pool:
                    my_i = iu
                    my_c = cu
                n_commit += 1
    if my_i < 0:
        # free list exhausted: nothing committed for this pool
        if writer:
            t_new[t] = -1
        return
    x0 = cand_pos[my_c]
    phi0 = cand_phi[my_c]
    kch = channel_k(P)
    # arc length of the new node and its local potential reference (visible to every block:
    # the parent's arrays are not written this step)
    s_new = float(0.0)
    if not is_claimed:
        ppar0 = cand_parent[my_c]
        if ppar0 >= 0:
            s_new = s_arc[ppar0] + wp.length(x0 - pos[ppar0])
    ref_new = float(0.0)
    if not is_claimed:
        ppar1 = cand_parent[my_c]
        if ppar1 >= 0:
            ref_new = kch * wp.length(x0 - pos[ppar1])
    if writer:
        i = my_i
        pos[i] = x0
        prev_pos[i] = x0
        tree[i] = t
        birth[i] = frame
        # the new node must sit at phi_ch(s_new) = phi_ch(s_parent) + kch * link: phi0 is local
        q[i] = -gain * (phi0 - ref_new) / self_potential(x0, r1, a, images)
        cand_alive[my_c] = 0
        if is_claimed:
            parent[i] = -1
            s_arc[i] = 0.0
            flags[i] = ALIVE | CHARGED | ROOT
            t_state[t] = GROW
            t_root[t] = i
            t_root_dir[t] = wp.normalize(x0)
            t_birth[t] = frame
            t_nodes[t] = 1
            t_L[t] = 0.0
            t_chord[t] = 0.0
            t_ring_age[t] = 0.0
            t_retract_len[t] = 0.0
            t_starve_I[t] = 0
            t_fork_s[t] = 0.0
            t_foot[t] = -1
            t_foot_dir[t] = wp.vec3(0.0, 0.0, 0.0)
            t_stretch[t] = 1.0
            t_timer[t] = reroute_timer(IP[IP_SEED], step, t, SALT_TIMER_STEP)
            t_regrow[t] = 0
            t_regrow_frame[t] = 0
            t_leader_frame[t] = 0
            t_prune_side[t] = 0
            t_brush[t] = 0
            for j in range(BRUSH_FEET):
                t_foot2[t * BRUSH_FEET + j] = -1
            t_committed[f_max] = 1 + t
        else:
            pp = cand_parent[my_c]
            parent[i] = pp
            s_arc[i] = s_arc[pp] + wp.length(x0 - pos[pp])
            flags[i] = ALIVE | CHARGED
            t_nodes[t] = t_nodes[t] + 1
            t_L[t] = wp.max(t_L[t], s_arc[i])
            t_committed[t] = 1
        t_starve[t] = 0
        t_tip[t] = i
        t_new[t] = i
        wp.atomic_max(cnt, CNT_NODE_HW, i + 1)
        if wp.length(x0) >= r2 - a - 1.0e-6:
            flags[i] = flags[i] | FOOT
            if t_state[t] != ATTACHED:
                t_state[t] = ATTACHED
                t_foot[t] = i
                t_foot_dir[t] = wp.normalize(x0)
                t_ring_age[t] = 0.0
                t_chord[t] = wp.length(x0 - pos[t_root[t]])
                t_stretch[t] = 1.0
                t_timer[t] = reroute_timer(IP[IP_SEED], step, t, SALT_TIMER_STEP)
                t_prune_side[t] = 1
            elif t_regrow[t] == 1 and t_brush[t] != 0:
                # a brush leader reached the glass: a secondary foot of the same channel; the
                # leader's own branches fade at the prune, its chain is kept (marked from the foot)
                fs = int(-1)
                for j in range(BRUSH_FEET):
                    if fs < 0 and t_foot2[t * BRUSH_FEET + j] < 0:
                        fs = t * BRUSH_FEET + j
                if fs >= 0:
                    t_foot2[fs] = i
                t_brush[t] = 0
                t_regrow[t] = 0
                t_prune_side[t] = 1
            elif t_regrow[t] == 1:
                # the re-route leader reached the glass: the foot moves, the old distal channel
                # is pruned by the next frame's k_node_frame; the brush feet of the old path go
                for j in range(BRUSH_FEET):
                    t_foot2[t * BRUSH_FEET + j] = -1
                t_foot[t] = i
                t_foot_dir[t] = wp.normalize(x0)
                t_ring_age[t] = 0.0
                t_chord[t] = wp.length(x0 - pos[t_root[t]])
                t_L[t] = s_arc[i]
                t_stretch[t] = 1.0
                t_timer[t] = reroute_timer(IP[IP_SEED], step, t, SALT_TIMER_STEP)
                t_regrow[t] = 2
                t_prune_side[t] = 1
    # spawn candidate kk of the new node
    tail = t_tail[t]
    c_new = t * c_f + (tail + kk) % c_f
    if c_new == my_c:
        # the ring is full and would overwrite the winner slot other blocks still read
        return
    # growth direction of the new node: from its parent (radial for a root)
    dirz = wp.normalize(x0)
    if not is_claimed:
        ppar = cand_parent[my_c]
        if ppar >= 0:
            dv = x0 - pos[ppar]
            if wp.length(dv) > 1.0e-6:
                dirz = wp.normalize(dv)
    st = wp.rand_init(step_seed(IP[IP_SEED], step, SALT_SPAWN), t)
    phase = 6.283185307 * wp.randf(st)
    x = x0 + h * cone_dir(kk, dirz, phase)
    r = wp.length(x)
    valid = int(1)
    if r < r1 + a:
        valid = 0
    if r > r2 - a:
        x = x * ((r2 - a) / r)
    if wp.length(x - x0) < 0.5 * h:
        # the shell projection pulled the candidate back onto its parent: same h/2 rule as the
        # re-base applies to every node, the parent included
        valid = 0
    acc = float(0.0)
    dmin = float(BIG)
    n = cnt[CNT_NODE_HW]
    for base in range(0, n, wp.block_dim()):
        j = base + lane
        if j < n:
            fresh = int(0)
            for u in range(n_commit):
                if ids[u] == j:
                    fresh = 1
            if fresh == 0:
                fj = flags[j]
                if (fj & ALIVE) != 0:
                    xj = pos[j]
                    dmin = wp.min(dmin, wp.length(x - xj))
                    if (fj & CHARGED) != 0:
                        acc += q[j] * pair_potential(x, xj, r1, a, images)
    sacc = wp.tile_sum(wp.tile(acc))
    smin = wp.tile_min(wp.tile(dmin))
    if lane == 0:
        total = sacc[0]
        dm = smin[0]
        for u in range(n_commit):
            xu = cand_pos[cs[u]]
            refu = float(0.0)
            if cs[u] < c_f * f_max:
                pu = cand_parent[cs[u]]
                if pu >= 0:
                    refu = kch * wp.length(xu - pos[pu])
            total += (-gain * (cand_phi[cs[u]] - refu) / self_potential(xu, r1, a, images)) \
                * pair_potential(x, xu, r1, a, images)
            if ids[u] != my_i:
                dm = wp.min(dm, wp.length(x - xu))
        if dm < 0.5 * h:
            valid = 0
        phi = boundary_potential(x, P, finger_dir, finger_q, n_fingers[0]) + total - kch * s_new
        cand_pos[c_new] = x
        cand_phi[c_new] = phi
        cand_parent[c_new] = my_i
        cand_alive[c_new] = valid
        cand_stamp[c_new] = step
        if valid != 0:
            wp.atomic_min(stage_min, nxt + t, phi)
            wp.atomic_max(stage_max, nxt + t, phi)


@wp.kernel
def k_settle(cnt: wp.array(dtype=int), t_committed: wp.array(dtype=int), t_tail: wp.array(dtype=int),
             t_state: wp.array(dtype=int), t_free_snapshot: wp.array(dtype=int), k: int, f_max: int):
    settle_step(cnt, t_committed, t_tail, t_state, t_free_snapshot, k, f_max)


@wp.kernel
def k_prepare(stage_min: wp.array(dtype=float), stage_max: wp.array(dtype=float),
              t_new: wp.array(dtype=int), buf: int, f1: int):
    """Before a (mid-frame) re-base into buffer `buf`: reset it and drop the incremental list."""
    t = wp.tid()
    stage_min[buf * f1 + t] = BIG
    stage_max[buf * f1 + t] = -BIG
    t_new[t] = -1


# --- conductor charges: Jacobi-preconditioned CG on M q = b -------------------------------------

@wp.kernel
def k_matvec(P: wp.array(dtype=float), IP: wp.array(dtype=int), cnt: wp.array(dtype=int),
             pos: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int),
             finger_dir: wp.array(dtype=wp.vec3), finger_q: wp.array(dtype=float),
             n_fingers: wp.array(dtype=int),
             v: wp.array(dtype=float), out: wp.array(dtype=float),
             b: wp.array(dtype=float), r: wp.array(dtype=float), z: wp.array(dtype=float),
             p: wp.array(dtype=float), inv_diag: wp.array(dtype=float), part: wp.array(dtype=float),
             s_arc: wp.array(dtype=float), init: int):
    """out = M v over the charged nodes (ROWS rows per block, lanes stride the columns) and the
    block's partial of v . out (init: of r . z).

    With init != 0 it also builds the right-hand side and the warm-started CG state:
    r = b - M q, z = r / diag, p = z."""
    blk, lane = wp.tid()
    i0 = blk * ROWS
    n = cnt[CNT_NODE_HW]
    if i0 >= n:
        if lane == 0:
            part[blk] = 0.0
        return
    f0 = flags[i0] & CHARGED
    f1 = flags[i0 + 1] & CHARGED
    f2 = flags[i0 + 2] & CHARGED
    f3 = flags[i0 + 3] & CHARGED
    if f0 == 0 and f1 == 0 and f2 == 0 and f3 == 0:
        if lane == 0:
            part[blk] = 0.0
            for rr in range(ROWS):
                out[i0 + rr] = 0.0
                if init != 0:
                    b[i0 + rr] = 0.0
                    r[i0 + rr] = 0.0
                    z[i0 + rr] = 0.0
                    p[i0 + rr] = 0.0
        return
    x0 = pos[i0]
    x1 = pos[i0 + 1]
    x2 = pos[i0 + 2]
    x3 = pos[i0 + 3]
    a0 = float(0.0)
    a1 = float(0.0)
    a2 = float(0.0)
    a3 = float(0.0)
    er1 = P[P_R1]
    a = P[P_A]
    images = IP[IP_IMAGES]
    for base in range(0, n, wp.block_dim()):
        j = base + lane
        if j < n:
            if (flags[j] & CHARGED) != 0:
                xj = pos[j]
                vj = v[j]
                if j != i0:
                    a0 += vj * pair_potential(x0, xj, er1, a, images)
                if j != i0 + 1:
                    a1 += vj * pair_potential(x1, xj, er1, a, images)
                if j != i0 + 2:
                    a2 += vj * pair_potential(x2, xj, er1, a, images)
                if j != i0 + 3:
                    a3 += vj * pair_potential(x3, xj, er1, a, images)
    s0 = wp.tile_sum(wp.tile(a0))
    s1 = wp.tile_sum(wp.tile(a1))
    s2 = wp.tile_sum(wp.tile(a2))
    s3 = wp.tile_sum(wp.tile(a3))
    if lane == 0:
        a = P[P_A]
        dot = float(0.0)
        for rr in range(ROWS):
            i = i0 + rr
            acc = s0[0]
            if rr == 1:
                acc = s1[0]
            elif rr == 2:
                acc = s2[0]
            elif rr == 3:
                acc = s3[0]
            val = float(0.0)
            diag = float(1.0)
            if (flags[i] & CHARGED) != 0:
                diag = self_potential(pos[i], er1, a, images)
                val = acc + diag * v[i]
            out[i] = val
            if init != 0:
                bi = float(0.0)
                if (flags[i] & CHARGED) != 0:
                    # the conductor sits at the channel potential phi_ch(s), not at 0
                    bi = channel_k(P) * s_arc[i] - boundary_potential(pos[i], P, finger_dir, finger_q, n_fingers[0])
                b[i] = bi
                ri = bi - val
                r[i] = ri
                z[i] = ri / diag
                p[i] = ri / diag
                inv_diag[i] = 1.0 / diag
                dot += ri * ri / diag
            else:
                dot += v[i] * val
        part[blk] = dot


@wp.kernel
def k_reduce(part: wp.array(dtype=float), cg: wp.array(dtype=float), n_part: int, mode: int):
    """Single-block, fixed-order sum of the matvec partials; mode 0: rz, 1: pAp + alpha."""
    blk, lane = wp.tid()
    acc = float(0.0)
    for i in range(lane, n_part, wp.block_dim()):
        acc += part[i]
    total = wp.tile_sum(wp.tile(acc))
    if lane == 0:
        if mode == 0:
            cg[CG_RZ] = total[0]
        else:
            cg[CG_PAP] = total[0]
            alpha = float(0.0)
            if total[0] > 0.0:
                alpha = cg[CG_RZ] / total[0]
            cg[CG_ALPHA] = alpha


@wp.kernel
def k_cg_update(cnt: wp.array(dtype=int), flags: wp.array(dtype=int), cg: wp.array(dtype=float),
                q: wp.array(dtype=float), r: wp.array(dtype=float), z: wp.array(dtype=float),
                p: wp.array(dtype=float), Ap: wp.array(dtype=float), inv_diag: wp.array(dtype=float),
                b: wp.array(dtype=float), last: int):
    """Single block: q += alpha p, r -= alpha Ap, z = r / diag, and the deterministic r . z
    reduction (beta); on the last sweep also |r|^2 and |b|^2 for the residual readback."""
    blk, lane = wp.tid()
    n = cnt[CNT_NODE_HW]
    alpha = cg[CG_ALPHA]
    rz = float(0.0)
    rr = float(0.0)
    bb = float(0.0)
    for i in range(lane, n, wp.block_dim()):
        if (flags[i] & CHARGED) != 0:
            q[i] = q[i] + alpha * p[i]
            ri = r[i] - alpha * Ap[i]
            r[i] = ri
            zi = inv_diag[i] * ri
            z[i] = zi
            rz += ri * zi
            if last != 0:
                rr += ri * ri
                bb += b[i] * b[i]
    t_rz = wp.tile_sum(wp.tile(rz))
    t_rr = wp.tile_sum(wp.tile(rr))
    t_bb = wp.tile_sum(wp.tile(bb))
    if lane == 0:
        beta = float(0.0)
        if cg[CG_RZ] > 0.0:
            beta = t_rz[0] / cg[CG_RZ]
        cg[CG_BETA] = beta
        cg[CG_RZ_NEW] = t_rz[0]
        cg[CG_RZ] = t_rz[0]
        if last != 0:
            cg[CG_RR] = t_rr[0]
            cg[CG_BB] = t_bb[0]


@wp.kernel
def k_cg_direction(cnt: wp.array(dtype=int), flags: wp.array(dtype=int), cg: wp.array(dtype=float),
                   z: wp.array(dtype=float), p: wp.array(dtype=float)):
    i = wp.tid()
    if i >= cnt[CNT_NODE_HW] or (flags[i] & CHARGED) == 0:
        return
    p[i] = z[i] + cg[CG_BETA] * p[i]


@wp.kernel
def k_init_free(free_ids: wp.array(dtype=int), cnt: wp.array(dtype=int), n_max: int):
    """Full free list and the counters of an empty globe (a kernel, so reset() captures)."""
    i = wp.tid()
    free_ids[i] = n_max - 1 - i
    if i == 0:
        cnt[CNT_FRAME] = 0
        cnt[CNT_FREE_TOP] = n_max
        cnt[CNT_NODE_HW] = 0


# --- engine --------------------------------------------------------------------------------------

class Dbm:
    """Persistent dielectric-breakdown growth state with a graph-capturable per-frame ``step``.

    All capacities are fixed at construction; nothing allocates, synchronises or reads back
    inside ``step``. ``mid_cg=(every, iterations)`` optionally inserts `iterations` CG sweeps
    plus a candidate re-base after every `every` growth steps (default off = the plan's one
    solve per frame; measured in the lab as the mitigation of the in-frame Born error). Knobs
    live in the pinned host arrays ``P`` / ``IP`` (copied to the device as the first graph node)
    and in the small device arrays ``finger_dir/finger_q/n_fingers``, ``hot_*``, ``admit``,
    ``t_I``, ``t_stretch`` and ``cand_s`` that the coupled app (gas, circuit, advection, sigma)
    writes in-graph. Every host-side setter copies from a persistent pinned staging array, so
    ``reset()`` and the setters are safe inside a ``wp.ScopedCapture`` too."""

    def __init__(self, device="cuda:0", r1=R1, r2=R2I, h=H, a_over_h=A_OVER_H,
                 n_max=N_MAX, f_max=F_MAX, c_f=C_F, k=K, s_max=S_MAX, e_pool=E_POOL,
                 n_cg=6, cg_stats=True, seed=1, images=1, mid_cg=(0, 0)):
        if c_f % ROWS or e_pool % ROWS or n_max % ROWS:
            raise ValueError("c_f, e_pool and n_max must be multiples of ROWS")
        if f_max > F_MAX:
            raise ValueError(f"f_max must be <= {F_MAX}")
        self.device = device
        self.n_max, self.f_max, self.c_f, self.k, self.s_max, self.e_pool = n_max, f_max, c_f, k, s_max, e_pool
        self.c_max = f_max * c_f
        self.c_total = self.c_max + e_pool
        self.n_cg = n_cg
        self.mid_cg_every, self.mid_cg_iters = mid_cg
        self.cg_stats = cg_stats
        self.graph = None
        with wp.ScopedDevice(device):
            self._allocate()
        self.P_host = wp.zeros(P_COUNT, dtype=float, device="cpu", pinned=(device != "cpu"))
        self.IP_host = wp.zeros(IP_COUNT, dtype=int, device="cpu", pinned=(device != "cpu"))
        self.configure(E_ch=0.0, r1=r1, r2=r2, h=h, a=a_over_h * h, V=V_DEFAULT, T0=T0, E_bd0=E_BD0,
                       E_prop0=E_PROP0, eta=ETA, gamma=GAMMA, hf=H_F, dt=DT,
                       retract_speed=RETRACT_SPEED, starve_age=STARVE_AGE, I_sus=0.0,
                       decay_time=DECAY_TIME, born_gain=BORN_GAIN, seed=seed, global_norm=0,
                       use_sigma=0,
                       grow_after_attach=1, pause=0, enable_reroute=0, starve_steps=STARVE_STEPS,
                       images=images)
        self.reset()

    # -- allocation -------------------------------------------------------------------------

    def _allocate(self):
        n, ct, f1 = self.n_max, self.c_total, self.f_max + 1
        v3, f32, i32 = wp.vec3, wp.float32, wp.int32
        self.P = wp.zeros(P_COUNT, dtype=f32)
        self.IP = wp.zeros(IP_COUNT, dtype=i32)
        self.cnt = wp.zeros(CNT_COUNT, dtype=i32)
        self.admit = wp.ones(1, dtype=i32)
        # node SoA
        self.pos = wp.zeros(n, dtype=v3)
        self.prev_pos = wp.zeros(n, dtype=v3)
        self.parent = wp.full(n, -1, dtype=i32)
        self.tree = wp.full(n, -1, dtype=i32)
        self.birth_frame = wp.zeros(n, dtype=i32)
        self.stamp = wp.zeros(n, dtype=i32)
        self.s_arc = wp.zeros(n, dtype=f32)
        self.q = wp.zeros(n, dtype=f32)
        self.flags = wp.zeros(n, dtype=i32)
        self.free_ids = wp.zeros(n, dtype=i32)
        self.mark = wp.zeros(n, dtype=i32)
        self.scan = wp.zeros(n, dtype=i32)
        # candidates (tree pools followed by the electrode pool)
        self.cand_pos = wp.zeros(ct, dtype=v3)
        self.cand_phi = wp.zeros(ct, dtype=f32)
        self.cand_parent = wp.full(ct, -1, dtype=i32)
        self.cand_alive = wp.zeros(ct, dtype=i32)
        self.cand_stamp = wp.full(ct, -1, dtype=i32)
        self.cand_s = wp.ones(ct, dtype=f32)
        self.cand_T = wp.zeros(ct, dtype=f32)   # coupled app: absolute T at the candidate (0 = default sampler)
        # per-tree state (+1 for the electrode pool where indexed by pool)
        self.t_state = wp.zeros(self.f_max, dtype=i32)
        self.t_root = wp.full(self.f_max, -1, dtype=i32)
        self.t_tip = wp.full(self.f_max, -1, dtype=i32)
        self.t_foot = wp.full(self.f_max, -1, dtype=i32)
        self.t_root_dir = wp.zeros(self.f_max, dtype=v3)
        self.t_foot_dir = wp.zeros(self.f_max, dtype=v3)
        self.t_birth = wp.zeros(self.f_max, dtype=i32)
        self.t_nodes = wp.zeros(self.f_max, dtype=i32)
        self.t_I = wp.full(self.f_max, I_DEFAULT, dtype=f32)
        self.t_starve_I = wp.zeros(self.f_max, dtype=i32)
        self.t_L = wp.zeros(self.f_max, dtype=f32)
        self.t_chord = wp.zeros(self.f_max, dtype=f32)
        self.t_ring_age = wp.zeros(self.f_max, dtype=f32)
        self.t_timer = wp.zeros(self.f_max, dtype=f32)
        self.t_starve = wp.zeros(self.f_max, dtype=i32)
        self.t_retract_len = wp.zeros(self.f_max, dtype=f32)
        self.t_fork_s = wp.zeros(self.f_max, dtype=f32)
        self.t_stretch = wp.ones(self.f_max, dtype=f32)
        self.t_fork_key = wp.zeros(self.f_max, dtype=wp.int64)
        self.t_regrow = wp.zeros(self.f_max, dtype=i32)
        self.t_regrow_frame = wp.zeros(self.f_max, dtype=i32)
        self.t_leader_frame = wp.zeros(self.f_max, dtype=i32)
        self.t_prune_side = wp.zeros(self.f_max, dtype=i32)
        self.t_reroute_req = wp.zeros(self.f_max, dtype=i32)
        self.t_foot2 = wp.full(self.f_max * BRUSH_FEET, -1, dtype=i32)
        self.t_brush = wp.zeros(self.f_max, dtype=i32)
        self.t_brush_req = wp.zeros(self.f_max, dtype=i32)
        self.t_foot2_drop = wp.zeros(self.f_max * BRUSH_FEET, dtype=i32)
        self.t_hold = wp.zeros(self.f_max, dtype=i32)
        self.main_mark = wp.zeros(n, dtype=i32)
        self.t_tail = wp.zeros(self.f_max, dtype=i32)
        self.t_free_snapshot = wp.zeros(self.f_max, dtype=i32)
        self.t_new = wp.full(f1, -1, dtype=i32)
        self.t_committed = wp.zeros(f1, dtype=i32)
        self.stage_min = wp.zeros(2 * f1, dtype=f32)
        self.stage_max = wp.zeros(2 * f1, dtype=f32)
        self.slot = wp.zeros(2 * f1, dtype=wp.int64)
        # touch and thermal hooks
        self.finger_dir = wp.zeros(FINGER_MAX, dtype=v3)
        self.finger_q = wp.zeros(FINGER_MAX, dtype=f32)
        self.n_fingers = wp.zeros(1, dtype=i32)
        self.hot_p0 = wp.zeros(HOT_MAX, dtype=v3)
        self.hot_p1 = wp.zeros(HOT_MAX, dtype=v3)
        self.hot_r = wp.zeros(HOT_MAX, dtype=f32)
        self.hot_T = wp.zeros(HOT_MAX, dtype=f32)
        # CG workspace
        self.cg = wp.zeros(CG_COUNT, dtype=f32)
        self.cg_b = wp.zeros(n, dtype=f32)
        self.cg_r = wp.zeros(n, dtype=f32)
        self.cg_z = wp.zeros(n, dtype=f32)
        self.cg_p = wp.zeros(n, dtype=f32)
        self.cg_Ap = wp.zeros(n, dtype=f32)
        self.cg_inv_diag = wp.ones(n, dtype=f32)
        self.cg_part = wp.zeros(n // ROWS, dtype=f32)
        # pinned host staging of the setters (persistent: safe to copy from inside a capture)
        pinned = self.device != "cpu"
        self.finger_dir_host = wp.zeros(FINGER_MAX, dtype=v3, device="cpu", pinned=pinned)
        self.finger_q_host = wp.zeros(FINGER_MAX, dtype=f32, device="cpu", pinned=pinned)
        self.n_fingers_host = wp.zeros(1, dtype=i32, device="cpu", pinned=pinned)
        self.hot_p0_host = wp.zeros(HOT_MAX, dtype=v3, device="cpu", pinned=pinned)
        self.hot_p1_host = wp.zeros(HOT_MAX, dtype=v3, device="cpu", pinned=pinned)
        self.hot_r_host = wp.zeros(HOT_MAX, dtype=f32, device="cpu", pinned=pinned)
        self.hot_T_host = wp.zeros(HOT_MAX, dtype=f32, device="cpu", pinned=pinned)
        self.admit_host = wp.ones(1, dtype=i32, device="cpu", pinned=pinned)
        self.t_state_host = wp.zeros(self.f_max, dtype=i32, device="cpu", pinned=pinned)
        # the mutable simulation state (reset / snapshot / restore); params and hooks excluded
        self._zeroed = [self.pos, self.prev_pos, self.birth_frame, self.stamp, self.s_arc, self.q,
                        self.flags, self.mark, self.scan, self.cand_pos, self.cand_phi,
                        self.cand_alive, self.t_state, self.t_root_dir, self.t_foot_dir,
                        self.t_birth, self.t_nodes, self.t_starve_I, self.t_L, self.t_chord,
                        self.t_ring_age, self.t_timer, self.t_starve, self.t_retract_len,
                        self.t_fork_s, self.t_fork_key, self.t_regrow, self.t_regrow_frame,
                        self.t_leader_frame, self.t_prune_side, self.main_mark, self.t_reroute_req,
                        self.t_brush, self.t_brush_req, self.t_foot2_drop, self.t_hold, self.t_tail,
                        self.t_free_snapshot,
                        self.t_committed, self.slot, self.cg, self.cg_b, self.cg_r, self.cg_z, self.cg_p, self.cg_Ap,
                        self.cg_part]
        self._minus_one = [self.parent, self.tree, self.cand_parent, self.cand_stamp, self.t_root,
                           self.t_tip, self.t_foot, self.t_new, self.t_foot2]
        self._state = self._zeroed + self._minus_one + [
            self.cnt, self.admit, self.free_ids, self.cand_s, self.t_I, self.t_stretch,
            self.stage_min, self.stage_max, self.cg_inv_diag]
        # warm up array_scan's temporary storage outside any capture
        wp.utils.array_scan(self.mark, self.scan, inclusive=True)

    # -- configuration ------------------------------------------------------------------

    _P_KEYS = dict(r1=P_R1, r2=P_R2, h=P_H, a=P_A, V=P_V, T0=P_T0, E_bd0=P_E_BD0, E_prop0=P_E_PROP0,
                   eta=P_ETA, gamma=P_GAMMA, hf=P_HF, dt=P_DT, retract_speed=P_RETRACT_SPEED,
                   starve_age=P_STARVE_AGE, I_sus=P_I_SUS, decay_time=P_DECAY_TIME,
                   born_gain=P_BORN_GAIN, E_ch=P_E_CH)
    _IP_KEYS = dict(seed=IP_SEED, global_norm=IP_GLOBAL_NORM, use_sigma=IP_USE_SIGMA,
                    grow_after_attach=IP_GROW_AFTER_ATTACH, pause=IP_PAUSE,
                    enable_reroute=IP_ENABLE_REROUTE, starve_steps=IP_STARVE_STEPS,
                    images=IP_IMAGES)

    def configure(self, **knobs):
        """Sets float / int knobs in the pinned params (picked up by the next step / replay)."""
        P = self.P_host.numpy()
        IP = self.IP_host.numpy()
        for key, value in knobs.items():
            if key in self._P_KEYS:
                P[self._P_KEYS[key]] = value
            elif key in self._IP_KEYS:
                IP[self._IP_KEYS[key]] = value
            else:
                raise KeyError(key)

    def param(self, key):
        if key in self._P_KEYS:
            return float(self.P_host.numpy()[self._P_KEYS[key]])
        return int(self.IP_host.numpy()[self._IP_KEYS[key]])

    def set_fingers(self, fingers):
        """fingers: iterable of (unit direction (3,), q_f); up to FINGER_MAX."""
        fingers = list(fingers)[:FINGER_MAX]
        d = self.finger_dir_host.numpy()
        qf = self.finger_q_host.numpy()
        d[:] = 0.0
        qf[:] = 0.0
        for i, (n_f, q_f) in enumerate(fingers):
            n_f = np.asarray(n_f, dtype=np.float64)
            d[i] = n_f / np.linalg.norm(n_f)
            qf[i] = q_f
        self.n_fingers_host.numpy()[0] = len(fingers)
        with wp.ScopedDevice(self.device):
            wp.copy(self.finger_dir, self.finger_dir_host)
            wp.copy(self.finger_q, self.finger_q_host)
            wp.copy(self.n_fingers, self.n_fingers_host)

    def set_admit(self, flag):
        """The 'circuit admits a new tree' device flag (the M4 circuit writes it in-graph)."""
        self.admit_host.numpy()[0] = 1 if flag else 0
        with wp.ScopedDevice(self.device):
            wp.copy(self.admit, self.admit_host)

    def set_tree_state(self, t, state):
        """Host write of one tree's state (lab protocols: FROZEN / back to GROW)."""
        self.t_state_host.numpy()[t] = state
        with wp.ScopedDevice(self.device):
            wp.copy(self.t_state, self.t_state_host, dest_offset=t, src_offset=t, count=1)

    def set_hot_channels(self, segments):
        """segments: iterable of (p0 (3,), p1 (3,), radius, T); the default temperature sampler."""
        segments = list(segments)[:HOT_MAX]
        p0, p1 = self.hot_p0_host.numpy(), self.hot_p1_host.numpy()
        rad, temp = self.hot_r_host.numpy(), self.hot_T_host.numpy()
        for arr in (p0, p1, rad, temp):
            arr[:] = 0.0
        for i, (a, b, r, t) in enumerate(segments):
            p0[i], p1[i], rad[i], temp[i] = a, b, r, t
        with wp.ScopedDevice(self.device):
            wp.copy(self.hot_p0, self.hot_p0_host)
            wp.copy(self.hot_p1, self.hot_p1_host)
            wp.copy(self.hot_r, self.hot_r_host)
            wp.copy(self.hot_T, self.hot_T_host)
        self.IP_host.numpy()[IP_HOT_COUNT] = len(segments)

    # -- state ------------------------------------------------------------------------------

    def reset(self):
        """Returns to the empty globe (no trees); capturable (fills + one kernel + the pinned
        params copy: no host temporaries)."""
        with wp.ScopedDevice(self.device):
            for a in self._zeroed:
                a.zero_()
            for a in self._minus_one:
                a.fill_(-1)
            self.admit.fill_(1)
            self.t_I.fill_(I_DEFAULT)
            self.t_stretch.fill_(1.0)
            self.cand_s.fill_(1.0)
            self.cand_T.zero_()
            self.cg_inv_diag.fill_(1.0)
            self.stage_min.fill_(BIG)
            self.stage_max.fill_(-BIG)
            wp.launch(k_init_free, dim=self.n_max, inputs=[self.free_ids, self.cnt, self.n_max])
            self.upload_params()

    def snapshot(self):
        """Device clones of the whole mutable state (outside the graph; allocates)."""
        with wp.ScopedDevice(self.device):
            return [wp.clone(a) for a in self._state]

    def restore(self, snap):
        """Copies a ``snapshot()`` back; the captured graph stays valid (same arrays)."""
        with wp.ScopedDevice(self.device):
            for dst, src in zip(self._state, snap):
                wp.copy(dst, src)

    # -- one frame ------------------------------------------------------------------------

    def step(self):
        """One frame of growth: link subdivision, prev_pos <- pos, bookkeeping, re-base, S_MAX
        steps, CG solve."""
        n, f, f1, k, s_max = self.n_max, self.f_max, self.f_max + 1, self.k, self.s_max
        c_f, c_max = self.c_f, self.c_max
        with wp.ScopedDevice(self.device):
            self.upload_params()
            wp.launch(k_subdivide_mark, dim=n, inputs=[self.P, self.IP, self.cnt, self.pos, self.parent,
                                                       self.flags, self.tree, self.t_nodes, self.mark])
            wp.utils.array_scan(self.mark, self.scan, inclusive=True)
            wp.launch(k_subdivide_insert, dim=n, inputs=[
                self.cnt, self.scan, self.free_ids, self.pos, self.prev_pos, self.parent, self.tree,
                self.birth_frame, self.s_arc, self.q, self.flags, self.stamp, self.mark, self.t_nodes])
            wp.launch(k_subdivide_finish, dim=1, inputs=[self.cnt, self.scan, n])
            wp.launch(k_frame_begin, dim=n, inputs=[self.pos, self.prev_pos, self.cnt])
            wp.launch(k_tree_frame, dim=f1, inputs=[
                self.P, self.IP, self.cnt, self.pos, self.parent, self.s_arc, self.t_foot,
                self.t_state, self.t_birth, self.t_I, self.t_starve_I,
                self.t_L, self.t_chord, self.t_ring_age, self.t_timer, self.t_retract_len,
                self.t_fork_s, self.t_stretch, self.t_fork_key, self.t_regrow, self.t_regrow_frame,
                self.t_leader_frame, self.t_prune_side, self.main_mark, self.t_reroute_req,
                self.t_foot2, self.t_brush, self.t_brush_req, self.t_foot2_drop, self.t_hold,
                self.t_free_snapshot, self.t_new, self.stage_min, self.stage_max, self.slot])
            wp.launch(k_node_frame, dim=n, inputs=[
                self.P, self.IP, self.cnt, self.flags, self.tree, self.s_arc, self.stamp, self.q,
                self.birth_frame, self.pos, self.parent, self.t_state, self.t_L, self.t_retract_len, self.t_fork_s,
                self.t_fork_key, self.t_regrow, self.t_leader_frame, self.t_prune_side, self.main_mark,
                self.t_nodes, self.mark])
            wp.utils.array_scan(self.mark, self.scan, inclusive=True)
            wp.launch(k_free_push, dim=n, inputs=[self.mark, self.scan, self.cnt, self.free_ids])
            wp.launch(k_free_finish, dim=f, inputs=[
                self.P, self.IP, self.cnt, self.scan, self.pos, self.cand_pos, self.cand_phi,
                self.cand_parent, self.cand_alive, self.cand_stamp, self.t_state, self.t_nodes,
                self.t_L, self.t_fork_s, self.t_fork_key, self.t_tip, self.t_tail, self.t_new,
                self.t_regrow, self.t_leader_frame, self.t_prune_side, self.t_foot, self.t_foot_dir,
                self.t_starve, self.t_foot2, self.t_brush, c_f, k, n])
            wp.launch(k_electrode_pool, dim=self.e_pool, inputs=[
                self.P, self.IP, self.cnt, self.cand_pos, self.cand_parent, self.cand_alive,
                self.cand_stamp, c_max, self.e_pool])
            self._rebase(0)
            for s in range(s_max):
                self._growth_step(s)
                if self.mid_cg_every and (s + 1) % self.mid_cg_every == 0 and s + 1 < s_max:
                    wp.launch(k_settle, dim=1, inputs=[self.cnt, self.t_committed, self.t_tail,
                                                       self.t_state, self.t_free_snapshot, k, f])
                    self._cg(self.mid_cg_iters, stats=False)
                    wp.launch(k_prepare, dim=f1, inputs=[self.stage_min, self.stage_max, self.t_new,
                                                         (s + 1) % 2, f1])
                    self._rebase((s + 1) % 2)
            wp.launch(k_settle, dim=1, inputs=[self.cnt, self.t_committed, self.t_tail, self.t_state,
                                               self.t_free_snapshot, k, f])
            self._cg()

    def _rebase(self, buf):
        """Re-bases every live candidate's potential on the current charges (k_cand_phi_full),
        staging the extrema into buffer `buf` (read by the next step as its previous extrema)."""
        wp.launch_tiled(k_cand_phi_full, dim=[self.c_total // ROWS], block_dim=TILE, inputs=[
            self.P, self.IP, self.cnt, self.pos, self.q, self.flags, self.tree, self.s_arc,
            self.birth_frame, self.finger_dir, self.finger_q, self.n_fingers, self.t_state,
            self.t_regrow, self.t_leader_frame, self.t_fork_s, self.cand_pos, self.cand_phi,
            self.cand_parent, self.cand_alive, self.stage_min, self.stage_max, self.c_f, self.c_max,
            self.f_max, buf])

    def _growth_step(self, s):
        f, k, s_max, c_f, c_max = self.f_max, self.k, self.s_max, self.c_f, self.c_max
        wp.launch(k_update_key, dim=self.c_total, inputs=[
            self.P, self.IP, self.cnt, self.pos, self.q, self.t_state, self.t_new, self.t_committed,
            self.t_tail, self.t_free_snapshot, self.cand_pos, self.cand_phi, self.cand_alive,
            self.cand_stamp, self.cand_s, self.cand_T, self.t_regrow, self.hot_p0, self.hot_p1,
            self.hot_r, self.hot_T, self.stage_min, self.stage_max, self.slot, c_f, c_max, f, k, s_max, s])
        wp.launch_tiled(k_commit_spawn, dim=[f * k], block_dim=SPAWN_TILE, inputs=[
            self.P, self.IP, self.cnt, self.admit, self.pos, self.prev_pos, self.parent, self.tree,
            self.birth_frame, self.s_arc, self.q, self.flags, self.free_ids, self.finger_dir,
            self.finger_q, self.n_fingers, self.cand_pos, self.cand_phi, self.cand_parent,
            self.cand_alive, self.cand_stamp, self.t_state, self.t_root, self.t_tip, self.t_foot,
            self.t_root_dir, self.t_foot_dir, self.t_birth, self.t_nodes, self.t_L, self.t_chord,
            self.t_ring_age, self.t_starve, self.t_starve_I, self.t_retract_len, self.t_fork_s,
            self.t_timer, self.t_stretch, self.t_tail, self.t_new, self.t_free_snapshot,
            self.t_committed, self.t_regrow, self.t_regrow_frame, self.t_leader_frame, self.t_prune_side,
            self.t_foot2, self.t_brush, self.stage_min, self.stage_max, self.slot, c_f, f, k, s_max, s])

    def _matvec(self, v, out, init):
        wp.launch_tiled(k_matvec, dim=[self.n_max // ROWS], block_dim=TILE, inputs=[
            self.P, self.IP, self.cnt, self.pos, self.flags, self.finger_dir, self.finger_q,
            self.n_fingers, v, out, self.cg_b, self.cg_r, self.cg_z, self.cg_p, self.cg_inv_diag,
            self.cg_part, self.s_arc, init])
        wp.launch_tiled(k_reduce, dim=[1], block_dim=DOT_TILE, inputs=[self.cg_part, self.cg,
                                                                      self.n_max // ROWS, init == 0])

    def _cg(self, iterations=None, stats=True):
        """`iterations` warm-started Jacobi-PCG sweeps on M q = b (default n_cg): per sweep one
        matvec (+ p.Ap partials), one single-block reduction, one fused update + r.z reduction
        and one direction update; every reduction has a fixed order (bit-identical replays)."""
        iterations = self.n_cg if iterations is None else iterations
        self._matvec(self.q, self.cg_Ap, 1)
        for it in range(iterations):
            self._matvec(self.cg_p, self.cg_Ap, 0)
            last = 1 if (it == iterations - 1 and self.cg_stats and stats) else 0
            wp.launch_tiled(k_cg_update, dim=[1], block_dim=DOT_TILE, inputs=[
                self.cnt, self.flags, self.cg, self.q, self.cg_r, self.cg_z, self.cg_p,
                self.cg_Ap, self.cg_inv_diag, self.cg_b, last])
            wp.launch(k_cg_direction, dim=self.n_max, inputs=[self.cnt, self.flags, self.cg,
                                                              self.cg_z, self.cg_p])

    def solve(self, iterations):
        """Extra CG sweeps on the current node set (outside the graph; for lab comparisons)."""
        with wp.ScopedDevice(self.device):
            self.upload_params()
            self._cg(iterations)

    def upload_params(self):
        wp.copy(self.P, self.P_host)
        wp.copy(self.IP, self.IP_host)

    # -- graph ------------------------------------------------------------------------------

    def capture(self):
        with wp.ScopedDevice(self.device):
            with wp.ScopedCapture() as capture:
                self.step()
        self.graph = capture.graph
        return self.graph

    def run(self, frames):
        """Runs `frames` frames through the captured graph (captures on first use; CPU: direct)."""
        if self.device == "cpu":
            for _ in range(frames):
                self.step()
            return
        if self.graph is None:
            self.capture()
        with wp.ScopedDevice(self.device):
            for _ in range(frames):
                wp.capture_launch(self.graph)

    # -- readback (never inside step) ---------------------------------------------------

    def frame(self):
        return int(self.cnt.numpy()[CNT_FRAME])

    def node_count(self):
        return int((self.flags.numpy() & ALIVE).astype(bool).sum())

    def nodes(self):
        """Host copy of the live nodes: dict of numpy arrays plus their ids."""
        flags = self.flags.numpy()
        ids = np.nonzero(flags & ALIVE)[0]
        return dict(ids=ids, pos=self.pos.numpy()[ids], prev_pos=self.prev_pos.numpy()[ids],
                    parent=self.parent.numpy()[ids], tree=self.tree.numpy()[ids],
                    birth_frame=self.birth_frame.numpy()[ids], s_arc=self.s_arc.numpy()[ids],
                    q=self.q.numpy()[ids], flags=flags[ids])

    def candidates(self, include_electrode=False):
        alive = self.cand_alive.numpy().astype(bool)
        if not include_electrode:
            alive[self.c_max:] = False
        ids = np.nonzero(alive)[0]
        return dict(ids=ids, pos=self.cand_pos.numpy()[ids], phi=self.cand_phi.numpy()[ids],
                    parent=self.cand_parent.numpy()[ids],
                    pool=np.where(ids < self.c_max, ids // self.c_f, self.f_max))

    def trees(self):
        return dict(state=self.t_state.numpy(), root=self.t_root.numpy(), tip=self.t_tip.numpy(),
                    foot=self.t_foot.numpy(), nodes=self.t_nodes.numpy(), L=self.t_L.numpy(),
                    chord=self.t_chord.numpy(), birth=self.t_birth.numpy(),
                    root_dir=self.t_root_dir.numpy(), foot_dir=self.t_foot_dir.numpy(),
                    I=self.t_I.numpy(), ring_age=self.t_ring_age.numpy(),
                    regrow=self.t_regrow.numpy(), regrow_frame=self.t_regrow_frame.numpy(),
                    foot2=self.t_foot2.numpy().reshape(self.f_max, BRUSH_FEET))

    def cg_residual(self):
        """(|r| / |b|, |r|, |b|) after the last frame's CG sweeps (cg_stats must be on)."""
        cg = self.cg.numpy()
        rr, bb = float(cg[CG_RR]), float(cg[CG_BB])
        return (math.sqrt(rr / bb) if bb > 0 else 0.0, math.sqrt(rr), math.sqrt(bb))

    def potential(self, points):
        """Host-side phi at arbitrary points from the current node charges (numpy, lab only)."""
        P = self.P_host.numpy()
        nodes = self.nodes()
        charged = (nodes["flags"] & CHARGED) != 0
        x = np.asarray(points, dtype=np.float64)
        r = np.linalg.norm(x, axis=1)
        phi = (1.0 / P[P_R1] - 1.0 / r) / (1.0 / P[P_R1] - 1.0 / P[P_R2])
        nf = int(self.n_fingers.numpy()[0])
        fd = self.finger_dir.numpy()[:nf].astype(np.float64)
        fq = self.finger_q.numpy()[:nf].astype(np.float64)
        for n_f, q_f in zip(fd, fq):
            d = (P[P_R2] + P[P_HF]) * n_f
            dl = np.linalg.norm(d)
            image = d * (P[P_R1] ** 2 / dl ** 2)
            phi += q_f * (1.0 / np.linalg.norm(x - d, axis=1)
                          - (P[P_R1] / dl) / np.linalg.norm(x - image, axis=1))
        xj = nodes["pos"][charged].astype(np.float64)
        qj = nodes["q"][charged].astype(np.float64)
        images = int(self.IP_host.numpy()[IP_IMAGES])
        lj = np.linalg.norm(xj, axis=1)
        xj_img = xj * (P[P_R1] ** 2 / lj ** 2)[:, None]
        for start in range(0, len(x), 2048):
            xs = x[start:start + 2048]
            d = np.linalg.norm(xs[:, None, :] - xj[None, :, :], axis=2)
            g = 1.0 / np.sqrt(d * d + P[P_A] ** 2)
            if images:
                di = np.linalg.norm(xs[:, None, :] - xj_img[None, :, :], axis=2)
                g -= (P[P_R1] / lj)[None, :] / np.sqrt(di * di + P[P_A] ** 2)
            phi[start:start + 2048] += (qj[None, :] * g).sum(1)
        return phi
