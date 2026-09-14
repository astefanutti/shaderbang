# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Circuit, surface charge and filament lifecycle for the plasma globe.

Everything here operates on the persistent tree / node arrays owned by ``plasma.dbm`` and is
graph-capturable (fixed launch dims, no allocation, no host sync).

Current budget (plan 4.5)
-------------------------
A plasma globe is a dielectric-barrier discharge fed through a small series capacitance: the
total current is limited, each attached filament draws a share set by its termination admittance
through the glass, and a finger on the glass multiplies that admittance ~215x so the touched
filament takes most of the budget while the others dim (an established channel sustains at
I_SUSTAIN_RATIO of the strike current, so they survive at ~10 %).  The filament
COUNT is therefore emergent (a new tree is admitted only while its projected share exceeds the
strike current), not prescribed.

    Z_s      = 1 / (omega C_e)                       source series impedance (C_e calibrated once)
    C_term,k = C_d A_foot (1 + 214 touch_k)          termination capacitance of foot k
    g_k      = 1 / (R'_ch L_k + 1 / (omega C_term,k))
    I_tot    = min(I_supply, max(0, V - V_ch) / (Z_s + 1 / sum g_k))
    I_k      = I_tot g_k / sum g_k

Surface charge (plan 4.6, DECISION: analytic per-foot records instead of a 256x128 texture)
--------------------------------------------------------------------------------------------
Each attached foot deposits charge that saturates the local barrier within a half-cycle
(activator core of fixed width ~1.5 foot radii: the next half-cycle re-strikes the same spot)
and inhibits a ring around it whose radius starts at r_inh and expands laterally with
diffusivity D_s up to a cap (Boeuf et al. 2012; Burin et al. 2015 'expanding circular
structures on the glass').  With <= 64 feet the field is an
analytic sum of Gaussian records, which is exact for the growth weight, needs no PDE on the
sphere (no equirect pole singularities, no explicit-diffusion stability limit) and hands the
renderer the foot disc / ring transient for free.

Lifecycle (plan 4.4, hybrid model)
----------------------------------
ATTACHED trees re-route from a fork at U[0.35, 0.75] of their arc length when stretched by
advection (L > 1.5 chord), when the channel has lost its thermal stability, or on a Poisson
timer floor; starved trees (I_k below the sustain current for 3 frames) retract from the tip.
The kernels here only MARK nodes (``NODE_DECAYING`` / ``NODE_MAINCUT``) and issue per-tree
commands; the growth engine reclaims freed nodes and re-seeds its candidate pool from the fork.
"""

import math

import numpy as np
import warp as wp

from plasma.params import (
    PlasmaParams, finger_dir,
    R2I, C_D, A_FOOT, R_CH, TOUCH_GAIN, I_SUPPLY, V_CH, I_STRIKE, I_SUSTAIN_RATIO, FREQ0, VOLT0,
    TAU_ENV, D_SIGMA, TAU_FINGER, SIGMA_A, SIGMA_B, R_INH, FOOT_RADIUS, FINGER_HWHM, NODE_SPACING,
    NODE_ALIVE, NODE_DECAYING, NODE_FOOT, NODE_MAINCUT, NODE_FREE_PENDING,
    TREE_FREE, TREE_GROW, TREE_ATTACHED, TREE_RETRACT,
    CMD_NONE, CMD_REROUTE, CMD_RETRACT,
)

F_MAX = 32                 # tree slots (must match plasma.dbm)
SIG_MAX = 2 * F_MAX        # surface-charge records: one per tree slot + an orphan ring buffer
STARVE_FRAMES = 3          # latch debounce (frames below sustain before retracting)
REROUTE_MEAN_S = 2.0       # Poisson re-route timer floor (s)                       CHOSEN
STRETCH_RATIO = 1.5        # re-route when arc length > ratio * chord               CHOSEN
COLD_RATIO = 0.3           # re-route when mean channel overheat < ratio * steady    CHOSEN
COLD_SECONDS = 0.2
DECAY_SECONDS = 0.1        # decaying-pool fade (s)
RETRACT_SPEED = 2.0e-2 * 60.0   # m/s (2 cm per frame at 60 Hz)                    plan 4.4
SIGMA_SAT_REF = C_D * VOLT0     # sigma_sat at the reference voltage (C/m^2), ~105 uC/m^2
RING_MAX_RADIUS = 1.5e-2        # cap on the expanding inhibition ring radius (m)      CHOSEN
ACT_WIDTH = 1.5 * FOOT_RADIUS   # angular width (in m on the glass) of the activator core        CHOSEN

# Source series capacitance, calibrated once so that the reference operating point (5 kV, 26 kHz,
# 20 untouched 6 cm filaments) draws 1.0 mA (PPPL-4485 Table 1).
def _calibrate_ce(n_ref=20, l_ref=0.06, i_ref=1.0e-3):
    omega = 2.0 * math.pi * FREQ0
    c_term = C_D * A_FOOT
    g = 1.0 / (R_CH * l_ref + 1.0 / (omega * c_term))
    z_s = (VOLT0 - V_CH) / i_ref - 1.0 / (n_ref * g)
    return 1.0 / (omega * z_s)

C_E = _calibrate_ce()


@wp.func
def foot_admittance(omega: float, length: float, touch: float) -> float:
    c_term = C_D * A_FOOT * (1.0 + TOUCH_GAIN * touch)
    return 1.0 / (R_CH * length + 1.0 / (omega * c_term))


@wp.func
def strike_current(frequency: float) -> float:
    """Frequency dependence of the strike current -- CHOSEN so that the filament count peaks
    near 24 kHz and falls at high frequency (PPPL-4485 Fig. 5)."""
    x = (frequency - 24.0e3) / 20.0e3
    return I_STRIKE * (1.0 + x * x)


@wp.func
def touch_coverage(p: PlasmaParams, direction: wp.vec3) -> float:
    """Gaussian finger coverage of a point on the glass given by its unit direction."""
    w = FINGER_HWHM / R2I
    cov = float(0.0)
    for f in range(p.num_fingers):
        c = wp.clamp(wp.dot(direction, finger_dir(p, f)), -1.0, 1.0)
        theta = wp.acos(c)
        cov += wp.exp(-0.6931472 * (theta / w) * (theta / w))
    return cov


BRUSH_FEET = 4                 # secondary feet per tree (must match plasma.dbm)
BRUSH_REACH = 0.8              # rad: an unserved finger this close to a touched channel's main foot is
                               # served by a brush leader of that channel, not by a new strike  CHOSEN
FINGER_REROUTE_ANGLE = 0.6     # rad: an unserved finger without a host re-routes the attached
                               # filaments whose foot lies within this angle                    CHOSEN
BRUSH_DROP_SECONDS = 0.5       # a secondary foot uncovered by any finger for this long is dropped
OWN_COVERAGE = 0.15            # a foot with at least this contact coverage (within ~1.65 half widths,
                               # ~1 cm) is under the finger: the finger's capacitive coupling extends
                               # over the whole contact, and the channel cannot land closer than its
                               # own charged predecessor allows                               CHOSEN
owners_t = wp.types.vector(length=10, dtype=wp.int32)


@wp.func
def finger_cov(p: PlasmaParams, f: int, direction: wp.vec3) -> float:
    w = FINGER_HWHM / R2I
    theta = wp.acos(wp.clamp(wp.dot(direction, finger_dir(p, f)), -1.0, 1.0))
    return wp.exp(-0.6931472 * (theta / w) * (theta / w))


@wp.kernel
def k_fingers(params: wp.array(dtype=PlasmaParams),
              tree_state: wp.array(dtype=wp.int32),
              tree_foot_dir: wp.array(dtype=wp.vec3),
              tree_foot2_dir: wp.array(dtype=wp.vec3),
              tree_touch: wp.array(dtype=wp.float32),
              tree_gscale: wp.array(dtype=wp.float32),
              tree_reroute_req: wp.array(dtype=wp.int32),
              tree_brush_req: wp.array(dtype=wp.int32),
              tree_foot2_drop: wp.array(dtype=wp.int32),
              tree_foot2_cold: wp.array(dtype=wp.float32),
              finger_served: wp.array(dtype=wp.int32),
              tree_targets: wp.array(dtype=wp.vec3),
              circuit: wp.array(dtype=wp.float32)):
    """Single thread: who serves which finger (``finger_served[f]`` = 1 when owned).

    ``tree_targets[k * (BRUSH_FEET + 1) + m]`` is the direction of the finger served by foot m of
    tree k (0 = main foot, 1.. = brush feet; zero when none): the app slides that foot and the
    channel behind it towards the finger every frame, so a moving finger is followed
    continuously instead of being re-struck.

    * A finger is owned by the attached filament with the largest contact coverage over its
      feet (main or brush) if that coverage exceeds OWN_COVERAGE (~1 cm). Only the owner
      gets the touched termination (``tree_touch``): one bright filament per finger.
    * Another attached filament whose main foot lies within two half widths of an owned finger
      sits on a barrier the owner already saturates: its admittance is scaled down
      (``tree_gscale``) and it falls under the sustain current and retracts.
    * An unserved finger within BRUSH_REACH of a touched filament's main foot asks that filament
      for a brush leader (a hand: one channel splitting to the fingers); otherwise it re-routes
      the untouched filaments within FINGER_REROUTE_ANGLE and counts as unserved for the
      admission of a strike (circuit[7]).
    * A brush foot no finger covers for BRUSH_DROP_SECONDS is dropped."""
    p = params[0]
    dt = p.dt
    w = FINGER_HWHM / R2I
    for k in range(F_MAX):
        tree_touch[k] = 0.0
        tree_gscale[k] = 1.0
        tree_reroute_req[k] = 0
        tree_brush_req[k] = 0
        for m in range(BRUSH_FEET + 1):
            tree_targets[k * (BRUSH_FEET + 1) + m] = wp.vec3(0.0, 0.0, 0.0)
    owner = owners_t()
    for f in range(10):
        owner[f] = -1
    nf = wp.min(p.num_fingers, 10)
    # pass 1: owners
    for f in range(nf):
        best = int(-1)
        best_foot = int(0)
        best_cov = float(OWN_COVERAGE)
        for k in range(F_MAX):
            if tree_state[k] == TREE_ATTACHED:
                cov = finger_cov(p, f, tree_foot_dir[k])
                foot = int(0)
                for j in range(BRUSH_FEET):
                    d2 = tree_foot2_dir[k * BRUSH_FEET + j]
                    if wp.length_sq(d2) > 0.5:
                        c2 = finger_cov(p, f, d2)
                        if c2 > cov:
                            cov = c2
                            foot = j + 1
                if cov > best_cov:
                    best_cov = cov
                    best = k
                    best_foot = foot
        owner[f] = best
        finger_served[f] = wp.where(best >= 0, 1, 0)
        if best >= 0:
            tree_touch[best] = tree_touch[best] + best_cov
            tree_targets[best * (BRUSH_FEET + 1) + best_foot] = finger_dir(p, f)
    # pass 2: intruders, hosts, re-routes
    unserved = int(0)
    for f in range(nf):
        d = finger_dir(p, f)
        if owner[f] >= 0:
            for k in range(F_MAX):
                if tree_state[k] == TREE_ATTACHED and k != owner[f] and tree_touch[k] == 0.0:
                    if wp.acos(wp.clamp(wp.dot(tree_foot_dir[k], d), -1.0, 1.0)) < 2.0 * w:
                        tree_gscale[k] = 0.01          # below sustain: the intruder retracts
        else:
            host = int(-1)
            host_ang = float(BRUSH_REACH)
            for k in range(F_MAX):
                if tree_state[k] == TREE_ATTACHED and tree_touch[k] > 0.0:
                    ang = wp.acos(wp.clamp(wp.dot(tree_foot_dir[k], d), -1.0, 1.0))
                    if ang < host_ang:
                        host_ang = ang
                        host = k
            if host >= 0:
                tree_brush_req[host] = 1
            else:
                unserved += 1
                for k in range(F_MAX):
                    if tree_state[k] == TREE_ATTACHED and tree_touch[k] == 0.0:
                        if wp.acos(wp.clamp(wp.dot(tree_foot_dir[k], d), -1.0, 1.0)) < FINGER_REROUTE_ANGLE:
                            tree_reroute_req[k] = 1
    # brush feet without a finger
    for k in range(F_MAX):
        for j in range(BRUSH_FEET):
            fs = k * BRUSH_FEET + j
            tree_foot2_drop[fs] = 0
            d2 = tree_foot2_dir[fs]
            if tree_state[k] == TREE_ATTACHED and wp.length_sq(d2) > 0.5:
                cov = float(0.0)
                for f in range(nf):
                    cov = wp.max(cov, finger_cov(p, f, d2))
                if cov < OWN_COVERAGE:
                    tree_foot2_cold[fs] = tree_foot2_cold[fs] + dt
                else:
                    tree_foot2_cold[fs] = 0.0
                if tree_foot2_cold[fs] > BRUSH_DROP_SECONDS:
                    tree_foot2_drop[fs] = 1
                    tree_foot2_cold[fs] = 0.0
            else:
                tree_foot2_cold[fs] = 0.0
    circuit[7] = float(unserved)


@wp.kernel
def k_current_division(params: wp.array(dtype=PlasmaParams),
                       tree_state: wp.array(dtype=wp.int32),
                       tree_length: wp.array(dtype=wp.float32),
                       tree_touch: wp.array(dtype=wp.float32),
                       tree_gscale: wp.array(dtype=wp.float32),
                       tree_g: wp.array(dtype=wp.float32),
                       tree_current: wp.array(dtype=wp.float32),
                       circuit: wp.array(dtype=wp.float32)):
    """Single-thread kernel (F_MAX <= 32 trees): admittances, total current, per-tree shares.

    circuit[0] = I_tot, [1] = sum g, [2] = admit flag for a new tree, [3] = I_strike(f),
    [4] = I_sustain, [5] = number of attached trees, [6] = projected share of a new tree,
    [7] = number of *unserved* fingers (no attached foot within the contact half width): a strike
    towards such a finger is admitted on the touched admittance (its terminal capacitance is
    x(1 + TOUCH_GAIN)), independently of the nominal admission.
    """
    p = params[0]
    omega = 2.0 * 3.14159265 * p.frequency
    z_s = 1.0 / (omega * C_E)
    sum_g = float(0.0)
    n_att = int(0)
    n_grow = int(0)
    for k in range(F_MAX):
        if tree_state[k] == TREE_ATTACHED:
            g = foot_admittance(omega, tree_length[k], tree_touch[k]) * tree_gscale[k]
            tree_g[k] = g
            sum_g += g
            n_att += 1
        else:
            tree_g[k] = 0.0
            if tree_state[k] != TREE_FREE:
                n_grow += 1          # growing trees reserve a nominal share in the admission test
    drive = wp.max(p.voltage - V_CH, 0.0)
    i_tot = float(0.0)
    if sum_g > 0.0:
        i_tot = wp.min(I_SUPPLY, drive / (z_s + 1.0 / sum_g))
    for k in range(F_MAX):
        if tree_state[k] == TREE_ATTACHED:
            tree_current[k] = i_tot * tree_g[k] / sum_g
        else:
            tree_current[k] = 0.0
    # admission: projected share of one more nominal (untouched, 6 cm) filament, with the trees
    # still growing counted as nominal filaments too (they will draw current once attached)
    g_new = foot_admittance(omega, 0.06, 0.0)
    g_proj = sum_g + g_new * float(n_grow + 1)
    i_new_tot = wp.min(I_SUPPLY, drive / (z_s + 1.0 / g_proj))
    i_new = i_new_tot * g_new / g_proj
    i_str = strike_current(p.frequency)
    unserved = int(circuit[7])               # k_fingers: unserved fingers without a brush host
    admit_touch = float(0.0)
    if unserved > 0:
        g_touch = foot_admittance(omega, 0.06, 1.0)
        g_proj_t = sum_g + g_touch + g_new * float(n_grow)
        i_touch = wp.min(I_SUPPLY, drive / (z_s + 1.0 / g_proj_t)) * g_touch / g_proj_t
        admit_touch = wp.where(i_touch >= i_str, float(unserved), 0.0)
    circuit[0] = i_tot
    circuit[1] = sum_g
    circuit[2] = wp.where(i_new >= i_str, 1.0, 0.0)
    circuit[3] = i_str
    circuit[4] = I_SUSTAIN_RATIO * i_str
    circuit[5] = float(n_att)
    circuit[6] = i_new
    circuit[7] = admit_touch


@wp.kernel
def k_lifecycle(params: wp.array(dtype=PlasmaParams),
                circuit: wp.array(dtype=wp.float32),
                tree_state: wp.array(dtype=wp.int32),
                tree_current: wp.array(dtype=wp.float32),
                tree_length: wp.array(dtype=wp.float32),
                tree_chord: wp.array(dtype=wp.float32),
                tree_hot: wp.array(dtype=wp.float32),
                tree_hot_ref: wp.array(dtype=wp.float32),
                tree_age: wp.array(dtype=wp.float32),
                tree_cold_time: wp.array(dtype=wp.float32),
                tree_starve: wp.array(dtype=wp.int32),
                tree_cmd: wp.array(dtype=wp.int32),
                tree_fork_u: wp.array(dtype=wp.float32)):
    """Per-tree latch and re-route triggers; writes a command the growth engine executes."""
    k = wp.tid()
    p = params[0]
    tree_cmd[k] = CMD_NONE
    state = tree_state[k]
    if state == TREE_FREE:
        return
    tree_age[k] = tree_age[k] + p.dt
    if state != TREE_ATTACHED:
        tree_starve[k] = 0
        return
    # starvation latch: below sustain for STARVE_FRAMES consecutive frames -> retract
    if tree_current[k] < circuit[4]:
        tree_starve[k] = tree_starve[k] + 1
    else:
        tree_starve[k] = 0
    if tree_starve[k] >= STARVE_FRAMES:
        tree_cmd[k] = CMD_RETRACT
        return
    if p.hybrid == 0:
        return
    rng = wp.rand_init(p.seed, 7919 * k + p.frame)
    reroute = int(0)
    # (a) stretched by advection
    if tree_length[k] > STRETCH_RATIO * tree_chord[k]:
        reroute = 1
    # (b) lost thermal stability for COLD_SECONDS
    if tree_hot_ref[k] > 0.0 and tree_hot[k] < COLD_RATIO * tree_hot_ref[k]:
        tree_cold_time[k] = tree_cold_time[k] + p.dt
    else:
        tree_cold_time[k] = 0.0
    if tree_cold_time[k] > COLD_SECONDS:
        reroute = 1
    # (c) Poisson timer floor
    if wp.randf(rng) < p.dt / REROUTE_MEAN_S:
        reroute = 1
    if reroute == 1:
        tree_cmd[k] = CMD_REROUTE
        tree_fork_u[k] = 0.35 + 0.4 * wp.randf(rng)
        tree_cold_time[k] = 0.0


# ---- surface-charge records -----------------------------------------------------------------

@wp.kernel
def k_sigma_records(params: wp.array(dtype=PlasmaParams),
                    tree_state: wp.array(dtype=wp.int32),
                    tree_foot_dir: wp.array(dtype=wp.vec3),
                    tree_current: wp.array(dtype=wp.float32),
                    tree_touch: wp.array(dtype=wp.float32),
                    sig_dir: wp.array(dtype=wp.vec3),
                    sig_amp: wp.array(dtype=wp.float32),
                    sig_radius: wp.array(dtype=wp.float32),
                    sig_age: wp.array(dtype=wp.float32),
                    sig_alive: wp.array(dtype=wp.int32),
                    sig_cursor: wp.array(dtype=wp.int32),
                    sig_I: wp.array(dtype=wp.float32)):
    """Advance the per-tree records (index k < F_MAX) and the orphan records (k >= F_MAX).
    ``sig_I`` carries the foot's current (A) for the renderer (foot size and brightness).

    A tree record follows its foot while attached; when the tree detaches, the record is
    copied into the orphan ring buffer so the deposited charge keeps decaying in place.
    """
    k = wp.tid()
    p = params[0]
    dt = p.dt
    sigma_sat = C_D * wp.max(p.voltage, 1.0)
    if k < F_MAX:
        attached = tree_state[k] == TREE_ATTACHED
        if attached:
            if sig_alive[k] == 0:
                # new attachment: start a fresh record at the foot
                sig_alive[k] = 1
                sig_amp[k] = 0.0
                sig_age[k] = 0.0
            sig_dir[k] = tree_foot_dir[k]
            g0 = 1.0 / (2.0 * 3.14159265 * FOOT_RADIUS * FOOT_RADIUS)   # peak of the unit-integral foot Gaussian
            r = tree_current[k] * g0 / sigma_sat
            d = 1.0 / TAU_ENV + tree_touch[k] / TAU_FINGER
            amp_eq = r / (r + d)
            sig_amp[k] = amp_eq + (sig_amp[k] - amp_eq) * wp.exp(-(r + d) * dt)
            sig_age[k] = sig_age[k] + dt
            sig_radius[k] = wp.min(wp.sqrt(R_INH * R_INH + 2.0 * D_SIGMA * sig_age[k]), RING_MAX_RADIUS)
            sig_I[k] = tree_current[k]
        elif sig_alive[k] == 1:
            # detached this frame: hand the record to the orphan pool
            slot = F_MAX + (wp.atomic_add(sig_cursor, 0, 1) % F_MAX)
            sig_dir[slot] = sig_dir[k]
            sig_amp[slot] = sig_amp[k]
            sig_radius[slot] = sig_radius[k]
            sig_age[slot] = sig_age[k]
            sig_I[slot] = 0.0                  # no current: the orphan keeps its charge, not its glow
            sig_alive[slot] = 1
            sig_alive[k] = 0
            sig_amp[k] = 0.0
            sig_I[k] = 0.0
    else:
        if sig_alive[k] == 1:
            d = 1.0 / TAU_ENV + touch_coverage(p, sig_dir[k]) / TAU_FINGER
            sig_amp[k] = sig_amp[k] * wp.exp(-d * dt)
            sig_age[k] = sig_age[k] + dt
            sig_radius[k] = wp.min(wp.sqrt(R_INH * R_INH + 2.0 * D_SIGMA * sig_age[k]), RING_MAX_RADIUS)
            if sig_amp[k] < 1.0e-3:
                sig_alive[k] = 0
                sig_amp[k] = 0.0


@wp.func
def sigma_weight(x: wp.vec3,
                 quincunx: int,
                 sig_dir: wp.array(dtype=wp.vec3),
                 sig_amp: wp.array(dtype=wp.float32),
                 sig_radius: wp.array(dtype=wp.float32),
                 sig_alive: wp.array(dtype=wp.int32)) -> float:
    """Growth-weight factor s(x) = 1 + A * activator - B * charged disc from the surface records.

    The barrier under a foot is charged over a disc that spreads by surface diffusion
    (``sig_radius``, 8 -> 15 mm): the gap field is cancelled there, so no other filament lands on
    it (plateau inside the radius, Gaussian edge of half the initial radius). Only meaningful for
    candidates within ~2h of the glass; callers should return 1 elsewhere.
    """
    r = wp.length(x)
    if r < R2I - 2.0 * NODE_SPACING:
        return 1.0
    n = x / r
    act = float(0.0)
    inh = float(0.0)
    s_act = ACT_WIDTH / R2I
    ring_w = 0.5 * R_INH / R2I
    for k in range(SIG_MAX):
        if sig_alive[k] == 1:
            c = wp.clamp(wp.dot(n, sig_dir[k]), -1.0, 1.0)
            theta = wp.acos(c)
            act += sig_amp[k] * wp.exp(-0.5 * (theta / s_act) * (theta / s_act))
            dr = wp.max(theta - sig_radius[k] / R2I, 0.0)
            inh += sig_amp[k] * wp.exp(-0.5 * (dr / ring_w) * (dr / ring_w))
    if quincunx == 1:
        act = -act
    return wp.max(1.0 + SIGMA_A * act - SIGMA_B * inh, 0.0)


@wp.kernel
def k_sigma_weight_probe(points: wp.array(dtype=wp.vec3), quincunx: int,
                         sig_dir: wp.array(dtype=wp.vec3), sig_amp: wp.array(dtype=wp.float32),
                         sig_radius: wp.array(dtype=wp.float32), sig_alive: wp.array(dtype=wp.int32),
                         out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = sigma_weight(points[i], quincunx, sig_dir, sig_amp, sig_radius, sig_alive)


# ---- re-route / retract marking on the persistent tree ----------------------------------------

@wp.kernel
def k_mark_main_cut(tree_cmd: wp.array(dtype=wp.int32),
                    tree_fork_u: wp.array(dtype=wp.float32),
                    tree_foot: wp.array(dtype=wp.int32),
                    tree_length: wp.array(dtype=wp.float32),
                    params: wp.array(dtype=PlasmaParams),
                    node_parent: wp.array(dtype=wp.int32),
                    node_s_arc: wp.array(dtype=wp.float32),
                    node_flags: wp.array(dtype=wp.int32),
                    tree_fork_node: wp.array(dtype=wp.int32)):
    """Walk the main channel foot -> root and flag the part beyond the fork (re-route) or beyond
    the retraction front (retract) with NODE_MAINCUT; records the fork node for the engine."""
    k = wp.tid()
    cmd = tree_cmd[k]
    tree_fork_node[k] = -1
    if cmd == CMD_NONE:
        return
    cut = float(0.0)
    if cmd == CMD_REROUTE:
        cut = tree_fork_u[k] * tree_length[k]
    else:
        cut = wp.max(tree_length[k] - RETRACT_SPEED * params[0].dt, 0.0)
    n = tree_foot[k]
    fork = int(-1)
    steps = int(0)
    while n >= 0 and steps < 4096:
        if node_s_arc[n] > cut:
            node_flags[n] = node_flags[n] | NODE_MAINCUT
        else:
            if fork < 0:
                fork = n
        n = node_parent[n]
        steps += 1
    tree_fork_node[k] = fork
    if cmd == CMD_RETRACT:
        tree_length[k] = cut


@wp.kernel
def k_mark_descendants(node_flags: wp.array(dtype=wp.int32),
                       node_parent: wp.array(dtype=wp.int32),
                       node_tree: wp.array(dtype=wp.int32),
                       tree_cmd: wp.array(dtype=wp.int32),
                       node_alpha: wp.array(dtype=wp.float32)):
    """Every live node whose ancestor chain crosses a NODE_MAINCUT node joins the decaying pool."""
    i = wp.tid()
    f = node_flags[i]
    if (f & NODE_ALIVE) == 0 or (f & NODE_DECAYING) != 0:
        return
    if tree_cmd[node_tree[i]] == CMD_NONE:
        return
    n = i
    steps = int(0)
    hit = int(0)
    while n >= 0 and steps < 4096:
        if (node_flags[n] & NODE_MAINCUT) != 0:
            hit = 1
            break
        n = node_parent[n]
        steps += 1
    if hit == 1:
        node_flags[i] = (f | NODE_DECAYING) & ~(NODE_FOOT)
        node_alpha[i] = 1.0


@wp.kernel
def k_clear_main_cut(node_flags: wp.array(dtype=wp.int32)):
    i = wp.tid()
    node_flags[i] = node_flags[i] & ~NODE_MAINCUT


@wp.kernel
def k_decay_pool(params: wp.array(dtype=PlasmaParams),
                 node_flags: wp.array(dtype=wp.int32),
                 node_alpha: wp.array(dtype=wp.float32)):
    """Fade decaying nodes; fully faded nodes are handed back to the engine's free list."""
    i = wp.tid()
    f = node_flags[i]
    if (f & NODE_DECAYING) == 0:
        return
    a = node_alpha[i] - params[0].dt / DECAY_SECONDS
    if a <= 0.0:
        node_alpha[i] = 0.0
        node_flags[i] = NODE_FREE_PENDING
    else:
        node_alpha[i] = a


@wp.kernel
def k_apply_tree_commands(tree_cmd: wp.array(dtype=wp.int32),
                          tree_fork_node: wp.array(dtype=wp.int32),
                          tree_length: wp.array(dtype=wp.float32),
                          tree_state: wp.array(dtype=wp.int32),
                          tree_foot: wp.array(dtype=wp.int32),
                          tree_starve: wp.array(dtype=wp.int32)):
    """State transitions after the marking kernels ran."""
    k = wp.tid()
    cmd = tree_cmd[k]
    if cmd == CMD_REROUTE:
        tree_state[k] = TREE_GROW          # regrow from the fork node (engine re-seeds its pool)
        tree_foot[k] = -1
        tree_starve[k] = 0
    elif cmd == CMD_RETRACT:
        if tree_length[k] <= 0.0:
            tree_state[k] = TREE_FREE
            tree_foot[k] = -1
        else:
            tree_state[k] = TREE_RETRACT
            tree_cmd[k] = CMD_RETRACT      # keep retracting next frame


class CircuitState:
    """Device arrays for the circuit / sigma / lifecycle kernels (F_MAX trees, SIG_MAX records)."""

    def __init__(self, device="cuda:0"):
        z = lambda dtype, n=F_MAX: wp.zeros(n, dtype=dtype, device=device)
        self.device = device
        self.tree_touch = z(wp.float32)
        self.tree_g = z(wp.float32)
        self.tree_current = z(wp.float32)
        self.tree_hot = z(wp.float32)
        self.tree_hot_ref = z(wp.float32)
        self.tree_age = z(wp.float32)
        self.tree_cold_time = z(wp.float32)
        self.tree_starve = z(wp.int32)
        self.tree_cmd = z(wp.int32)
        self.tree_fork_u = z(wp.float32)
        self.tree_fork_node = z(wp.int32)
        self.circuit = wp.zeros(8, dtype=wp.float32, device=device)
        self.sig_dir = z(wp.vec3, SIG_MAX)
        self.sig_amp = z(wp.float32, SIG_MAX)
        self.sig_radius = z(wp.float32, SIG_MAX)
        self.sig_age = z(wp.float32, SIG_MAX)
        self.sig_alive = z(wp.int32, SIG_MAX)
        self.sig_I = z(wp.float32, SIG_MAX)
        self.tree_gscale = wp.ones(F_MAX, dtype=wp.float32, device=device)
        self.tree_reroute_req = z(wp.int32)
        self.tree_brush_req = z(wp.int32)
        self.tree_foot2_drop = z(wp.int32, F_MAX * BRUSH_FEET)
        self.tree_foot2_cold = z(wp.float32, F_MAX * BRUSH_FEET)
        self.no_foot2_dir = z(wp.vec3, F_MAX * BRUSH_FEET)
        self.finger_served = z(wp.int32, 10)
        self.tree_targets = z(wp.vec3, F_MAX * (BRUSH_FEET + 1))
        self.sig_cursor = wp.zeros(1, dtype=wp.int32, device=device)

    def zero(self):
        for a in (self.tree_touch, self.tree_g, self.tree_current, self.tree_hot, self.tree_hot_ref,
                  self.tree_age, self.tree_cold_time, self.tree_starve, self.tree_cmd, self.tree_fork_u,
                  self.tree_fork_node, self.circuit, self.sig_dir, self.sig_amp, self.sig_radius, self.sig_I,
                  self.tree_reroute_req, self.tree_brush_req, self.tree_foot2_drop, self.tree_foot2_cold,
                  self.tree_targets,
                  self.sig_age, self.sig_alive, self.sig_cursor):
            a.zero_()


def launch_circuit(params, cs, tree_state, tree_foot_dir, tree_length, tree_chord, device, tree_foot2_dir=None):
    """The per-frame circuit slice: fingers -> current division -> sigma records -> lifecycle."""
    wp.launch(k_fingers, dim=1,
              inputs=[params, tree_state, tree_foot_dir,
                      tree_foot2_dir if tree_foot2_dir is not None else cs.no_foot2_dir,
                      cs.tree_touch, cs.tree_gscale, cs.tree_reroute_req, cs.tree_brush_req,
                      cs.tree_foot2_drop, cs.tree_foot2_cold, cs.finger_served, cs.tree_targets, cs.circuit],
              device=device)
    wp.launch(k_current_division, dim=1,
              inputs=[params, tree_state, tree_length, cs.tree_touch, cs.tree_gscale, cs.tree_g, cs.tree_current,
                      cs.circuit],
              device=device)
    wp.launch(k_sigma_records, dim=SIG_MAX,
              inputs=[params, tree_state, tree_foot_dir, cs.tree_current, cs.tree_touch,
                      cs.sig_dir, cs.sig_amp, cs.sig_radius, cs.sig_age, cs.sig_alive, cs.sig_cursor, cs.sig_I],
              device=device)
    wp.launch(k_lifecycle, dim=F_MAX,
              inputs=[params, cs.circuit, tree_state, cs.tree_current, tree_length, tree_chord,
                      cs.tree_hot, cs.tree_hot_ref, cs.tree_age, cs.tree_cold_time, cs.tree_starve,
                      cs.tree_cmd, cs.tree_fork_u],
              device=device)


def launch_tree_commands(params, cs, tree_state, tree_foot, tree_length,
                         node_parent, node_s_arc, node_flags, node_tree, node_alpha, n_max, device):
    """Execute the lifecycle commands on the node arrays (marking only; the engine reclaims)."""
    wp.launch(k_mark_main_cut, dim=F_MAX,
              inputs=[cs.tree_cmd, cs.tree_fork_u, tree_foot, tree_length, params,
                      node_parent, node_s_arc, node_flags, cs.tree_fork_node], device=device)
    wp.launch(k_mark_descendants, dim=n_max,
              inputs=[node_flags, node_parent, node_tree, cs.tree_cmd, node_alpha], device=device)
    wp.launch(k_clear_main_cut, dim=n_max, inputs=[node_flags], device=device)
    wp.launch(k_apply_tree_commands, dim=F_MAX,
              inputs=[cs.tree_cmd, cs.tree_fork_node, tree_length, tree_state, tree_foot, cs.tree_starve],
              device=device)
    wp.launch(k_decay_pool, dim=n_max, inputs=[params, node_flags, node_alpha], device=device)


# ---- self-test -----------------------------------------------------------------------------------

def _selftest(device="cuda:0"):
    import sys
    sys.path.insert(0, __file__.rsplit("/", 2)[0])
    from plasma.params import ParamsBuffer, TREE_ATTACHED as ATT

    pb = ParamsBuffer(device)
    cs = CircuitState(device)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print(f"C_e calibrated = {C_E*1e12:.2f} pF; C_d = {C_D*1e9:.1f} nF/m^2; sigma_sat(5 kV) = {SIGMA_SAT_REF*1e6:.0f} uC/m^2")

    # 1) calibration: 20 attached 6 cm untouched filaments at 5 kV / 26 kHz draw 1.0 mA
    n = 20
    state = np.zeros(F_MAX, np.int32); state[:n] = ATT
    length = np.full(F_MAX, 0.06, np.float32); chord = np.full(F_MAX, 0.06, np.float32)
    dirs = np.zeros((F_MAX, 3), np.float32); dirs[:, 2] = 1.0
    for k in range(n):
        th = 2 * np.pi * k / n; dirs[k] = (np.cos(th), 0.2, np.sin(th)); dirs[k] /= np.linalg.norm(dirs[k])
    tree_state = wp.array(state, dtype=wp.int32, device=device)
    tree_len = wp.array(length, dtype=wp.float32, device=device)
    tree_chord = wp.array(chord, dtype=wp.float32, device=device)
    tree_dir = wp.array(dirs, dtype=wp.vec3, device=device)
    pb.upload()
    launch_circuit(pb.device_array, cs, tree_state, tree_dir, tree_len, tree_chord, device)
    c = cs.circuit.numpy(); I = cs.tree_current.numpy()
    check("I_tot at calibration point ~ 1.0 mA", abs(c[0] - 1e-3) < 2e-5, f"I_tot={c[0]*1e3:.3f} mA, I_k={I[0]*1e6:.1f} uA")

    # 2) emergent N(V): admit trees one by one while the circuit allows
    print("  N(V) by admission (26 kHz):")
    curve = {}
    for V in (1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000):
        pb.host[0]["voltage"] = V; pb.upload()
        st = np.zeros(F_MAX, np.int32)
        for k in range(F_MAX):
            tree_state.assign(st)
            launch_circuit(pb.device_array, cs, tree_state, tree_dir, tree_len, tree_chord, device)
            if cs.circuit.numpy()[2] < 0.5:
                break
            st[k] = ATT
        curve[V] = int(st.sum() // ATT)
        print(f"    V={V:5d} V -> N={curve[V]:2d}  (I_tot={cs.circuit.numpy()[0]*1e3:.2f} mA)")
    check("no filaments below V_ch", curve[1500] == 0 and curve[2000] == 0)
    check("N rises with V and saturates <= 32", curve[3000] < curve[5000] <= curve[8000] <= 32, str(curve))

    # 3) touch: one finger on foot 0 -> that tree takes >= 60 % of I_tot, the others dim but survive
    pb.host[0]["voltage"] = 5000; pb.set_fingers([]); pb.upload()
    tree_state.assign(state)
    launch_circuit(pb.device_array, cs, tree_state, tree_dir, tree_len, tree_chord, device)
    i_k_untouched = float(cs.tree_current.numpy()[1])
    pb.set_fingers([dirs[0]]); pb.upload()
    tree_state.assign(state)
    launch_circuit(pb.device_array, cs, tree_state, tree_dir, tree_len, tree_chord, device)
    c = cs.circuit.numpy(); I = cs.tree_current.numpy(); t = cs.tree_touch.numpy()
    check("touched tree takes >= 30 % of I_tot (footage: white channel, the others stay lit)", I[0] / c[0] >= 0.3, f"share={I[0]/c[0]:.2f}, touch={t[0]:.2f}, I_tot={c[0]*1e3:.2f} mA")
    check("others dim below their untouched share but stay above I_sustain", I[1] < i_k_untouched and I[1] >= c[4],
          f"I_1={I[1]*1e6:.1f} uA (untouched {i_k_untouched*1e6:.1f} uA), I_sus={c[4]*1e6:.1f} uA")
    pb.set_fingers([]); pb.upload()

    # 4) sigma records: attached feet saturate, spread and inhibit an annulus
    cs.zero()
    for _ in range(60):
        launch_circuit(pb.device_array, cs, tree_state, tree_dir, tree_len, tree_chord, device)
    amp = cs.sig_amp.numpy(); rad = cs.sig_radius.numpy()
    check("foot record saturates near sigma_sat within 1 s", 0.9 < amp[0] <= 1.0, f"amp={amp[0]:.3f}, ring radius={rad[0]*1e3:.1f} mm")
    d0 = dirs[0]
    up = np.cross(d0, [0, 1, 0]); up /= np.linalg.norm(up)
    def on_glass(theta):
        v = np.cos(theta) * d0 + np.sin(theta) * up
        return (R2I - 0.5e-3) * v / np.linalg.norm(v)
    sdirs = cs.sig_dir.numpy()[cs.sig_alive.numpy() == 1]
    cands = [on_glass(t) for t in np.linspace(1.0, np.pi, 12)]
    far = max(cands, key=lambda v: float(np.min(np.arccos(np.clip(sdirs @ (v / np.linalg.norm(v)), -1.0, 1.0)))))
    probes = np.array([on_glass(0.0), on_glass(rad[0] / R2I), far, 0.5 * R2I * d0], np.float32)
    out = wp.zeros(len(probes), dtype=wp.float32, device=device)
    wp.launch(k_sigma_weight_probe, dim=len(probes),
              inputs=[wp.array(probes, dtype=wp.vec3, device=device), 0, cs.sig_dir, cs.sig_amp, cs.sig_radius, cs.sig_alive, out],
              device=device)
    w = out.numpy()
    check("charged footprint inhibits (s < 0.6 at the foot and at its radius), far = 1, interior = 1",
          w[0] < 0.6 and w[1] < 0.6 and 0.95 < w[2] <= 1.05 and abs(w[3] - 1.0) < 1e-6, f"s={np.round(w, 3)}")

    # 5) lifecycle marking on a synthetic tree: chain of 100 nodes + a 20-node side branch at node 60
    N = 8192
    parent = -np.ones(N, np.int32); s_arc = np.zeros(N, np.float32); flags = np.zeros(N, np.int32); tree = np.zeros(N, np.int32)
    for i in range(100):
        parent[i] = i - 1; s_arc[i] = i * NODE_SPACING; flags[i] = NODE_ALIVE
    for j in range(20):
        i = 100 + j; parent[i] = 60 if j == 0 else i - 1; s_arc[i] = (61 + j) * NODE_SPACING; flags[i] = NODE_ALIVE
    flags[99] |= NODE_FOOT
    node_parent = wp.array(parent, dtype=wp.int32, device=device)
    node_s = wp.array(s_arc, dtype=wp.float32, device=device)
    node_flags = wp.array(flags, dtype=wp.int32, device=device)
    node_tree = wp.array(tree, dtype=wp.int32, device=device)
    node_alpha = wp.zeros(N, dtype=wp.float32, device=device)
    st = np.zeros(F_MAX, np.int32); st[0] = ATT
    tree_state.assign(st)
    tree_foot = wp.array(np.where(np.arange(F_MAX) == 0, 99, -1).astype(np.int32), dtype=wp.int32, device=device)
    tree_len.assign(np.where(np.arange(F_MAX) == 0, 99 * NODE_SPACING, 0.06).astype(np.float32))
    cs.tree_cmd.assign(np.where(np.arange(F_MAX) == 0, CMD_REROUTE, CMD_NONE).astype(np.int32))
    cs.tree_fork_u.fill_(0.5)
    launch_tree_commands(pb.device_array, cs, tree_state, tree_foot, tree_len, node_parent, node_s, node_flags, node_tree, node_alpha, N, device)
    f = node_flags.numpy(); fork = cs.tree_fork_node.numpy()[0]
    decaying = int(((f & NODE_DECAYING) != 0).sum())
    # nodes with s_arc > 0.5 * L on the main chain: indices 50..99 (50 nodes) + the whole side branch (20)
    check("re-route marks main channel beyond the fork + the side branch", decaying == 70 and 48 <= fork <= 50,
          f"decaying={decaying}, fork node={fork}, tree_state={tree_state.numpy()[0]} (GROW={TREE_GROW})")
    # decay to free
    for _ in range(int(DECAY_SECONDS * 60) + 2):
        wp.launch(k_decay_pool, dim=N, inputs=[pb.device_array, node_flags, node_alpha], device=device)
    freed = int((node_flags.numpy() == NODE_FREE_PENDING).sum())
    check("decayed nodes become FREE_PENDING", freed == 70, f"freed={freed}")

    print("circuit self-test:", "ALL PASS" if ok else "FAILURES")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    wp.init()
    raise SystemExit(0 if _selftest(args.device) else 1)
