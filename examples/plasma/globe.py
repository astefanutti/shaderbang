# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
The simulation Input of the plasma globe: one fixed-shape CUDA graph per frame that runs

    params (pinned -> device) -> gas convection step -> circuit / surface charge -> engine hooks
    (fingers, admission, currents, sigma weights, gas temperature at the candidates)
    -> DBM growth + lifecycle (plasma.dbm: prev_pos snapshot, S_MAX steps, conductor solve,
    retract / re-route / decay) -> node advection by the gas -> stretch / colour / heat sources
    -> publish (segments, CSR, lights) -> volume pack -> counters

plus a tiny reset graph. Nothing is allocated or read back per frame; every host input travels
in ``PlasmaParams`` (plasma.params), the engine's pinned knob arrays and the gas control slots.

Module ownership: growth + lifecycle = plasma.dbm, gas = plasma.gas, circuit / sigma =
plasma.circuit, render buffers = plasma.publish. This module owns the couplings: node advection
by the gas (the hybrid model's coherence, with the engine's ``t_stretch`` re-route contract),
the reduced-field inputs (gas temperature at the candidates, sigma weights), heat deposition
along the channels (P_gas top-down), the ionisation-fraction colour law and the frame order.
"""

import math
import os

import numpy as np
import warp as wp

from shaderbang.input import Input

from plasma import circuit, dbm, gas, publish
from plasma.params import (
    ParamsBuffer, PlasmaParams, finger_dir, R1, R2I, CHARGE_RADIUS, NODE_SPACING, FINGER_HWHM,
    VOLT0, FREQ0, Q_FINGER0, V_CH, NODE_ALIVE, NODE_ROOT, NODE_FOOT, NODE_DECAYING,
)

P_GAS_NOMINAL = 0.7           # W total at 1 mA (CHOSEN top-down; gas lab: 0.6-1.3 W/m -> 1.2-1.9 cm/s fronts;
                              # with the drift weighting this gives the ~1 cm/s filament rise of PPPL-4485)
I_TOT_NOMINAL = 1.0e-3
VIDEO_NEUTRAL = (1.00, 0.30, 0.62)   # 'video' preset: pink (feet, electrode)               CHOSEN (footage)
VIDEO_ION = (0.22, 0.17, 0.75)       # 'video' preset: violet-blue shaft                    CHOSEN (footage)
X_ION_FOOT = 0.45             # ionised-line fraction at the glass foot: pink (the footage shows magenta
                              # feet and branches under a hand, not the pure Ne I orange)
X_ION_ROOT = 0.35            # ionised-line fraction at the bulb (pink-white root flares)
X_ION_ROOT_LEN = 5.0e-3      # m: blend length of the root colour
X_ION_FOOT_LEN = 8.0e-3       # blend length before the glass (m)
I_STRIKE_REF = 3.0e-5
CORE_RADIUS = 0.45e-3         # rendered core radius at 50 uA; r ~ I^0.4, clamped at 1.5 mm (the cross-section
                              # grows with the current: a touched filament at ~20x the current is ~3x
                              # thicker, as in the reference footage)
ETA_GLOBE = 120.0             # growth exponent for the globe (CHOSEN 2026-09-14). The potential of a tip's
                              # straight candidate exceeds a 35-degree one by only ~1 %, so eta <= 10
                              # leaves the direction to the Gumbel noise (channels 1.6-1.9x longer than
                              # their chord, lightning look); a streamer resolves such differences because
                              # ionisation is exponential in the field. Lab: eta 120 -> tortuosity 1.09,
                              # single 44-node channels; 300 -> perfectly straight; 3 -> D 1.68 lightning
SERVED_SCREEN = 0.85          # fraction of a served finger's charge cancelled by the barrier charge its
                              # filament deposits (CHOSEN: leaders aimed at a second finger otherwise
                              # fall back next to the first one)
MAX_GROWING = 4               # filaments strike one after another: admit a new tree only while at most
                              # this many are still growing (CHOSEN; a real filament forms in ~5 us)
E_BD0_GLOBE = float(os.environ.get("PLASMA_E_BD0", 110.0e3))
# Root strike field (V/m) for the globe (CHOSEN): the footage shows ~35 filaments at the nominal
# drive; with R1 = 1.1 cm the lab default (190 kV/m) gives 13 at 5 kV, 130 kV/m 21 after 30 s,
# 100 kV/m 28. The count is limited by the screening of the electrode by the attached channels,
# which the model overstates because a channel is held at the electrode potential: a real
# filament drops ~2 kV along its length (V_CH), so its far half screens much less. Modelling that
# resistive channel (phi_ch(s) on the conductor right-hand side) is the physical fix; the lower
# strike field (V_th ~ 1.0 kV) stands in for it until then.
E_CH_GLOBE = V_CH / 0.06      # V/m along a channel: the sustaining drop V_CH over the nominal 6 cm (plan 4.1);
                              # the conductor potential rises by E_ch s / V with arc length (resistive
                              # channel), which is what keeps the attached filaments from screening the
                              # electrode and the glass as if they were at the electrode potential
E_PROP_FACTOR = 0.05          # propagation / strike threshold ratio (CHOSEN): a started channel keeps
                              # propagating through the field screened by the attached channels (the
                              # plan's 0.25 starved every tree seeded next to attached ones)
FOOT_PIN_LEN = 4.0e-3         # m: the channel does not ride the gas within this distance of the glass
PLUME_MIN_RISE = 4.0e-3       # m/s: a channel never sinks; its own plume gives it at least this rise (before the drift
                              # weight): the recording's feet walk up the glass at up to ~6 mm/s (median 0.6)
FOOT_CREEP = 1.0              # drift kept inside FOOT_PIN_LEN (1 = the foot walks with the channel; 0 = pinned)
ROOT_DRIFT_GAIN = 1.8         # roots ride the plume over the bulb faster than the drift weight alone gives (recording ~2 mm/s)
ROOT_SAMPLE_OFFSET = 2.0      # x h: roots ride the flow sampled this far above the electrode (the
                              # channel's attachment follows its hot column out of the no-slip layer)
FOLLOW_LEN = 0.02             # m: the last 2 cm of a channel under a finger follow the finger (the foot
                              # re-strikes each half cycle at the field maximum, which moves with it)
FOLLOW_TAU = 0.05             # s: time constant of the foot's convergence onto the finger
FOLLOW_ANGLE = 0.6            # rad: a near-glass node follows the nearest served finger within this angle
DRIFT_EXPONENT = 1.5          # channel drift with the gas is weighted by (r / R2)^DRIFT_EXPONENT: the
                              # thermal memory (a few % lower breakdown field in the hot channel) can
                              # only hold the re-struck path off the field lines where the ambient
                              # field is weak; near the electrode (E ~ 1/r^2, 25x the glass value) the
                              # field dictates the path and the channel barely moves (CHOSEN; without
                              # it every channel is sucked into the plume core above the electrode)
ADVECT_RADIUS = 8.0e-3        # m: node velocity = mean gas velocity over a 7-point stencil of this
                              # radius (CHOSEN): the re-strike follows the hot CHANNEL, whose centroid
                              # moves at ~1/4 of the plume-core speed (gas lab: centroid 0.8-1.4 cm/s
                              # vs core 5.5-8 cm/s; PPPL measured 1 cm/s filament drift)
F_MAX = circuit.F_MAX
ATTACHED = dbm.ATTACHED


class _SoA:
    """Attribute bag presenting the engine + globe arrays under the names plasma.publish expects."""


# ---- coupling kernels ------------------------------------------------------------------------------

@wp.kernel
def k_gas_control(params: wp.array(dtype=PlasmaParams), ctrl: wp.array(dtype=wp.float32)):
    p = params[0]
    ctrl[gas.CTRL_TIME_SCALE] = wp.where(p.running == 1, 1.0, 0.0)
    ctrl[gas.CTRL_G_SIGN] = p.g_sign
    ctrl[gas.CTRL_ICE] = float(p.ice_cap)


@wp.kernel
def k_engine_hooks(params: wp.array(dtype=PlasmaParams),
                   circuit_out: wp.array(dtype=wp.float32),
                   tree_state: wp.array(dtype=wp.int32),
                   tree_current: wp.array(dtype=wp.float32),
                   finger_dir_out: wp.array(dtype=wp.vec3),
                   finger_q_out: wp.array(dtype=wp.float32),
                   n_fingers_out: wp.array(dtype=wp.int32),
                   admit: wp.array(dtype=wp.int32),
                   t_I: wp.array(dtype=wp.float32),
                   tree_reroute_req: wp.array(dtype=wp.int32), tree_brush_req: wp.array(dtype=wp.int32),
                   tree_foot2_drop: wp.array(dtype=wp.int32),
                   t_reroute_req: wp.array(dtype=wp.int32), t_brush_req: wp.array(dtype=wp.int32),
                   t_foot2_drop: wp.array(dtype=wp.int32),
                   finger_served: wp.array(dtype=wp.int32), tree_touch: wp.array(dtype=wp.float32),
                   t_hold: wp.array(dtype=wp.int32)):
    """Fingers, admission flag, per-tree currents and the circuit's finger requests (re-route,
    brush leader, brush-foot drop) into the growth engine's device arrays.

    A served finger's charge is mostly screened: the barrier under it charges up to cancel the
    applied field once a filament feeds it (sigma saturates), so it stops pulling further
    leaders; a touched filament is held (no timer re-route)."""
    k = wp.tid()
    p = params[0]
    if k < dbm.FINGER_MAX:
        finger_dir_out[k] = finger_dir(p, k)
        finger_q_out[k] = p.q_finger * wp.where(finger_served[k] != 0, 1.0 - SERVED_SCREEN, 1.0)
    if k < F_MAX:
        t_reroute_req[k] = tree_reroute_req[k]
        t_brush_req[k] = tree_brush_req[k]
        t_hold[k] = wp.where(tree_touch[k] > 0.0, 1, 0)
    if k < F_MAX * dbm.BRUSH_FEET:
        t_foot2_drop[k] = tree_foot2_drop[k]
    if k == 0:
        n_fingers_out[0] = p.num_fingers
        growing = int(0)
        for t in range(F_MAX):
            st = tree_state[t]
            if st != 0 and st != ATTACHED:
                growing += 1
        nominal = circuit_out[2] > 0.5 and growing < MAX_GROWING
        # an unserved finger without a brush host admits one strike on the touched admittance
        touched = circuit_out[7] > 0.5 and growing == 0
        admit[0] = wp.where(nominal or touched, 1, 0)
    if k < F_MAX:
        # only attached trees draw current; growing trees keep the nominal share so they do not starve
        t_I[k] = wp.where(tree_state[k] == ATTACHED, tree_current[k], dbm.I_DEFAULT)


@wp.kernel
def k_cand_inputs(params: wp.array(dtype=PlasmaParams),
                  grid: gas.GasGrid,
                  cand_pos: wp.array(dtype=wp.vec3), cand_alive: wp.array(dtype=wp.int32),
                  sig_dir: wp.array(dtype=wp.vec3), sig_amp: wp.array(dtype=wp.float32),
                  sig_radius: wp.array(dtype=wp.float32), sig_alive: wp.array(dtype=wp.int32),
                  finger_served: wp.array(dtype=wp.int32),
                  cand_s: wp.array(dtype=wp.float32), cand_T: wp.array(dtype=wp.float32)):
    """Per-candidate surface-charge weight and absolute gas temperature (reduced-field gate)."""
    c = wp.tid()
    if cand_alive[c] == 0:
        cand_s[c] = 1.0
        cand_T[c] = 0.0
        return
    x = cand_pos[c]
    p = params[0]
    s = circuit.sigma_weight(x, p.quincunx, sig_dir, sig_amp, sig_radius, sig_alive)
    if s != 1.0 and p.num_fingers > 0:
        # a finger's ground plane dominates the local field: the barrier-charge inhibition fades
        # under a contact no filament serves yet (a served one is charged by its owner's foot,
        # which is what keeps a second filament from landing there)
        n = x / wp.max(wp.length(x), 1.0e-6)
        cov = float(0.0)
        for f in range(wp.min(p.num_fingers, 10)):
            if finger_served[f] == 0:
                cov = wp.max(cov, circuit.finger_cov(p, f, n))
        s = 1.0 + (s - 1.0) * (1.0 - wp.min(cov, 1.0))
    cand_s[c] = s
    cand_T[c] = wp.max(gas.sample_T(grid, x), 1.0)


@wp.func
def stencil_velocity(u: wp.array3d(dtype=wp.vec3), n: int, origin: wp.vec3, inv_dx: float, x: wp.vec3) -> wp.vec3:
    """Mean gas velocity over a 7-point stencil whose radius shrinks to the distance to the nearest
    wall (a channel next to the electrode or the glass sits in the no-slip layer and must not be
    dragged by fluid 8 mm away). Array-based sampler: a struct-by-value sampler called 7 times per
    thread overflows the CUDA stack."""
    r = wp.length(x)
    rad = wp.clamp(wp.min(r - R1, R2I - r), 0.0, ADVECT_RADIUS)
    v = gas.velocity(u, n, origin, inv_dx, x)
    v += gas.velocity(u, n, origin, inv_dx, x + wp.vec3(rad, 0.0, 0.0))
    v += gas.velocity(u, n, origin, inv_dx, x - wp.vec3(rad, 0.0, 0.0))
    v += gas.velocity(u, n, origin, inv_dx, x + wp.vec3(0.0, rad, 0.0))
    v += gas.velocity(u, n, origin, inv_dx, x - wp.vec3(0.0, rad, 0.0))
    v += gas.velocity(u, n, origin, inv_dx, x + wp.vec3(0.0, 0.0, rad))
    v += gas.velocity(u, n, origin, inv_dx, x - wp.vec3(0.0, 0.0, rad))
    return v / 7.0


@wp.kernel
def k_advect_nodes(params: wp.array(dtype=PlasmaParams),
                   gas_u: wp.array3d(dtype=wp.vec3), gas_n: int, gas_origin: wp.vec3, gas_inv_dx: float,
                   node_pos: wp.array(dtype=wp.vec3),
                   node_flags: wp.array(dtype=wp.int32),
                   node_tree: wp.array(dtype=wp.int32),
                   node_parent: wp.array(dtype=wp.int32),
                   tree_state: wp.array(dtype=wp.int32),
                   tree_targets: wp.array(dtype=wp.vec3)):
    """Persistent channels ride the gas: x += u(x) dt; roots slide on the electrode (driven by
    the flow just above the no-slip layer), feet walk on the glass, everything stays inside the
    annulus. Under a finger the foot and the last FOLLOW_LEN of the channel are pulled towards
    the finger (time constant FOLLOW_TAU), so the filament follows a moving finger.

    The whole channel, foot included, rises with its own buoyant plume (the gas grid cannot
    resolve it): the vertical drift is floored at PLUME_MIN_RISE everywhere, so the feet walk up
    the glass at a few mm/s (recording: median +0.6, upper quartile +6 mm/s) while the middle
    rises faster and the channel eventually breaks and re-strikes higher."""
    i = wp.tid()
    p = params[0]
    f = node_flags[i]
    if (f & NODE_ALIVE) == 0 or p.running == 0:
        return
    x = node_pos[i]
    rr = wp.length(x)
    if rr > R2I - FOLLOW_LEN and (f & NODE_ROOT) == 0:
        t = node_tree[i]
        if tree_state[t] == ATTACHED:
            n = x / wp.max(rr, 1.0e-6)
            best = wp.vec3(0.0, 0.0, 0.0)
            best_ang = float(FOLLOW_ANGLE)
            for m in range(dbm.BRUSH_FEET + 1):
                tg = tree_targets[t * (dbm.BRUSH_FEET + 1) + m]
                if wp.length_sq(tg) > 0.5:
                    ang = wp.acos(wp.clamp(wp.dot(n, tg), -1.0, 1.0))
                    if ang < best_ang:
                        best_ang = ang
                        best = tg
            if wp.length_sq(best) > 0.5:
                w = (rr - (R2I - FOLLOW_LEN)) / FOLLOW_LEN
                rate = wp.min(1.0, p.dt / FOLLOW_TAU) * w * w
                x = x + (best * rr - x) * rate
    r0 = wp.max(wp.length(x), 1.0e-6)
    drift = wp.pow(wp.min(r0 / R2I, 1.0), DRIFT_EXPONENT)
    # the foot is held by its surface-charge footprint: no sliding within FOOT_PIN_LEN of the glass
    # (the return flow along the cold wall dragged the feet down while the channel rose; in the
    # recordings the channel rises, breaks and re-strikes with the foot higher up)
    tp = wp.clamp((R2I - r0) / FOOT_PIN_LEN, 0.0, 1.0)
    drift = drift * (FOOT_CREEP + (1.0 - FOOT_CREEP) * tp * tp * (3.0 - 2.0 * tp))   # the foot creeps, it does not slide
    if (f & NODE_ROOT) != 0:
        # the roots walk up the anode (recording: ~2 mm/s, 70 % of crossings upward)
        xs = x * ((R1 + ROOT_SAMPLE_OFFSET * NODE_SPACING) / r0)
        vr = gas.velocity(gas_u, gas_n, gas_origin, gas_inv_dx, xs)
        vr = wp.vec3(vr[0], wp.max(vr[1] * p.g_sign, PLUME_MIN_RISE) * p.g_sign, vr[2])
        x = x + vr * (ROOT_DRIFT_GAIN * drift * p.dt)
        node_pos[i] = x * ((R1 + NODE_SPACING) / wp.max(wp.length(x), 1.0e-6))
        return
    v = stencil_velocity(gas_u, gas_n, gas_origin, gas_inv_dx, x)
    # a channel sits inside its own buoyant plume (dT ~ 100 K over ~1 mm: it rises at cm/s relative
    # to the ambient) which the 2.5 mm gas grid cannot resolve; the grid's return flow along the
    # cold glass (-1 cm/s) must not drag it down: its vertical drift is at least PLUME_MIN_RISE
    v = wp.vec3(v[0], wp.max(v[1] * p.g_sign, PLUME_MIN_RISE) * p.g_sign, v[2])
    x = x + v * (drift * p.dt)
    r = wp.length(x)
    if (f & NODE_FOOT) != 0:
        x = x * ((R2I - CHARGE_RADIUS) / wp.max(r, 1.0e-6))
    else:
        lo = R1 + CHARGE_RADIUS
        hi = R2I - CHARGE_RADIUS
        if r < lo:
            x = x * (lo / wp.max(r, 1.0e-6))
        elif r > hi:
            x = x * (hi / r)
    node_pos[i] = x


@wp.kernel
def k_tree_frame_geometry(tree_state: wp.array(dtype=wp.int32),
                          tree_foot: wp.array(dtype=wp.int32), tree_tip: wp.array(dtype=wp.int32),
                          tree_root: wp.array(dtype=wp.int32), tree_L: wp.array(dtype=wp.float32),
                          node_pos: wp.array(dtype=wp.vec3), node_parent: wp.array(dtype=wp.int32),
                          tree_foot_dir: wp.array(dtype=wp.vec3), tree_root_dir: wp.array(dtype=wp.vec3),
                          tree_chord: wp.array(dtype=wp.float32), tree_stretch: wp.array(dtype=wp.float32),
                          tree_foot2: wp.array(dtype=wp.int32), tree_foot2_dir: wp.array(dtype=wp.vec3)):
    """After advection: unit directions of foot/root (and of the brush feet), chord, and the
    engine's re-route contract t_stretch = geometric main-channel length / arc length at
    attachment (fires at > 1.5)."""
    k = wp.tid()
    for j in range(dbm.BRUSH_FEET):
        fs = k * dbm.BRUSH_FEET + j
        n2 = tree_foot2[fs]
        if tree_state[k] != 0 and n2 >= 0:
            p2 = node_pos[n2]
            tree_foot2_dir[fs] = p2 / wp.max(wp.length(p2), 1.0e-6)
        else:
            tree_foot2_dir[fs] = wp.vec3(0.0, 0.0, 0.0)
    if tree_state[k] == 0:
        return
    end = tree_foot[k]
    if end < 0:
        end = tree_tip[k]
    root = tree_root[k]
    if end < 0 or root < 0:
        return
    pe = node_pos[end]
    pr = node_pos[root]
    tree_foot_dir[k] = pe / wp.max(wp.length(pe), 1.0e-6)
    tree_root_dir[k] = pr / wp.max(wp.length(pr), 1.0e-6)
    tree_chord[k] = wp.max(wp.length(pe - pr), 1.0e-4)
    length = float(0.0)
    n = end
    steps = int(0)
    while n >= 0 and steps < 4096:
        pn = node_parent[n]
        if pn < 0:
            break
        length += wp.length(node_pos[n] - node_pos[pn])
        n = pn
        steps += 1
    if tree_state[k] == ATTACHED and tree_L[k] > 0.0:
        tree_stretch[k] = length / tree_L[k]


@wp.kernel
def k_node_alpha(node_flags: wp.array(dtype=wp.int32), node_stamp: wp.array(dtype=wp.int32),
                 cnt: wp.array(dtype=wp.int32), dt: float, decay_time: float,
                 node_alpha: wp.array(dtype=wp.float32)):
    """Decaying-pool fade for the renderer: 1 while alive, 1 -> 0 over decay_time once decaying."""
    i = wp.tid()
    f = node_flags[i]
    if (f & NODE_DECAYING) != 0:
        age = float(cnt[dbm.CNT_FRAME] - node_stamp[i]) * dt
        node_alpha[i] = wp.clamp(1.0 - age / decay_time, 0.0, 1.0)
    else:
        node_alpha[i] = 1.0


@wp.kernel
def k_heat_sources(node_pos: wp.array(dtype=wp.vec3), node_parent: wp.array(dtype=wp.int32),
                   node_tree: wp.array(dtype=wp.int32), node_flags: wp.array(dtype=wp.int32),
                   tree_current: wp.array(dtype=wp.float32), tree_length: wp.array(dtype=wp.float32),
                   circuit_out: wp.array(dtype=wp.float32),
                   src_p0: wp.array(dtype=wp.vec3), src_p1: wp.array(dtype=wp.vec3),
                   src_q: wp.array(dtype=wp.float32), src_e: wp.array(dtype=wp.float32),
                   src_count: wp.array(dtype=wp.int32)):
    """One Gaussian line source per live charged segment: q' = P_gas (I_k / I_tot) / L_k (W/m)."""
    i = wp.tid()
    if i == 0:
        src_count[0] = 0
    f = node_flags[i]
    p = node_parent[i]
    if (f & NODE_ALIVE) == 0 or (f & NODE_DECAYING) != 0 or p < 0:
        return
    t = node_tree[i]
    i_tot = circuit_out[0]
    ik = tree_current[t]
    if ik <= 0.0 or i_tot <= 0.0 or tree_length[t] <= 0.0:
        return
    slot = wp.atomic_add(src_count, 0, 1)
    if slot >= gas.SEG_MAX:
        return
    src_p0[slot] = node_pos[p]
    src_p1[slot] = node_pos[i]
    q = P_GAS_NOMINAL * (i_tot / I_TOT_NOMINAL) * (ik / i_tot) / tree_length[t]
    src_q[slot] = q
    src_e[slot] = q


@wp.kernel
def k_fix_source_count(src_count: wp.array(dtype=wp.int32)):
    src_count[0] = wp.min(src_count[0], gas.SEG_MAX)


@wp.kernel
def k_node_xion(node_pos: wp.array(dtype=wp.vec3), node_tree: wp.array(dtype=wp.int32),
                node_flags: wp.array(dtype=wp.int32), tree_current: wp.array(dtype=wp.float32),
                node_xion: wp.array(dtype=wp.float32), tree_radius: wp.array(dtype=wp.float32)):
    """Ionised-line fraction along a channel (plan 4.9) and the rendered core radius per tree."""
    i = wp.tid()
    if i < F_MAX:
        tree_radius[i] = wp.clamp(CORE_RADIUS * wp.pow(wp.max(tree_current[i], 1.0e-6) / 5.0e-5, 0.3), 0.2e-3, 1.5e-3)
    if (node_flags[i] & NODE_ALIVE) == 0:
        return
    ik = tree_current[node_tree[i]]
    shaft = wp.clamp(0.93 + 0.05 * wp.log(wp.max(ik, 1.0e-9) / I_STRIKE_REF) / wp.log(10.0), 0.85, 0.97)
    rr = wp.length(node_pos[i])
    # the root flares are pink-white in the footage (the neutral lines of the electrode's glow
    # region): blend towards X_ION_ROOT within X_ION_ROOT_LEN of the bulb
    wr = wp.clamp((rr - R1) / X_ION_ROOT_LEN, 0.0, 1.0)
    shaft = X_ION_ROOT + (shaft - X_ION_ROOT) * wr
    w = wp.clamp((R2I - rr) / X_ION_FOOT_LEN, 0.0, 1.0)
    node_xion[i] = X_ION_FOOT + (shaft - X_ION_FOOT) * w


@wp.kernel
def k_counters(node_flags: wp.array(dtype=wp.int32), tree_state: wp.array(dtype=wp.int32),
               circuit_out: wp.array(dtype=wp.float32), counters: wp.array(dtype=wp.float32)):
    i = wp.tid()
    if i == 0:
        counters[0] = 0.0
        counters[1] = 0.0
        counters[2] = 0.0
        counters[3] = circuit_out[0]
        counters[4] = 0.0
    f = node_flags[i]
    if (f & NODE_ALIVE) != 0:
        wp.atomic_add(counters, 0, 1.0)
    if (f & NODE_DECAYING) != 0:
        wp.atomic_add(counters, 1, 1.0)
    if i < F_MAX:
        if tree_state[i] == ATTACHED:
            wp.atomic_add(counters, 2, 1.0)
        if tree_state[i] != 0:
            wp.atomic_add(counters, 4, 1.0)


class SimFlags:
    """Per-frame toggles the application derives from its State flags."""

    def __init__(self, running=True, invert=False, ice=False, hybrid=True, quincunx=False):
        self.running = running
        self.invert = invert
        self.ice = ice
        self.hybrid = hybrid
        self.quincunx = quincunx


class Globe(Input):
    """See the module docstring. ``fingers.active()`` returns unit directions on the glass;
    ``state_fn`` returns a SimFlags; ``args`` carries seed / gas_res."""

    def __init__(self, camera, fingers, state_fn, args, device="cuda:0"):
        super().__init__("globe")
        self.camera = camera
        self.fingers = fingers
        self.state_fn = state_fn
        self.args = args
        self.device = device
        # eta 6: the near-unbranched rope regime of a plasma-globe filament (the lab's eta 3 gives
        # D ~ 1.7 lightning trees that need ~1000 nodes to cross the 6 cm gap)  CHOSEN, plan 4.3
        self.knobs = {"voltage": VOLT0, "frequency": FREQ0, "eta": ETA_GLOBE, "gamma": dbm.GAMMA, "q_finger": Q_FINGER0}
        self.preset_index = 0
        self.presets = ["video", "ne_xe", "ne", "ar", "kr"]
        self._reset_requested = True
        self.frame = 0
        self.t = 0.0
        self.built = False

    # ---- construction --------------------------------------------------------------------------
    def build(self):
        d = self.device
        n = publish.N_MAX
        self.params = ParamsBuffer(d)
        self.engine = dbm.Dbm(device=d, seed=int(getattr(self.args, "seed", 1)), n_max=n, f_max=F_MAX)
        # a filament stops branching once it reaches the glass (re-routes replace branches); the
        # persistent channel count is then set by the current budget, not by the node capacity
        self.engine.configure(r1=R1, E_bd0=E_BD0_GLOBE, E_ch=E_CH_GLOBE, use_sigma=1, enable_reroute=1,
                              grow_after_attach=0, dt=1.0 / 60.0,
                              E_prop0=E_PROP_FACTOR * E_BD0_GLOBE)
        self.gas = gas.GasSolver(res=int(getattr(self.args, "gas_res", 64)), r1=R1, device=d).allocate()
        self.gas_state = [self.gas.state(), self.gas.state()]
        self.gas_ctrl = self.gas.control()
        self.circuit = circuit.CircuitState(d)
        self.pub = publish.Publisher(d)
        self.node_alpha = wp.ones(n, dtype=wp.float32, device=d)
        self.node_xion = wp.zeros(n, dtype=wp.float32, device=d)
        self.tree_radius = wp.full(F_MAX, CORE_RADIUS, dtype=wp.float32, device=d)
        self.color_neutral = wp.zeros(F_MAX, dtype=wp.vec3, device=d)
        self.color_ion = wp.zeros(F_MAX, dtype=wp.vec3, device=d)
        self.counters = wp.zeros(8, dtype=wp.float32, device=d)
        self.tree_foot2_dir = wp.zeros(F_MAX * dbm.BRUSH_FEET, dtype=wp.vec3, device=d)
        self.counters_host = wp.zeros(8, dtype=wp.float32, device="cpu", pinned=True)
        e = self.engine
        self.nodes = _SoA()
        self.nodes.pos, self.nodes.prev_pos, self.nodes.parent = e.pos, e.prev_pos, e.parent
        self.nodes.tree, self.nodes.flags, self.nodes.s_arc = e.tree, e.flags, e.s_arc
        self.nodes.alpha, self.nodes.x_ion = self.node_alpha, self.node_xion
        self.trees = _SoA()
        self.trees.state, self.trees.foot, self.trees.tip, self.trees.root = e.t_state, e.t_foot, e.t_tip, e.t_root
        self.trees.length, self.trees.chord = e.t_L, e.t_chord
        self.trees.foot_dir, self.trees.root_dir = e.t_foot_dir, e.t_root_dir
        self.trees.foot2 = e.t_foot2
        self.trees.current = self.circuit.tree_current
        self.trees.radius = self.tree_radius
        self.trees.color_neutral, self.trees.color_ion = self.color_neutral, self.color_ion
        self.apply_preset()
        self.built = True

    def apply_preset(self):
        name = self.presets[self.preset_index]
        if name == "video":
            # calibrated on the user's reference footage (linear sRGB): filament shaft violet-blue
            # (peak ~ (120, 110, 210) 8-bit), feet and electrode pink; the spectral presets stay
            # selectable with T
            neutral, ion = VIDEO_NEUTRAL, VIDEO_ION
        else:
            try:
                from plasma import spectra
                pr = spectra.preset(name)
                neutral, ion = pr["neutral_rgb"], pr["ion_rgb"]
            except Exception:
                neutral, ion = (2.884, 0.541, 0.0), (1.167, 0.809, 2.404)
        self.color_neutral.fill_(wp.vec3(*[float(c) for c in neutral]))
        self.color_ion.fill_(wp.vec3(*[float(c) for c in ion]))
        self.preset_name = name
        self.preset_rgb = (tuple(float(c) for c in neutral), tuple(float(c) for c in ion))

    def cycle_preset(self):
        self.preset_index = (self.preset_index + 1) % len(self.presets)
        self.apply_preset()

    def request_reset(self):
        self._reset_requested = True

    # ---- the frame (recorded once into a CUDA graph) -------------------------------------------
    def frame_launches(self):
        d = self.device
        params = self.params.device_array
        e, cs, nodes, trees = self.engine, self.circuit, self.nodes, self.trees
        s0, s1 = self.gas_state
        n = publish.N_MAX
        self.params.upload()
        wp.launch(k_gas_control, dim=1, inputs=[params, self.gas_ctrl.array], device=d)
        self.gas.step(s0, s1, self.gas_ctrl)
        s0.assign(s1)
        circuit.launch_circuit(params, cs, e.t_state, e.t_foot_dir, e.t_L, e.t_chord, d,
                               tree_foot2_dir=self.tree_foot2_dir)
        wp.launch(k_engine_hooks, dim=max(F_MAX * dbm.BRUSH_FEET, dbm.FINGER_MAX),
                  inputs=[params, cs.circuit, e.t_state, cs.tree_current, e.finger_dir, e.finger_q, e.n_fingers,
                          e.admit, e.t_I, cs.tree_reroute_req, cs.tree_brush_req, cs.tree_foot2_drop,
                          e.t_reroute_req, e.t_brush_req, e.t_foot2_drop, cs.finger_served, cs.tree_touch,
                          e.t_hold], device=d)
        wp.launch(k_cand_inputs, dim=e.c_total,
                  inputs=[params, s0.grid, e.cand_pos, e.cand_alive, cs.sig_dir, cs.sig_amp, cs.sig_radius,
                          cs.sig_alive, cs.finger_served, e.cand_s, e.cand_T], device=d)
        e.step()                                      # prev_pos <- pos, growth, lifecycle, conductor solve
        wp.launch(k_advect_nodes, dim=n, inputs=[params, s0.u, s0.grid.n, s0.grid.origin, s0.grid.inv_dx, e.pos, e.flags,
                                                 e.tree, e.parent, e.t_state, cs.tree_targets], device=d)
        wp.launch(k_tree_frame_geometry, dim=F_MAX,
                  inputs=[e.t_state, e.t_foot, e.t_tip, e.t_root, e.t_L, e.pos, e.parent,
                          e.t_foot_dir, e.t_root_dir, e.t_chord, e.t_stretch, e.t_foot2, self.tree_foot2_dir], device=d)
        wp.launch(k_node_alpha, dim=n, inputs=[e.flags, e.stamp, e.cnt, 1.0 / 60.0, dbm.DECAY_TIME, self.node_alpha], device=d)
        wp.launch(k_heat_sources, dim=n,
                  inputs=[e.pos, e.parent, e.tree, e.flags, cs.tree_current, e.t_L, cs.circuit,
                          self.gas.sources.p0, self.gas.sources.p1, self.gas.sources.q, self.gas.sources.e,
                          self.gas.sources.count], device=d)
        wp.launch(k_fix_source_count, dim=1, inputs=[self.gas.sources.count], device=d)
        wp.launch(k_node_xion, dim=n, inputs=[e.pos, e.tree, e.flags, cs.tree_current, self.node_xion, self.tree_radius], device=d)
        self.pub.launch(nodes, trees, pack_volume=False)
        self.gas.pack_volume(s0, self.pub.volume)
        wp.launch(k_counters, dim=n, inputs=[e.flags, e.t_state, cs.circuit, self.counters], device=d)
        wp.copy(self.counters_host, self.counters)

    def reset_launches(self):
        self.engine.reset()
        self.gas.reset(*self.gas_state)
        self.circuit.zero()
        self.node_alpha.fill_(1.0)
        self.node_xion.zero_()
        self.gas.sources.count.zero_()

    def capture(self):
        d = self.device
        self.reset_launches()
        self.frame_launches()                        # warm-up (module loads, scan scratch) outside the capture
        self.reset_launches()
        wp.synchronize()
        with wp.ScopedCapture(device=d) as cap:
            self.frame_launches()
        self.graph = cap.graph
        with wp.ScopedCapture(device=d) as cap:
            self.reset_launches()
        self.reset_graph = cap.graph
        wp.synchronize()

    # ---- Input lifecycle -----------------------------------------------------------------------
    def init(self, width, height):
        if not self.built:
            self.build()
        self.capture()
        print(f"[globe] frame graph captured: gas {self.gas.n}^3, nodes {publish.N_MAX}, trees {F_MAX}, "
              f"S_MAX {self.engine.s_max}, candidates {self.engine.c_total}")

    def fill_params(self):
        flags = self.state_fn()
        h = self.params.host[0]
        running = 1 if flags.running else 0
        h["frame"] = self.frame
        h["seed"] = int(getattr(self.args, "seed", 1))
        h["dt"] = 1.0 / 60.0
        h["time"] = self.t
        h["running"] = running
        h["voltage"] = self.knobs["voltage"]
        h["frequency"] = self.knobs["frequency"]
        h["eta"] = self.knobs["eta"]
        h["gamma"] = self.knobs["gamma"]
        h["q_finger"] = self.knobs["q_finger"]
        h["g_sign"] = -1.0 if flags.invert else 1.0
        h["ice_cap"] = 1 if flags.ice else 0
        h["hybrid"] = 1 if flags.hybrid else 0
        h["quincunx"] = 1 if flags.quincunx else 0
        self.params.set_fingers(self.fingers.active())
        # engine knobs (pinned host arrays, uploaded by the first node of the engine step)
        i_sus = circuit.I_SUSTAIN_RATIO * circuit.I_STRIKE * (1.0 + ((self.knobs["frequency"] - 24.0e3) / 20.0e3) ** 2)
        self.engine.configure(V=self.knobs["voltage"], eta=self.knobs["eta"], gamma=self.knobs["gamma"],
                              hf=dbm.H_F, I_sus=i_sus, enable_reroute=1 if flags.hybrid else 0,
                              pause=0 if running else 1)
        return running

    def pre_render(self, **kwargs):
        if self._reset_requested:
            wp.capture_launch(self.reset_graph)
            self._reset_requested = False
            self.t = 0.0
        running = self.fill_params()
        wp.capture_launch(self.graph)
        if running:
            self.t += 1.0 / 60.0
        self.frame += 1

    def counters_now(self):
        """Counters of the previous frame (pinned mirror, read after a stream sync by the caller)."""
        c = self.counters_host.numpy()
        return dict(live=int(c[0]), decaying=int(c[1]), attached=int(c[2]), i_tot=float(c[3]), trees=int(c[4]))

    def print_counters(self):
        c = self.counters_now()
        print(f"[globe] frame {self.frame}: live nodes {c['live']}, decaying {c['decaying']}, trees {c['trees']} "
              f"(attached {c['attached']}), I_tot {c['i_tot']*1e3:.2f} mA, V {self.knobs['voltage']:.0f} V, "
              f"f {self.knobs['frequency']/1e3:.0f} kHz")

    def dump_state(self, path=None):
        wp.synchronize()
        path = path or f"/tmp/plasma_state_{self.frame:06d}.npz"
        e = self.engine
        np.savez(path, pos=e.pos.numpy(), prev_pos=e.prev_pos.numpy(), parent=e.parent.numpy(), tree=e.tree.numpy(),
                 flags=e.flags.numpy(), s_arc=e.s_arc.numpy(), tree_state=e.t_state.numpy(), tree_foot=e.t_foot.numpy(),
                 tree_tip=e.t_tip.numpy(), tree_root=e.t_root.numpy(), tree_current=self.circuit.tree_current.numpy())
        print(f"[globe] state dumped to {path}")
