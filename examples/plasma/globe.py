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
    T0, FOOT_RADIUS, TAU_ENV, D_SIGMA, C_D,
    ParamsBuffer, PlasmaParams, finger_dir, R1, R2I, CHARGE_RADIUS, NODE_SPACING, FINGER_HWHM,
    VOLT0, FREQ0, Q_FINGER0, V_CH, NODE_ALIVE, NODE_ROOT, NODE_FOOT, NODE_DECAYING,
)

P_GAS_NOMINAL = 0.7           # W total at 1 mA (CHOSEN top-down; gas lab: 0.6-1.3 W/m -> 1.2-1.9 cm/s fronts;
                              # with the drift weighting this gives the ~1 cm/s filament rise of PPPL-4485)
I_TOT_NOMINAL = 1.0e-3
# ---- emission: the corona-model line spectrum of the gas at the local electron temperature
# (plasma.spectra, 2026-09-16). Each node's colour is the gas mixture's spectrum at Te = a (E/N)^b,
# with the reduced field of the channel phase EN_CH x (T / T0) (the hot channel is thinner gas)
# raised near both ends (the sheaths at the electrode and at the dielectric, where the voltage
# drop concentrates), plus a continuum (~ n_e^2, unit-luminance bremsstrahlung colour) whose share
# grows with the current. The luminance of the lines (which the model makes ~30x larger at the
# sheath fields) is NOT used along the channel: the recordings measure the brightness falling
# towards the glass (publisher POWER_FALL), the sheath emission is transient; the table's absolute
# scale drives the electrode glow layer and the gas around the electrode in the tracer.
EN_CH = 3.0                   # Td: channel-phase reduced field (V_CH / L ~ 33 kV/m at 2.4e25 m^-3 is 1.4 Td; CHOSEN 3
                              # with the channel's own heating; calibrated: a Ne/Kr/Xe channel is lavender-blue there)
END_ROOT, L_ROOT = 15.0, 4.0e-3     # electrode-side sheath: E/N x (1 + END_ROOT exp(-d / L_ROOT))        CHOSEN
END_FOOT, L_FOOT = 25.0, 8.0e-3     # dielectric-side sheath (cathode fall: hundreds of Td at the surface); the
                                    # recordings' red tips run 15-20 mm                                  CHOSEN
CORE_T_COEF = 12.0 / 50.0e-6  # K/A: steady conduction heating of the core, dT = I (V_CH/L) ln(b/r) / (2 pi k) ~ 12 K at 50 uA
C_CONT = 1.5                  # continuum luminance / line luminance at I_CONT_REF, ~ (I / I_CONT_REF)^CONT_EXP
CONT_EXP = 0.7                # n_e ~ I / r_core^2 with a core radius ~ I^0.15 (the conducting core is thinner than the glow)
I_CONT_REF = 1.0e-3           # A: a touched channel: ~60 % continuum, white core (the recordings)
EMIS_BINS = 64                # table bins over log Te (plasma.spectra.TE_GRID)
EN_FACE = 60.0                # Td: reduced field in the glow layer under the electrode's envelope     CHOSEN (deep red face)
# morphology per preset (2026-09-16 recordings): the pink / neon globes are 'rope' (15-30 smooth
# unbranched channels), the green globes 'coral' (6-10 channels with 2-3 Y-forks each at 20-45 mm
# from the electrode, branches at 0.4-0.6 of the trunk, many ending in the gas)
MORPHOLOGY = {
    "rope": dict(class_secondary=0.30, class_side=0.02, power_fall=0.5, foot_widen=1.0, taper=0.15, root_gain=1.15,
                 fork_rate=0.0, fork_max=0, roots_max=32),
    # coral: thick bright trunks at the electrode, thin sharp branches and tips (the user's target look):
    # radius x root_gain (R1 / r)^taper along the channel (0.71x at the glass), no foot flare
    "coral": dict(class_secondary=0.50, class_side=0.12, power_fall=0.6, foot_widen=0.0, taper=0.4, root_gain=1.5,
                  fork_rate=5.0, fork_max=3, roots_max=10),
}
PRESET_MORPHOLOGY = {"coral": "coral"}                     # every other preset is a rope globe
PRESET_POWER_FALL = {"ne": 0.2}                             # neon's feet stay bright red (r3: 96 at the glass vs 92 mid)
PRESET_ALIASES = {"video": "tyrian"}                       # the former footage-calibrated preset
I_STRIKE_REF = 3.0e-5
CORE_RADIUS = 0.45e-3         # rendered core radius at 50 uA; r ~ I^0.3, clamped at 1.5 mm (the cross-section
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
FOOT_CREEP = 0.0              # the gas velocity is zero at a wall (no-slip): an attachment does not ride the gas; it
                              # creeps down the surface-charge gradient (K_SIG) and re-strikes. Inside FOOT_PIN_LEN the
                              # channel's advection is damped to this share (the 2026-09-16 tracker: feet at 0.6 mm/s)
# ---- the attachments as surface discharges on dielectrics (2026-09-16): both the outer glass and the
# electrode's glass envelope charge under an attachment (barrier capacitance), the local field
# collapses and the attachment creeps onto fresh surface, down the surface-charge gradient with a
# mobility K_SIG; the channel a core radius above the surface is dragged by the surface's own thermal
# boundary layer. The electrode is the hottest object in the globe (every current's sheath power lands
# on it): its laminar free-convection layer carries the roots up the ball at mm/s (the recordings'
# 7 mm/s at full power, 3 at low, 70 % upward), the outer glass sits within a kelvin of the gas.
K_SIG = 4.5e-6                # m^2/s: creep mobility down the surface-charge gradient (sigma / sigma_sat per m -> m/s):
                              # ~3 mm/s at the edge of a saturated 1.5 mm footprint                            CHOSEN
V_SHEATH = 300.0              # V: electrode sheath (fall) voltage; P_ball = I_tot V_SHEATH heats the envelope      CHOSEN
C_BL = 0.4                    # boundary-layer velocity scale U = C_BL sqrt(g beta dT 2 R1) (laminar free convection) CHOSEN
DELTA_BL = 5.0                # layer thickness delta = DELTA_BL R1 Gr^-1/4                                        CHOSEN
SIGE_NLON, SIGE_NLAT = 64, 32 # the envelope's surface-charge grid (equirectangular, sigma / sigma_sat)
SIGE_FOOT_R = 1.0e-3          # m: a root's charging footprint on the envelope (the close-ups' spot e-fold 0.55 mm)
SIGE_MEM = 1.0                # relative weight of the memory re-ignition on charged surface (tracer J_MEM)
I_REROUTE_REF = 1.0e-3        # A: the re-strike timer's reference total current (REROUTE_MEAN at this drive)
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
SIG_MAX = circuit.SIG_MAX
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
    ctrl[gas.CTRL_T_ELECTRODE] = p.t_ball


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
                   t_hold: wp.array(dtype=wp.int32), t_stalled: wp.array(dtype=wp.int32),
                   morph: wp.array(dtype=wp.float32), t_foot2: wp.array(dtype=wp.int32)):
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
        # a coral globe forks its channels: a Poisson request (rate morph[0] per second per attached
        # channel) for a fork leader while the channel has fewer than morph[1] secondary feet
        if tree_brush_req[k] == 0 and morph[0] > 0.0 and tree_state[k] == ATTACHED:
            rng = wp.rand_init(p.seed * 7919 + p.frame, k)
            if wp.randf(rng) < morph[0] * p.dt:
                nfeet = int(0)
                for j in range(dbm.BRUSH_FEET):
                    if t_foot2[k * dbm.BRUSH_FEET + j] >= 0:
                        nfeet += 1
                if float(nfeet) < morph[1]:
                    t_brush_req[k] = 2
    if k < F_MAX * dbm.BRUSH_FEET:
        t_foot2_drop[k] = tree_foot2_drop[k]
    if k == 0:
        n_fingers_out[0] = p.num_fingers
        growing = int(0)
        for t in range(F_MAX):
            st = tree_state[t]
            if st != 0 and st != ATTACHED and t_stalled[t] == 0:     # a stalled partial channel does not block a strike
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


@wp.func
def sigma_e_cell(n: wp.vec3):
    """Equirectangular cell coordinates (continuous) of the unit direction n on the envelope."""
    lon = wp.atan2(n[2], n[0])                                  # (-pi, pi]
    lat = wp.asin(wp.clamp(n[1], -1.0, 1.0))                    # [-pi/2, pi/2]
    u = (lon / 6.283185307 + 0.5) * float(SIGE_NLON)
    v = (lat / 3.14159265 + 0.5) * float(SIGE_NLAT)
    return wp.vec2(u, v)


@wp.func
def sigma_e_sample(sig_e: wp.array(dtype=wp.float32), n: wp.vec3) -> float:
    """Bilinear sample of the envelope's surface charge at the unit direction n (longitude wraps)."""
    c = sigma_e_cell(n)
    u = c[0] - 0.5
    v = wp.clamp(c[1] - 0.5, 0.0, float(SIGE_NLAT - 1) - 1.0e-4)
    i0 = int(wp.floor(u))
    j0 = int(wp.floor(v))
    fu = u - float(i0)
    fv = v - float(j0)
    ia = (i0 % SIGE_NLON + SIGE_NLON) % SIGE_NLON
    ib = (ia + 1) % SIGE_NLON
    j1 = wp.min(j0 + 1, SIGE_NLAT - 1)
    return (sig_e[j0 * SIGE_NLON + ia] * (1.0 - fu) * (1.0 - fv) + sig_e[j0 * SIGE_NLON + ib] * fu * (1.0 - fv)
            + sig_e[j1 * SIGE_NLON + ia] * (1.0 - fu) * fv + sig_e[j1 * SIGE_NLON + ib] * fu * fv)


@wp.func
def sigma_e_gradient(sig_e: wp.array(dtype=wp.float32), n: wp.vec3) -> wp.vec3:
    """Tangential gradient of the envelope's surface charge (per metre) at the unit direction n."""
    up = wp.vec3(0.0, 1.0, 0.0)
    if wp.abs(n[1]) > 0.99:
        up = wp.vec3(1.0, 0.0, 0.0)
    t1 = wp.normalize(up - wp.dot(up, n) * n)
    t2 = wp.cross(n, t1)
    h = 0.5e-3 / R1                                              # 0.5 mm step, in radians
    g1 = (sigma_e_sample(sig_e, wp.normalize(n + h * t1)) - sigma_e_sample(sig_e, wp.normalize(n - h * t1))) / (2.0 * h * R1)
    g2 = (sigma_e_sample(sig_e, wp.normalize(n + h * t2)) - sigma_e_sample(sig_e, wp.normalize(n - h * t2))) / (2.0 * h * R1)
    return t1 * g1 + t2 * g2


@wp.func
def sigma_glass_gradient(sig_dir: wp.array(dtype=wp.vec3), sig_amp: wp.array(dtype=wp.float32),
                         sig_alive: wp.array(dtype=wp.int32), n: wp.vec3) -> wp.vec3:
    """Tangential gradient (per metre) of the outer glass's surface charge at the unit direction n:
    the sum of the feet's Gaussian footprints (radius FOOT_RADIUS) from the circuit's records."""
    g = wp.vec3(0.0, 0.0, 0.0)
    s2 = FOOT_RADIUS * FOOT_RADIUS
    for j in range(SIG_MAX):
        if sig_alive[j] == 0 or sig_amp[j] <= 0.0:
            continue
        d = (n - sig_dir[j]) * R2I                               # chord offset on the glass (m)
        d = d - wp.dot(d, n) * n                                 # tangential part
        d2 = wp.dot(d, d)
        if d2 > 25.0 * s2:
            continue
        g = g - d * (sig_amp[j] * wp.exp(-0.5 * d2 / s2) / s2)
    return g


@wp.kernel
def k_sigma_envelope(params: wp.array(dtype=PlasmaParams),
                     tree_state: wp.array(dtype=wp.int32), tree_root_dir: wp.array(dtype=wp.vec3),
                     tree_current: wp.array(dtype=wp.float32),
                     sig_prev: wp.array(dtype=wp.float32), sig_e: wp.array(dtype=wp.float32)):
    """The electrode envelope's surface charge (sigma / sigma_sat on an equirectangular grid): every
    attached root deposits under its footprint at the rate its current gives (the barrier charges
    towards saturation), the charge relaxes with TAU_ENV and spreads with D_SIGMA (Burin 2015), as
    the outer glass's footprints do."""
    idx = wp.tid()
    p = params[0]
    if p.running == 0:
        return
    j = idx // SIGE_NLON
    i = idx - j * SIGE_NLON
    lon = (float(i) + 0.5) / float(SIGE_NLON) * 6.283185307 - 3.14159265
    lat = (float(j) + 0.5) / float(SIGE_NLAT) * 3.14159265 - 1.570796327
    n = wp.vec3(wp.cos(lat) * wp.cos(lon), wp.sin(lat), wp.cos(lat) * wp.sin(lon))
    dt = p.dt
    sigma_sat = C_D * wp.max(p.voltage, 1.0)
    g0 = 1.0 / (2.0 * 3.14159265 * SIGE_FOOT_R * SIGE_FOOT_R)
    rate = float(0.0)
    for k in range(F_MAX):
        if tree_state[k] == 3 and tree_current[k] > 0.0:
            c = wp.dot(n, tree_root_dir[k])
            d2 = 2.0 * R1 * R1 * wp.max(1.0 - c, 0.0)             # chord distance squared on the envelope
            if d2 < 16.0 * SIGE_FOOT_R * SIGE_FOOT_R:
                rate += tree_current[k] * g0 / sigma_sat * wp.exp(-0.5 * d2 / (SIGE_FOOT_R * SIGE_FOOT_R))
    # deposit towards saturation and relaxation (exact over dt), then lateral spreading
    d = 1.0 / TAU_ENV
    amp_eq = rate / (rate + d)
    v = amp_eq + (sig_prev[idx] - amp_eq) * wp.exp(-(rate + d) * dt)
    dl = R1 * 3.14159265 / float(SIGE_NLAT)                      # cell size (m) along the latitude
    a = wp.min(D_SIGMA * dt / (dl * dl), 0.2)
    iw = (i + SIGE_NLON - 1) % SIGE_NLON
    ie = (i + 1) % SIGE_NLON
    jn = wp.max(j - 1, 0)
    js = wp.min(j + 1, SIGE_NLAT - 1)
    lap = sig_prev[j * SIGE_NLON + iw] + sig_prev[j * SIGE_NLON + ie] + sig_prev[jn * SIGE_NLON + i] + sig_prev[js * SIGE_NLON + i] - 4.0 * sig_prev[idx]
    sig_e[idx] = wp.clamp(v + a * lap, 0.0, 1.0)


@wp.kernel
def k_advect_nodes(params: wp.array(dtype=PlasmaParams),
                   gas_u: wp.array3d(dtype=wp.vec3), gas_n: int, gas_origin: wp.vec3, gas_inv_dx: float,
                   node_pos: wp.array(dtype=wp.vec3),
                   node_flags: wp.array(dtype=wp.int32),
                   node_tree: wp.array(dtype=wp.int32),
                   node_parent: wp.array(dtype=wp.int32),
                   tree_state: wp.array(dtype=wp.int32),
                   tree_targets: wp.array(dtype=wp.vec3),
                   tree_radius: wp.array(dtype=wp.float32),
                   sig_e: wp.array(dtype=wp.float32),
                   sig_dir: wp.array(dtype=wp.vec3), sig_amp: wp.array(dtype=wp.float32),
                   sig_alive: wp.array(dtype=wp.int32)):
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
        # a root is a surface discharge on the electrode's glass envelope: it creeps down the
        # envelope's surface-charge gradient (its own footprint charges, the field there collapses,
        # fresh surface attracts it) and the channel a core radius above the surface rides the
        # ball's free-convection boundary layer: u = U sin(theta) 6.75 eta (1 - eta)^2, eta = r_k / delta,
        # tangential towards the top (theta from the bottom stagnation point); the gas velocity at
        # the surface itself is zero
        t = node_tree[i]
        n = x / r0
        up = wp.vec3(0.0, p.g_sign, 0.0)
        t_up = up - wp.dot(up, n) * n
        sin_t = wp.length(t_up)
        vel = wp.vec3(0.0, 0.0, 0.0)
        if sin_t > 1.0e-4:
            t_up = t_up / sin_t
            eta = wp.clamp(tree_radius[t] / wp.max(p.delta_bl, 1.0e-4), 0.0, 1.0)
            vel = t_up * (p.u_bl * sin_t * 6.75 * eta * (1.0 - eta) * (1.0 - eta))
        vel = vel - K_SIG * sigma_e_gradient(sig_e, n)
        x = x + vel * p.dt
        node_pos[i] = x * ((R1 + NODE_SPACING) / wp.max(wp.length(x), 1.0e-6))
        return
    v = stencil_velocity(gas_u, gas_n, gas_origin, gas_inv_dx, x)
    # a channel sits inside its own buoyant plume (dT ~ 100 K over ~1 mm: it rises at cm/s relative
    # to the ambient) which the 2.5 mm gas grid cannot resolve; the grid's return flow along the
    # cold glass (-1 cm/s) must not drag it down: its vertical drift is at least PLUME_MIN_RISE
    v = wp.vec3(v[0], wp.max(v[1] * p.g_sign, PLUME_MIN_RISE) * p.g_sign, v[2])
    x = x + v * (drift * p.dt)
    if (f & NODE_FOOT) != 0:
        # the foot is a surface discharge on the outer glass: pinned by its own footprint charge,
        # creeping down the gradient of the surface charge its neighbours and the old feet left
        x = x - (K_SIG * p.dt) * sigma_glass_gradient(sig_dir, sig_amp, sig_alive, x / wp.max(wp.length(x), 1.0e-6))
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
def k_node_emission(node_pos: wp.array(dtype=wp.vec3), node_tree: wp.array(dtype=wp.int32),
                    node_flags: wp.array(dtype=wp.int32), tree_current: wp.array(dtype=wp.float32),
                    emis_rgb: wp.array(dtype=wp.vec3), emis_params: wp.array(dtype=wp.float32),
                    node_rgb: wp.array(dtype=wp.vec3), node_te: wp.array(dtype=wp.float32),
                    tree_radius: wp.array(dtype=wp.float32)):
    """Per-node emission colour (unit-luminance line spectrum at the local Te + the continuum share,
    per unit current) and the electron temperature; the rendered core radius per tree.

    emis_rgb: the preset's line colour table over log Te (plasma.spectra.emission_table);
    emis_params: [log Te_min, log Te_max, a, b] of Te = a (E/N)^b."""
    i = wp.tid()
    if i < F_MAX:
        # the rendered core radius follows the current (2026-09-16 sweeps: width ratio 1.43 for ~3.5x current)
        tree_radius[i] = wp.clamp(CORE_RADIUS * wp.pow(wp.max(tree_current[i], 1.0e-6) / 5.0e-5, 0.3), 0.2e-3, 1.5e-3)
    if (node_flags[i] & NODE_ALIVE) == 0:
        return
    ik = wp.max(tree_current[node_tree[i]], 1.0e-9)
    rr = wp.length(node_pos[i])
    # reduced field along the channel: channel phase x thinner hot gas x the sheaths at both ends
    t_gas = T0 + CORE_T_COEF * ik
    en = EN_CH * (t_gas / T0) * (1.0 + END_ROOT * wp.exp(-wp.max(rr - R1, 0.0) / L_ROOT)
                                 + END_FOOT * wp.exp(-wp.max(R2I - rr, 0.0) / L_FOOT))
    te = emis_params[2] * wp.pow(en, emis_params[3])
    # table lookup (linear in log Te)
    u = (wp.log(te) - emis_params[0]) / (emis_params[1] - emis_params[0]) * float(EMIS_BINS - 1)
    u = wp.clamp(u, 0.0, float(EMIS_BINS - 1) - 1.0e-3)
    j0 = int(u)
    fr = u - float(j0)
    c = emis_rgb[j0] * (1.0 - fr) + emis_rgb[j0 + 1] * fr
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    lines = c / wp.max(lum, 1.0e-9)                                    # unit-luminance line colour
    cont = C_CONT * wp.pow(ik / I_CONT_REF, CONT_EXP)                  # continuum share (~ n_e^2 / n_e)
    node_rgb[i] = lines + wp.vec3(1.0086, 0.9775, 1.1981) * cont       # bremsstrahlung colour (unit luminance)
    node_te[i] = te


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


def electrode_thermal(i_tot):
    """(surface excess K, boundary-layer velocity scale m/s, thickness m) of the electrode heated by
    P = I_tot V_SHEATH in neon at 740 Torr (gas constants from plasma.gas), laminar free convection."""
    power = i_tot * V_SHEATH
    area = 4.0 * np.pi * R1 * R1
    beta = 1.0 / T0
    pr = gas.NU / gas.ALPHA
    dT = 10.0
    for _ in range(6):
        ra = max(gas.G * beta * dT * (2.0 * R1) ** 3 / (gas.NU * gas.ALPHA), 1e-6)
        nu = 2.0 + 0.589 * ra ** 0.25 / (1.0 + (0.469 / pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)
        h_conv = nu * gas.K_GAS / (2.0 * R1)
        dT = power / max(h_conv * area, 1e-9)
    dT = float(min(dT, 400.0))
    gr = max(gas.G * beta * dT * R1 ** 3 / gas.NU ** 2, 1e-6)
    u_bl = C_BL * np.sqrt(gas.G * beta * dT * 2.0 * R1)
    delta = DELTA_BL * R1 * gr ** -0.25
    return dT, float(u_bl), float(min(delta, 0.05))


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
        self.presets = ["tyrian", "ne_xe", "ne", "ar", "kr", "coral"]
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
        self.node_rgb = wp.zeros(n, dtype=wp.vec3, device=d)                 # emission colour per unit current (k_node_emission)
        self.node_te = wp.zeros(n, dtype=wp.float32, device=d)               # electron temperature (eV)
        self.tree_radius = wp.full(F_MAX, CORE_RADIUS, dtype=wp.float32, device=d)
        self.emis_rgb = wp.zeros(EMIS_BINS, dtype=wp.vec3, device=d)         # the preset's line colour table over log Te
        self.emis_params = wp.zeros(4, dtype=wp.float32, device=d)           # [log Te_min, log Te_max, a, b]
        self.morph = wp.zeros(4, dtype=wp.float32, device=d)                # [fork rate (1/s), fork max, roots max, -] (k_engine_hooks, circuit)
        self.sig_e = wp.zeros(SIGE_NLON * SIGE_NLAT, dtype=wp.float32, device=d)       # the envelope's surface charge
        self.sig_e_prev = wp.zeros(SIGE_NLON * SIGE_NLAT, dtype=wp.float32, device=d)
        self.counters = wp.zeros(8, dtype=wp.float32, device=d)
        self.tree_foot2_dir = wp.zeros(F_MAX * dbm.BRUSH_FEET, dtype=wp.vec3, device=d)
        self.counters_host = wp.zeros(8, dtype=wp.float32, device="cpu", pinned=True)
        e = self.engine
        self.nodes = _SoA()
        self.nodes.pos, self.nodes.prev_pos, self.nodes.parent = e.pos, e.prev_pos, e.parent
        self.nodes.tree, self.nodes.flags, self.nodes.s_arc = e.tree, e.flags, e.s_arc
        self.nodes.alpha, self.nodes.rgb, self.nodes.te = self.node_alpha, self.node_rgb, self.node_te
        self.trees = _SoA()
        self.trees.state, self.trees.foot, self.trees.tip, self.trees.root = e.t_state, e.t_foot, e.t_tip, e.t_root
        self.trees.length, self.trees.chord = e.t_L, e.t_chord
        self.trees.foot_dir, self.trees.root_dir = e.t_foot_dir, e.t_root_dir
        self.trees.foot2 = e.t_foot2
        self.trees.current = self.circuit.tree_current
        self.trees.radius = self.tree_radius
        self.apply_preset()
        self.built = True

    def apply_preset(self):
        """The gas preset: its corona-model emission table (plasma.spectra), the anode colours the
        tracer's discs and spots use, and the morphology of its channels."""
        from plasma import spectra
        name = self.presets[self.preset_index]
        table = spectra.emission_table(name)
        te = table["te"]
        self.emis_rgb.assign(np.ascontiguousarray(table["rgb"], dtype=np.float32))
        a, b = table["te_law"]
        self.emis_params.assign(np.array([np.log(te[0]), np.log(te[-1]), a, b], np.float32))
        self.emis_table = table                                     # host copy for the renderer's UBO
        # the anode colours of the tracer's discs and spots (unit max): the glow layer under the
        # envelope (EN_FACE), the gas around the electrode near (20 Td) and far (8 Td), the feet
        def colour_at(en_td):
            c = np.interp(np.log(a * en_td ** b), np.log(te), np.arange(te.size))
            j = int(min(c, te.size - 2)); f = c - j
            rgb = table["rgb"][j] * (1.0 - f) + table["rgb"][j + 1] * f
            return tuple(float(v) for v in rgb / max(float(rgb.max()), 1e-9))
        self.anode_rgb = (colour_at(EN_FACE), colour_at(8.0), colour_at(20.0), colour_at(EN_CH * (1.0 + END_FOOT)))
        self.preset_name = name
        self.EN_FACE = EN_FACE
        morph = dict(MORPHOLOGY[PRESET_MORPHOLOGY.get(name, "rope")])
        morph["power_fall"] = PRESET_POWER_FALL.get(name, morph["power_fall"])
        self.morphology = morph
        self.pub.set_look(class_secondary=morph["class_secondary"], class_side=morph["class_side"],
                          power_fall=morph["power_fall"], foot_widen=morph["foot_widen"],
                          taper=morph["taper"], root_gain=morph["root_gain"])
        self.morph.assign(np.array([morph["fork_rate"], morph["fork_max"], morph["roots_max"], 0.0], np.float32))

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
        wp.launch(gas.k_electrode_temperature, dim=(s0.grid.n, s0.grid.n, s0.grid.n), inputs=[s0.grid, self.gas_ctrl.array], device=d)
        self.gas.step(s0, s1, self.gas_ctrl)
        s0.assign(s1)
        circuit.launch_circuit(params, cs, e.t_state, e.t_foot_dir, e.t_L, e.t_chord, d,
                               tree_foot2_dir=self.tree_foot2_dir, morph=self.morph)
        wp.launch(k_engine_hooks, dim=max(F_MAX * dbm.BRUSH_FEET, dbm.FINGER_MAX),
                  inputs=[params, cs.circuit, e.t_state, cs.tree_current, e.finger_dir, e.finger_q, e.n_fingers,
                          e.admit, e.t_I, cs.tree_reroute_req, cs.tree_brush_req, cs.tree_foot2_drop,
                          e.t_reroute_req, e.t_brush_req, e.t_foot2_drop, cs.finger_served, cs.tree_touch,
                          e.t_hold, e.t_stalled, self.morph, e.t_foot2], device=d)
        wp.launch(k_cand_inputs, dim=e.c_total,
                  inputs=[params, s0.grid, e.cand_pos, e.cand_alive, cs.sig_dir, cs.sig_amp, cs.sig_radius,
                          cs.sig_alive, cs.finger_served, e.cand_s, e.cand_T], device=d)
        e.step()                                      # prev_pos <- pos, growth, lifecycle, conductor solve
        wp.copy(self.sig_e_prev, self.sig_e)
        wp.launch(k_sigma_envelope, dim=SIGE_NLON * SIGE_NLAT,
                  inputs=[params, e.t_state, e.t_root_dir, cs.tree_current, self.sig_e_prev, self.sig_e], device=d)
        wp.launch(k_advect_nodes, dim=n, inputs=[params, s0.u, s0.grid.n, s0.grid.origin, s0.grid.inv_dx, e.pos, e.flags,
                                                 e.tree, e.parent, e.t_state, cs.tree_targets, self.tree_radius, self.sig_e,
                                                 cs.sig_dir, cs.sig_amp, cs.sig_alive], device=d)
        wp.launch(k_tree_frame_geometry, dim=F_MAX,
                  inputs=[e.t_state, e.t_foot, e.t_tip, e.t_root, e.t_L, e.pos, e.parent,
                          e.t_foot_dir, e.t_root_dir, e.t_chord, e.t_stretch, e.t_foot2, self.tree_foot2_dir], device=d)
        wp.launch(k_node_alpha, dim=n, inputs=[e.flags, e.stamp, e.cnt, 1.0 / 60.0, dbm.DECAY_TIME, self.node_alpha], device=d)
        wp.launch(k_heat_sources, dim=n,
                  inputs=[e.pos, e.parent, e.tree, e.flags, cs.tree_current, e.t_L, cs.circuit,
                          self.gas.sources.p0, self.gas.sources.p1, self.gas.sources.q, self.gas.sources.e,
                          self.gas.sources.count], device=d)
        wp.launch(k_fix_source_count, dim=1, inputs=[self.gas.sources.count], device=d)
        wp.launch(k_node_emission, dim=n, inputs=[e.pos, e.tree, e.flags, cs.tree_current, self.emis_rgb, self.emis_params,
                                                  self.node_rgb, self.node_te, self.tree_radius], device=d)
        self.pub.launch(nodes, trees, pack_volume=False)
        self.gas.pack_volume(s0, self.pub.volume)
        wp.launch(k_counters, dim=n, inputs=[e.flags, e.t_state, cs.circuit, self.counters], device=d)
        wp.copy(self.counters_host, self.counters)

    def reset_launches(self):
        self.engine.reset()
        self.gas.reset(*self.gas_state)
        self.circuit.zero()
        self.node_alpha.fill_(1.0)
        self.node_rgb.zero_()
        self.node_te.zero_()
        self.sig_e.zero_()
        self.sig_e_prev.zero_()
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
        # the electrode as a heated sphere: its sheath power P = I_tot V_SHEATH leaves by laminar free
        # convection (Churchill's sphere correlation Nu = 2 + 0.589 Ra^1/4 / (1 + (0.469/Pr)^9/16)^4/9,
        # solved for the surface excess), which sets the boundary layer the roots ride
        i_prev = max(float(self.counters_host.numpy()[3]), 0.0)
        t_ball, u_bl, delta_bl = electrode_thermal(i_prev)
        h["t_ball"] = t_ball
        h["u_bl"] = u_bl
        h["delta_bl"] = delta_bl
        self.params.set_fingers(self.fingers.active())
        # engine knobs (pinned host arrays, uploaded by the first node of the engine step)
        i_sus = circuit.I_SUSTAIN_RATIO * circuit.I_STRIKE * (1.0 + ((self.knobs["frequency"] - 24.0e3) / 20.0e3) ** 2)
        # the re-strike rate follows the drive (2026-09-16 sweeps: 3.5 / s per filament at full power,
        # 1.5 / s at low): the Poisson mean scales with (I_ref / I_tot)^0.5 on the previous frame's total
        i_tot = max(float(self.counters_host.numpy()[3]), 1.0e-5)
        reroute_mean = dbm.REROUTE_MEAN * min(max((I_REROUTE_REF / i_tot) ** 0.5, 0.6), 1.8)
        self.engine.configure(V=self.knobs["voltage"], eta=self.knobs["eta"], gamma=self.knobs["gamma"],
                              hf=dbm.H_F, I_sus=i_sus, enable_reroute=1 if flags.hybrid else 0,
                              pause=0 if running else 1, reroute_mean=reroute_mean)
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

    def dump_state(self, path=None, **extra):
        """Dump the simulation arrays (plus any extra arrays, e.g. the camera pose) to an .npz."""
        wp.synchronize()
        path = path or f"/tmp/plasma_state_{self.frame:06d}.npz"
        e = self.engine
        np.savez(path, pos=e.pos.numpy(), prev_pos=e.prev_pos.numpy(), parent=e.parent.numpy(), tree=e.tree.numpy(),
                 flags=e.flags.numpy(), s_arc=e.s_arc.numpy(), tree_state=e.t_state.numpy(), tree_foot=e.t_foot.numpy(),
                 tree_tip=e.t_tip.numpy(), tree_root=e.t_root.numpy(), tree_current=self.circuit.tree_current.numpy(),
                 **extra)
        print(f"[globe] state dumped to {path}")
